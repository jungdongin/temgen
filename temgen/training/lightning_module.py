"""
lightning_module.py

PyTorch Lightning module for TEMGen contrastive learning.

Wraps TEMGenModel and handles:
    - Training / validation step
    - Optimizer: AdamW
    - LR schedule: linear warmup + cosine decay
    - Gradient clipping
    - Mixed precision (FP16 via Lightning's built-in autocast)
    - Logging: loss, temperature τ, learning rate, top-1/top-5 retrieval accuracy
    - Checkpoint: monitors val/retrieval_top1

All hyperparameters come from cfg (OmegaConf DictConfig).

Usage:
    from omegaconf import OmegaConf
    from temgen.models.temgen_model import TEMGenModel
    from temgen.training.lightning_module import TEMGenLightningModule

    cfg     = OmegaConf.load("configs/cuau_101010.yaml")
    model   = TEMGenModel(cfg)
    lit     = TEMGenLightningModule(model, cfg)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig


class TEMGenLightningModule(pl.LightningModule):
    """
    Lightning wrapper for TEMGenModel.

    Args:
        model : TEMGenModel instance
        cfg   : OmegaConf config from configs/cuau_101010.yaml
    """

    def __init__(self, model: torch.nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg   = cfg

        t_cfg = cfg.training
        self.lr            = t_cfg.lr
        self.weight_decay  = t_cfg.weight_decay
        self.betas         = tuple(t_cfg.betas)
        self.eps           = t_cfg.eps
        self.warmup_epochs = t_cfg.warmup_epochs
        self.max_epochs    = t_cfg.epochs
        self.grad_clip     = t_cfg.grad_clip_norm

        # Save hyperparameters to checkpoint (excludes model weights)
        self.save_hyperparameters(ignore=["model"])

    # ── Forward (delegates to model) ──────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        return self.model(batch)

    # ── Shared step ───────────────────────────────────────────────────────────

    def _shared_step(self, batch: dict, stage: str) -> dict:
        """
        Runs forward pass, computes loss + metrics, logs everything.

        Returns dict with loss and projected embeddings for metric computation.
        """
        out = self.model(batch)

        loss        = out["loss"]
        tau         = out["tau"]
        z_TEM_proj  = out["z_TEM_proj"]    # (B, 128)
        z_cell_proj = out["z_cell_proj"]   # (B, 128)

        # ── Retrieval accuracy (top-1, top-5) ─────────────────────────────────
        top1, top5 = self._retrieval_accuracy(z_TEM_proj, z_cell_proj)

        # ── Logging ───────────────────────────────────────────────────────────
        self.log(f"{stage}/loss",             loss,  on_step=(stage=="train"),
                 on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/retrieval_top1",   top1,  on_step=False,
                 on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/retrieval_top5",   top5,  on_step=False,
                 on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/temperature",      tau,   on_step=False,
                 on_epoch=True, prog_bar=False, sync_dist=True)

        if stage == "train":
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False)

        return {"loss": loss}

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")["loss"]

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    # ── Retrieval accuracy ────────────────────────────────────────────────────

    @torch.no_grad()
    def _retrieval_accuracy(
        self,
        z_tem : torch.Tensor,   # (B, 128)
        z_cell: torch.Tensor,   # (B, 128)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image→structure retrieval accuracy within the batch.

        For each TEM embedding T_i, rank all structure embeddings C_j by
        cosine similarity and check if C_i appears in the top-k.

        Returns:
            top1 : scalar  fraction of samples where correct match is rank-1
            top5 : scalar  fraction of samples where correct match is in top-5
        """
        B = z_tem.shape[0]

        z_tem  = F.normalize(z_tem,  dim=-1)
        z_cell = F.normalize(z_cell, dim=-1)

        # Similarity matrix: (B, B)
        sim    = z_tem @ z_cell.T

        # For each row i, rank of correct match C_i (diagonal)
        # argsort descending → position of correct match
        labels  = torch.arange(B, device=sim.device)
        ranks   = (sim >= sim[labels, labels].unsqueeze(1)).sum(dim=1)  # (B,)

        top1 = (ranks == 1).float().mean()
        top5 = (ranks <= 5).float().mean()

        return top1, top5

    # ── Optimiser + LR schedule ───────────────────────────────────────────────

    def configure_optimizers(self):
        """
        AdamW optimiser with linear warmup + cosine decay.

        Warmup: lr increases linearly from 0 → cfg.training.lr over warmup_epochs.
        Decay : cosine annealing from cfg.training.lr → 0 over remaining epochs.
        """
        # Separate weight-decayed and non-decayed parameters
        # Bias, LayerNorm, and embedding weights should NOT have weight decay
        decay_params     = []
        no_decay_params  = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_no_decay(name):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params,    "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr    = self.lr,
            betas = self.betas,
            eps   = self.eps,
        )

        # Steps per epoch — Lightning provides this after setup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self._lr_lambda,
        )

        return {
            "optimizer"  : optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval" : "epoch",
                "frequency": 1,
                "name"     : "lr/cosine_warmup",
            },
        }

    def _lr_lambda(self, epoch: int) -> float:
        """
        LR multiplier for epoch (0-indexed).

        Linear warmup: 0 → 1 over warmup_epochs.
        Cosine decay : 1 → 0 over (max_epochs - warmup_epochs).
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            return float(epoch + 1) / float(max(1, self.warmup_epochs))
        else:
            # Cosine decay
            progress = float(epoch - self.warmup_epochs) / float(
                max(1, self.max_epochs - self.warmup_epochs)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # ── Gradient clipping (Lightning hook) ────────────────────────────────────

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val    : float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Override Lightning's default clipping to use our cfg value."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.grad_clip
        )

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TEMGenLightningModule(\n"
            f"  model          : {type(self.model).__name__}\n"
            f"  optimizer      : AdamW  lr={self.lr}  wd={self.weight_decay}\n"
            f"  schedule       : linear warmup ({self.warmup_epochs} epochs) "
            f"+ cosine decay ({self.max_epochs} epochs total)\n"
            f"  grad_clip_norm : {self.grad_clip}\n"
            f")"
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_no_decay(param_name: str) -> bool:
    """
    Return True if this parameter should NOT have weight decay applied.

    Standard practice: exclude bias, LayerNorm weights/biases, embeddings.
    """
    no_decay_keywords = ("bias", "layer_norm", "layernorm", "ln.", "embedding")
    return any(kw in param_name.lower() for kw in no_decay_keywords)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from temgen.models.temgen_model import TEMGenModel

    cfg   = OmegaConf.load("configs/cuau_101010.yaml")
    model = TEMGenModel(cfg)
    lit   = TEMGenLightningModule(model, cfg)

    print(lit)
    print()

    # Verify LR schedule shape over training
    print("LR schedule (selected epochs):")
    for epoch in [0, 1, 5, 9, 10, 50, 100, 150, 199]:
        mult = lit._lr_lambda(epoch)
        lr   = cfg.training.lr * mult
        print(f"  epoch {epoch:3d}: multiplier={mult:.4f}  lr={lr:.2e}")

    # Verify warmup is linear
    for e in range(cfg.training.warmup_epochs):
        expected = (e + 1) / cfg.training.warmup_epochs
        actual   = lit._lr_lambda(e)
        assert abs(actual - expected) < 1e-6, f"Warmup mismatch at epoch {e}"

    # Verify decay ends at ~0
    final = lit._lr_lambda(cfg.training.epochs - 1)
    assert final < 0.02, f"Final LR multiplier too high: {final}"

    # Verify no-decay detection
    assert _is_no_decay("model.cnn_frontend.bn1.bias")         # BN bias
    assert _is_no_decay("model.geometry_tokens.f_geo.0.bias")  # Linear bias
    assert _is_no_decay("model.structure_encoder.node_embedding.weight")  # embedding
    assert not _is_no_decay("model.cnn_frontend.layer1.0.conv1.weight")   # conv weight

    print("\nno-decay detection ... OK")
    print("All assertions passed!")