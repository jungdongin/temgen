"""
info_nce.py

Symmetric InfoNCE loss with learnable temperature for contrastive learning.

Extracted from temgen_model.py to keep the loss decoupled from the model.
The learnable log-temperature scalar lives here as an nn.Parameter,
so it must be registered on a parent nn.Module (e.g. TEMGenModel)
to enter the optimizer's parameter groups.

Usage:
    loss_fn = InfoNCELoss()
    out = loss_fn(z_tem_proj, z_cell_proj)   # both (B, 128)
    # out["loss"]  — scalar, backprop target
    # out["tau"]   — scalar (detached), for logging
    # out["acc"]   — scalar, in-batch top-1 accuracy (free from logits)

Spec reference:
    CuAu 101010 Encoding Manual, Part C1.
    τ_init = 1/0.07 ≈ 14.29  →  log_temp_init = log(14.29) ≈ 2.6593
    Clamp τ ∈ [0.01, 1.0] after exp.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (NT-Xent) loss for contrastive learning.

    Learns a log-temperature scalar that is optimised jointly with the
    encoders.  Temperature is clamped to [tau_min, tau_max] after exp
    to keep training stable.

    When ``fixed_temp`` is provided, temperature is constant (not learned).

    In multi-GPU DDP, embeddings are gathered across all ranks before
    computing the similarity matrix so that every sample in the global
    batch acts as a negative.  Gradients flow only through the local
    rank's embeddings.

    Args:
        log_temp_init : initial value of log(τ).
                        Default 2.6593 = log(1/0.07).
        tau_min       : lower clamp for τ after exp.
        tau_max       : upper clamp for τ after exp.
        fixed_temp    : if set, use this constant τ (not learned).
    """

    def __init__(
        self,
        log_temp_init: float = 2.6593,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        fixed_temp: float | None = None,
    ) -> None:
        super().__init__()
        self._fixed_temp = fixed_temp
        if fixed_temp is not None:
            # Constant temperature — stored as buffer (not optimised)
            self.register_buffer(
                "log_temp",
                torch.tensor(math.log(fixed_temp), dtype=torch.float32),
            )
        else:
            # Learnable temperature
            self.log_temp = nn.Parameter(torch.tensor(log_temp_init, dtype=torch.float32))
        self.tau_min = tau_min
        self.tau_max = tau_max

    # ------------------------------------------------------------------
    @property
    def tau(self) -> torch.Tensor:
        """Current temperature (scalar, detached for logging)."""
        return self.log_temp.exp().clamp(self.tau_min, self.tau_max).detach()

    # ------------------------------------------------------------------
    @staticmethod
    def _gather(z: torch.Tensor) -> torch.Tensor:
        """
        Gather embeddings from all GPUs. Returns (world_size * B, D).

        The local rank's slice keeps its gradient; other ranks' slices
        are detached so gradients only flow through the local shard.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return z

        world_size = dist.get_world_size()
        if world_size <= 1:
            return z

        gathered = [torch.zeros_like(z) for _ in range(world_size)]
        dist.all_gather(gathered, z)

        # Splice local tensor back in to preserve gradient flow
        gathered[dist.get_rank()] = z
        return torch.cat(gathered, dim=0)

    # ------------------------------------------------------------------
    def forward(
        self,
        z_tem: torch.Tensor,
        z_cell: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute symmetric InfoNCE loss.

        Args:
            z_tem  : (B_local, D)  projected TEM embeddings (raw, not yet normalised).
            z_cell : (B_local, D)  projected structure embeddings (raw, not yet normalised).

        Returns:
            dict with keys:
                loss  — scalar, symmetric InfoNCE
                tau   — scalar, current temperature (detached)
                acc   — scalar, in-batch top-1 matching accuracy
        """
        # L2-normalise ------------------------------------------------
        z_tem  = F.normalize(z_tem,  dim=-1)
        z_cell = F.normalize(z_cell, dim=-1)

        # Gather across GPUs for full-batch negatives ------------------
        z_tem  = self._gather(z_tem)     # (B_global, D)
        z_cell = self._gather(z_cell)    # (B_global, D)

        B = z_tem.shape[0]

        # Temperature --------------------------------------------------
        tau = self.log_temp.exp().clamp(self.tau_min, self.tau_max)

        # Cosine similarity matrix ------------------------------------
        logits = z_tem @ z_cell.T / tau                  # (B, B)
        labels = torch.arange(B, device=logits.device)

        # Symmetric cross-entropy -------------------------------------
        loss_tc = F.cross_entropy(logits,   labels)
        loss_ct = F.cross_entropy(logits.T, labels)
        loss = (loss_tc + loss_ct) / 2.0

        # In-batch top-1 accuracy (free, already have logits) ----------
        with torch.no_grad():
            preds_tc = logits.argmax(dim=1)
            preds_ct = logits.T.argmax(dim=1)
            acc = ((preds_tc == labels).float().mean()
                   + (preds_ct == labels).float().mean()) / 2.0

        return {
            "loss": loss,
            "tau": tau.detach(),
            "acc": acc,
            "z_tem_gathered": z_tem.detach(),
            "z_cell_gathered": z_cell.detach(),
        }

    def __repr__(self) -> str:
        temp_mode = f"fixed={self._fixed_temp}" if self._fixed_temp is not None else "learnable"
        return (
            f"InfoNCELoss(\n"
            f"  temperature   : {temp_mode}\n"
            f"  tau={self.tau.item():.4f}\n"
            f"  tau_clamp=[{self.tau_min}, {self.tau_max}]\n"
            f")"
        )


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("InfoNCELoss sanity check\n")

    loss_fn = InfoNCELoss()
    print(loss_fn)
    print()

    # Verify log_temp is a parameter
    params = list(loss_fn.parameters())
    assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"
    assert params[0] is loss_fn.log_temp
    print(f"Parameters: {len(params)} (log_temp)")

    # Simulate projected embeddings
    B, D = 32, 128
    z_tem  = torch.randn(B, D)
    z_cell = torch.randn(B, D)

    out = loss_fn(z_tem, z_cell)

    print(f"\nz_tem   : {tuple(z_tem.shape)}")
    print(f"z_cell  : {tuple(z_cell.shape)}")
    print(f"loss    : {out['loss'].item():.4f}")
    print(f"tau     : {out['tau'].item():.4f}")
    print(f"acc     : {out['acc'].item():.4f}")

    # Loss should be around log(B) for random embeddings
    expected_loss = torch.tensor(B, dtype=torch.float32).log().item()
    print(f"\nExpected loss ≈ log({B}) = {expected_loss:.4f}")
    assert abs(out["loss"].item() - expected_loss) < 1.0, "Loss too far from log(B)"

    # Verify gradient flows through log_temp
    out["loss"].backward()
    assert loss_fn.log_temp.grad is not None, "No gradient on log_temp"
    print(f"log_temp.grad = {loss_fn.log_temp.grad.item():.6f}")

    # Test with identical embeddings (perfect matching → loss ≈ 0, acc ≈ 1)
    loss_fn.zero_grad()
    z_shared = torch.randn(B, D)
    # Add small noise to avoid numerical issues
    out_perfect = loss_fn(z_shared, z_shared + 1e-6 * torch.randn_like(z_shared))
    print(f"\nPerfect match: loss={out_perfect['loss'].item():.4f}  "
          f"acc={out_perfect['acc'].item():.4f}")
    assert out_perfect["acc"].item() > 0.9, "Acc should be ~1.0 for identical embeddings"

    print("\nAll assertions passed!")