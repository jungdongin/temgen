"""
train.py

Python entry point for TEMGen contrastive learning.

Wires together:
    - Config          : OmegaConf from cuau_101010.yaml
    - Model           : TEMGenModel
    - Lightning module: TEMGenLightningModule
    - Dataloaders     : build_dataloaders from HDF5 files
    - Callbacks       : RetrievalAccuracyCallback + ModelCheckpoint
    - Trainer         : PyTorch Lightning DDP

Called by scripts/train.sh via srun.

Usage:
    # Single-GPU debug (no SLURM):
    python scripts/train.py --config configs/cuau_101010.yaml --gpus-per-node 1

    # Via SLURM (train.sh calls this):
    srun python scripts/train.py --config ... --nodes 1 --gpus-per-node 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

# Ensure temgen package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from temgen.models.temgen_model import TEMGenModel
from temgen.training.lightning_module import TEMGenLightningModule
from temgen.training.callbacks import RetrievalAccuracyCallback, get_checkpoint_callback
from temgen.data.cuau_dataset import build_dataloaders


def main():
    parser = argparse.ArgumentParser(description="TEMGen training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt-dir", default="checkpoints/", help="Checkpoint directory")
    parser.add_argument("--log-dir", default="logs/tensorboard", help="TensorBoard log dir")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per node")
    parser.add_argument("--resume-from", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)

    # ── Dataloaders ───────────────────────────────────────────────────────────
    d_cfg = cfg.data
    train_loader, val_loader, test_loader = build_dataloaders(
        train_h5     = d_cfg.train_h5,
        test_h5      = d_cfg.test_h5,
        val_fraction = d_cfg.val_fraction,
        val_seed     = d_cfg.val_seed,
        batch_size   = cfg.training.batch_size,
        num_workers  = d_cfg.num_workers,
        pin_memory   = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TEMGenModel(cfg)
    lit   = TEMGenLightningModule(model, cfg)

    print(model)
    print(lit)
    print()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        RetrievalAccuracyCallback(
            test_loader         = test_loader,
            eval_every_n_epochs = cfg.training.get("eval_every_n_epochs", 5),
            ks                  = (1, 5, 10),
        ),
        get_checkpoint_callback(
            dirpath    = args.ckpt_dir,
            monitor    = "val/retrieval_top1",
            save_top_k = 3,
            save_last  = True,
        ),
    ]

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir = args.log_dir,
        name     = "temgen",
        version  = None,      # auto-increment
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    total_gpus = args.nodes * args.gpus_per_node

    # DDP strategy for multi-GPU / multi-node
    strategy = "auto"
    if total_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        # Hardware
        accelerator       = "gpu",
        devices           = args.gpus_per_node,
        num_nodes         = args.nodes,
        strategy          = strategy,
        precision         = "16-mixed",          # FP16 autocast

        # Training
        max_epochs        = cfg.training.epochs,
        gradient_clip_val = None,                # handled by lightning_module
        accumulate_grad_batches = cfg.training.get("accumulate_grad_batches", 1),

        # Logging & checkpointing
        logger            = logger,
        callbacks         = callbacks,
        log_every_n_steps = 10,

        # Validation
        val_check_interval = 1.0,                # validate every epoch
        check_val_every_n_epoch = 1,

        # Performance
        enable_progress_bar = True,
        enable_model_summary = True,

        # Determinism
        deterministic     = False,               # True slows things down
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Starting training: {cfg.training.epochs} epochs, "
          f"{total_gpus} GPU(s), strategy={strategy}")
    print(f"  Effective batch size: "
          f"{cfg.training.batch_size} x {total_gpus} x "
          f"{cfg.training.get('accumulate_grad_batches', 1)} = "
          f"{cfg.training.batch_size * total_gpus * cfg.training.get('accumulate_grad_batches', 1)}")
    print()

    trainer.fit(
        model       = lit,
        train_dataloaders = train_loader,
        val_dataloaders   = val_loader,
        ckpt_path   = args.resume_from,          # None = fresh, path = resume
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\nTraining complete.")
    print(f"Best checkpoint: {callbacks[1].best_model_path}")
    print(f"Best val/retrieval_top1: {callbacks[1].best_model_score}")


if __name__ == "__main__":
    main()