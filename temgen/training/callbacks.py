"""
callbacks.py

PyTorch Lightning callbacks for TEMGen contrastive learning.

Two components:

    1. RetrievalAccuracyCallback
       Runs full-set retrieval accuracy on the test DataLoader every N epochs.
       Logs test/retrieval_top1, top5, top10 — these are the *real* metrics
       (ranking among all N test samples, not just the in-batch B samples).

    2. get_checkpoint_callback()
       Factory that returns a properly configured ModelCheckpoint monitoring
       val/retrieval_top1 (in-batch, computed every epoch in validation_step).

Why a separate callback for test-set retrieval?
    The validation_step already logs in-batch retrieval accuracy every epoch,
    but that ranks among B=256 candidates. Full-set evaluation collects ALL
    embeddings from the test set (N=252), builds the full (N,N) similarity
    matrix, and ranks against every candidate. This is more expensive, so
    we run it less frequently (e.g. every 5 epochs).

Usage:
    from temgen.training.callbacks import (
        RetrievalAccuracyCallback,
        get_checkpoint_callback,
    )

    callbacks = [
        RetrievalAccuracyCallback(
            test_loader       = test_loader,
            eval_every_n_epochs = 5,
            ks                = (1, 5, 10),
        ),
        get_checkpoint_callback(
            dirpath = "checkpoints/",
            monitor = "val/retrieval_top1",
        ),
    ]
    trainer = pl.Trainer(callbacks=callbacks, ...)
"""

from __future__ import annotations

import logging
from typing import Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from temgen.eval.retrieval import compute_retrieval_accuracy

log = logging.getLogger(__name__)


# ─── Full-set retrieval accuracy callback ─────────────────────────────────────

class RetrievalAccuracyCallback(pl.Callback):
    """
    Evaluate full-set retrieval accuracy on the test DataLoader.

    Runs every `eval_every_n_epochs` epochs (and always on the last epoch).
    Collects all projected embeddings, builds the full (N, N) similarity
    matrix, and logs top-k retrieval accuracy.

    Args:
        test_loader         : DataLoader for the test set
        eval_every_n_epochs : run evaluation every N epochs (default 5)
        ks                  : tuple of k values for top-k accuracy
    """

    def __init__(
        self,
        test_loader: DataLoader,
        eval_every_n_epochs: int = 5,
        ks: Sequence[int] = (1, 5, 10),
    ):
        super().__init__()
        self.test_loader = test_loader
        self.eval_every_n_epochs = eval_every_n_epochs
        self.ks = tuple(ks)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Hook: runs after every validation epoch.

        Only performs full-set evaluation when:
            - current_epoch % eval_every_n_epochs == 0, OR
            - current_epoch == max_epochs - 1 (last epoch)
        """
        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs or 0

        is_eval_epoch = (current_epoch % self.eval_every_n_epochs == 0)
        is_last_epoch = (current_epoch == max_epochs - 1)

        if not (is_eval_epoch or is_last_epoch):
            return

        log.info(
            f"[RetrievalAccuracyCallback] epoch {current_epoch}: "
            f"running full-set evaluation on {len(self.test_loader.dataset)} "
            f"test samples..."
        )

        # Use the underlying model (not the LightningModule) for evaluation
        # so we get the raw model forward dict with z_TEM_proj / z_cell_proj
        metrics = compute_retrieval_accuracy(
            model      = pl_module.model,
            dataloader = self.test_loader,
            ks         = self.ks,
        )

        # Log to Lightning (shows up in TensorBoard / WandB / CSV)
        for k_name, value in metrics.items():
            pl_module.log(
                f"test/retrieval_{k_name}",
                value,
                on_step  = False,
                on_epoch = True,
                prog_bar = (k_name == "top1"),
                sync_dist = True,
            )

        log.info(
            f"[RetrievalAccuracyCallback] epoch {current_epoch}: "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )


# ─── Checkpoint callback factory ──────────────────────────────────────────────

def get_checkpoint_callback(
    dirpath  : str  = "checkpoints/",
    monitor  : str  = "val/retrieval_top1",
    mode     : str  = "max",
    save_top_k: int = 3,
    save_last : bool = True,
    filename  : str = "temgen-epoch{epoch:03d}-top1{val/retrieval_top1:.4f}",
) -> ModelCheckpoint:
    """
    Return a configured ModelCheckpoint callback.

    Monitors val/retrieval_top1 (in-batch, computed every epoch) and saves
    the top-k best checkpoints plus the last checkpoint.

    Args:
        dirpath    : directory for checkpoint files
        monitor    : metric to monitor for best model selection
        mode       : "max" for accuracy, "min" for loss
        save_top_k : keep top-k best checkpoints
        save_last  : always save the last epoch checkpoint
        filename   : checkpoint filename template

    Returns:
        ModelCheckpoint callback instance
    """
    return ModelCheckpoint(
        dirpath    = dirpath,
        monitor    = monitor,
        mode       = mode,
        save_top_k = save_top_k,
        save_last  = save_last,
        filename   = filename,
        verbose    = True,
        auto_insert_metric_name = False,
    )


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("callbacks.py sanity check\n")

    # Verify imports work
    print(f"RetrievalAccuracyCallback : {RetrievalAccuracyCallback}")
    print(f"get_checkpoint_callback   : {get_checkpoint_callback}")

    # Verify checkpoint callback factory
    ckpt_cb = get_checkpoint_callback(dirpath="/tmp/test_ckpt")
    print(f"\nCheckpoint callback:")
    print(f"  dirpath    : {ckpt_cb.dirpath}")
    print(f"  monitor    : {ckpt_cb.monitor}")
    print(f"  mode       : {ckpt_cb.mode}")
    print(f"  save_top_k : {ckpt_cb.save_top_k}")
    print(f"  save_last  : {ckpt_cb.save_last}")

    assert ckpt_cb.monitor == "val/retrieval_top1"
    assert ckpt_cb.mode == "max"

    print("\nAll assertions passed!")
    print("\nNOTE: Full integration test requires a model + test DataLoader.")
    print("Run via: python -c 'from temgen.training.callbacks import *; print(\"OK\")'")