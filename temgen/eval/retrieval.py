"""
retrieval.py

Full-set retrieval accuracy evaluation for TEMGen.

Collects all projected embeddings from a DataLoader, builds the full
(N, N) cosine similarity matrix, and computes top-k retrieval accuracy.

Two use cases:
    1. Called by training/callbacks.py every N epochs on the test set
    2. Run standalone for post-training evaluation

Why not just use in-batch accuracy?
    In-batch accuracy ranks among B=256 candidates.  Full-set accuracy
    ranks among all N candidates (e.g. N=252 test, N=15502 train).
    The full-set metric is much harder and more meaningful.

Memory note:
    For N=252  (test):  similarity matrix = 252² × 4B ≈ 0.25 MB  — trivial
    For N=15502 (train): similarity matrix = 15502² × 4B ≈ 0.96 GB — fits A100

Usage:
    from temgen.eval.retrieval import compute_retrieval_accuracy

    metrics = compute_retrieval_accuracy(model, test_loader, ks=(1, 5, 10))
    # metrics = {"top1": 0.xx, "top5": 0.xx, "top10": 0.xx}

    # Or standalone:
    python -m temgen.eval.retrieval \\
        --checkpoint path/to/best.ckpt \\
        --config configs/cuau_101010.yaml \\
        --test-h5 data/hdf5/test_20260304.h5
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the model on every batch and collect projected embeddings.

    Args:
        model      : TEMGenModel (or LightningModule wrapping it).
                     Must return dict with "z_TEM_proj" and "z_cell_proj".
        dataloader : DataLoader yielding batch dicts.

    Returns:
        all_z_tem  : (N, D_proj)  float32, on CPU
        all_z_cell : (N, D_proj)  float32, on CPU
    """
    model.eval()

    z_tem_list  = []
    z_cell_list = []

    for batch in dataloader:
        # Move tensors to model device
        device = next(model.parameters()).device

        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device, non_blocking=True)
            else:
                batch_gpu[k] = v

        out = model(batch_gpu)

        z_tem_list.append(out["z_TEM_proj"].cpu())
        z_cell_list.append(out["z_cell_proj"].cpu())

    all_z_tem  = torch.cat(z_tem_list,  dim=0)    # (N, D_proj)
    all_z_cell = torch.cat(z_cell_list, dim=0)    # (N, D_proj)

    return all_z_tem, all_z_cell


def retrieval_metrics(
    z_tem : torch.Tensor,     # (N, D)
    z_cell: torch.Tensor,     # (N, D)
    ks    : Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """
    Compute top-k retrieval accuracy from pre-collected embeddings.

    For each TEM embedding i, rank all structure embeddings j by cosine
    similarity and check whether the correct match j=i falls in the top-k.

    Computes both directions (TEM→struct, struct→TEM) and averages.

    Args:
        z_tem  : (N, D)  projected TEM embeddings
        z_cell : (N, D)  projected structure embeddings
        ks     : tuple of k values to evaluate

    Returns:
        dict like {"top1": 0.xx, "top5": 0.xx, "top10": 0.xx}
    """
    # L2-normalise
    z_tem  = F.normalize(z_tem,  dim=-1)
    z_cell = F.normalize(z_cell, dim=-1)

    # Full similarity matrix: (N, N)
    sim = z_tem @ z_cell.T

    N = sim.shape[0]
    labels = torch.arange(N, device=sim.device)

    metrics = {}
    for k in ks:
        # TEM → structure: for each row, check if correct column is in top-k
        topk_tc = sim.topk(min(k, N), dim=1).indices          # (N, k)
        hits_tc = (topk_tc == labels.unsqueeze(1)).any(dim=1)  # (N,)

        # Structure → TEM: for each column, check if correct row is in top-k
        topk_ct = sim.T.topk(min(k, N), dim=1).indices        # (N, k)
        hits_ct = (topk_ct == labels.unsqueeze(1)).any(dim=1)  # (N,)

        # Average both directions
        acc = (hits_tc.float().mean() + hits_ct.float().mean()) / 2.0
        metrics[f"top{k}"] = acc.item()

    return metrics


@torch.no_grad()
def compute_retrieval_accuracy(
    model     : nn.Module,
    dataloader: DataLoader,
    ks        : Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """
    End-to-end: collect embeddings from dataloader, then compute metrics.

    This is the main entry point for callbacks and standalone evaluation.

    Args:
        model      : TEMGenModel or LightningModule
        dataloader : test (or val) DataLoader
        ks         : tuple of k values

    Returns:
        dict like {"top1": 0.xx, "top5": 0.xx, "top10": 0.xx}
    """
    was_training = model.training
    model.eval()

    all_z_tem, all_z_cell = collect_embeddings(model, dataloader)
    metrics = retrieval_metrics(all_z_tem, all_z_cell, ks=ks)

    if was_training:
        model.train()

    return metrics


# ─── Standalone CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from omegaconf import OmegaConf
    from temgen.models.temgen_model import TEMGenModel
    from temgen.training.lightning_module import TEMGenLightningModule
    from temgen.data.cuau_dataset import build_dataloaders

    parser = argparse.ArgumentParser(description="Standalone retrieval evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--config", default="configs/cuau_101010.yaml")
    parser.add_argument("--test-h5", required=True, help="Path to test HDF5 file")
    parser.add_argument("--train-h5", default=None, help="Path to train HDF5 (for val split)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model from checkpoint
    model = TEMGenModel(cfg)
    lit = TEMGenLightningModule.load_from_checkpoint(
        args.checkpoint,
        model=model,
        cfg=cfg,
    )
    lit = lit.to(device)
    lit.eval()

    # Build test loader
    # We only need the test loader; pass a dummy train_h5 if not provided
    if args.train_h5:
        _, _, test_loader = build_dataloaders(
            train_h5    = args.train_h5,
            test_h5     = args.test_h5,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
        )
    else:
        # Build test-only loader
        from temgen.data.cuau_dataset import CuAuHDF5Dataset, cuau_collate_fn
        test_ds = CuAuHDF5Dataset(args.test_h5)
        test_loader = DataLoader(
            test_ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            collate_fn  = cuau_collate_fn,
            pin_memory  = True,
        )
        print(f"[eval] test={len(test_ds):,} samples")

    # Evaluate
    print(f"\nEvaluating on {len(test_loader.dataset):,} samples...")
    metrics = compute_retrieval_accuracy(lit.model, test_loader, ks=tuple(args.ks))

    print("\n" + "=" * 40)
    print("Retrieval Accuracy (full-set)")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:>6s} : {v:.4f}  ({v*100:.1f}%)")
    print("=" * 40)