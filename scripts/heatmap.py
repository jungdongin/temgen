"""
heatmap.py — Generate cosine similarity heatmaps on the test set.

Loads the best checkpoint for each method, computes the (N, N) similarity
matrix on the test set, and saves a side-by-side heatmap figure.

Usage:
    python scripts/heatmap.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from temgen.models.temgen_model import TEMGenModel
from temgen.training.lightning_module import TEMGenLightningModule
from temgen.data.cuau_dataset import CuAuHDF5Dataset, cuau_collate_fn
from temgen.eval.retrieval import collect_embeddings


METHODS = [
    {
        "name": "Method 1: Perceiver\n(τ=0.07, best val=39.2%)",
        "config": "configs/cuau_101010_1.yaml",
        "ckpt": "checkpoints/49790143/temgen-epoch288-top10.3924.ckpt",
    },
    {
        "name": "Method 2: GeometryAware\n(τ=0.07, best val=42.9%)",
        "config": "configs/cuau_101010_2.yaml",
        "ckpt": "checkpoints/49790144/temgen-epoch272-top10.4291.ckpt",
    },
    {
        "name": "Method 3: CrossViewVoxel\n(τ=0.07, best val=45.8%)",
        "config": "configs/cuau_101010_3.yaml",
        "ckpt": "checkpoints/49790145/temgen-epoch264-top10.4581.ckpt",
    },
]

TEST_H5 = "data/hdf5/test_20260304.h5"
BATCH_SIZE = 64
NUM_WORKERS = 4
OUT_PATH = "plots/test_heatmap_tau007.png"


def load_and_eval(method, device):
    cfg = OmegaConf.load(method["config"])
    model = TEMGenModel(cfg)
    lit = TEMGenLightningModule.load_from_checkpoint(
        method["ckpt"], model=model, cfg=cfg,
    )
    lit = lit.to(device)
    lit.eval()

    test_ds = CuAuHDF5Dataset(TEST_H5)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=cuau_collate_fn, pin_memory=True,
    )

    z_tem, z_cell = collect_embeddings(lit.model, test_loader)
    z_tem = F.normalize(z_tem, dim=-1)
    z_cell = F.normalize(z_cell, dim=-1)
    sim = (z_tem @ z_cell.T).numpy()
    return sim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sims = []
    for m in METHODS:
        print(f"\nLoading {m['name']}  ({m['ckpt']})")
        sim = load_and_eval(m, device)
        print(f"  Similarity matrix: {sim.shape}, diag mean={np.diag(sim).mean():.4f}")
        sims.append(sim)

    # Sort by diagonal similarity (correct-pair score) for cleaner visualization
    # Use last method's diagonal to define sort order
    sort_idx = np.argsort(-np.diag(sims[-1]))  # descending by correct-pair sim

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Compute color range from data: use off-diagonal stats for contrast
    all_off = []
    for sim in sims:
        mask = ~np.eye(sim.shape[0], dtype=bool)
        all_off.append(sim[mask])
    off_all = np.concatenate(all_off)
    vmin = float(np.percentile(off_all, 1))
    vmax = 1.0
    print(f"Color range: [{vmin:.3f}, {vmax:.3f}]")

    n = len(METHODS)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
    if n == 1:
        axes = [axes]
    for ax, sim, m in zip(axes, sims, METHODS):
        sim_sorted = sim[sort_idx][:, sort_idx]
        im = ax.imshow(sim_sorted, cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")
        diag_mean = np.diag(sim).mean()
        off_mean = sim[~np.eye(sim.shape[0], dtype=bool)].mean()
        ax.set_title(f"{m['name']}\ndiag={diag_mean:.3f}  off-diag={off_mean:.3f}", fontsize=11)
        ax.set_xlabel("Structure index")
        ax.set_ylabel("TEM image index")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Cosine similarity", pad=0.02)
    fig.suptitle("Test Set Cosine Similarity (N=252, best checkpoints, τ=0.07 runs)",
                 fontsize=14, y=1.02)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
