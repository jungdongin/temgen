"""
Quick dry-run: loads config, data, model, runs 1 forward+backward pass.
Tests the full pipeline without SLURM. Delete after use.

Usage:
    python scripts/dry_run.py
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from omegaconf import OmegaConf

from temgen.models.temgen_model import TEMGenModel
from temgen.data.cuau_dataset import build_dataloaders

def main():
    cfg = OmegaConf.load("configs/cuau_101010.yaml")

    # ── Step 1: Build dataloaders with tiny batch ──
    print("[1/4] Building dataloaders (batch_size=2) ...")
    t0 = time.time()
    train_loader, val_loader, test_loader = build_dataloaders(
        train_h5     = cfg.data.train_h5,
        test_h5      = cfg.data.test_h5,
        val_fraction = cfg.data.val_fraction,
        val_seed     = cfg.data.val_seed,
        batch_size   = 2,        # tiny batch for dry run
        num_workers  = 0,        # no multiprocessing
        pin_memory   = False,
    )
    print(f"      Done in {time.time()-t0:.1f}s  "
          f"(train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
          f"test={len(test_loader.dataset)})")

    # ── Step 2: Build model ──
    print("[2/4] Building model ...")
    t0 = time.time()
    model = TEMGenModel(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Done in {time.time()-t0:.1f}s  ({n_params:,} parameters)")

    # ── Step 3: Forward pass ──
    print("[3/4] Forward pass (1 batch) ...")
    batch = next(iter(train_loader))
    # Move tensors to GPU
    batch["dp"] = batch["dp"].cuda()
    batch["alpha"] = batch["alpha"].cuda()
    batch["lengths"] = batch["lengths"].cuda()
    batch["angles"] = batch["angles"].cuda()
    batch["frac_coords"] = [c.cuda() for c in batch["frac_coords"]]
    batch["atom_types"] = [a.cuda() for a in batch["atom_types"]]

    t0 = time.time()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = model(batch)
    torch.cuda.synchronize()
    print(f"      Done in {time.time()-t0:.1f}s")
    print(f"      loss={out['loss'].item():.4f}  tau={out['tau'].item():.4f}  acc={out['acc'].item():.4f}")
    print(f"      z_TEM={tuple(out['z_TEM'].shape)}  z_cell={tuple(out['z_cell'].shape)}")

    # ── Step 4: Backward pass ──
    print("[4/4] Backward pass ...")
    t0 = time.time()
    out["loss"].backward()
    torch.cuda.synchronize()
    print(f"      Done in {time.time()-t0:.1f}s")

    # Check gradients exist
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"      All gradients computed: {has_grad}")

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak VRAM: {vram:.2f} GB")
    print("\nDry run PASSED — pipeline is functional.")


if __name__ == "__main__":
    main()
