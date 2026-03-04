"""
cuau_dataset.py

PyTorch Dataset reading from pre-built HDF5 files (train/test).

Each __getitem__ does exactly:
  1. One HDF5 row read for dp       → (15, 1, 409, 409)
  2. One HDF5 slice read for struct → frac_coords, atom_types
  3. A few scalar reads             → lengths, angles, a_frac

No zarr, no CIF parsing at training time.

Usage
-----
    from temgen.data.cuau_dataset import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# ─── Constants ────────────────────────────────────────────────────────────────
TILT_ANGLES_RAD = torch.tensor(
    [math.radians(a) for a in range(-7, 8)], dtype=torch.float32
)  # (15,)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CuAuHDF5Dataset(Dataset):
    """
    Reads CuAu samples from a single HDF5 file (train or test).

    HDF5 layout expected:
        dp                (N, 15, 409, 409)  float16
        lengths           (N, 3)             float32
        angles            (N, 3)             float32
        a_frac            (N,)               float32
        num_atoms         (N,)               int32
        atom_offsets      (N+1,)             int64
        frac_coords_flat  (total_atoms, 3)   float32
        atom_types_flat   (total_atoms,)     int16
        sample_ids        (N,)               bytes

    The HDF5 file is opened lazily per worker (opened on first __getitem__
    call rather than __init__) to be compatible with num_workers > 0.
    """

    def __init__(
        self,
        h5_path: str | Path,
        indices: Optional[List[int]] = None,
    ):
        """
        Args:
            h5_path : path to .h5 file
            indices : optional subset of sample indices (for train/val split)
        """
        self.h5_path = Path(h5_path)
        assert self.h5_path.exists(), f"HDF5 not found: {self.h5_path}"

        # read metadata once (cheap)
        with h5py.File(self.h5_path, "r") as f:
            self._n_total    = len(f["dp"])
            self._num_atoms  = f["num_atoms"][:]          # (N,) int32
            self._atom_offsets = f["atom_offsets"][:]     # (N+1,) int64
            self._a_frac     = f["a_frac"][:]             # (N,) float32
            self._sample_ids = [
                sid.decode() if isinstance(sid, bytes) else sid
                for sid in f["sample_ids"][:]
            ]

        self.indices = indices if indices is not None else list(range(self._n_total))

        # HDF5 handle — opened lazily per worker
        self._h5: Optional[h5py.File] = None

    def _open(self):
        """Open HDF5 file if not already open (called per worker on first access)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        self._open()
        i = self.indices[idx]

        # ── DP ────────────────────────────────────────────────────────────────
        dp = torch.from_numpy(
            self._h5["dp"][i].astype(np.float32)
        ).unsqueeze(1)                                    # (15, 1, 409, 409)

        # ── Structure ─────────────────────────────────────────────────────────
        s = int(self._atom_offsets[i])
        e = int(self._atom_offsets[i + 1])

        frac_coords = torch.from_numpy(
            self._h5["frac_coords_flat"][s:e]             # (N_i, 3)
        )
        atom_types = torch.from_numpy(
            self._h5["atom_types_flat"][s:e].astype(np.int32)  # (N_i,) → long-safe
        ).long()

        lengths = torch.from_numpy(self._h5["lengths"][i])   # (3,)
        angles  = torch.from_numpy(self._h5["angles"][i])    # (3,)

        return dict(
            dp          = dp,                               # (15, 1, 409, 409) float32
            alpha       = TILT_ANGLES_RAD.clone(),          # (15,) float32
            frac_coords = frac_coords,                      # (N_i, 3)  float32
            atom_types  = atom_types,                       # (N_i,)    int64
            lengths     = lengths,                          # (3,)      float32
            angles      = angles,                           # (3,)      float32
            num_atoms   = int(self._num_atoms[i]),
            a_frac      = float(self._a_frac[i]),
            sample_id   = self._sample_ids[i],
        )

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# ─── Collate ──────────────────────────────────────────────────────────────────

def cuau_collate_fn(batch: List[dict]) -> dict:
    """
    Collate a list of samples into a batch.
    dp / alpha / lengths / angles are stacked normally.
    frac_coords / atom_types are kept as lists (variable N per sample)
    — the GNN graph builder handles batching these.
    """
    return dict(
        dp          = torch.stack([b["dp"]     for b in batch]),   # (B,15,1,409,409)
        alpha       = torch.stack([b["alpha"]  for b in batch]),   # (B,15)
        lengths     = torch.stack([b["lengths"]for b in batch]),   # (B,3)
        angles      = torch.stack([b["angles"] for b in batch]),   # (B,3)
        frac_coords = [b["frac_coords"] for b in batch],           # list[Tensor(N_i,3)]
        atom_types  = [b["atom_types"]  for b in batch],           # list[Tensor(N_i,)]
        num_atoms   = [b["num_atoms"]   for b in batch],           # list[int]
        a_frac      = torch.tensor([b["a_frac"] for b in batch], dtype=torch.float32),
        sample_ids  = [b["sample_id"]   for b in batch],           # list[str]
    )


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    train_h5      : str | Path,
    test_h5       : str | Path,
    val_fraction  : float = 0.1,
    val_seed      : int   = 42,
    batch_size    : int   = 256,
    num_workers   : int   = 8,
    pin_memory    : bool  = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from HDF5 files.

    Train/val split is done by random permutation (seeded) on train.h5.
    Test loader reads directly from test.h5.

    Returns:
        train_loader, val_loader, test_loader
    """
    # ── full train dataset ─────────────────────────────────────────────────────
    full_train = CuAuHDF5Dataset(train_h5)
    N          = len(full_train)

    rng      = torch.Generator().manual_seed(val_seed)
    perm     = torch.randperm(N, generator=rng).tolist()
    n_val    = int(N * val_fraction)
    n_train  = N - n_val

    train_ds = CuAuHDF5Dataset(train_h5, indices=perm[:n_train])
    val_ds   = CuAuHDF5Dataset(train_h5, indices=perm[n_train:])
    test_ds  = CuAuHDF5Dataset(test_h5)

    loader_kwargs = dict(
        batch_size       = batch_size,
        num_workers      = num_workers,
        pin_memory       = pin_memory,
        collate_fn       = cuau_collate_fn,
        persistent_workers = (num_workers > 0),
        prefetch_factor  = 2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"[CuAuDataset] train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
    return train_loader, val_loader, test_loader


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--test-h5",  required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    train_l, val_l, test_l = build_dataloaders(
        train_h5    = args.train_h5,
        test_h5     = args.test_h5,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    print("\n--- Train batch ---")
    b = next(iter(train_l))
    print(f"  dp         : {b['dp'].shape}   dtype={b['dp'].dtype}")
    print(f"  alpha      : {b['alpha'].shape}")
    print(f"  lengths    : {b['lengths'].shape}")
    print(f"  frac_coords: {b['frac_coords'][0].shape}  (sample 0)")
    print(f"  atom_types : {b['atom_types'][0].shape}")
    print(f"  a_frac     : {b['a_frac']}")
    print(f"  sample_ids : {b['sample_ids']}")

    print("\n--- Test batch ---")
    b = next(iter(test_l))
    print(f"  dp         : {b['dp'].shape}")
    print(f"  sample_ids : {b['sample_ids']}")
    print("\nAll good!")