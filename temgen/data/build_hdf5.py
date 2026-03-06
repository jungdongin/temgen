"""
build_hdf5.py

Converts raw CuAu 10×10×10 data (zarr DPs + CIF structures) into compressed HDF5 files

--------------------------------------------------------------------
train_YYYYMMDD.h5 / test_YYYYMMDD.h5
    dp                (N, 15, 409, 409)  float16   — log1p-normalised
    lengths           (N, 3)             float32   — ROI cell lengths in Å
    angles            (N, 3)             float32   — ROI cell angles in degrees
    a_frac            (N,)               float32   — Au fraction
    num_atoms         (N,)               int32
    atom_offsets      (N+1,)             int64     — CSR: slice i = [off[i]:off[i+1]]
    frac_coords_flat  (total_atoms, 3)   float32
    atom_types_flat   (total_atoms,)     int16     — atomic number (Cu=29, Au=79)
    sample_ids        (N,)               bytes/str — e.g. b"00001"

Usage
-----
    # Build train set (13k + 2502)
    python build_hdf5.py --split train 

    # Build test set (252)
    python build_hdf5.py --split test

    # Dry run (process only first 10 samples)
    python build_hdf5.py --split train --dry-run

    # Resume from checkpoint if job was interrupted
    python build_hdf5.py --split train --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import h5py
import numpy as np
import zarr
from pymatgen.core import Structure
from tqdm import tqdm

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_CFS    = Path("/global/cfs/cdirs/m1090/dongin")
BASE_PSCRATCH = Path("/pscratch/sd/d/dongin/temgen")

DATA_13K    = BASE_CFS / "cuau_fcc_101010_data"
DATA_2502   = BASE_CFS / "cuau_fcc_101010_data_2502"
DATA_252    = BASE_CFS / "cuau_fcc_101010_data_252"

HDF5_DIR    = BASE_PSCRATCH / "data" / "hdf5"

# ─── Dataset definitions ──────────────────────────────────────────────────────
# Each entry: (root_dir, list_of_5digit_ids)
# Train = 13k (00001–13000) + 2502 (00001–02502, same root naming after fix)
# Test  = 252  (00001–00252)

def get_sample_list(split: str) -> List[Tuple[Path, str]]:
    """
    Returns list of (root_dir, id5_string) for every sample in the split.
    Train: 13k set first (ids 00001–13000), then 2502 set (ids 00001–02502).
    Test : 252 set (ids 00001–00252).
    """
    samples = []
    if split == "train":
        for i in range(1, 13001):
            samples.append((DATA_13K, f"{i:05d}"))
        for i in range(1, 2503):
            samples.append((DATA_2502, f"{i:05d}"))
    elif split == "test":
        for i in range(1, 253):
            samples.append((DATA_252, f"{i:05d}"))
    else:
        raise ValueError(f"Unknown split: {split}")
    return samples


# ─── Per-sample reader ────────────────────────────────────────────────────────

def read_sample(root: Path, sid: str) -> dict | None:
    """
    Read one sample from disk. Returns None if any file is missing.

    Returns dict with keys:
        dp          np.float16  (15, 409, 409)
        frac_coords np.float32  (N, 3)
        atom_types  np.int16    (N,)
        lengths     np.float32  (3,)
        angles      np.float32  (3,)
        a_frac      float
        num_atoms   int
        sample_id   str
    """
    sample_dir = root / sid

    # ── Diffraction patterns ─────────────────────────────────────────────────
    zarr_path = sample_dir / f"{sid}_dp_convAngle_2" / "dp"
    if not zarr_path.exists():
        log.warning(f"Missing zarr: {zarr_path}")
        return None
    try:
        z  = zarr.open(str(zarr_path), mode="r")
        dp = np.array(z, dtype=np.float32)          # (15, 409, 409)
    except Exception as e:
        log.warning(f"Failed to read zarr {zarr_path}: {e}")
        return None

    # log1p normalise, then min-max per tilt → float16
    dp = np.log1p(dp)
    mins = dp.reshape(15, -1).min(axis=1)[:, None, None]
    maxs = dp.reshape(15, -1).max(axis=1)[:, None, None]
    dp   = ((dp - mins) / (maxs - mins + 1e-8)).astype(np.float16)

    # ── ROI structure ─────────────────────────────────────────────────────────
    cif_path = sample_dir / f"{sid}_structure_roi.cif"
    if not cif_path.exists():
        log.warning(f"Missing CIF: {cif_path}")
        return None
    try:
        struct = Structure.from_file(str(cif_path))
    except Exception as e:
        log.warning(f"Failed to parse CIF {cif_path}: {e}")
        return None

    frac_coords = struct.frac_coords.astype(np.float32)            # (N, 3)
    atom_types  = np.array([s.specie.Z for s in struct], dtype=np.int16)  # (N,)
    lengths     = np.array(struct.lattice.abc,    dtype=np.float32)       # (3,)
    angles      = np.array(struct.lattice.angles, dtype=np.float32)       # (3,)

    # ── Meta ──────────────────────────────────────────────────────────────────
    meta_path = sample_dir / f"{sid}_meta.json"
    a_frac = float("nan")
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            a_frac = float(meta.get("a_frac_actual", float("nan")))
        except Exception:
            pass

    return dict(
        dp          = dp,
        frac_coords = frac_coords,
        atom_types  = atom_types,
        lengths     = lengths,
        angles      = angles,
        a_frac      = a_frac,
        num_atoms   = len(atom_types),
        sample_id   = sid,
    )


# ─── Two-pass HDF5 builder ────────────────────────────────────────────────────

def build_hdf5(split: str, date_str: str, dry_run: bool, resume: bool) -> None:
    """
    Two-pass strategy:
      Pass 1 (fast): scan all CIF files to get num_atoms per sample → pre-allocate
      Pass 2       : read zarr + write into pre-allocated datasets
    This avoids loading everything into RAM at once.
    """
    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HDF5_DIR / f"{split}_{date_str}.h5"
    tmp_path = HDF5_DIR / f"{split}_{date_str}.h5.tmp"

    samples = get_sample_list(split)
    if dry_run:
        samples = samples[:10]
        log.info(f"DRY RUN: processing first {len(samples)} samples only")

    N = len(samples)
    log.info(f"Split={split}  N={N}  output={out_path}")

    # ── Resume: find how many samples already written ─────────────────────────
    start_idx = 0
    if resume and tmp_path.exists():
        with h5py.File(tmp_path, "r") as f:
            start_idx = int(f.attrs.get("n_written", 0))
        log.info(f"Resuming from sample {start_idx}")

    # ── Pass 1: count atoms (only for samples not yet written) ────────────────
    if start_idx == 0:
        log.info("Pass 1: scanning CIF files for atom counts...")
        num_atoms_arr = np.zeros(N, dtype=np.int32)
        valid_mask    = np.ones(N,  dtype=bool)

        for i, (root, sid) in enumerate(tqdm(samples, desc="Pass1-CIF")):
            cif_path = root / sid / f"{sid}_structure_roi.cif"
            if not cif_path.exists():
                log.warning(f"[{i}] Missing CIF: {cif_path}")
                valid_mask[i] = False
                continue
            try:
                struct = Structure.from_file(str(cif_path))
                num_atoms_arr[i] = len(struct)
            except Exception as e:
                log.warning(f"[{i}] CIF parse error {cif_path}: {e}")
                valid_mask[i] = False

        total_atoms = int(num_atoms_arr[valid_mask].sum())
        atom_offsets = np.zeros(N + 1, dtype=np.int64)
        atom_offsets[1:] = np.cumsum(num_atoms_arr)

        log.info(f"  Valid samples : {valid_mask.sum()} / {N}")
        log.info(f"  Total atoms   : {total_atoms:,}")
        log.info(f"  Avg atoms/sample: {total_atoms / max(valid_mask.sum(),1):.1f}")

        # ── Pre-allocate HDF5 ─────────────────────────────────────────────────
        log.info("Pre-allocating HDF5 datasets...")
        with h5py.File(tmp_path, "w") as f:
            f.attrs["n_written"]   = 0
            f.attrs["n_total"]     = N
            f.attrs["split"]       = split
            f.attrs["date"]        = date_str

            # chunked along sample axis for efficient row reads
            f.create_dataset("dp",
                shape=(N, 15, 409, 409), dtype="float16",
                chunks=(1, 15, 409, 409),
                compression="lzf")          # lzf: fast compress, good for float16

            f.create_dataset("lengths",     shape=(N, 3),    dtype="float32")
            f.create_dataset("angles",      shape=(N, 3),    dtype="float32")
            f.create_dataset("a_frac",      shape=(N,),      dtype="float32")
            f.create_dataset("num_atoms",   shape=(N,),      dtype="int32")
            f.create_dataset("atom_offsets",shape=(N + 1,),  dtype="int64",
                data=atom_offsets)

            f.create_dataset("frac_coords_flat",
                shape=(total_atoms, 3), dtype="float32",
                chunks=(min(total_atoms, 100000), 3))

            f.create_dataset("atom_types_flat",
                shape=(total_atoms,), dtype="int16",
                chunks=(min(total_atoms, 100000),))

            # sample_ids as fixed-length bytes
            dt = h5py.string_dtype(encoding="ascii", length=5)
            f.create_dataset("sample_ids", shape=(N,), dtype=dt)

            # store valid_mask so resume pass knows which to skip
            f.create_dataset("valid_mask", data=valid_mask.astype(np.int8))

        log.info(f"HDF5 pre-allocated at {tmp_path}")

    # ── Pass 2: read zarr + write data ────────────────────────────────────────
    log.info(f"Pass 2: writing data (starting at sample {start_idx})...")
    t0 = time.time()

    with h5py.File(tmp_path, "a") as f:
        valid_mask   = f["valid_mask"][:].astype(bool)
        atom_offsets = f["atom_offsets"][:]

        for i in tqdm(range(start_idx, N), desc="Pass2-write", initial=start_idx, total=N):
            root, sid = samples[i]

            if not valid_mask[i]:
                # fill with zeros / empty so indices stay aligned
                f["dp"][i]       = np.zeros((15, 409, 409), dtype=np.float16)
                f["lengths"][i]  = np.zeros(3,  dtype=np.float32)
                f["angles"][i]   = np.zeros(3,  dtype=np.float32)
                f["a_frac"][i]   = float("nan")
                f["num_atoms"][i]= 0
                f["sample_ids"][i] = "00000"
                continue

            sample = read_sample(root, sid)
            if sample is None:
                log.warning(f"read_sample returned None for {sid}, skipping")
                f["num_atoms"][i] = 0
                f["sample_ids"][i] = "00000"
                continue

            # scalar / small arrays
            f["dp"][i]        = sample["dp"]
            f["lengths"][i]   = sample["lengths"]
            f["angles"][i]    = sample["angles"]
            f["a_frac"][i]    = sample["a_frac"]
            f["num_atoms"][i] = sample["num_atoms"]
            f["sample_ids"][i]= sample["sample_id"]

            # flat structure arrays
            s = int(atom_offsets[i])
            e = int(atom_offsets[i + 1])
            f["frac_coords_flat"][s:e] = sample["frac_coords"]
            f["atom_types_flat"][s:e]  = sample["atom_types"]

            # checkpoint every 500 samples
            if (i + 1) % 500 == 0:
                f.attrs["n_written"] = i + 1
                elapsed = time.time() - t0
                rate = (i + 1 - start_idx) / elapsed
                eta  = (N - i - 1) / max(rate, 1e-6)
                log.info(f"  [{i+1}/{N}]  rate={rate:.1f} samples/s  ETA={eta/60:.1f} min")

        f.attrs["n_written"] = N

    # ── Rename tmp → final ────────────────────────────────────────────────────
    tmp_path.rename(out_path)
    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9
    log.info(f"Done! {out_path}  size={size_gb:.1f} GB  time={elapsed/60:.1f} min")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split",   choices=["train", "test"], required=True)
    parser.add_argument("--date",    default="20260304",
                        help="Date tag for output filename, e.g. 20260304")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 10 samples only")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from a .h5.tmp checkpoint file")
    args = parser.parse_args()

    build_hdf5(
        split    = args.split,
        date_str = args.date,
        dry_run  = args.dry_run,
        resume   = args.resume,
    )


if __name__ == "__main__":
    main()