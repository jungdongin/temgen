# TEMGen — TEM Diffraction + Crystal Structure Contrastive Learning

Contrastive learning framework that aligns TEM diffraction patterns with crystal structures for the CuAu FCC 10×10×10 system.

## Project Structure

```
temgen/                              # repo root (on pscratch)
├── configs/
│   └── cuau_101010.yaml             # all hyperparameters and paths
├── scripts/
│   ├── build_hdf5.py                # one-time: raw zarr+cif → .h5 files
│   ├── build_hdf5.sh                # SLURM wrapper for HDF5 build
│   ├── train.py                     # training entry point (called by train.sh)
│   └── train.sh                     # SLURM training script (Perlmutter A100)
├── data/                            # data assets (NOT in Python package)
│   ├── raw/                         # symlinks to CFS
│   │   ├── cuau_13k -> /global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data
│   │   ├── cuau_2502 -> /global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data_2502
│   │   └── cuau_252  -> /global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data_252
│   └── hdf5/
│       ├── train_20260305.h5        # 15,502 samples (~185 GB)
│       └── test_20260304.h5         # 252 samples (~3 GB)
├── temgen/                          # installable Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── cuau_dataset.py          # Dataset + DataLoader factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_encoder/
│   │   │   ├── __init__.py
│   │   │   ├── cnn_frontend.py      # ResNet-18 1-ch backbone
│   │   │   ├── geometry_tokens.py   # reciprocal grid, rotation, geo+angle embed
│   │   │   ├── aggregator.py        # Method 1: Perceiver Latent
│   │   │   ├── geometry_aware_perceiver.py  # Method 2: Geometry-Aware Perceiver
│   │   │   └── cross_view_voxel_aggregator.py  # Method 3: Voxel Latent
│   │   ├── structure_encoder/
│   │   │   ├── __init__.py
│   │   │   ├── graph_builder.py     # radius graph + RBF edge features
│   │   │   └── gnn.py              # CSPLayerCartesian × 4
│   │   ├── losses/
│   │   │   ├── __init__.py
│   │   │   └── info_nce.py         # InfoNCE with learnable temperature
│   │   └── temgen_model.py         # top-level model (orchestrator)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lightning_module.py      # LightningModule
│   │   └── callbacks.py            # retrieval accuracy, checkpointing
│   └── eval/
│       ├── __init__.py
│       └── retrieval.py            # full-set top-1 / top-5 / top-10 accuracy
├── pyproject.toml                   # pip install -e .
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### 1. Environment (Perlmutter)
```bash
module load conda
conda create -n temgen python=3.10
conda activate temgen

# PyTorch — use Perlmutter-optimised build
module load pytorch/2.1.0-cu12
# or: conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse

# Everything else
pip install -r requirements.txt

# Install temgen as editable package
pip install -e .
```

### 2. Create data symlinks (one-time)
```bash
cd /pscratch/sd/d/dongin/temgen
mkdir -p data/raw
ln -s /global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data      data/raw/cuau_13k
ln -s /global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data_2502  data/raw/cuau_2502
ln -s /global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data_252   data/raw/cuau_252
```

### 3. Build HDF5 files (one-time)
```bash
# Dry run first
python scripts/build_hdf5.py --split train --date 20260305 --dry-run

# Submit SLURM job
sbatch scripts/build_hdf5.sh
```

### 4. Train
```bash
# Default: preempt QOS (75% cheaper), 1 node (4× A100 80GB)
sbatch scripts/train.sh

# Override to regular QOS
sbatch --qos=regular scripts/train.sh

# 4-node DDP (16 GPUs)
sbatch --nodes=4 scripts/train.sh
```

### 5. Evaluate
```bash
# Standalone retrieval accuracy on test set
python -m temgen.eval.retrieval \
    --checkpoint checkpoints/best.ckpt \
    --config configs/cuau_101010.yaml \
    --test-h5 data/hdf5/test_20260304.h5
```

## Data

Raw data lives on CFS (read-only):
- `13k set` : `/global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data/`
- `2502 set` : `/global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data_2502/`
- `252 set`  : `/global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data_252/`  ← test

Processed HDF5 on pscratch (fast NVMe):
- `train_20260305.h5` : 15,502 samples (~185 GB)
- `test_20260304.h5`  :   252 samples (~3 GB)

## Dataset Summary

| Parameter | Value |
|---|---|
| System | Cu–Au FCC random alloy, 10×10×10 supercell |
| Tilts | 15 (−7° to +7°, step 1°, y-axis) |
| DP resolution | 409 × 409 px |
| ROI | 15 × 15 × 50 Å |
| Beam energy | 300 keV |
| Detector max angle | 80 mrad |
| Train samples | 15,502 (13k + 2502) |
| Test samples | 252 |

## Architecture

Three image encoder aggregation methods (selected via `aggregator_method` in config):

| Method | Description | Latent tokens |
|---|---|---|
| 1 | Perceiver Latent (baseline) | 32 learned |
| 2 | Geometry-Aware Perceiver (+ anchor bias) | 32 with 3D anchors |
| 3 | Cross-View Voxel Latent | 128 voxel grid (8×8×2) |

Structure encoder: 4 × CSPLayerCartesian (adapted from DiffCSP) with radius graph (r_c=5.0 Å) and Gaussian RBF edge features.

Loss: Symmetric InfoNCE with learnable temperature.

## QOS Guide (Perlmutter GPU)

| QOS | Cost factor | Max time | Notes |
|---|---|---|---|
| `preempt` | 0.25 | 48hr (preemptible after 2hr) | **Default.** Use with `--requeue`. 75% cheaper. |
| `regular` | 1.0 | 48hr | Full price. 50% off at 128+ nodes (big job discount). |
| `debug` | — | 0.5hr | Fast turnaround for dry runs. |
| `shared` | fraction | 48hr | 1–2 GPUs, fraction-node cost. |