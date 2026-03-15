# TEMGen — TEM Diffraction + Crystal Structure Contrastive Learning

Contrastive learning framework that aligns TEM diffraction patterns with crystal structures for the CuAu FCC 10×10×10 system.

## Project Structure

```
temgen/                              # repo root (on pscratch)
├── configs/
│   ├── cuau_101010_m1_t01.yaml      # Method 1, τ = 0.1
│   ├── cuau_101010_m1_t05.yaml      # Method 1, τ = 0.5
│   ├── cuau_101010_m1_t10.yaml      # Method 1, τ = 1.0
│   ├── cuau_101010_m3_t01.yaml      # Method 3, τ = 0.1
│   ├── cuau_101010_m3_t05.yaml      # Method 3, τ = 0.5
│   └── cuau_101010_m3_t10.yaml      # Method 3, τ = 1.0
├── scripts/
│   ├── build_hdf5.py                # one-time: raw zarr+cif → .h5 files
│   ├── build_hdf5.sh                # SLURM wrapper for HDF5 build
│   ├── train.py                     # training entry point (called by train.sh)
│   └── train.sh                     # SLURM training script (Perlmutter A100)
├── data/                            # data assets (NOT in Python package)
│   ├── raw/                         # symlinks / copies of raw data
│   │   ├── cuau_2502/
│   │   ├── cuau_26k/
│   │   └── cuau_26k_var/
│   └── hdf5/
│       ├── train_20260312.h5        # 60K samples (~600 GB)
│       └── test_20260312.h5         # 2502 samples (~30 GB)
├── temgen/                          # installable Python package
│   ├── __init__.py
│   ├── utils.py                     # shared utilities (fourier_encode)
│   ├── data/
│   │   ├── __init__.py
│   │   └── cuau_dataset.py          # HDF5 Dataset + DataLoader factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── temgen_model.py          # top-level model (orchestrator)
│   │   ├── image_encoder/
│   │   │   ├── __init__.py
│   │   │   ├── cnn_frontend.py      # ResNet-18 1-ch backbone
│   │   │   ├── geometry_tokens.py   # reciprocal grid, rotation, geo+angle embed
│   │   │   ├── aggregator.py        # Method 1: Perceiver Latent + shared blocks
│   │   │   ├── geometry_aware_aggregator.py  # Method 2: Geometry-Aware Perceiver
│   │   │   └── cross_view_voxel_aggregator.py  # Method 3: Voxel Latent
│   │   ├── structure_encoder/
│   │   │   ├── __init__.py
│   │   │   ├── graph_builder.py     # radius graph + RBF edge features
│   │   │   └── gnn.py              # CSPLayerCartesian GNN
│   │   └── losses/
│   │       ├── __init__.py
│   │       └── info_nce.py         # symmetric InfoNCE with temperature
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lightning_module.py      # LightningModule (AdamW + cosine LR)
│   │   └── callbacks.py            # retrieval accuracy, checkpointing
│   └── eval/
│       ├── __init__.py
│       └── retrieval.py            # full-set top-1 / top-5 / top-10 accuracy
├── pyproject.toml
├── setup.py
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Environment (Perlmutter)
```bash
module load conda
conda create -n temgen python=3.12
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

### 2. Build HDF5 files (one-time)
```bash
# Dry run first
python scripts/build_hdf5.py --split train --date 20260312 --dry-run

# Submit SLURM job
sbatch scripts/build_hdf5.sh
```

### 3. Train
```bash
# Default: regular QOS, 1 node (4× A100 80GB), Method 3 τ=0.1
sbatch scripts/train.sh

# Specify a different config
sbatch scripts/train.sh configs/cuau_101010_m1_t05.yaml

# Override QOS
sbatch --qos=preempt scripts/train.sh

# 4-node DDP (16 GPUs)
sbatch --nodes=4 scripts/train.sh
```

Training auto-resumes from `last.ckpt` on preemption (`--requeue`) or wall-time timeout.

### 4. Evaluate
```bash
python -m temgen.eval.retrieval \
    --checkpoint checkpoints/<job_id>/best.ckpt \
    --config configs/cuau_101010_m3_t01.yaml \
    --test-h5 data/hdf5/test_20260312.h5
```

## Data

Raw data on CFS:
- `cuau_26k` : 26k original + augmented samples (train)
- `cuau_26k_var` : variant augmented samples (train)
- `cuau_2502` : 2,502 samples (train supplement)

Processed HDF5 on pscratch (fast NVMe):
- `train_20260312.h5` : 15,502 samples (~185 GB)
- `test_20260312.h5`  :   252 samples (~3 GB)

## Dataset Summary

| Parameter | Value |
|---|---|
| System | Cu-Au FCC random alloy, 10x10x10 supercell |
| Tilts | 15 (-7 to +7 deg, step 1 deg, y-axis) |
| DP resolution | 409x409 px (downsampled to 256x256) |
| ROI | 15x15x50 A |
| Beam energy | 300 keV |
| Detector max angle | 80 mrad |
| Train samples | 15,502 (26k + 2502 subsets) |
| Test samples | 252 |

## Architecture

Three image encoder aggregation methods (selected via `aggregator_method` in config):

| Method | Description | Latent tokens |
|---|---|---|
| 1 | Perceiver Latent (baseline) | 32 learned |
| 2 | Geometry-Aware Perceiver (+ anchor bias) | 32 with 3D anchors |
| 3 | Cross-View Voxel Latent | 128 voxel grid (8x8x2) |

Structure encoder: 4x CSPLayerCartesian GNN with radius graph (r_c=5.0 A) and Gaussian RBF edge features.

Loss: Symmetric InfoNCE with fixed or learnable temperature.

Attention uses `F.scaled_dot_product_attention` (FlashAttention-compatible). Dropout (default 0.1) is applied in FFN and attention layers.

## Config Naming

`cuau_101010_m{method}_t{temp}.yaml`

- `m1` / `m3` : aggregator method (1 = Perceiver, 3 = Voxel)
- `t01` / `t05` / `t10` : fixed temperature (0.1, 0.5, 1.0)

All configs use: batch_size=64/GPU (effective 256 on 4 GPUs), lr=3e-4, AdamW, 500 epochs, cosine LR with 10-epoch warmup.

## QOS Guide (Perlmutter GPU)

| QOS | Cost factor | Max time | Notes |
|---|---|---|---|
| `regular` | 1.0 | 48hr | Default. 50% off at 128+ nodes. |
| `preempt` | 0.25 | 48hr (preemptible after 2hr) | Use with `--requeue`. 75% cheaper. |
| `debug` | -- | 0.5hr | Fast turnaround for dry runs. |
| `shared` | fraction | 48hr | 1-2 GPUs, fraction-node cost. |

## Analysis

Warren-Cowley SRO parameters are computed for up to 40 neighbor shells across the 252 test samples:

```bash
# Run SRO analysis
python analysis/sro_40nn.py
```

Results are saved to `analysis/sro_40nn_results.csv`. Jupyter notebooks in `analysis/` provide visualization and exploration at different shell depths (1NN, 4NN, 12NN, 40NN).
