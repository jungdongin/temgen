# TEMGen — TEM Diffraction + Crystal Structure Contrastive Learning

Contrastive learning framework that aligns TEM diffraction patterns with crystal structures for the CuAu FCC 10×10×10 system.

## Project Structure

```
temgen/
├── data/
│   ├── build_hdf5.py        # one-time: raw zarr+cif → train.h5 / test.h5
│   └── cuau_dataset.py      # Dataset + DataLoader factory
├── models/
│   ├── image_encoder/
│   │   ├── cnn_frontend.py       # ResNet-18 1-ch backbone
│   │   ├── geometry_tokens.py    # reciprocal grid, rotation, geo+angle embed
│   │   └── aggregator.py         # Perceiver (Methods 1/2/3)
│   ├── structure_encoder/
│   │   ├── graph_builder.py      # radius graph + RBF edge features
│   │   └── gnn.py                # CSPLayerCartesian × 4
│   └── temgen_model.py           # top-level model
├── losses/
│   └── info_nce.py               # InfoNCE with learnable temperature
├── training/
│   ├── lightning_module.py       # LightningModule
│   └── callbacks.py              # retrieval accuracy, checkpointing
└── eval/
    └── retrieval.py              # top-1 / top-5 accuracy
configs/
└── cuau_101010.yaml              # all hyperparameters and paths
scripts/
├── build_hdf5.sh
└── train.sh
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

### 2. Build HDF5 files (one-time)
```bash
# Dry run first
python temgen/data/build_hdf5.py --split train --date 20260304 --dry-run

# Submit SLURM job
sbatch scripts/build_hdf5.sh
```

### 3. Train
```bash
sbatch scripts/train.sh
```

## Data

Raw data lives on CFS (read-only):
- `13k set` : `/global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data/`
- `2502 set` : `/global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data_2502/`
- `252 set`  : `/global/cfs/cdirs/m1090/dongin/cuau_fcc_101010_data_252/`  ← test

Processed HDF5 on pscratch (fast NVMe):
- `train.h5` : 15502 samples (~185 GB)
- `test.h5`  :   252 samples (~3 GB)

## Dataset Summary

| Parameter | Value |
|---|---|
| System | Cu–Au FCC random alloy, 10×10×10 supercell |
| Tilts | 15 (−7° to +7°, step 1°, y-axis) |
| DP resolution | 409 × 409 px |
| ROI | 15 × 15 × 50 Å |
| Train samples | 15,502 |
| Test samples | 252 |