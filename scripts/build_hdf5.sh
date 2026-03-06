#!/bin/bash
#SBATCH --job-name=cuau_build_hdf5
#SBATCH --account=m3828
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH -o /pscratch/sd/d/dongin/temgen/logs/slurm/build_hdf5_%j.out
#SBATCH -e /pscratch/sd/d/dongin/temgen/logs/slurm/build_hdf5_%j.err

# ─── Config ───────────────────────────────────────────────────────────────────
TEMGEN_DIR="/pscratch/sd/d/dongin/temgen"
SCRIPT="$TEMGEN_DIR/temgen/data/build_hdf5.py"
DATE_TAG="20260304"       # ← update this when rebuilding with new data

# ─── Environment ──────────────────────────────────────────────────────────────
module load conda
conda activate temgen

# ─── Setup dirs ───────────────────────────────────────────────────────────────
mkdir -p "$TEMGEN_DIR/data/hdf5"
mkdir -p "$TEMGEN_DIR/logs/slurm"

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Date tag : $DATE_TAG"
echo "Start    : $(date)"
echo "========================================"

# ─── Build train.h5 ───────────────────────────────────────────────────────────
echo ""
echo ">>> Building train_${DATE_TAG}.h5"
python "$SCRIPT" --split train --date "$DATE_TAG"

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: train build failed (exit $TRAIN_EXIT). Check logs."
    exit $TRAIN_EXIT
fi

# ─── Build test.h5 ────────────────────────────────────────────────────────────
echo ""
echo ">>> Building test_${DATE_TAG}.h5"
python "$SCRIPT" --split test --date "$DATE_TAG"

TEST_EXIT=$?
if [ $TEST_EXIT -ne 0 ]; then
    echo "ERROR: test build failed (exit $TEST_EXIT). Check logs."
    exit $TEST_EXIT
fi

echo ""
echo "========================================"
echo "All done at $(date)"
echo "Files:"
ls -lh "$TEMGEN_DIR/data/hdf5/"
echo "========================================"