#!/bin/bash
#──────────────────────────────────────────────────────────────────────────────
# train_smoke.sh — Quick smoke test (3 epochs, 1 GPU, 30 min max)
#──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=temgen_smoke
#SBATCH --account=m3828
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o /pscratch/sd/d/dongin/temgen/logs/slurm/smoke_%j.out
#SBATCH -e /pscratch/sd/d/dongin/temgen/logs/slurm/smoke_%j.err

# ─── Config ───────────────────────────────────────────────────────────────────
TEMGEN_DIR="/pscratch/sd/d/dongin/temgen"
CONFIG="$TEMGEN_DIR/configs/cuau_101010_smoke.yaml"
CKPT_DIR="$TEMGEN_DIR/checkpoints/smoke_$SLURM_JOB_ID"
LOG_DIR="$TEMGEN_DIR/logs"

# ─── Environment ──────────────────────────────────────────────────────────────
module load conda
conda activate temgen

cd "$TEMGEN_DIR"
export PYTHONPATH="$TEMGEN_DIR:$PYTHONPATH"

# ─── Create directories ──────────────────────────────────────────────────────
mkdir -p "$CKPT_DIR"
mkdir -p "$LOG_DIR/slurm"
mkdir -p "$LOG_DIR/tensorboard"

# ─── Print job info ───────────────────────────────────────────────────────────
echo "========================================"
echo "SMOKE TEST"
echo "Job ID        : $SLURM_JOB_ID"
echo "QOS           : debug (30 min, fast queue)"
echo "GPUs          : 4"
echo "Config        : $CONFIG"
echo "Checkpoint dir: $CKPT_DIR"
echo "Start time    : $(date)"
echo "========================================"

# ─── Launch training ──────────────────────────────────────────────────────────
srun python "$TEMGEN_DIR/scripts/train.py" \
    --config "$CONFIG" \
    --ckpt-dir "$CKPT_DIR" \
    --log-dir "$LOG_DIR/tensorboard" \
    --nodes 1 \
    --gpus-per-node 4

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Smoke test finished at $(date)"
echo "Exit code     : $EXIT_CODE"
echo "Checkpoints   :"
ls -lht "$CKPT_DIR/" 2>/dev/null | head -5
echo "========================================"

exit $EXIT_CODE
