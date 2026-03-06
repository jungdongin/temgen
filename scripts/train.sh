#!/bin/bash
#──────────────────────────────────────────────────────────────────────────────
# train.sh — TEMGen contrastive learning on Perlmutter A100 GPUs
#
# QOS cheat sheet (GPU):
#   regular  : CF=1.0,  max 48hr,  full price
#   preempt  : CF=0.25, min 2hr guaranteed, then preemptible → 75% cheaper
#   premium  : CF=2–4,  higher priority
#   shared   : fraction-node, 1–2 GPUs
#   debug    : max 0.5hr, 8 nodes, fast turnaround
#
# Big job discount: 128+ GPU nodes in regular QOS → 50% off (CF=0.5).
#   Relevant only for massive scaling (512+ GPUs). Not typical for training.
#   If you ever scale to 128+ nodes, switch to regular QOS for the discount.
#
# Cost estimates (1 node = 4× A100 80GB):
#   1 node × 24hr × regular = 24 node-hours
#   1 node × 24hr × preempt = 6 node-hours  (75% savings!)
#   128 nodes × 24hr × regular + big job = 1536 node-hours (50% off)
#
# Usage:
#   sbatch scripts/train.sh                    # default: preempt, 1 node
#   NODES=4 sbatch scripts/train.sh            # 4-node DDP
#   sbatch --qos=regular scripts/train.sh      # override to regular QOS
#──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=temgen_train
#SBATCH --account=m3828
#SBATCH --qos=preempt
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120
#SBATCH -o /pscratch/sd/d/dongin/temgen/logs/slurm/train_%j.out
#SBATCH -e /pscratch/sd/d/dongin/temgen/logs/slurm/train_%j.err

# ─── Config ───────────────────────────────────────────────────────────────────
TEMGEN_DIR="/pscratch/sd/d/dongin/temgen"
CONFIG="$TEMGEN_DIR/configs/cuau_101010.yaml"
CKPT_DIR="$TEMGEN_DIR/checkpoints"
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

# ─── Multi-node DDP setup ─────────────────────────────────────────────────────
# Perlmutter: 4 × A100 80GB per node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

NNODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# ─── Resume from checkpoint if requeued ───────────────────────────────────────
# preempt QOS + --requeue: job is requeued on preemption.
# Lightning saves last.ckpt; we auto-resume from it.
RESUME_ARG=""
LAST_CKPT="$CKPT_DIR/last.ckpt"
if [ -f "$LAST_CKPT" ]; then
    echo ">>> Found $LAST_CKPT — resuming training"
    RESUME_ARG="--resume-from $LAST_CKPT"
fi

# ─── Print job info ───────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Job name      : $SLURM_JOB_NAME"
echo "QOS           : $SLURM_JOB_QOS"
echo "Nodes         : $NNODES  ($SLURM_JOB_NODELIST)"
echo "GPUs/node     : $GPUS_PER_NODE"
echo "World size    : $WORLD_SIZE"
echo "Master        : $MASTER_ADDR:$MASTER_PORT"
echo "Config        : $CONFIG"
echo "Checkpoint dir: $CKPT_DIR"
echo "Resume        : ${RESUME_ARG:-fresh start}"
echo "Start time    : $(date)"
echo "========================================"

# ─── Launch training ──────────────────────────────────────────────────────────
srun python "$TEMGEN_DIR/scripts/train.py" \
    --config "$CONFIG" \
    --ckpt-dir "$CKPT_DIR" \
    --log-dir "$LOG_DIR/tensorboard" \
    --nodes "$NNODES" \
    --gpus-per-node "$GPUS_PER_NODE" \
    $RESUME_ARG

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job finished at $(date)"
echo "Exit code     : $EXIT_CODE"
echo "Checkpoints   :"
ls -lht "$CKPT_DIR/" 2>/dev/null | head -5
echo "========================================"

exit $EXIT_CODE