#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Unified R50 Deformable DETR Training Script for PDestre
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage: Edit the CONFIGURATION section below, then submit with:
#   sbatch train_r50.sh
#
# Supported configurations:
#   NUM_CONCEPTS: 0 (base), 2 (2concepts), 3 (3concepts)
#   FOLD: 0, 1, 2, 3, 4
#   RESUME_MODE: "none", "auto", or "manual:/path/to/checkpoint.pth"
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── CONFIGURATION (edit these) ────────────────────────────────────────────────
NUM_CONCEPTS=2               # 0=base, 2=2concepts, 3=3concepts
FOLD=0                       # Fold number for cross-validation
RESUME_MODE="auto"           # "none", "auto", or "manual:/path/to/checkpoint.pth"
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=motip_r50
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=logs/motip_r50_%j.out
#SBATCH --error=logs/motip_r50_%j.err
#SBATCH --exclude=uc3n073
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# ═══════════════════════════════════════════════════════════════════════════════
# SCRIPT BODY (do not edit below unless necessary)
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
SCRIPT_PATH="${SCRIPT_DIR}/train_r50.sh"

# ── Build config path and experiment name based on NUM_CONCEPTS ───────────────
case $NUM_CONCEPTS in
    0)
        CONFIG="./configs/r50_motip_pdestre_base.yaml"
        EXP_NAME="r50_motip_pdestre_base_fold_${FOLD}"
        ;;
    2)
        CONFIG="./configs/r50_motip_pdestre_2concepts.yaml"
        EXP_NAME="r50_motip_pdestre_2concepts_fold_${FOLD}"
        ;;
    3)
        CONFIG="./configs/r50_motip_pdestre_3concepts.yaml"
        EXP_NAME="r50_motip_pdestre_3concepts_fold_${FOLD}"
        ;;
    7)
        CONFIG="./configs/r50_motip_pdestre_7concepts_lw.yaml"
        EXP_NAME="r50_motip_pdestre_7concepts_learnable_fold_${FOLD}"
        ;;
    *)
        echo "ERROR: NUM_CONCEPTS must be 0, 2, 3, or 7. Got: $NUM_CONCEPTS"
        exit 1
        ;;
esac

# ── Signal handler: stop training, resubmit self ──────────────────────────────
TRAINING_PID=""
resubmit() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SIGUSR1: time limit approaching — stopping and resubmitting..."
    if [ -n "$TRAINING_PID" ] && kill -0 "$TRAINING_PID" 2>/dev/null; then
        kill -TERM "$TRAINING_PID" 2>/dev/null
        for i in $(seq 1 60); do
            sleep 1
            kill -0 "$TRAINING_PID" 2>/dev/null || { echo "Training stopped."; break; }
        done
        kill -KILL "$TRAINING_PID" 2>/dev/null
    fi
    NEW_JOB=$(sbatch --parsable "$SCRIPT_PATH")
    echo "Resubmitted as job $NEW_JOB"
    exit 0
}
trap 'resubmit' SIGUSR1

# ── Environment setup ─────────────────────────────────────────────────────────
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p logs
    exec > >(tee -a "logs/motip_r50_${NUM_CONCEPTS}c_$(date +%Y%m%d_%H%M%S).log") 2>&1
fi

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
[ ! -d "$CONDA_BASE" ] && CONDA_BASE="$HOME/miniconda3"
[ ! -d "$CONDA_BASE" ] && CONDA_BASE="$HOME/anaconda3"
[ ! -d "$CONDA_BASE" ] && { echo "ERROR: conda not found"; exit 1; }

eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate MOTIP
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
if [[ "$PYTHON_PATH" != *"miniconda3/envs/MOTIP"* ]] && [[ "$PYTHON_PATH" != *"anaconda3/envs/MOTIP"* ]]; then
    echo "ERROR: Not using MOTIP environment! Python: $PYTHON_PATH"
    exit 1
fi

if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || true
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *)      export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" ;;
esac
export CUDA_VISIBLE_DEVICES=0
[ -d "/usr/local/cuda" ] && export CUDA_HOME="/usr/local/cuda"

echo "═══════════════════════════════════════════════════════════════"
echo "MOTIP R50 Training — ${NUM_CONCEPTS} concepts, fold ${FOLD}"
echo "───────────────────────────────────────────────────────────────"
echo "Config:  $CONFIG"
echo "ExpName: $EXP_NAME"
echo "Start:   $(date)"
echo "Node:    $(hostname)"
echo "GPU:     $GPU_NAME"
echo "Python:  $PYTHON_PATH"
echo "═══════════════════════════════════════════════════════════════"

# ── Determine resume strategy ─────────────────────────────────────────────────
OUTPUTS_DIR="./outputs/${EXP_NAME}"
INTRA_DIR="${OUTPUTS_DIR}/intra_epoch_checkpoints"
RESUME_ARGS=()

if [ "$RESUME_MODE" = "auto" ]; then
    if ls "${INTRA_DIR}"/*.pth 2>/dev/null | head -1 | grep -q . || \
       ls "${OUTPUTS_DIR}"/checkpoint_*.pth 2>/dev/null | head -1 | grep -q .; then
        echo "[checkpoint] Found — resuming from latest (--use-previous-checkpoint True)"
        RESUME_ARGS=(--use-previous-checkpoint True)
    else
        echo "[checkpoint] None found — starting fresh"
    fi
elif [[ "$RESUME_MODE" == manual:* ]]; then
    RESUME_PATH="${RESUME_MODE#manual:}"
    echo "[checkpoint] Manual resume from: $RESUME_PATH"
    RESUME_ARGS=(--resume-model "$RESUME_PATH")
else
    echo "[checkpoint] No resume — starting fresh"
fi

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "Starting training at $(date)..."
echo ""

accelerate launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name "$EXP_NAME" \
    --config-path "$CONFIG" \
    "${RESUME_ARGS[@]}" \
    > >(while IFS= read -r line; do echo "$(date +'%Y-%m-%d %H:%M:%S') $line"; done) 2>&1 &
TRAINING_PID=$!
echo "Training PID: $TRAINING_PID"
wait $TRAINING_PID
EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Training exited with code $EXIT_CODE at $(date)"
echo "═══════════════════════════════════════════════════════════════"
