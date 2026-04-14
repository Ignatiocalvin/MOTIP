#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate a Single Checkpoint on a Chosen Test Split
# ═══════════════════════════════════════════════════════════════════════════════
#
# Interactive SLURM script — user specifies checkpoint path and split.
#
# Usage:
#   sbatch eval_checkpoint.sh <checkpoint> <split>
#
# Examples:
#   sbatch eval_checkpoint.sh outputs/r50_motip_pdestre_base_fold0/checkpoint_2.pth Test_0
#   sbatch eval_checkpoint.sh outputs/rfdetr_large_motip_pdestre_2concepts_fold0/checkpoint_0.pth val_0
#   sbatch eval_checkpoint.sh /absolute/path/to/checkpoint_5.pth Test_1
#
# The config is auto-discovered from the checkpoint's parent directory
# (looks for train/config.yaml next to the checkpoint).
#
# ═══════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=eval_ckpt
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/eval_ckpt_%j.out
#SBATCH --error=logs/eval_ckpt_%j.err
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# ── Parse arguments ──────────────────────────────────────────────────────────
CHECKPOINT="${1:?Usage: sbatch eval_checkpoint.sh <checkpoint_path> <split>}"
SPLIT="${2:?Usage: sbatch eval_checkpoint.sh <checkpoint_path> <split>}"

# ── Resolve checkpoint to absolute path ──────────────────────────────────────
if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="$(pwd)/$CHECKPOINT"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# ── Auto-discover config ─────────────────────────────────────────────────────
CKPT_DIR="$(dirname "$CHECKPOINT")"
CONFIG=""

# Try train/config.yaml in the same directory as the checkpoint
if [ -f "$CKPT_DIR/train/config.yaml" ]; then
    CONFIG="$CKPT_DIR/train/config.yaml"
# Try config.yaml in the same directory (some older runs)
elif [ -f "$CKPT_DIR/config.yaml" ]; then
    CONFIG="$CKPT_DIR/config.yaml"
# Try one level up (checkpoint might be in a subdirectory)
elif [ -f "$(dirname "$CKPT_DIR")/train/config.yaml" ]; then
    CONFIG="$(dirname "$CKPT_DIR")/train/config.yaml"
fi

if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Could not auto-discover config.yaml from checkpoint path."
    echo "        Looked in: $CKPT_DIR/train/, $CKPT_DIR/, $(dirname "$CKPT_DIR")/train/"
    echo "        You can set CONFIG manually by editing this script."
    exit 1
fi

# ── Derive output directory ──────────────────────────────────────────────────
CKPT_NAME="$(basename "$CHECKPOINT" .pth)"
OUTPUT_DIR="$CKPT_DIR/eval/PDESTRE_${SPLIT}/${CKPT_NAME}"

echo "═══════════════════════════════════════════════════════════════"
echo " Checkpoint Evaluation"
echo " Checkpoint: $CHECKPOINT"
echo " Config:     $CONFIG"
echo " Split:      $SPLIT"
echo " Output:     $OUTPUT_DIR"
echo " Started:    $(date)"
echo " Node:       $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

# ── Environment setup ─────────────────────────────────────────────────────────
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
[ ! -d "$CONDA_BASE" ] && CONDA_BASE="$HOME/miniconda3"
[ ! -d "$CONDA_BASE" ] && CONDA_BASE="$HOME/anaconda3"
[ ! -d "$CONDA_BASE" ] && { echo "ERROR: conda not found"; exit 1; }

eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate MOTIP
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

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

echo ""
echo "GPU:    $GPU_NAME"
echo "Python: $(python -c 'import sys; print(sys.executable)')"
echo ""

# ── Run evaluation ────────────────────────────────────────────────────────────
mkdir -p logs

accelerate launch evaluation/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --data-root ./data/ \
    --dataset P-DESTRE \
    --split "$SPLIT" \
    --output-dir "$OUTPUT_DIR" \
    --skip-existing

EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Finished — exit code $EXIT_CODE"
echo " Results:  $OUTPUT_DIR"
echo " Time:     $(date)"
echo "═══════════════════════════════════════════════════════════════"
