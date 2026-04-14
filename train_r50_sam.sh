#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# R50 Deformable DETR + SAM Concept Bottleneck Training Script for PDestre
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage: Edit the CONFIGURATION section below, then submit with:
#   sbatch train_r50_sam.sh
#
# This script trains the R50 model with SAM concept bottleneck:
#   - Uses precomputed SAM masks for concept feature extraction
#   - Pools spatial features through segmentation masks
#   - Requires precomputed_sam_masks/ directory with mask files
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── CONFIGURATION (edit these) ────────────────────────────────────────────────
FOLD=0                       # Fold number for cross-validation
RESUME_MODE="none"           # "none", "auto", or "manual:/path/to/checkpoint.pth"
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=r50_sam
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=logs/r50_sam_%j.out
#SBATCH --error=logs/r50_sam_%j.err
#SBATCH --exclude=uc3n073
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# ═══════════════════════════════════════════════════════════════════════════════
# SCRIPT BODY (do not edit below unless necessary)
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
SCRIPT_PATH="${SCRIPT_DIR}/train_r50_sam.sh"

# Signal handler for requeue
handle_requeue() {
    echo "Received signal, requeuing job..."
    scontrol requeue $SLURM_JOB_ID
    exit 0
}
trap handle_requeue SIGUSR1

# Activate conda environment
source /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/anaconda3/etc/profile.d/conda.sh
conda activate MOTIP
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# R50 + SAM DanceTrack: single canonical config (no fold-based split)
CONFIG_FILE="configs/r50_deformable_detr_motip_sam_concept_dancetrack.yaml"

# ── Determine output directory ────────────────────────────────────────────────
OUTPUT_DIR="outputs/r50_motip_sam_concept_dancetrack"
mkdir -p "$OUTPUT_DIR"

# ── RESUME LOGIC ────────────────────────────────────────────────────────────
RESUME_ARGS=""
if [[ "$RESUME_MODE" == "auto" ]]; then
    # Find latest checkpoint
    LATEST_CKPT=$(ls -t ${OUTPUT_DIR}/checkpoint_*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Auto-resuming from: $LATEST_CKPT"
        RESUME_ARGS="--resume-model $LATEST_CKPT --resume-optimizer --resume-scheduler --resume-states"
    else
        echo "No checkpoint found for auto-resume. Starting fresh."
    fi
elif [[ "$RESUME_MODE" == manual:* ]]; then
    MANUAL_CKPT="${RESUME_MODE#manual:}"
    if [ -f "$MANUAL_CKPT" ]; then
        echo "Manual resume from: $MANUAL_CKPT"
        RESUME_ARGS="--resume-model $MANUAL_CKPT --resume-optimizer --resume-scheduler --resume-states"
    else
        echo "ERROR: Manual checkpoint not found: $MANUAL_CKPT"
        exit 1
    fi
else
    echo "Starting training from scratch (RESUME_MODE=$RESUME_MODE)"
fi

# ── VERIFY SAM MASKS EXIST ───────────────────────────────────────────────────
SAM_MASK_ROOT="./precomputed_sam_masks"
if [ ! -d "$SAM_MASK_ROOT" ]; then
    echo "WARNING: SAM mask directory not found: $SAM_MASK_ROOT"
    echo "Please precompute masks with precompute_sam_masks.py first."
fi

# ── RUN TRAINING ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo " R50 + SAM Concept Bottleneck Training"
echo " Config: $CONFIG_FILE"
echo " Output: $OUTPUT_DIR"
echo " SAM Masks: $SAM_MASK_ROOT"
echo " Resume: $RESUME_MODE"
echo "════════════════════════════════════════════════════════════════════"

python train.py \
    --config-path "$CONFIG_FILE" \
    --outputs-dir "$OUTPUT_DIR" \
    $RESUME_ARGS

echo "Training complete."
