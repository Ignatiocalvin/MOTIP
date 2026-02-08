#!/bin/bash
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_motip
#SBATCH --mem=32G
#SBATCH --output=logs/eval_fold_%j.out
#SBATCH --error=logs/eval_fold_%j.err

# Evaluation script for MOTIP fold
# Usage: sbatch evaluate_fold.sh <fold_number>
# Example: sbatch evaluate_fold.sh 0

FOLD=${1:-0}

echo "=========================================="
echo "MOTIP Evaluation Script"
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "Fold: $FOLD"
echo "=========================================="

# Check GPU
echo "Checking GPU availability..."
nvidia-smi
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

# Activate environment
source ~/miniconda3/bin/activate MOTIP

# Change to MOTIP root directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$0")")}"

# Run evaluation
echo "Starting evaluation for fold_${FOLD}..."
python evaluation/submit_and_evaluate.py \
    --config-path configs/r50_deformable_detr_motip_pdestre_fast.yaml \
    --inference-model outputs/r50_motip_pdestre_fold_${FOLD}/checkpoint_2.pth \
    --inference-group fold_${FOLD} \
    --inference-dataset P-DESTRE \
    --inference-split val_${FOLD} \
    --outputs-dir outputs/r50_motip_pdestre_fold_${FOLD}

echo "=========================================="
echo "Evaluation completed at $(date)"
echo "=========================================="
