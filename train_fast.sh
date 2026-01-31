#!/usr/bin/env bash
#SBATCH --job-name=motip_fast
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=32:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/test_motip_fast_fold_3_%j.out
#SBATCH --error=logs/test_motip_fast_fold_3_%j.err

echo "=========================================="
echo "MOTIP Fast Training Script"
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Load CUDA module FIRST - this is critical!
module load devel/cuda/11.8

# Make sure we're not in any virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

# Set CUDA environment for H100 GPUs (compute capability 9.0)
export TORCH_CUDA_ARCH_LIST="9.0" 
export CUDA_HOME='/opt/bwhpc/common/devel/cuda/11.8'
export CUDA_VISIBLE_DEVICES=0

# Debug: Check GPU availability
echo "Checking GPU availability..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not available"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# If CUDA is not available, exit with error
if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: CUDA is not available! Check GPU allocation."
    echo "Trying to diagnose..."
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo "PATH: $PATH"
    lspci | grep -i nvidia || echo "No NVIDIA GPU found in lspci"
    exit 1
fi

# ========================================
# USAGE OPTIONS:
# ========================================
# 1. Fresh training with fast config:
#    accelerate launch --num_processes=1 train.py \
#      --data-root ./data/ \
#      --exp-name r50_motip_pdestre_fast \
#      --config-path ./configs/r50_deformable_detr_motip_pdestre_fast.yaml

# 2. Resume from mid-epoch checkpoint (e.g., after timeout):
#    RESUME_STEP=15000  # Set to the step from checkpoint filename
#    accelerate launch --num_processes=1 train.py \
#      --data-root ./data/ \
#      --exp-name r50_motip_pdestre_fast \
#      --config-path ./configs/r50_deformable_detr_motip_pdestre_fast.yaml \
#      --resume-model ./outputs/r50_motip_pdestre_fast/intra_epoch_checkpoints/epoch_0_step_${RESUME_STEP}.pth \
#      --resume-from-step ${RESUME_STEP}

# ========================================
# Run fast training (fresh start)
# ========================================
echo "Starting FAST training with optimized config..."
echo "Expected time per epoch: ~24 hours (down from ~80 hours)"
echo "Intra-epoch checkpoints saved every 5000 steps"
echo "Resuming from checkpoint_3.pth (after epoch 1)"

accelerate launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name test_r50_motip_pdestre_fold_3 \
    --config-path ./configs/r50_deformable_detr_motip_pdestre_fast.yaml \
    --resume-model ./outputs/r50_motip_pdestre_fold_3/checkpoint_1.pth

echo "Finished at $(date)"
# Testing with Deformable DETR + MOTIP fast model
# first run - batch 2879446 (outputs: r50_motip_pdestre_fast_first)
# second run - batch 2912323 (outputs: r50_motip_pdestre_fast_second)
# third run - batch 2945733 (outputs: r50_motip_pdestre_fast) This should work! Fixed errors of the second run



