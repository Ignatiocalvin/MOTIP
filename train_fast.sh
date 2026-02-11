#!/usr/bin/env bash
#SBATCH --job-name=motip_fixed_fold_3
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/motip_fixed_fold_3_%j.out
#SBATCH --error=logs/motip_fixed_fold_3_%j.err

# ========================================
# Setup logging for non-sbatch execution
# ========================================
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running under Slurm, setup manual logging
    mkdir -p logs
    LOG_FILE="logs/motip_fixed_fold_3_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to: $LOG_FILE"
    # Redirect all output to log file while also displaying on terminal
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=========================================="
echo "MOTIP Fixed Training Script - Fold 3"
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "========================================="

# Load CUDA module (HPC only)
if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || echo "Could not load CUDA module, assuming CUDA is already available"
fi

# Make sure we're not in any virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

# Source conda (if available)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate MOTIP
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate MOTIP
else
    echo "Conda not found, using system Python"
fi

# Set CUDA environment
export TORCH_CUDA_ARCH_LIST="8.9"
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif [ -d "/opt/bwhpc/common/devel/cuda/11.8" ]; then
    export CUDA_HOME="/opt/bwhpc/common/devel/cuda/11.8"
fi
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
#      --resume-model ./outputs/r50_motip_pdestre_fast/intra_epoch_checkpoints/epoch_3_step_${RESUME_STEP}.pth \
#      --resume-from-step ${RESUME_STEP}

# ========================================
# Run fast training (fresh start)
# ========================================
echo "Starting training with FIXED bbox format and 7 concepts..."
echo "Training fold 3 from scratch with corrected annotations"
echo "Expected: 30 epochs, ~24 hours per epoch"
echo "Intra-epoch checkpoints saved every 5000 steps"

accelerate launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name r50_motip_pdestre_fixed_fold_3 \
    --config-path ./configs/r50_deformable_detr_motip_pdestre_fast.yaml

echo "Finished at $(date)"
# Testing with Deformable DETR + MOTIP fast model
# first run - batch 2879446 (outputs: r50_motip_pdestre_fast_first)
# second run - batch 2912323 (outputs: r50_motip_pdestre_fast_second)
# third run - batch 2945733 (outputs: r50_motip_pdestre_fast) This should work! Fixed errors of the second run



