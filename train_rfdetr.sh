#!/usr/bin/env bash
#SBATCH --job-name=motip_rfdetr
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time 10:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/motip_rfdetr_%j.out
#SBATCH --error=logs/motip_rfdetr_%j.err

echo "=========================================="
echo "MOTIP RF-DETR Training Script"
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
# Install RF-DETR dependencies (first run only)
# ========================================
echo "Checking RF-DETR dependencies..."
python -c "import transformers, timm, supervision, pydantic, pycocotools, fairscale, einops, peft, scipy, open_clip_torch, pylabel, pandas, roboflow" 2>/dev/null || {
    echo "Installing RF-DETR dependencies (this may take a few minutes)..."
    pip install --quiet \
        cython \
        pycocotools \
        fairscale \
        scipy \
        timm \
        transformers \
        peft \
        einops \
        pandas \
        pylabel \
        open_clip_torch \
        rf100vl \
        pydantic \
        supervision \
        matplotlib \
        roboflow
    echo "RF-DETR dependencies installed successfully!"
}

# ========================================
# USAGE OPTIONS:
# ========================================
# 1. Fresh training with RF-DETR:
#    accelerate launch --num_processes=1 train.py \
#      --data-root ./data/ \
#      --exp-name rfdetr_motip_pdestre \
#      --config-path ./configs/rfdetr_medium_motip_pdestre.yaml

# 2. Resume from checkpoint:
#    accelerate launch --num_processes=1 train.py \
#      --data-root ./data/ \
#      --exp-name rfdetr_motip_pdestre \
#      --config-path ./configs/rfdetr_medium_motip_pdestre.yaml \
#      --resume-model ./outputs/rfdetr_motip_pdestre/checkpoint_epoch_X.pth

# ========================================
# Run RF-DETR training
# ========================================
echo "Starting RF-DETR + MOTIP training..."
echo "Model: RF-DETR Medium (DINOv2 backbone)"
echo "Resolution: 576x576"
echo "Dataset: P-DESTRE"

accelerate launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name rfdetr_motip_pdestre \
    --config-path ./configs/rfdetr_medium_motip_pdestre.yaml

echo "Finished at $(date)"

# First run - batch 2947635 (outputs: rfdetr_motip_pdestre)
