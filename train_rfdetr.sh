#!/usr/bin/env bash
#SBATCH --job-name=motip_rfdetr_all_folds
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/motip_rfdetr_all_folds_%j.out
#SBATCH --error=logs/motip_rfdetr_all_folds_%j.err

# ========================================
# Setup logging for non-sbatch execution
# ========================================
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running under Slurm, setup manual logging
    mkdir -p logs
    LOG_FILE="logs/motip_rfdetr_all_folds_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to: $LOG_FILE"
    # Redirect all output to log file while also displaying on terminal
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=========================================="
echo "MOTIP RF-DETR Training Script - All Folds"
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

# Set CUDA environment (adjust based on your GPU)
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4070 Ti SUPER (Ada Lovelace)
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
# Install dependencies (first run only)
# ========================================
echo "Checking dependencies..."
python -c "import wandb, transformers, timm, supervision, pydantic, pycocotools, fairscale, einops, peft, scipy, open_clip_torch, pylabel, pandas, roboflow" 2>/dev/null || {
    echo "Installing missing dependencies (this may take a few minutes)..."
    pip install --quiet \
        wandb \
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
    echo "Dependencies installed successfully!"
}

# Install rfdetr package if not already installed
echo "Checking rfdetr package..."
python -c "import rfdetr" 2>/dev/null || {
    echo "Installing rfdetr package..."
    pip install --quiet git+https://github.com/lyuwenyu/RT-DETR.git
    echo "rfdetr package installed successfully!"
}

# ========================================
# Build CUDA operators if needed
# ========================================
echo "Checking if CUDA operators are built..."
python -c "import MultiScaleDeformableAttention" 2>/dev/null || {
    echo "Building CUDA operators (first run only, may take a few minutes)..."
    cd models/ops
    python setup.py build install
    cd ../..
    echo "CUDA operators built successfully!"
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
# Run RF-DETR training for all folds
# ========================================
echo "Starting RF-DETR + MOTIP training for all folds..."
echo "Model: RF-DETR Medium (DINOv2 backbone)"
echo "Resolution: 588x588"
echo "Dataset: P-DESTRE with 7 concepts"
echo "Training 10 folds sequentially"
echo ""

# Loop through folds 0-9
for FOLD in {0..0}; do
    echo "=========================================="
    echo "Starting Fold ${FOLD} at $(date)"
    echo "=========================================="
    
    # Create a temporary config file for this fold
    FOLD_CONFIG="/tmp/rfdetr_fold_${FOLD}.yaml"
    cat > "$FOLD_CONFIG" << EOF
# Temporary config for fold ${FOLD}
SUPER_CONFIG_PATH: ./configs/rfdetr_medium_motip_pdestre.yaml

# Override dataset splits for this fold
DATASETS: [P-DESTRE]
DATASET_SPLITS: [Train_${FOLD}]
INFERENCE_DATASET: P-DESTRE
INFERENCE_SPLIT: val_${FOLD}
EOF
    
    accelerate launch --num_processes=1 train.py \
        --data-root ./data/ \
        --exp-name rfdetr_motip_pdestre_fold_${FOLD} \
        --config-path "$FOLD_CONFIG"
    
    # Clean up temporary config
    rm -f "$FOLD_CONFIG"
    
    echo "Completed Fold ${FOLD} at $(date)"
    echo ""
done

echo "=========================================="
echo "All folds completed at $(date)"
echo "=========================================="
