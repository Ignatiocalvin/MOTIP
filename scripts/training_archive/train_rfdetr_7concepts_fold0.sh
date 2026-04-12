#!/bin/bash
#SBATCH --job-name=motip_rfdetr_7c_fold0
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/motip_rfdetr_7c_fold0_%j.out
#SBATCH --error=logs/motip_rfdetr_7c_fold0_%j.err
#SBATCH --exclude=uc3n073
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# IMMEDIATELY strip .venv from PATH before anything else
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

# ========================================
# Setup logging for non-sbatch execution
# ========================================
if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p logs
    LOG_FILE="logs/motip_rfdetr_7c_fold0_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to: $LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=========================================="
echo "MOTIP RF-DETR Large 7-Concept Learnable - Fold 0"
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Load CUDA module (HPC only)
if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || echo "Could not load CUDA module, assuming CUDA is already available"
fi

# Initialize conda properly for batch scripts
CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
if [ ! -d "$CONDA_BASE" ]; then
    CONDA_BASE="$HOME/miniconda3"
fi
if [ ! -d "$CONDA_BASE" ]; then
    CONDA_BASE="$HOME/anaconda3"
fi

if [ ! -d "$CONDA_BASE" ]; then
    echo "ERROR: Could not find conda installation"
    exit 1
fi

echo "Using conda from: $CONDA_BASE"

# Initialize conda for this shell
eval "$($CONDA_BASE/bin/conda shell.bash hook)"

# Activate MOTIP environment
conda activate MOTIP

# Force conda python to be first in PATH
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

echo "Activated conda MOTIP environment"

# Verify we're using the right Python
PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
if [[ "$PYTHON_PATH" == *".venv"* ]]; then
    echo "ERROR: Still using .venv! Python: $PYTHON_PATH"
    exit 1
fi
if [[ "$PYTHON_PATH" != *"miniconda3/envs/MOTIP"* ]] && [[ "$PYTHON_PATH" != *"anaconda3/envs/MOTIP"* ]]; then
    echo "ERROR: Not using MOTIP environment! Python: $PYTHON_PATH"
    exit 1
fi
echo "✓ Using Python from: $PYTHON_PATH"
echo "Python version: $(python --version)"

# Set CUDA environment
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *4070*|*4080*|*4090*) export TORCH_CUDA_ARCH_LIST="8.9" ;;
    *3090*|*3080*) export TORCH_CUDA_ARCH_LIST="8.6" ;;
    *) export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" ;;
esac
echo "Detected GPU: $GPU_NAME -> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif [ -d "/opt/bwhpc/common/devel/cuda/11.8" ]; then
    export CUDA_HOME="/opt/bwhpc/common/devel/cuda/11.8"
fi
export CUDA_VISIBLE_DEVICES=0

# Change to MOTIP directory (hardcoded - BASH_SOURCE[0] is unreliable under SLURM)
SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
cd "$SCRIPT_DIR"

echo "Working directory: $PWD"
echo ""
echo "Config: configs/rfdetr_large_motip_pdestre_7concepts_learnable.yaml"
echo "Exp name: rfdetr_large_motip_pdestre_7concepts_learnable_fold0"
echo "Splits: Train_0 / val_0 (fold 0)"
echo "Epochs: 3"
echo "N_CONCEPTS: 7 (learnable weights)"
echo ""

# Quick sanity check
env PYTHONPATH="${SCRIPT_DIR}/rf-detr:${PYTHONPATH}" \
python -u -c "
import sys
print(f'Python executable: {sys.executable}')
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
" || { echo "ERROR: Sanity check failed"; exit 1; }

# ========================================
# Train RF-DETR Large 7-Concept Fold 0
# ========================================
echo "Starting training at $(date)..."

env PYTHONPATH="${SCRIPT_DIR}/rf-detr:${PYTHONPATH}" \
python -u -m accelerate.commands.launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name rfdetr_large_motip_pdestre_7concepts_lw_fold0 \
    --config-path ./configs/rfdetr_large_motip_pdestre_7concepts_learnable.yaml \
    2>&1 | while IFS= read -r line; do echo "$(date +'%Y-%m-%d %H:%M:%S') $line"; done

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
