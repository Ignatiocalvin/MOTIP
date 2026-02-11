#!/usr/bin/env bash
#SBATCH --job-name=motip_no_cbm
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/motip_fold_3_no_concepts_%j.out
#SBATCH --error=logs/motip_fold_3_no_concepts_%j.err

# ========================================
# Setup logging for non-sbatch execution
# ========================================
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running under Slurm, setup manual logging
    mkdir -p logs
    LOG_FILE="logs/motip_fold_3_no_concepts_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to: $LOG_FILE"
    # Redirect all output to log file while also displaying on terminal
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=========================================="
echo "MOTIP Training WITHOUT Concept Bottleneck"
echo "Baseline MOTIP for comparison (fold_3)"
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "========================================="

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

# Load CUDA module (HPC only)
if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || echo "Could not load CUDA module, assuming CUDA is already available"
fi

# Set CUDA environment
export TORCH_CUDA_ARCH_LIST="8.9"
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif [ -d "/opt/bwhpc/common/devel/cuda/11.8" ]; then
    export CUDA_HOME="/opt/bwhpc/common/devel/cuda/11.8"
fi
export CUDA_VISIBLE_DEVICES=0
# Force unbuffered Python output so logs appear immediately
export PYTHONUNBUFFERED=1

echo "Checking GPU availability..."
nvidia-smi || echo "nvidia-smi not available"

echo "=========================================="
echo "Training baseline MOTIP (no concepts)"
echo "Expected time per epoch: ~24 hours"
echo "NUM_WORKERS=0 (single-threaded data loading)"
echo "=========================================="

# Use python directly instead of accelerate to simplify debugging
python train.py \
    --data-root ./data/ \
    --exp-name r50_motip_pdestre_fold_3_no_concepts \
    --config-path ./configs/r50_deformable_detr_motip_pdestre_standard.yaml

echo "Finished at $(date)"