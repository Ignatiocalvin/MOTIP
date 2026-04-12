#!/bin/bash
#SBATCH --job-name=motip_rfdetr_7c_ep6
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/motip_rfdetr_7c_ep6_%j.out
#SBATCH --error=logs/motip_rfdetr_7c_ep6_%j.err
#SBATCH --exclude=uc3n073
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# IMMEDIATELY strip .venv from PATH before anything else
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p logs
    LOG_FILE="logs/motip_rfdetr_7c_ep6_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to: $LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=========================================="
echo "MOTIP RF-DETR Large 7-Concept (NO learnable weights) - Resume from checkpoint_6"
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "=========================================="

if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || echo "Could not load CUDA module, assuming CUDA is already available"
fi

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
if [ ! -d "$CONDA_BASE" ]; then CONDA_BASE="$HOME/miniconda3"; fi
if [ ! -d "$CONDA_BASE" ]; then CONDA_BASE="$HOME/anaconda3"; fi
if [ ! -d "$CONDA_BASE" ]; then echo "ERROR: Could not find conda installation"; exit 1; fi

eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate MOTIP
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
if [[ "$PYTHON_PATH" != *"miniconda3/envs/MOTIP"* ]] && [[ "$PYTHON_PATH" != *"anaconda3/envs/MOTIP"* ]]; then
    echo "ERROR: Not using MOTIP environment! Python: $PYTHON_PATH"; exit 1
fi
echo "✓ Using Python from: $PYTHON_PATH"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *) export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" ;;
esac

if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif [ -d "/opt/bwhpc/common/devel/cuda/11.8" ]; then
    export CUDA_HOME="/opt/bwhpc/common/devel/cuda/11.8"
fi
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
cd "$SCRIPT_DIR"

echo ""
echo "Resuming: rfdetr_large_motip_pdestre_7concepts_learnable_fold0 (checkpoint_6)"
echo "Config: rfdetr_large_motip_pdestre_7concepts_nolw_fold0.yaml (USE_LEARNABLE_TASK_WEIGHTS=False)"
echo "Output folder: outputs/rfdetr_large_motip_pdestre_7concepts_learnable_fold0 (same as before)"
echo "NaN fix applied in models/rfdetr/criterion.py (nan_to_num before linear_sum_assignment)"
echo ""

env PYTHONPATH="${SCRIPT_DIR}/rf-detr:${PYTHONPATH}" \
python -u -m accelerate.commands.launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name rfdetr_large_motip_pdestre_7concepts_learnable_fold0 \
    --config-path ./configs/rfdetr_large_motip_pdestre_7concepts_nolw_fold0.yaml \
    --resume-model ./outputs/rfdetr_large_motip_pdestre_7concepts_learnable_fold0/checkpoint_6.pth \
    2>&1 | while IFS= read -r line; do echo "$(date +'%Y-%m-%d %H:%M:%S') $line"; done

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
