#!/bin/bash
#SBATCH --job-name=motip_rfdetr_base_v4
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/motip_rfdetr_base_v4_%j.out
#SBATCH --error=logs/motip_rfdetr_base_v4_%j.err
#SBATCH --exclude=uc3n073
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate MOTIP
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
if [[ "$PYTHON_PATH" != *"miniconda3/envs/MOTIP"* ]]; then
    echo "ERROR: Not using MOTIP env: $PYTHON_PATH"; exit 1
fi
echo "Using Python: $PYTHON_PATH"

if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || true
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *)      export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" ;;
esac
[ -d "/usr/local/cuda" ] && export CUDA_HOME="/usr/local/cuda"
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "RF-DETR Large MOTIP P-DESTRE Base (v4) - Fold 0"
echo "v4 resumes from v3/checkpoint_1.pth after NaN divergence at epoch 2 step 30k."
echo "Key fix: RESUME_OPTIMIZER=false (fresh Adam moments) + fresh epoch 2 shuffle."
echo "Started at $(date)"
echo "Node: $(hostname)  GPU: $GPU_NAME"
echo "=========================================="

env PYTHONPATH="${SCRIPT_DIR}/rf-detr:${PYTHONPATH}" \
python -u -m accelerate.commands.launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name rfdetr_large_motip_pdestre_base_fold0_v4 \
    --config-path ./configs/rfdetr_large_motip_pdestre_base_fold0_v4.yaml \
    2>&1

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
