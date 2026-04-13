#!/bin/bash
#SBATCH --job-name=motip_rfdetr_sam_smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=./logs/smoke_rfdetr_sam_%j.out
#SBATCH --error=./logs/smoke_rfdetr_sam_%j.err

set -e

WORKDIR=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP
cd "$WORKDIR"

echo "=== MOTIP RF-DETR + SAM Concept Bottleneck Smoke Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate conda environment
source /home/ma/ma_ma/ma_ighidaya/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

# Ensure PyTorch CUDA ops can resolve shared libs
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
echo "TORCH_LIB: $TORCH_LIB"

# Force HuggingFace offline mode - no network on compute nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Disable Python stdout buffering so log output appears immediately in file
export PYTHONUNBUFFERED=1

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo ""
echo "=== Starting RF-DETR + SAM smoke test (5 steps per log) ==="
echo ""

python train.py \
    --config-path ./configs/rfdetr_test_smoke_sam.yaml \
    --num-workers 0

echo ""
echo "=== RF-DETR SAM smoke test DONE ==="
echo "Date: $(date)"
