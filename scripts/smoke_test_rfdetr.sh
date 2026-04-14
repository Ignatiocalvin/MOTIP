#!/bin/bash
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --job-name=motip_rfdetr_smoke
#SBATCH --output=logs/smoke_test_rfdetr_%j.out
#SBATCH --error=logs/smoke_test_rfdetr_%j.err

########################################
# RF-DETR Smoke Test for MOTIP
# Tests DanceTrack dataset with RF-DETR backbone
########################################

echo "========================================"
echo "[RFDETR SMOKE TEST] Starting..."
echo "  Start: $(date)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "========================================"

# Load CUDA module (bwUniCluster specific)
module load devel/cuda/11.8
export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

# Navigate to MOTIP directory
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# Create logs directory if needed
mkdir -p logs

# Verify CUDA availability
echo "[CHECK] Verifying CUDA..."
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[ERROR] CUDA is not available! Exiting."
    exit 1
fi
echo "[CHECK] CUDA OK: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

# Build CUDA operators if needed
if ! python -c "from models.ops import MultiScaleDeformableAttention" 2>/dev/null; then
    echo "[BUILD] Building MultiScaleDeformableAttention CUDA extension..."
    cd models/ops
    python setup.py build install
    cd ../..
    echo "[BUILD] Done!"
fi

# Verify RF-DETR pretrain exists
if [ ! -f "pretrains/rf-detr-base-coco.pth" ]; then
    echo "[ERROR] RF-DETR pretrain not found: pretrains/rf-detr-base-coco.pth"
    exit 1
fi
echo "[CHECK] RF-DETR pretrain OK"

# Verify DanceTrack data exists
DANCETRACK_PATH="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP_SAM/datasets/DanceTrack"
if [ ! -d "$DANCETRACK_PATH/train" ]; then
    echo "[ERROR] DanceTrack train data not found: $DANCETRACK_PATH/train"
    exit 1
fi
echo "[CHECK] DanceTrack data OK"

# Clear any previous smoke test outputs
rm -rf outputs/smoke_test_rfdetr

# Run smoke test training
echo ""
echo "[TRAIN] Starting RF-DETR smoke test training..."
python train.py \
    --config-path configs/smoke_test_rfdetr.yaml \
    --exp-name smoke_test_rfdetr

echo ""
echo "[DONE] RF-DETR smoke test completed!"
echo "  End: $(date)"
echo "========================================"

# Verify outputs
if [ -f "outputs/smoke_test_rfdetr/checkpoint_1.pth" ]; then
    echo "[CHECK] ✅ Checkpoint found: outputs/smoke_test_rfdetr/checkpoint_1.pth"
else
    echo "[CHECK] ❌ Checkpoint NOT found!"
    exit 1
fi
