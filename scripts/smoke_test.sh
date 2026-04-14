#!/bin/bash
#SBATCH --job-name=motip_smoke_test
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# ============================================================
# MOTIP Smoke Test
# Quick validation: 2 epochs, 20 steps per epoch
# Expected runtime: ~5-10 minutes on H100
# ============================================================

set -e

# Navigate to MOTIP directory
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

# Load CUDA toolkit (required for building CUDA extensions)
module load devel/cuda/11.8 2>/dev/null || true
export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create logs directory if needed
mkdir -p logs

echo "========================================"
echo "  MOTIP Smoke Test"
echo "  Start: $(date)"
echo "  Node : $(hostname)"
echo "========================================"
echo "[ENV] Python: $(which python) ($(python --version))"
echo "[ENV] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[ENV] CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "[ENV] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[ENV] CUDA_HOME: ${CUDA_HOME:-not set}"
echo ""

# Verify CUDA is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[ERROR] CUDA is not available! Exiting."
    exit 1
fi

# Build CUDA operators if needed
if ! python -c "from models.ops import MultiScaleDeformableAttention" 2>/dev/null; then
    echo "[BUILD] Building MultiScaleDeformableAttention CUDA extension..."
    cd models/ops
    python setup.py build install
    cd ../..
    echo "[BUILD] Done!"
fi

# Clear any previous smoke test outputs
rm -rf outputs/smoke_test

# Run smoke test training
echo ""
echo "[TRAIN] Starting smoke test training..."
python train.py \
    --config-path configs/smoke_test.yaml \
    --exp-name smoke_test

echo ""
echo "[DONE] Smoke test completed successfully!"
echo "  End: $(date)"
echo "========================================"

# Verify outputs
if [ -f "outputs/smoke_test/checkpoint_1.pth" ]; then
    echo "[CHECK] ✅ Checkpoint found: outputs/smoke_test/checkpoint_1.pth"
else
    echo "[CHECK] ❌ Checkpoint NOT found!"
    exit 1
fi
