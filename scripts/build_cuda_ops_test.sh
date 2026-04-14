#!/bin/bash
#SBATCH --job-name=build_cuda_ops
#SBATCH --partition=gpu_h100_short 
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/logs/build_cuda_%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/logs/build_cuda_%j.err

# Build CUDA ops for MOTIP_test environment
echo "=== Building MultiScaleDeformableAttention CUDA Extension ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

export PYTHON=~/miniconda3/envs/MOTIP_test/bin/python

# Set CUDA
module load devel/cuda/11.8
export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8
echo "CUDA_HOME: $CUDA_HOME"
echo ""

# Change to ops directory
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/models/ops

echo "=== Current directory: $(pwd) ==="
echo ""

echo "=== Building CUDA extension ==="
$PYTHON setup.py build install 2>&1

echo ""
echo "=== Verifying installation ==="
$PYTHON -c "from models.ops import MultiScaleDeformableAttention; print('SUCCESS: MultiScaleDeformableAttention imported!')" 2>&1 || \
$PYTHON -c "import sys; sys.path.insert(0, '/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP'); from models.ops import MultiScaleDeformableAttention; print('SUCCESS: MultiScaleDeformableAttention imported!')"

echo ""
echo "=== Build Complete ==="
