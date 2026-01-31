#!/bin/bash

#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=compile_ops_h100
#SBATCH --output=compile_h100_%j.out
#SBATCH --error=compile_h100_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Load modules
module load devel/cuda/11.8
module load devel/python/3.12.3_intel_2023.1.0

echo "=== Compiling Deformable Attention CUDA ops for A100 (sm_80) and H100 (sm_90) ==="
echo "Current directory: $(pwd)"

# Activate conda environment
source /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

echo "Using Python from: $(which python)"
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import sys; print(f'Python executable: {sys.executable}')"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Set CUDA architecture for both A100 (8.0) and H100 (9.0)
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CUDA_HOME='/opt/bwhpc/common/devel/cuda/11.8'

echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "CUDA_HOME: $CUDA_HOME"

# Check GPU is available
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Clean previous build
rm -rf build/
rm -rf *.egg-info
rm -rf dist/

# Build and install
echo "Building and installing..."
python setup.py build install

echo "=== Compilation complete ==="
