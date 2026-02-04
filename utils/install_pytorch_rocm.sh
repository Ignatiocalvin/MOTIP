#!/bin/bash

#SBATCH --partition=gpu_mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=install_pytorch_rocm
#SBATCH --output=install_pytorch_rocm_%j.out
#SBATCH --error=install_pytorch_rocm_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

echo "=== Installing PyTorch with ROCm support for MI300 ==="

# Load ROCm module
module load toolkit/rocm/6.3.1

# Activate conda environment
source /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

echo "Current PyTorch version:"
python -c "import torch; print(torch.__version__)"

echo ""
echo "Uninstalling CUDA version of PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Installing PyTorch with ROCm 6.2 support..."
# PyTorch 2.5.1 supports ROCm 6.2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/rocm6.2

echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'ROCm version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Installation complete ==="
