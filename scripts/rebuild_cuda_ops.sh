#!/bin/bash
#SBATCH --job-name=rebuild_ops
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=16gb
#SBATCH --output=logs/rebuild_ops_%j.out
#SBATCH --error=logs/rebuild_ops_%j.err

# Force conda environment (disable .venv)
if [ -d ".venv" ]; then
    deactivate 2>/dev/null || true
    export VIRTUAL_ENV=""
fi

# Initialize conda for this shell
source /home/ma/ma_ma/ma_ighidaya/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

# Load CUDA module
module load devel/cuda/11.8

# Build for BOTH A100 (8.0) and H100 (9.0) - universal binary
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
echo "Building CUDA operators for architectures: $TORCH_CUDA_ARCH_LIST"

# Navigate to ops directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MOTIP_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "${MOTIP_ROOT}/models/ops"

# Clean old builds (including installed egg in site-packages)
rm -rf build dist *.egg-info
pip uninstall -y MultiScaleDeformableAttention 2>/dev/null || true

# Also clean from conda site-packages
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "$SITE_PKG"/MultiScaleDeformableAttention* 2>/dev/null || true

# Rebuild
python setup.py build install

echo "CUDA operators rebuilt successfully for A100 and H100!"
echo "Testing import..."
python -c "import MultiScaleDeformableAttention; print('Import successful!')"
