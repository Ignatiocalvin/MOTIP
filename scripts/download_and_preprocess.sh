#!/bin/bash

# ================================================================== #
#  MOTIP Setup Script                                                 #
#  Creates conda env, installs dependencies, builds CUDA ops         #
# ================================================================== #

set -e  # Exit on any error

# Get the MOTIP root directory (parent of scripts/)
MOTIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "${MOTIP_ROOT}"

echo "========================================"
echo "MOTIP Environment Setup"
echo "========================================"
echo "Working directory: ${MOTIP_ROOT}"
echo ""

# ------------------------------------------------------------------ #
# 1. Create conda environment with Python 3.12                        #
# ------------------------------------------------------------------ #
ENV_NAME="MOTIP"

# Initialize conda for this shell
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda. Please install miniconda or anaconda first."
    exit 1
fi

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[ENV] Conda env '$ENV_NAME' already exists."
else
    echo "[ENV] Creating conda env '$ENV_NAME' with Python 3.12 ..."
    conda create -n "$ENV_NAME" python=3.12 -y
fi

echo "[ENV] Activating $ENV_NAME ..."
conda activate "$ENV_NAME"
echo "[ENV] Python: $(which python) ($(python --version))"
echo ""

# ------------------------------------------------------------------ #
# 2. Install Python dependencies                                      #
# ------------------------------------------------------------------ #
echo "[PIP] Installing Python dependencies..."
pip install -r "${MOTIP_ROOT}/requirements.txt"
echo "[PIP] Done."
echo ""

# ------------------------------------------------------------------ #
# 3. Download pretrained model                                        #
# ------------------------------------------------------------------ #
mkdir -p "${MOTIP_ROOT}/pretrains"
if [ ! -f "${MOTIP_ROOT}/pretrains/r50_deformable_detr_coco.pth" ]; then
    echo "[PRETRAIN] Downloading R50 pretrained model..."
    wget -O "${MOTIP_ROOT}/pretrains/r50_deformable_detr_coco.pth" \
        https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco.pth
    echo "[PRETRAIN] Done."
else
    echo "[PRETRAIN] r50_deformable_detr_coco.pth already exists, skipping."
fi
echo ""

# ------------------------------------------------------------------ #
# 4. P-DESTRE Dataset (already downloaded — uncomment to re-run)     #
# ------------------------------------------------------------------ #
echo "========================================"
echo "P-DESTRE Dataset Setup"
echo "========================================"
echo "Working directory: ${MOTIP_ROOT}"
echo ""

# Create data directory if it doesn't exist
# mkdir -p data
# cd data

# Download the dataset
# echo "Downloading P-DESTRE dataset..."
# wget https://socia-lab.di.ubi.pt/%7Ehugomcp/dataset.tar

# Extract the tar file
# echo "Extracting dataset..."
# tar -xf dataset.tar

# Check if P-DESTRE directory exists
# if [ ! -d "P-DESTRE" ]; then
#     echo "Error: P-DESTRE directory not found after extraction"
#     exit 1
# fi

# cd P-DESTRE

# Rename annotation folder to annotations if needed
# if [ -d "annotation" ] && [ ! -d "annotations" ]; then
#     mv annotation annotations
# fi

# Remove excluded sequence files
# rm -f annotations/22-10-2019-1-2.txt videos/22-10-2019-1-2.MP4
# rm -f annotations/13-11-2019-4-3.txt videos/13-11-2019-4-3.MP4

# Run the preprocessing script from data/P-DESTRE/
# python3 preprocess_pdestre.py

# Clean up the downloaded tar file
# cd ..
# rm -f dataset.tar

cd "${MOTIP_ROOT}"
echo "[P-DESTRE] Dataset already preprocessed — sections above commented out."
echo ""

# ------------------------------------------------------------------ #
# 5. Build CUDA Operators (skip if no GPU - will build in SLURM job) #
# ------------------------------------------------------------------ #
echo "========================================"
echo "Building CUDA Operators"
echo "========================================"

# Check if CUDA is available
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Building MultiScaleDeformableAttention CUDA extension..."
    cd "${MOTIP_ROOT}/models/ops"
    python setup.py build install
    cd "${MOTIP_ROOT}"
    echo "CUDA operators built successfully!"
else
    echo "[SKIP] No GPU available on login node."
    echo "       CUDA ops will be built when you submit a SLURM job."
fi

# ------------------------------------------------------------------ #
# 6. Done                                                             #
# ------------------------------------------------------------------ #
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment in future sessions:"
echo "    conda activate $ENV_NAME"
echo ""
echo "To run a smoke test, submit a SLURM job:"
echo "    sbatch setup_and_smoke.sh"
echo ""