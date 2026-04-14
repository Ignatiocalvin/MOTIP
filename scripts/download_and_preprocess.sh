#!/bin/bash

# ================================================================== #
#  MOTIP Setup, Download & Preprocess Script                          #
#                                                                      #
#  This script handles the full setup pipeline:                        #
#    1. Create conda environment & install dependencies                #
#    2. Preprocess P-DESTRE (extract frames from videos)               #
#    3. Download pretrained weights (R50, RF-DETR, SAM)                #
#    4. Download DanceTrack dataset from HuggingFace                   #
#    5. Clone RF-DETR and apply compatibility fixes                    #
#    6. Build CUDA extension (if on a GPU node)                        #
#                                                                      #
#  PREREQUISITE: Download P-DESTRE manually from Google Drive and      #
#  extract it into data/ BEFORE running this script. See README.md.    #
#                                                                      #
# ================================================================== #
#
# Usage:
#   ./scripts/download_and_preprocess.sh             # Full setup
#   ./scripts/download_and_preprocess.sh --env-only  # Only conda env + pip
#   ./scripts/download_and_preprocess.sh --skip-env  # Skip env, do the rest
#
# Environment variables:
#   ENV_NAME  - Conda environment name (default: MOTIP)
#
# ================================================================== #

set -e  # Exit on any error

# ------------------------------------------------------------------ #
# Parse command-line arguments                                        #
# ------------------------------------------------------------------ #
DO_ENV=true
DO_REST=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-only)
            DO_REST=false
            shift
            ;;
        --skip-env)
            DO_ENV=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-only   Only setup conda environment + pip install"
            echo "  --skip-env   Skip environment setup, do everything else"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "PREREQUISITE: Download P-DESTRE from Google Drive first."
            echo "  See README.md Section 2 for the download link."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #
MOTIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "${MOTIP_ROOT}"

ENV_NAME="${ENV_NAME:-MOTIP}"
DATA_ROOT="${MOTIP_ROOT}/data"
PDESTRE_DIR="${DATA_ROOT}/P-DESTRE"
PRETRAINS_DIR="${MOTIP_ROOT}/pretrains"

echo "========================================================"
echo "  MOTIP Setup, Download & Preprocess"
echo "========================================================"
echo ""
echo "  Working directory:  ${MOTIP_ROOT}"
echo "  Conda environment:  ${ENV_NAME}"
echo "  Data directory:     ${DATA_ROOT}"
echo ""

# ------------------------------------------------------------------ #
# Helper: Initialize conda                                            #
# ------------------------------------------------------------------ #
init_conda() {
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: Could not find conda installation."
        echo "       Install miniconda3 or anaconda3 first."
        exit 1
    fi
}

# ================================================================== #
# 1. Environment Setup                                                #
# ================================================================== #
if [ "$DO_ENV" = true ]; then
    echo "========================================================"
    echo "  Step 1: Environment Setup"
    echo "========================================================"
    echo ""

    init_conda

    # Create conda environment if it doesn't exist
    if conda env list | grep -qE "^${ENV_NAME}\s"; then
        echo "[ENV] Conda environment '${ENV_NAME}' already exists."
    else
        echo "[ENV] Creating conda environment '${ENV_NAME}' (Python 3.12)..."
        conda create -n "${ENV_NAME}" python=3.12 -y
    fi

    echo "[ENV] Activating ${ENV_NAME}..."
    conda activate "${ENV_NAME}"
    echo "[ENV] Python: $(which python) ($(python --version))"
    echo ""

    # Install Python dependencies
    echo "[PIP] Installing requirements.txt..."
    pip install -r "${MOTIP_ROOT}/requirements.txt"

    # Extra packages needed for setup scripts
    pip install huggingface_hub --quiet
    pip install segment-anything --quiet

    echo "[PIP] Done."
    echo ""
fi

if [ "$DO_REST" = false ]; then
    echo "Environment setup complete (--env-only). Exiting."
    exit 0
fi

# Ensure conda is initialized and env is active for remaining steps
init_conda
conda activate "${ENV_NAME}" 2>/dev/null || true

# ================================================================== #
# 2. Preprocess P-DESTRE Dataset                                      #
# ================================================================== #
echo "========================================================"
echo "  Step 2: Preprocess P-DESTRE Dataset"
echo "========================================================"
echo ""

# Check that P-DESTRE was downloaded and extracted
if [ ! -d "${PDESTRE_DIR}" ]; then
    echo "ERROR: P-DESTRE directory not found at ${PDESTRE_DIR}"
    echo ""
    echo "  You must download P-DESTRE manually before running this script."
    echo "  See README.md Section 2 for the Google Drive download link."
    echo ""
    echo "  After downloading, upload and extract it:"
    echo "    scp dataset.tar <user>@bwunicluster.scc.kit.edu:${DATA_ROOT}/"
    echo "    cd ${DATA_ROOT} && tar -xf dataset.tar"
    echo ""
    exit 1
fi

# 2a. Rename annotation/ → annotations/ (raw tar uses singular)
if [ -d "${PDESTRE_DIR}/annotation" ] && [ ! -d "${PDESTRE_DIR}/annotations" ]; then
    mv "${PDESTRE_DIR}/annotation" "${PDESTRE_DIR}/annotations"
    echo "[P-DESTRE] Renamed 'annotation/' → 'annotations/'"
elif [ -d "${PDESTRE_DIR}/annotations" ]; then
    echo "[P-DESTRE] 'annotations/' directory already exists."
else
    echo "WARNING: Neither 'annotation/' nor 'annotations/' found in ${PDESTRE_DIR}"
fi

# 2b. Remove known problematic sequences
#     These have corrupted annotations or missing data.
REMOVE_SEQUENCES=("22-10-2019-1-2" "13-11-2019-4-3")
for seq in "${REMOVE_SEQUENCES[@]}"; do
    removed=false
    if [ -f "${PDESTRE_DIR}/annotations/${seq}.txt" ]; then
        rm "${PDESTRE_DIR}/annotations/${seq}.txt"
        removed=true
    fi
    if [ -f "${PDESTRE_DIR}/videos/${seq}.MP4" ]; then
        rm "${PDESTRE_DIR}/videos/${seq}.MP4"
        removed=true
    fi
    if [ "$removed" = true ]; then
        echo "[P-DESTRE] Removed problematic sequence: ${seq}"
    fi
done

# 2c. Extract frames from MP4 videos → images/{seq}/img1/*.jpg
if [ -d "${PDESTRE_DIR}/videos" ] && [ "$(ls -A "${PDESTRE_DIR}/videos"/*.MP4 2>/dev/null)" ]; then
    echo "[P-DESTRE] Extracting frames from videos (this may take a while)..."
    cd "${PDESTRE_DIR}"
    python preprocess_pdestre.py \
        --ann_dir ./annotations \
        --video_dir ./videos \
        --converted_img_dir ./images
    cd "${MOTIP_ROOT}"
    echo "[P-DESTRE] Frame extraction complete."

    # Delete videos directory to reclaim ~30GB of space
    if [ -d "${PDESTRE_DIR}/videos" ]; then
        echo "[P-DESTRE] Removing videos/ directory to free disk space..."
        rm -rf "${PDESTRE_DIR}/videos"
        echo "[P-DESTRE] Removed videos/ directory."
    fi
elif [ -d "${PDESTRE_DIR}/images" ] && [ "$(ls -A "${PDESTRE_DIR}/images" 2>/dev/null)" ]; then
    echo "[P-DESTRE] Frames already extracted (images/ directory exists). Skipping."
else
    echo "WARNING: No videos found and no images directory exists."
    echo "         P-DESTRE may not have been extracted correctly."
fi

# 2d. Verify splits exist
if [ -d "${PDESTRE_DIR}/splits" ]; then
    echo "[P-DESTRE] Splits directory found."
else
    echo "WARNING: No splits/ directory in ${PDESTRE_DIR}."
    echo "         Pre-generated splits are in ${MOTIP_ROOT}/splits/"
fi

# Clean up the downloaded tar file if it's still in data/
for tarfile in "${DATA_ROOT}/dataset.tar" "${DATA_ROOT}/dataset.tar.1"; do
    if [ -f "$tarfile" ]; then
        echo "[P-DESTRE] Removing ${tarfile} to free disk space..."
        rm "$tarfile"
    fi
done

echo ""

# ================================================================== #
# 3. Download Pretrained Weights                                      #
# ================================================================== #
echo "========================================================"
echo "  Step 3: Download Pretrained Weights"
echo "========================================================"
echo ""

mkdir -p "${PRETRAINS_DIR}"

# R50 Deformable DETR (required for all R50 experiments)
if [ -f "${PRETRAINS_DIR}/r50_deformable_detr_coco.pth" ]; then
    echo "[WEIGHTS] r50_deformable_detr_coco.pth already exists."
else
    echo "[WEIGHTS] Downloading R50 Deformable DETR COCO pretrain (467 MB)..."
    wget -q --show-progress \
        https://github.com/fundamentalvision/Deformable-DETR/releases/download/v0.1/r50_deformable_detr-checkpoint.pth \
        -O "${PRETRAINS_DIR}/r50_deformable_detr_coco.pth"
    echo "[WEIGHTS] Done."
fi

# SAM ViT-B (needed for SAM mask experiments)
if [ -f "${PRETRAINS_DIR}/sam_vit_b_01ec64.pth" ]; then
    echo "[WEIGHTS] sam_vit_b_01ec64.pth already exists."
else
    echo "[WEIGHTS] Downloading SAM ViT-B (358 MB)..."
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
        -O "${PRETRAINS_DIR}/sam_vit_b_01ec64.pth"
    echo "[WEIGHTS] Done."
fi

# RF-DETR Large (needed for RF-DETR experiments)
if [ -f "${PRETRAINS_DIR}/rf-detr-large.pth" ]; then
    echo "[WEIGHTS] rf-detr-large.pth already exists."
else
    echo "[WEIGHTS] Downloading RF-DETR Large (1.5 GB)..."
    # RF-DETR weights are on HuggingFace
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='rafaelpadilla/RF-DETR',
    filename='rf-detr-large-coco.pth',
    local_dir='${PRETRAINS_DIR}',
    local_dir_use_symlinks=False,
)
import os, shutil
src = os.path.join('${PRETRAINS_DIR}', 'rf-detr-large-coco.pth')
dst = os.path.join('${PRETRAINS_DIR}', 'rf-detr-large.pth')
if os.path.exists(src) and not os.path.exists(dst):
    shutil.move(src, dst)
print('Downloaded rf-detr-large.pth')
" 2>/dev/null || {
        echo "[WEIGHTS] WARNING: Could not auto-download RF-DETR weights."
        echo "          Please download manually from https://github.com/roboflow/RF-DETR"
        echo "          and place as: ${PRETRAINS_DIR}/rf-detr-large.pth"
    }
fi

echo ""

# ================================================================== #
# 4. Download DanceTrack Dataset                                      #
# ================================================================== #
echo "========================================================"
echo "  Step 4: Download DanceTrack Dataset"
echo "========================================================"
echo ""

DANCETRACK_DIR="${DATA_ROOT}/DanceTrack"
if [ -d "${DANCETRACK_DIR}/train" ] && [ -d "${DANCETRACK_DIR}/val" ]; then
    echo "[DATA] DanceTrack already exists at ${DANCETRACK_DIR}. Skipping."
else
    echo "[DATA] Downloading DanceTrack from HuggingFace..."
    python "${MOTIP_ROOT}/scripts/download_dancetrack.py" \
        --output-dir "${DANCETRACK_DIR}" \
        --splits train val test
    echo "[DATA] DanceTrack download complete."
fi

echo ""

# ================================================================== #
# 5. Clone RF-DETR & Apply Compatibility Fixes                       #
# ================================================================== #
echo "========================================================"
echo "  Step 5: Clone RF-DETR Repository"
echo "========================================================"
echo ""

RFDETR_DIR="${MOTIP_ROOT}/rf-detr"
if [ -d "${RFDETR_DIR}" ]; then
    echo "[RF-DETR] Repository already exists at ${RFDETR_DIR}. Skipping clone."
else
    echo "[RF-DETR] Cloning RF-DETR repository..."
    git clone https://github.com/roboflow/rf-detr.git "${RFDETR_DIR}"
    echo "[RF-DETR] Cloned successfully."
fi

# Apply transformers v5.x compatibility fix to DINOv2 backbone
DINOV2_FIXED="${MOTIP_ROOT}/models/rfdetr/dinov2_with_windowed_attn.py"
DINOV2_TARGET="${RFDETR_DIR}/rfdetr/models/backbone/dinov2_with_windowed_attn.py"

if [ -f "${DINOV2_FIXED}" ] && [ -f "${DINOV2_TARGET}" ]; then
    cp "${DINOV2_FIXED}" "${DINOV2_TARGET}"
    echo "[RF-DETR] Applied transformers v5.x compatibility fix."
elif [ -f "${DINOV2_TARGET}" ]; then
    echo "[RF-DETR] DINOv2 backbone file exists (fix may already be applied)."
else
    echo "[RF-DETR] WARNING: Could not find DINOv2 files to patch."
fi

echo ""

# ================================================================== #
# 6. Build CUDA Extension                                             #
# ================================================================== #
echo "========================================================"
echo "  Step 6: Build CUDA Extension"
echo "========================================================"
echo ""

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[CUDA] GPU detected. Building MultiScaleDeformableAttention..."

    # Try to load CUDA module if on bwUniCluster
    module load devel/cuda/11.8 2>/dev/null || true
    export CUDA_HOME="${CUDA_HOME:-/opt/bwhpc/common/devel/cuda/11.8}"

    cd "${MOTIP_ROOT}/models/ops"
    python setup.py build install
    cd "${MOTIP_ROOT}"
    echo "[CUDA] CUDA extension built successfully!"
else
    echo "[CUDA] No GPU available (login node)."
    echo "       The CUDA extension will be built automatically"
    echo "       when you submit a training job to a GPU node."
    echo ""
    echo "       To build manually, start an interactive GPU session:"
    echo "         srun --partition=gpu_h100 --gres=gpu:1 --time=00:30:00 --mem=8G --pty bash"
    echo "         conda activate ${ENV_NAME}"
    echo "         module load devel/cuda/11.8"
    echo "         export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8"
    echo "         cd models/ops && python setup.py build install && cd ../.."
fi

echo ""

# ================================================================== #
# Done                                                                #
# ================================================================== #
echo "========================================================"
echo "  Setup Complete!"
echo "========================================================"
echo ""
echo "Summary:"
echo "  Conda environment:  ${ENV_NAME}"
echo "  P-DESTRE dataset:   ${PDESTRE_DIR}/"
echo "  DanceTrack dataset: ${DANCETRACK_DIR}/"
echo "  Pretrained weights: ${PRETRAINS_DIR}/"
echo ""

# Verification checklist
echo "Checklist:"
[ -d "${PDESTRE_DIR}/images" ] && echo "  ✓ P-DESTRE frames extracted" || echo "  ✗ P-DESTRE frames missing"
[ -d "${PDESTRE_DIR}/annotations" ] && echo "  ✓ P-DESTRE annotations present" || echo "  ✗ P-DESTRE annotations missing"
[ -d "${DANCETRACK_DIR}/train" ] && echo "  ✓ DanceTrack downloaded" || echo "  ✗ DanceTrack missing"
[ -f "${DANCETRACK_DIR}/val_seqmap.txt" ] && echo "  ✓ DanceTrack seqmaps generated" || echo "  ✗ DanceTrack seqmaps missing"
[ -f "${PRETRAINS_DIR}/r50_deformable_detr_coco.pth" ] && echo "  ✓ R50 pretrain downloaded" || echo "  ✗ R50 pretrain missing"
[ -f "${PRETRAINS_DIR}/rf-detr-large.pth" ] && echo "  ✓ RF-DETR pretrain downloaded" || echo "  ✗ RF-DETR pretrain missing"
[ -f "${PRETRAINS_DIR}/sam_vit_b_01ec64.pth" ] && echo "  ✓ SAM pretrain downloaded" || echo "  ✗ SAM pretrain missing"
[ -d "${RFDETR_DIR}" ] && echo "  ✓ RF-DETR repo cloned" || echo "  ✗ RF-DETR repo missing"
pip show MultiScaleDeformableAttention &>/dev/null && echo "  ✓ CUDA extension installed" || echo "  ✗ CUDA extension not yet built (needs GPU node)"
echo ""
echo "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. sbatch scripts/smoke_test.sh        # Quick validation"
echo "  3. sbatch train_r50.sh                 # Start training"
echo ""
