#!/bin/bash

# ================================================================== #
#  MOTIP Setup, Download & Preprocess Script                          #
#  Creates conda env, downloads data, precomputes SAM masks          #
# ================================================================== #
#
# Usage:
#   ./scripts/download_and_preprocess.sh                 # Full setup
#   ./scripts/download_and_preprocess.sh --env-only      # Only setup env
#   ./scripts/download_and_preprocess.sh --data-only     # Only download data
#   ./scripts/download_and_preprocess.sh --masks-only    # Only precompute masks
#   ./scripts/download_and_preprocess.sh --skip-masks    # Setup + data, no masks
#
# Environment variables:
#   ENV_NAME     - Conda environment name (default: MOTIP_fresh)
#   DATA_ROOT    - Data directory (default: ./data)
#   MASK_ROOT    - SAM masks directory (default: ./precomputed_sam_masks)
#   SAM_CKPT     - SAM checkpoint path (default: ./pretrains/sam_vit_b_01ec64.pth)
#
# ================================================================== #

set -e  # Exit on any error

# ------------------------------------------------------------------ #
# Parse command-line arguments                                        #
# ------------------------------------------------------------------ #
DO_ENV=true
DO_DATA=true
DO_MASKS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-only)
            DO_DATA=false
            DO_MASKS=false
            shift
            ;;
        --data-only)
            DO_ENV=false
            DO_MASKS=false
            shift
            ;;
        --masks-only)
            DO_ENV=false
            DO_DATA=false
            shift
            ;;
        --skip-masks)
            DO_MASKS=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-only     Only setup conda environment"
            echo "  --data-only    Only download datasets"
            echo "  --masks-only   Only precompute SAM masks"
            echo "  --skip-masks   Skip SAM mask precomputation"
            echo "  -h, --help     Show this help message"
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

# Configurable via environment variables
ENV_NAME="${ENV_NAME:-MOTIP_fresh}"
DATA_ROOT="${DATA_ROOT:-${MOTIP_ROOT}/data}"
MASK_ROOT="${MASK_ROOT:-${MOTIP_ROOT}/precomputed_sam_masks}"
SAM_CKPT="${SAM_CKPT:-${MOTIP_ROOT}/pretrains/sam_vit_b_01ec64.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"

echo "========================================"
echo "MOTIP Setup, Download & Preprocess"
echo "========================================"
echo "Working directory: ${MOTIP_ROOT}"
echo "Data root:         ${DATA_ROOT}"
echo "Mask root:         ${MASK_ROOT}"
echo "SAM checkpoint:    ${SAM_CKPT}"
echo ""
echo "Steps to run:"
echo "  Environment setup: ${DO_ENV}"
echo "  Data download:     ${DO_DATA}"
echo "  SAM masks:         ${DO_MASKS}"
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
        echo "ERROR: Could not find conda. Please install miniconda or anaconda first."
        exit 1
    fi
}

# ================================================================== #
# 1. Environment Setup                                                #
# ================================================================== #
if [ "$DO_ENV" = true ]; then
    echo "========================================"
    echo "1. Environment Setup"
    echo "========================================"

    init_conda

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

    # Install Python dependencies
    echo "[PIP] Installing Python dependencies..."
    pip install -r "${MOTIP_ROOT}/requirements.txt"
    
    # Install huggingface_hub for downloading datasets
    pip install huggingface_hub --quiet
    
    # Install segment-anything for SAM mask precomputation
    pip install segment-anything --quiet
    
    echo "[PIP] Done."
    echo ""

    # Build CUDA Operators (skip if no GPU - will build in SLURM job)
    echo "[CUDA] Checking for GPU..."
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[CUDA] Building MultiScaleDeformableAttention CUDA extension..."
        cd "${MOTIP_ROOT}/models/ops"
        python setup.py build install
        cd "${MOTIP_ROOT}"
        echo "[CUDA] CUDA operators built successfully!"
    else
        echo "[CUDA] No GPU available on login node."
        echo "       CUDA ops will be built when you submit a SLURM job."
    fi
    echo ""
fi

# ================================================================== #
# 2. Download Datasets                                                #
# ================================================================== #
if [ "$DO_DATA" = true ]; then
    echo "========================================"
    echo "2. Download Datasets"
    echo "========================================"

    init_conda
    conda activate "$ENV_NAME" 2>/dev/null || true

    mkdir -p "${DATA_ROOT}"

    # Download DanceTrack
    DANCETRACK_DIR="${DATA_ROOT}/DanceTrack"
    if [ -d "${DANCETRACK_DIR}/train" ] && [ -d "${DANCETRACK_DIR}/val" ]; then
        echo "[DATA] DanceTrack already exists at ${DANCETRACK_DIR}"
    else
        echo "[DATA] Downloading DanceTrack dataset..."
        python "${MOTIP_ROOT}/scripts/download_dancetrack.py" \
            --output-dir "${DANCETRACK_DIR}" \
            --splits train val test
    fi
    echo ""
fi

# ================================================================== #
# 3. Precompute SAM Masks                                             #
# ================================================================== #
if [ "$DO_MASKS" = true ]; then
    echo "========================================"
    echo "3. Precompute SAM Masks"
    echo "========================================"

    init_conda
    conda activate "$ENV_NAME" 2>/dev/null || true

    # Check if SAM checkpoint exists
    if [ ! -f "${SAM_CKPT}" ]; then
        echo "[WARN] SAM checkpoint not found at ${SAM_CKPT}"
        echo "       Please download it from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        echo "       Skipping SAM mask precomputation."
    else
        # Check if GPU is available (SAM needs GPU)
        if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            mkdir -p "${MASK_ROOT}"

            # Precompute masks for DanceTrack train split
            DANCETRACK_MASK_FLAG="${MASK_ROOT}/.dancetrack_train_done"
            if [ -f "${DANCETRACK_MASK_FLAG}" ]; then
                echo "[MASKS] DanceTrack train masks already computed."
            else
                echo "[MASKS] Precomputing SAM masks for DanceTrack train..."
                python "${MOTIP_ROOT}/scripts/precompute_sam_masks.py" \
                    --dataset DanceTrack \
                    --split train \
                    --data-root "${DATA_ROOT}" \
                    --sam-checkpoint "${SAM_CKPT}" \
                    --model-type "${SAM_MODEL_TYPE}" \
                    --save-root "${MASK_ROOT}" \
                    --device cuda
                touch "${DANCETRACK_MASK_FLAG}"
                echo "[MASKS] DanceTrack train masks done."
            fi

            # Precompute masks for DanceTrack val split
            DANCETRACK_VAL_FLAG="${MASK_ROOT}/.dancetrack_val_done"
            if [ -f "${DANCETRACK_VAL_FLAG}" ]; then
                echo "[MASKS] DanceTrack val masks already computed."
            else
                echo "[MASKS] Precomputing SAM masks for DanceTrack val..."
                python "${MOTIP_ROOT}/scripts/precompute_sam_masks.py" \
                    --dataset DanceTrack \
                    --split val \
                    --data-root "${DATA_ROOT}" \
                    --sam-checkpoint "${SAM_CKPT}" \
                    --model-type "${SAM_MODEL_TYPE}" \
                    --save-root "${MASK_ROOT}" \
                    --device cuda
                touch "${DANCETRACK_VAL_FLAG}"
                echo "[MASKS] DanceTrack val masks done."
            fi
        else
            echo "[MASKS] No GPU available. SAM mask precomputation requires a GPU."
            echo "        Submit a SLURM job with GPU to precompute masks:"
            echo ""
            echo "        python scripts/precompute_sam_masks.py \\"
            echo "            --dataset DanceTrack --split train \\"
            echo "            --data-root ${DATA_ROOT} \\"
            echo "            --save-root ${MASK_ROOT}"
        fi
    fi
    echo ""
fi

# ================================================================== #
# Done                                                                #
# ================================================================== #
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "  Conda environment: ${ENV_NAME}"
echo "  Data directory:    ${DATA_ROOT}"
echo "  SAM masks:         ${MASK_ROOT}"
echo ""
echo "To activate the environment:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To run training, submit a SLURM job:"
echo "    sbatch setup_and_smoke_fresh.sh"
echo ""
echo "To precompute SAM masks on a GPU node (if not done):"
echo "    python scripts/precompute_sam_masks.py --dataset DanceTrack --split train"
echo ""
