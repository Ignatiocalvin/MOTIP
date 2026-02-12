#!/bin/bash

# P-DESTRE Dataset Download and Preprocessing Script

set -e  # Exit on any error

# Get the MOTIP root directory (parent of scripts/)
MOTIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "${MOTIP_ROOT}"

echo "========================================"
echo "P-DESTRE Dataset Setup"
echo "========================================"
echo ""
echo "Working directory: ${MOTIP_ROOT}"
echo ""

# Create data directory if it doesn't exist
mkdir -p data
cd data

# Download the dataset
echo "Downloading P-DESTRE dataset..."
wget https://socia-lab.di.ubi.pt/%7Ehugomcp/dataset.tar

# Extract the tar file
echo "Extracting dataset..."
tar -xf dataset.tar

# Check if P-DESTRE directory exists
if [ ! -d "P-DESTRE" ]; then
    echo "Error: P-DESTRE directory not found after extraction"
    exit 1
fi

cd P-DESTRE

# Rename annotation folder to annotations if needed
if [ -d "annotation" ] && [ ! -d "annotations" ]; then
    mv annotation annotations
    echo "Renamed 'annotation' folder to 'annotations'"
elif [ -d "annotations" ]; then
    echo "Annotations folder already exists"
else
    echo "Warning: Neither 'annotation' nor 'annotations' folder found"
fi

# Remove specific annotation and video files
echo "Removing specified files..."

# Remove 22-10-2019-1-2 files
if [ -f "annotations/22-10-2019-1-2.txt" ]; then
    rm annotations/22-10-2019-1-2.txt
    echo "Removed annotations/22-10-2019-1-2.txt"
else
    echo "Warning: annotations/22-10-2019-1-2.txt not found"
fi

if [ -f "videos/22-10-2019-1-2.MP4" ]; then
    rm videos/22-10-2019-1-2.MP4
    echo "Removed videos/22-10-2019-1-2.MP4"
else
    echo "Warning: videos/22-10-2019-1-2.MP4 not found"
fi

# Remove 13-11-2019-4-3 files
if [ -f "annotations/13-11-2019-4-3.txt" ]; then
    rm annotations/13-11-2019-4-3.txt
    echo "Removed annotations/13-11-2019-4-3.txt"
else
    echo "Warning: annotations/13-11-2019-4-3.txt not found"
fi

if [ -f "videos/13-11-2019-4-3.MP4" ]; then
    rm videos/13-11-2019-4-3.MP4
    echo "Removed videos/13-11-2019-4-3.MP4"
else
    echo "Warning: videos/13-11-2019-4-3.MP4 not found"
fi

# Run the preprocessing script from data/P-DESTRE/
echo ""
echo "Extracting frames from videos..."
if [ -f "preprocess_pdestre.py" ]; then
    python preprocess_pdestre.py
else
    echo "Warning: preprocess_pdestre.py not found in data/P-DESTRE/"
    echo "Skipping frame extraction. You may need to run it manually."
fi

# Verify splits directory exists
echo ""
if [ ! -d "splits" ]; then
    echo "Warning: splits/ directory not found in data/P-DESTRE/"
    echo "Make sure to create train/val/test split files before training."
else
    echo "Splits directory found: data/P-DESTRE/splits/"
fi

# Clean up the downloaded tar file
cd ..
if [ -f "dataset.tar" ]; then
    echo ""
    echo "Cleaning up dataset.tar..."
    rm dataset.tar
fi

cd ..

echo ""
echo "=========================================="
echo "Cloning RF-DETR Repository"
echo "=========================================="
# Clone RF-DETR repository as sibling to MOTIP directory
RFDETR_DIR="${MOTIP_ROOT}/../rf-detr"
if [ ! -d "$RFDETR_DIR" ]; then
    echo "Cloning RF-DETR repository..."
    cd "${MOTIP_ROOT}/.."
    git clone https://github.com/roboflow/rf-detr.git rf-detr
    echo "RF-DETR repository cloned to: $RFDETR_DIR"
else
    echo "RF-DETR repository already exists at: $RFDETR_DIR"
fi

# Apply compatibility fixes to RF-DETR's DINOv2 backbone
echo ""
echo "Applying transformers compatibility fixes to RF-DETR..."
DINOV2_SOURCE="${MOTIP_ROOT}/rf-detr/rfdetr/models/backbone/dinov2_with_windowed_attn.py"
DINOV2_TARGET="${RFDETR_DIR}/rfdetr/models/backbone/dinov2_with_windowed_attn.py"

if [ -f "$DINOV2_SOURCE" ]; then
    echo "Copying fixed dinov2_with_windowed_attn.py to RF-DETR..."
    cp "$DINOV2_SOURCE" "$DINOV2_TARGET"
    echo "✓ Applied compatibility fixes for transformers v5.x"
else
    echo "Warning: Fixed DINOv2 file not found at: $DINOV2_SOURCE"
    echo "You may need to manually apply transformers compatibility fixes."
fi

cd "${MOTIP_ROOT}"

echo ""
echo "=========================================="
echo "Building CUDA Operators"
echo "=========================================="
echo "Building MultiScaleDeformableAttention CUDA extension..."
cd "${MOTIP_ROOT}/models/ops"
python setup.py build install
cd "${MOTIP_ROOT}"
echo "CUDA operators built successfully!"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Dataset location: ${MOTIP_ROOT}/data/P-DESTRE/"
if [ -d "${MOTIP_ROOT}/data/P-DESTRE/images" ]; then
    echo "Extracted frames: data/P-DESTRE/images/"
fi
if [ -d "${MOTIP_ROOT}/data/P-DESTRE/annotations" ]; then
    echo "Annotations: data/P-DESTRE/annotations/"
fi
if [ -d "${MOTIP_ROOT}/data/P-DESTRE/splits" ]; then
    echo "Splits: data/P-DESTRE/splits/"
fi
echo ""