#!/bin/bash

# P-DESTRE Dataset Download and Preprocessing Script

set -e  # Exit on any error

echo "Starting P-DESTRE dataset download and preprocessing..."

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

# Copy the preprocessing script to the current directory if it doesn't exist
PREPROCESS_SCRIPT="../../preprocess_pdestre.py"
if [ ! -f "preprocess_pdestre.py" ] && [ -f "$PREPROCESS_SCRIPT" ]; then
    cp "$PREPROCESS_SCRIPT" .
    echo "Copied preprocessing script to P-DESTRE directory"
fi

# Run the preprocessing script
echo "Running preprocessing script..."
python preprocess_pdestre.py

echo "Dataset download and preprocessing completed successfully!"
echo "Extracted frames are available in the 'images/' directory"

# Clean up the downloaded tar file
cd ..
if [ -f "dataset.tar" ]; then
    rm dataset.tar
    echo "Cleaned up dataset.tar file"
fi

# Return to original directory
cd ..

echo "All done!"
echo "P-DESTRE dataset is now available in: data/P-DESTRE/"