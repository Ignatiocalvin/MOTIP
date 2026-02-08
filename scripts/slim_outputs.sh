#!/bin/bash
# Script to create a slim version of outputs folder for cloud GPU inference
# This keeps only the necessary checkpoints and removes training artifacts

set -e  # Exit on error

# Get the MOTIP root directory (parent of scripts/)
MOTIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "${MOTIP_ROOT}"

OUTPUTS_DIR="outputs"
SLIM_DIR="outputs_slim"
ARCHIVE_NAME="outputs_for_inference.tar.gz"

echo "=========================================="
echo "Slimming outputs folder for cloud inference"
echo "=========================================="
echo ""

# Step 1: Create inventory of current checkpoints
echo "Step 1: Creating inventory of current checkpoints..."
find ${OUTPUTS_DIR}/ -name "*.pth" > checkpoint_inventory.txt
echo "Inventory saved to checkpoint_inventory.txt"
echo "Total checkpoint files: $(wc -l < checkpoint_inventory.txt)"
echo ""

# Step 2: Show current size
echo "Step 2: Current size breakdown..."
du -sh ${OUTPUTS_DIR}/
echo ""
echo "Detailed breakdown (top 15 largest):"
du -h --max-depth=2 ${OUTPUTS_DIR}/ | sort -hr | head -15
echo ""

# Step 3: Create slim directory structure
echo "Step 3: Creating slim directory with only inference checkpoints..."
rm -rf ${SLIM_DIR}
mkdir -p ${SLIM_DIR}

# Copy only the latest checkpoint from each fold
for fold_dir in ${OUTPUTS_DIR}/r50_motip_pdestre_fold_*/; do
    if [ -d "$fold_dir" ]; then
        fold_name=$(basename "$fold_dir")
        echo "Processing $fold_name..."
        mkdir -p "${SLIM_DIR}/$fold_name"
        
        # Find the highest numbered checkpoint (usually the best/latest)
        latest_checkpoint=$(ls -1 "$fold_dir"checkpoint_*.pth 2>/dev/null | sort -V | tail -1)
        
        if [ -n "$latest_checkpoint" ]; then
            echo "  Copying $(basename $latest_checkpoint)..."
            cp "$latest_checkpoint" "${SLIM_DIR}/$fold_name/"
        else
            echo "  Warning: No checkpoint found in $fold_name"
        fi
        
        # Copy results.json if it exists
        if [ -f "$fold_dir/results.json" ]; then
            echo "  Copying results.json..."
            cp "$fold_dir/results.json" "${SLIM_DIR}/$fold_name/"
        fi
        
        # Copy evaluate folder if it exists (contains evaluation results)
        if [ -d "$fold_dir/evaluate" ]; then
            echo "  Copying evaluation results..."
            cp -r "$fold_dir/evaluate" "${SLIM_DIR}/$fold_name/"
        fi
    fi
done

# Also copy the demo folder if you want example code
if [ -d "${OUTPUTS_DIR}/demo" ]; then
    echo "Copying demo folder..."
    cp -r "${OUTPUTS_DIR}/demo" "${SLIM_DIR}/"
fi

echo ""
echo "Step 4: Size comparison..."
echo "Original outputs size: $(du -sh ${OUTPUTS_DIR} | cut -f1)"
echo "Slim outputs size:     $(du -sh ${SLIM_DIR} | cut -f1)"
echo ""

# Step 5: Create compressed archive
echo "Step 5: Creating compressed archive..."
tar -czf ${ARCHIVE_NAME} ${SLIM_DIR}/
echo "Archive created: ${ARCHIVE_NAME}"
echo "Archive size: $(du -sh ${ARCHIVE_NAME} | cut -f1)"
echo ""

# Step 6: Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Original size:  $(du -sh ${OUTPUTS_DIR} | cut -f1)"
echo "Slim size:      $(du -sh ${SLIM_DIR} | cut -f1)"
echo "Archive size:   $(du -sh ${ARCHIVE_NAME} | cut -f1)"
echo ""
echo "What was kept:"
find ${SLIM_DIR}/ -name "*.pth" | wc -l | xargs echo "- Checkpoint files:"
find ${SLIM_DIR}/ -name "results.json" | wc -l | xargs echo "- Results files:"
echo ""
echo "Archive ready for download: ${ARCHIVE_NAME}"
echo ""
echo "To download locally, use:"
echo "  scp $(whoami)@$(hostname):$(pwd)/${ARCHIVE_NAME} ."
echo ""
echo "=========================================="
