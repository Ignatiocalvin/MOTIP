#!/bin/bash

# Visualize MOTIP Training Results
# This script generates comprehensive visualizations for all completed folds

# Configuration
EXP_PREFIX="${1:-r50_motip_pdestre}"  # Default experiment prefix
OUTPUTS_DIR="${2:-../outputs}"
OUTPUT_VIZ_DIR="${3:-${OUTPUTS_DIR}/${EXP_PREFIX}_visualizations}"

echo "=========================================="
echo "MOTIP Results Visualization"
echo "=========================================="
echo "Experiment: $EXP_PREFIX"
echo "Outputs directory: $OUTPUTS_DIR"
echo "Visualization output: $OUTPUT_VIZ_DIR"
echo ""

# Check if Python script exists
if [ ! -f "visualize_results.py" ]; then
    echo "Error: visualize_results.py not found!"
    echo "Make sure you're running this from the MOTIP/evaluation directory."
    exit 1
fi

# Run visualization
python visualize_results.py \
    --exp-prefix "$EXP_PREFIX" \
    --outputs-dir "$OUTPUTS_DIR" \
    --output-dir "$OUTPUT_VIZ_DIR"

EXITCODE=$?

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "✅ Visualization completed successfully!"
    echo ""
    echo "View your results:"
    echo "  cd $OUTPUT_VIZ_DIR"
    echo "  ls -lh"
    echo ""
    echo "To view images, you can:"
    echo "  1. Use 'eog' or 'display' on Linux"
    echo "  2. Copy to your local machine with scp"
    echo "  3. Open in VS Code"
else
    echo ""
    echo "❌ Visualization failed with exit code $EXITCODE"
    exit $EXITCODE
fi
