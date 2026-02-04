#!/bin/bash

# Complete Workflow: Extract Metrics and Generate Visualizations
# This script automates the entire process from evaluation results to visualizations

set -e  # Exit on error

# Configuration
EXP_PREFIX="${1:-r50_motip_pdestre}"
OUTPUTS_DIR="${2:-../outputs}"

echo "=========================================="
echo "MOTIP Complete Visualization Workflow"
echo "=========================================="
echo "Experiment: $EXP_PREFIX"
echo "Outputs: $OUTPUTS_DIR"
echo ""

# Step 1: Extract metrics from all folds
echo "Step 1/2: Extracting metrics from evaluation results..."
echo "--------------------------------------------------"
python extract_metrics.py --exp-prefix "$EXP_PREFIX" --outputs-dir "$OUTPUTS_DIR" --all-folds

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Metric extraction failed!"
    echo "Make sure evaluation has been run for at least some folds."
    exit 1
fi

echo ""
echo "Step 2/2: Generating visualizations..."
echo "--------------------------------------------------"
python visualize_results.py --exp-prefix "$EXP_PREFIX" --outputs-dir "$OUTPUTS_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Visualization generation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Complete! All visualizations generated."
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  ${OUTPUTS_DIR}/${EXP_PREFIX}_visualizations/"
echo ""
echo "View your results:"
echo "  cd ${OUTPUTS_DIR}/${EXP_PREFIX}_visualizations"
echo "  ls -lh"
echo ""
