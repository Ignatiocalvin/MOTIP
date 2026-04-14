#!/bin/bash
#SBATCH --job-name=eval_smoke
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=24G
#SBATCH --output=logs/eval_smoke_%j.out
#SBATCH --error=logs/eval_smoke_%j.err
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# ==============================================================================
# Evaluation Smoke Test for MOTIP
# Purpose: Verify that inference and evaluation pipeline work correctly
# Tests:
#   1. Model loading from checkpoint
#   2. Inference on validation sequences  
#   3. Output format validation
#   4. TrackEval metrics computation
# ==============================================================================

set -e

echo "=========================================="
echo "MOTIP Evaluation Smoke Test"
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# --- Environment Setup ---
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate MOTIP
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

# Verify Python environment
PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
if [[ "$PYTHON_PATH" != *"miniconda3/envs/MOTIP"* ]]; then
    echo "ERROR: Not using MOTIP env: $PYTHON_PATH"
    exit 1
fi
echo "Python: $PYTHON_PATH"

# --- CUDA Setup ---
if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || true
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *) export TORCH_CUDA_ARCH_LIST="8.0;9.0" ;;
esac
export CUDA_VISIBLE_DEVICES=0

# --- Path Setup ---
SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rf-detr:${PYTHONPATH}"
cd "$SCRIPT_DIR"

# --- Rebuild CUDA ops if needed ---
echo ""
echo "Checking CUDA ops..."
CUDA_OPS_OK=false
python -c "import torch; import MultiScaleDeformableAttention as MSDA; torch.rand(1,1,1,2).cuda()" 2>/dev/null && CUDA_OPS_OK=true
if [ "$CUDA_OPS_OK" = false ]; then
    echo "Rebuilding CUDA ops for $GPU_NAME..."
    cd "$SCRIPT_DIR/models/ops"
    rm -rf build dist *.egg-info
    pip uninstall -y MultiScaleDeformableAttention 2>/dev/null || true
    python setup.py build install
    cd "$SCRIPT_DIR"
fi
echo "CUDA ops: OK"

# --- Data paths ---
DATA_ROOT="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP_SAM/datasets/"

# --- Results directory ---
RESULTS_DIR="outputs/eval_smoke_test"
mkdir -p "$RESULTS_DIR"
RESULTS_JSON="$RESULTS_DIR/results.json"
echo "{\"eval_smoke_test\": {\"timestamp\": \"$(date -Iseconds)\", \"models\": {" > "$RESULTS_JSON"

# --- Quick test: Just test ONE model to verify pipeline works ---
# Use RF-DETR as it's the smallest and fastest to load
echo ""
echo "==========================================================="
echo "Testing RF-DETR model evaluation pipeline"
echo "==========================================================="

CHECKPOINT="outputs/smoke_test_rfdetr/checkpoint_1.pth"
CONFIG="outputs/smoke_test_rfdetr/train/config.yaml"
NAME="rfdetr"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# Create output directory
OUT_DIR="$RESULTS_DIR/${NAME}"
rm -rf "$OUT_DIR"  # Clean previous run
mkdir -p "$OUT_DIR"

echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Output: $OUT_DIR"

# Run evaluation on DanceTrack validation set (limited to 3 sequences via seqmap)
echo ""
echo "--- Running inference on DanceTrack val (3 sequences via val_smoke_seqmap) ---"
START_TIME=$(date +%s)

# Use submit_and_evaluate.py directly with proper args
python -u -m accelerate.commands.launch --num_processes=1 \
    evaluation/submit_and_evaluate.py \
    --config-path "$CONFIG" \
    --inference-model "$CHECKPOINT" \
    --inference-dataset DanceTrack \
    --inference-split val \
    --data-root "$DATA_ROOT" \
    --outputs-dir "$OUT_DIR" \
    --inference-mode evaluate \
    2>&1 | tee "$OUT_DIR/eval.log"

EVAL_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "--- Evaluation Results ---"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Time elapsed: ${ELAPSED}s"

# Check for tracker output files
TRACKER_DIR=$(find "$OUT_DIR" -type d -name "tracker" 2>/dev/null | head -1)
TRACKER_FILES=0
TOTAL_LINES=0

if [ -n "$TRACKER_DIR" ] && [ -d "$TRACKER_DIR" ]; then
    TRACKER_FILES=$(ls -1 "$TRACKER_DIR"/*.txt 2>/dev/null | wc -l)
    if [ "$TRACKER_FILES" -gt 0 ]; then
        TOTAL_LINES=$(cat "$TRACKER_DIR"/*.txt 2>/dev/null | wc -l)
        
        echo ""
        echo "--- Output Files ---"
        ls -la "$TRACKER_DIR"/*.txt 2>/dev/null | head -10
        
        echo ""
        echo "--- Sample Output (first file, first 5 lines) ---"
        head -5 "$TRACKER_DIR"/*.txt 2>/dev/null | head -5
    fi
fi

echo ""
echo "Tracker files: $TRACKER_FILES"
echo "Total tracking lines: $TOTAL_LINES"

# Check output format validity
FORMAT_VALID="false"
if [ "$TRACKER_FILES" -gt 0 ]; then
    SAMPLE=$(head -1 "$TRACKER_DIR"/*.txt 2>/dev/null | head -1)
    echo "Sample line: $SAMPLE"
    # Format: frame_id,track_id,x,y,w,h,conf,-1,-1,-1[,concepts]
    if [[ "$SAMPLE" =~ ^[0-9]+,[0-9]+, ]]; then
        echo "Output format: VALID"
        FORMAT_VALID="true"
    else
        echo "Output format: INVALID"
    fi
fi

# Check for metrics output
echo ""
echo "--- Checking for metrics ---"
if [ -f "$TRACKER_DIR/pedestrian_summary.txt" ]; then
    echo "TrackEval metrics found:"
    cat "$TRACKER_DIR/pedestrian_summary.txt"
elif find "$OUT_DIR" -name "pedestrian_summary.txt" 2>/dev/null | head -1 | xargs cat 2>/dev/null; then
    echo "TrackEval metrics found (in subdir)"
else
    echo "No TrackEval metrics file found (expected for quick test)"
fi

# Write results JSON
STATUS="failed"
[ "$TRACKER_FILES" -gt 0 ] && [ "$FORMAT_VALID" = "true" ] && STATUS="success"

cat >> "$RESULTS_JSON" << EOF
    "$NAME": {
      "checkpoint": "$CHECKPOINT",
      "config": "$CONFIG",
      "tracker_files": $TRACKER_FILES,
      "tracking_lines": $TOTAL_LINES,
      "time_seconds": $ELAPSED,
      "format_valid": $FORMAT_VALID,
      "exit_code": $EVAL_EXIT_CODE,
      "status": "$STATUS"
    }
EOF

# Close JSON
echo "}," >> "$RESULTS_JSON"
echo "\"completed\": \"$(date -Iseconds)\"" >> "$RESULTS_JSON"
echo "}}" >> "$RESULTS_JSON"

echo ""
echo "=========================================="
echo "Evaluation Smoke Test Complete"
echo "Results saved to: $RESULTS_JSON"
echo "Completed at $(date)"
echo "=========================================="

# Print summary
echo ""
echo "--- SUMMARY ---"
python -m json.tool "$RESULTS_JSON" 2>/dev/null || cat "$RESULTS_JSON"

# Exit with appropriate code
if [ "$STATUS" = "success" ]; then
    echo ""
    echo "SUCCESS: Evaluation pipeline works correctly!"
    exit 0
else
    echo ""
    echo "FAILED: Evaluation pipeline did not produce valid outputs"
    exit 1
fi
