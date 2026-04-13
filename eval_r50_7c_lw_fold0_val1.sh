#!/bin/bash
#SBATCH --job-name=eval_r50_7c_lw_val1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --output=logs/eval_r50_7c_lw_val1_%j.out
#SBATCH --error=logs/eval_r50_7c_lw_val1_%j.err
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate MOTIP
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
if [[ "$PYTHON_PATH" != *"miniconda3/envs/MOTIP"* ]]; then
    echo "ERROR: Not using MOTIP env: $PYTHON_PATH"; exit 1
fi

if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || true
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *) export TORCH_CUDA_ARCH_LIST="8.0;9.0" ;;
esac
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rf-detr:${PYTHONPATH}"

# Rebuild CUDA ops if needed
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

EXP="r50_motip_pdestre_7concepts_learnable_v2_fold_0"
CONFIG="./outputs/${EXP}/train/config.yaml"
OUT_BASE="./outputs/${EXP}/train/eval_during_train"
CHECKPOINT_JSON="./outputs/${EXP}/eval_results_val1_checkpoint.json"

echo "==========================================================="
echo "Evaluating on val_1: $EXP"
echo "Strategy: epochs 0-2 re-evaluated on val_1; epochs 3-8 already done"
echo "Started at $(date) on $(hostname)"
echo "Wall-clock time limit: 4 hours"
echo "==========================================================="

for EPOCH in 0 1 2 3 4 5 6 7 8; do
    CKPT="./outputs/${EXP}/checkpoint_${EPOCH}.pth"
    OUT_DIR="${OUT_BASE}/epoch_${EPOCH}"

    if [ ! -f "$CKPT" ]; then
        echo "Checkpoint not found, skipping: $CKPT"
        continue
    fi

    # Count non-empty tracker files
    NONEMPTY=0
    if [ -d "$OUT_DIR/tracker" ]; then
        for f in "$OUT_DIR/tracker"/*.txt; do
            [ -s "$f" ] && NONEMPTY=$((NONEMPTY + 1))
        done
    fi

    # Epochs 3-8 already have 15 val_1 files — skip
    if [ "$EPOCH" -ge 3 ] && [ "$NONEMPTY" -ge 15 ]; then
        echo "--- Epoch $EPOCH --- already on val_1 ($NONEMPTY files), skipping"
        continue
    fi

    # Skip if already fully done on val_1 (15 files)
    if [ "$NONEMPTY" -ge 15 ]; then
        echo "--- Epoch $EPOCH --- already on val_1 ($NONEMPTY files), skipping inference"
        continue
    fi

    # Epochs 0-2: re-evaluate on val_1
    echo ""
    echo "--- Epoch $EPOCH --- re-evaluating on val_1 ($NONEMPTY/15 files already present)"
    mkdir -p "$OUT_DIR/tracker"

    EPOCH_OK=false
    for RETRY in 1 2; do
        python -u -m accelerate.commands.launch --num_processes=1 \
            evaluation/evaluate_checkpoint.py \
            --checkpoint "$CKPT" \
            --config "$CONFIG" \
            --data-root ./data/ \
            --dataset P-DESTRE \
            --split val_1 \
            --output-dir "$OUT_DIR" \
            --skip-existing \
            2>&1 | tee -a logs/eval_r50_7c_lw_val1_epoch_${EPOCH}.log

        EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -eq 0 ]; then
            EPOCH_OK=true
            break
        fi
        echo "[WARN] Epoch $EPOCH attempt $RETRY failed (exit $EXIT_CODE). Retrying in 5s..."
        sleep 5
    done
    
    if [ "$EPOCH_OK" = false ]; then
        echo "[ERROR] Epoch $EPOCH failed after 2 attempts. Continuing to next epoch..."
        continue
    fi

    # Verify completion
    FINAL_COUNT=0
    if [ -d "$OUT_DIR/tracker" ]; then
        for f in "$OUT_DIR/tracker"/*.txt; do
            [ -s "$f" ] && FINAL_COUNT=$((FINAL_COUNT + 1))
        done
    fi
    echo "✓ Epoch $EPOCH: $FINAL_COUNT/15 tracker files generated at $(date)"

done

echo ""
echo "==========================================================="
echo "Inference evaluation completed at $(date)"
echo "==========================================================="
echo ""
echo "Next steps:"
echo "  1. Run: python evaluation/compute_tracking_metrics.py --fold 0"
echo "  2. Or run: python evaluation/extract_metrics.py --fold 0"
echo "=========================================================="
