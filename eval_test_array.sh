#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate Best-Epoch Checkpoints on Test_0 (SLURM Array Job)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs inference for each model's best epoch (selected on val_0) on the
# held-out Test_0 split.  Submitted as an array job so all 8 models run
# in parallel on separate GPUs.
#
# Usage:
#   sbatch eval_test_array.sh          # submit all 8 tasks
#   sbatch --array=0-3 eval_test_array.sh  # submit only the first 4
#
# ═══════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=test0_eval
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --array=0-7
#SBATCH --output=logs/test0_eval_%A_%a.out
#SBATCH --error=logs/test0_eval_%A_%a.err
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# ── Model definitions: EXP_DIR  EPOCH  CONFIG_PATH  LABEL ────────────────────
# Format: "output_dir|epoch|config_yaml|label"
MODELS=(
  "r50_motip_pdestre_base_fold0|2|outputs/r50_motip_pdestre_base_fold0/train/config.yaml|R50 Base"
  "r50_base_motip_2concepts_fold_0|9|outputs/r50_base_motip_2concepts_fold_0/train/config.yaml|R50 2C"
  "r50_motip_pdestre_3concepts_fold_0|2|outputs/r50_motip_pdestre_3concepts_fold_0/train/config.yaml|R50 3C"
  "r50_motip_pdestre_7concepts_learnable_v2_fold_0|3|outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/train/config.yaml|R50 7C-LW"
  "rfdetr_large_motip_pdestre_base_fold0_v3|1|outputs/rfdetr_large_motip_pdestre_base_fold0_v3/train/config.yaml|RFDETR-L Base"
  "rfdetr_large_motip_pdestre_2concepts_fold0|0|outputs/rfdetr_large_motip_pdestre_2concepts_fold0/train/config.yaml|RFDETR-L 2C"
  "rfdetr_large_motip_pdestre_7concepts_lw_fold0|2|outputs/rfdetr_large_motip_pdestre_7concepts_lw_fold0/train/config.yaml|RFDETR-L 7C-LW"
  "rfdetr_large_motip_pdestre_7concepts_learnable_fold0|0|outputs/rfdetr_large_motip_pdestre_7concepts_learnable_fold0/train/config.yaml|RFDETR-L 7C-Learnable"
)

# ── Pick this task's model ────────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r EXP_DIR EPOCH CONFIG LABEL <<< "${MODELS[$IDX]}"

CHECKPOINT="outputs/${EXP_DIR}/checkpoint_${EPOCH}.pth"
OUTPUT_DIR="outputs/${EXP_DIR}/eval/PDESTRE_Test_0/checkpoint_${EPOCH}"

echo "═══════════════════════════════════════════════════════════════"
echo " Test_0 Evaluation — Task $IDX"
echo " Model:      $LABEL"
echo " Experiment: $EXP_DIR"
echo " Epoch:      $EPOCH"
echo " Checkpoint: $CHECKPOINT"
echo " Config:     $CONFIG"
echo " Output:     $OUTPUT_DIR"
echo " Started:    $(date)"
echo " Node:       $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

# ── Validate inputs ──────────────────────────────────────────────────────────
if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

# ── Environment setup ─────────────────────────────────────────────────────────
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
[ ! -d "$CONDA_BASE" ] && CONDA_BASE="$HOME/miniconda3"
[ ! -d "$CONDA_BASE" ] && CONDA_BASE="$HOME/anaconda3"
[ ! -d "$CONDA_BASE" ] && { echo "ERROR: conda not found"; exit 1; }

eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate MOTIP
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || true
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *)      export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" ;;
esac
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "GPU:    $GPU_NAME"
echo "Python: $(python -c 'import sys; print(sys.executable)')"
echo ""

# ── Run evaluation ────────────────────────────────────────────────────────────
mkdir -p logs

accelerate launch evaluation/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --data-root ./data/ \
    --dataset P-DESTRE \
    --split Test_0 \
    --output-dir "$OUTPUT_DIR" \
    --skip-existing

EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Finished: $LABEL (epoch $EPOCH) — exit code $EXIT_CODE"
echo " Results:  $OUTPUT_DIR"
echo " Time:     $(date)"
echo "═══════════════════════════════════════════════════════════════"
