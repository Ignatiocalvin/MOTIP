#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --array=1-6
#SBATCH --output=logs/eval_test_%a_%j.out
#SBATCH --error=logs/eval_test_%a_%j.err
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# ============================================================
# One SLURM array task per model.  Each task evaluates exactly
# ONE best-epoch checkpoint on the held-out test split.
# Submit:  sbatch eval_test_array.sh
# Or single model: sbatch --array=3 eval_test_array.sh
#
# Array index → model mapping:
#   1  R50-MOTIP-2C        best=7  test=Test_0
#   2  R50-MOTIP-3C        best=2  test=Test_0
#   3  R50-MOTIP-7C-LW     best=3  test=Test_1
#   4  RF-DETR-MOTIP-2C    best=0  test=Test_0
#   5  RF-DETR-MOTIP-7C-NLW best=0 test=Test_0
#   6  RF-DETR-MOTIP-7C-LW  best=2 test=Test_0
# ============================================================

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
echo "✓ Using Python from: $PYTHON_PATH"

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

SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rf-detr:${PYTHONPATH}"
cd "$SCRIPT_DIR"

# Rebuild CUDA ops if needed
python -c "import torch; import MultiScaleDeformableAttention as MSDA; torch.rand(1,1,1,2).cuda()" 2>/dev/null || {
    echo "Rebuilding CUDA ops for $GPU_NAME..."
    cd "$SCRIPT_DIR/models/ops"
    rm -rf build dist *.egg-info
    pip uninstall -y MultiScaleDeformableAttention 2>/dev/null || true
    python setup.py build install
    cd "$SCRIPT_DIR"
}

# ---- Model definitions (index 1-6) ----
case "$SLURM_ARRAY_TASK_ID" in
    1) EXP="r50_base_motip_2concepts_fold_0";                             EPOCH=7; SPLIT="Test_0" ;;
    2) EXP="r50_motip_pdestre_3concepts_fold_0";                          EPOCH=2; SPLIT="Test_0" ;;
    3) EXP="r50_motip_pdestre_7concepts_learnable_v2_fold_0";             EPOCH=3; SPLIT="Test_1" ;;
    4) EXP="rfdetr_large_motip_pdestre_2concepts_fold0";                  EPOCH=0; SPLIT="Test_0" ;;
    5) EXP="rfdetr_large_motip_pdestre_7concepts_learnable_fold0";        EPOCH=0; SPLIT="Test_0" ;;
    6) EXP="rfdetr_large_motip_pdestre_7concepts_lw_fold0";               EPOCH=2; SPLIT="Test_0" ;;
    *)
        echo "ERROR: Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

CKPT="./outputs/${EXP}/checkpoint_${EPOCH}.pth"
CONFIG="./outputs/${EXP}/train/config.yaml"
OUT_DIR="./outputs/${EXP}/eval_test/${SPLIT}/epoch_${EPOCH}"

echo "============================================================"
echo "Array task $SLURM_ARRAY_TASK_ID: $EXP"
echo "  Checkpoint : $CKPT"
echo "  Test split : $SPLIT"
echo "  Output dir : $OUT_DIR"
echo "  Started    : $(date)  Node: $(hostname)"
echo "============================================================"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"; exit 1
fi

# Skip if already done (>=15 non-empty tracker files = Test_0/Test_1 have 15 seqs)
NONEMPTY=0
if [ -d "$OUT_DIR/tracker" ]; then
    for f in "$OUT_DIR/tracker"/*.txt; do
        [ -s "$f" ] && NONEMPTY=$((NONEMPTY + 1))
    done
fi
if [ "$NONEMPTY" -ge 15 ]; then
    echo "Already evaluated ($NONEMPTY non-empty tracker files). Skipping inference."
else
    mkdir -p "$OUT_DIR"
    rm -rf "$OUT_DIR/tracker"   # clear any partial results

    for RETRY in 1 2 3; do
        echo "[Attempt $RETRY] Running inference..."
        python -u -m accelerate.commands.launch --num_processes=1 \
            evaluation/evaluate_checkpoint.py \
            --checkpoint "$CKPT" \
            --config "$CONFIG" \
            --data-root ./data/ \
            --dataset P-DESTRE \
            --split "$SPLIT" \
            --output-dir "$OUT_DIR"
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Inference succeeded."
            break
        fi
        echo "[WARN] Attempt $RETRY failed (exit $EXIT_CODE)."
        [ $RETRY -lt 3 ] && sleep 15
    done
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Inference failed after 3 attempts."; exit 1
    fi
fi

echo ""
echo "Computing metrics for $EXP on $SPLIT..."

python - <<PYEOF
import sys, json
sys.path.insert(0, '.')
from evaluate_all_models import get_val_sequences, evaluate_epoch
from eval_hota import build_hota_data, compute_hota
from pathlib import Path

exp       = "${EXP}"
split     = "${SPLIT}"
epoch_num = ${EPOCH}
out_dir   = Path("outputs/${EXP}/eval_test/${SPLIT}/epoch_${EPOCH}")
tracker_dir = out_dir / "tracker"
split_file  = f"splits/{split}.txt"

seqs = get_val_sequences(split_file)
agg  = evaluate_epoch(tracker_dir, seqs)

if not agg:
    print("No results (empty tracker output).")
    sys.exit(0)

print(f"  MOTA:      {agg['MOTA']:.2f}%  (±{agg['MOTA_std']:.2f}%)")
print(f"  IDF1:      {agg['IDF1']:.2f}%  (±{agg['IDF1_std']:.2f}%)")
print(f"  Precision: {agg['Precision']:.2f}%")
print(f"  Recall:    {agg['Recall']:.2f}%")
print(f"  TP/FP/FN:  {agg['TP']} / {agg['FP']} / {agg['FN']}")
print(f"  IDsw:      {agg['IDsw']}")

# HOTA
try:
    gt_root  = Path("data/P-DESTRE")
    hota_results = {}
    hota_vals, deta_vals, assa_vals = [], [], []
    for seq in seqs:
        gt_file  = gt_root / seq / "gt" / "gt.txt"
        pred_file = tracker_dir / f"{seq}.txt"
        if not gt_file.exists() or not pred_file.exists():
            continue
        gt_lines   = gt_file.read_text().strip().splitlines()
        pred_lines = pred_file.read_text().strip().splitlines() if pred_file.stat().st_size > 0 else []
        hdata = build_hota_data(gt_lines, pred_lines)
        h = compute_hota(hdata)
        hota_results[seq] = h
        hota_vals.append(h["HOTA"])
        deta_vals.append(h["DetA"])
        assa_vals.append(h["AssA"])
    if hota_vals:
        import statistics
        hota_mean = statistics.mean(hota_vals)
        deta_mean = statistics.mean(deta_vals)
        assa_mean = statistics.mean(assa_vals)
        print(f"  HOTA:      {hota_mean:.1f}%  DetA={deta_mean:.1f}%  AssA={assa_mean:.1f}%")
        agg["HOTA"] = hota_mean
        agg["DetA"] = deta_mean
        agg["AssA"] = assa_mean
except Exception as e:
    print(f"  [WARN] HOTA computation failed: {e}")

result = {"exp": exp, "split": split, "epoch": epoch_num, "metrics": agg}
out_json = Path(f"outputs/{exp}/eval_test/{split}/epoch_{epoch_num}/results.json")
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(result, indent=2))
print(f"\nSaved to {out_json}")
PYEOF

echo ""
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID done at $(date)"
echo "============================================================"
