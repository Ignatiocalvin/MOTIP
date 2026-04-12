#!/bin/bash
#SBATCH --job-name=eval_r50_7c_lw_val1
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
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
echo "Checkpointing results to: $CHECKPOINT_JSON"
echo "==========================================================="

# Initialize checkpoint file (resume-safe: preserves already-computed epochs)
python - <<'INITEOF'
import json
from pathlib import Path
CHECKPOINT_JSON = "./outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val1_checkpoint.json"
Path(CHECKPOINT_JSON).parent.mkdir(parents=True, exist_ok=True)
if Path(CHECKPOINT_JSON).exists():
    with open(CHECKPOINT_JSON) as f:
        existing = json.load(f)
    existing['status'] = 'in_progress'
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Resuming from existing checkpoint ({len(existing.get('all_epochs', {}))} epochs done): {CHECKPOINT_JSON}")
else:
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump({'all_epochs': {}, 'best_epoch': None, 'best': None, 'status': 'in_progress'}, f, indent=2)
    print(f"Initialized new checkpoint file: {CHECKPOINT_JSON}")
INITEOF

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

    # Epochs 3-8 already have 15 val_1 files — just compute metrics
    if [ "$EPOCH" -ge 3 ] && [ "$NONEMPTY" -ge 15 ]; then
        echo "--- Epoch $EPOCH --- already on val_1 ($NONEMPTY files), computing metrics only"
        python - <<METEOF
import sys
sys.path.insert(0, '.')
from evaluate_all_models import get_val_sequences, evaluate_epoch
from pathlib import Path
import json

epoch_num = ${EPOCH}
CHECKPOINT_JSON = './outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val1_checkpoint.json'

try:
    with open(CHECKPOINT_JSON) as f:
        data = json.load(f)
except:
    data = {'all_epochs': {}, 'best_epoch': None, 'best': None, 'status': 'in_progress'}

# Skip if already checkpointed
if str(epoch_num) in data['all_epochs']:
    print(f"  Epoch {epoch_num}: already checkpointed, skipping")
else:
    val_seqs = get_val_sequences('data/P-DESTRE/splits/val_1.txt')
    tracker_dir = Path('outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/train/eval_during_train') / f'epoch_{epoch_num}' / 'tracker'
    agg = evaluate_epoch(tracker_dir, val_seqs)
    if agg:
        data['all_epochs'][str(epoch_num)] = agg
        if not data['best'] or agg['MOTA'] > data['best']['MOTA']:
            data['best_epoch'] = epoch_num
            data['best'] = agg
        with open(CHECKPOINT_JSON, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Epoch {epoch_num:2d}: MOTA={agg['MOTA']:6.2f}% ±{agg['MOTA_std']:5.2f}  IDF1={agg['IDF1']:6.2f}% ±{agg['IDF1_std']:5.2f}  Prec={agg['Precision']:6.2f}%  Rec={agg['Recall']:6.2f}%  [CHECKPOINTED]")
    else:
        print(f"✗ Epoch {epoch_num}: No results")
METEOF
        continue
    fi

    # Epochs 0-2: need to re-evaluate on val_1
    # Skip if already fully done on val_1 (15 files)
    if [ "$NONEMPTY" -ge 15 ]; then
        echo "--- Epoch $EPOCH --- already on val_1 ($NONEMPTY files), skipping inference"
    else
        # Clear any existing tracker (may have partial val_0 data) and re-run on val_1
        echo ""
        echo "--- Epoch $EPOCH --- re-evaluating on val_1 (clearing old tracker)"
        rm -rf "$OUT_DIR/tracker"
        mkdir -p "$OUT_DIR/tracker"

        EPOCH_OK=false
        for RETRY in 1 2 3; do
            env PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rf-detr:${PYTHONPATH}" \
            python -u -m accelerate.commands.launch --num_processes=1 \
                evaluation/evaluate_checkpoint.py \
                --checkpoint "$CKPT" \
                --config "$CONFIG" \
                --data-root ./data/ \
                --dataset P-DESTRE \
                --split val_1 \
                --output-dir "$OUT_DIR" \
                2>&1
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                EPOCH_OK=true
                break
            fi
            echo "[WARN] Epoch $EPOCH attempt $RETRY failed (exit $EXIT_CODE). Retrying in 10s..."
            sleep 10
        done
        if [ "$EPOCH_OK" = false ]; then
            echo "[ERROR] Epoch $EPOCH failed after 3 attempts, continuing."
            continue
        fi
    fi

    # Compute and checkpoint metrics
    python - <<METEOF
import sys
sys.path.insert(0, '.')
from evaluate_all_models import get_val_sequences, evaluate_epoch
from pathlib import Path
import json

epoch_num = ${EPOCH}
CHECKPOINT_JSON = './outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val1_checkpoint.json'

try:
    with open(CHECKPOINT_JSON) as f:
        data = json.load(f)
except:
    data = {'all_epochs': {}, 'best_epoch': None, 'best': None, 'status': 'in_progress'}

val_seqs = get_val_sequences('data/P-DESTRE/splits/val_1.txt')
tracker_dir = Path('outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/train/eval_during_train') / f'epoch_{epoch_num}' / 'tracker'
agg = evaluate_epoch(tracker_dir, val_seqs)
if agg:
    data['all_epochs'][str(epoch_num)] = agg
    if not data['best'] or agg['MOTA'] > data['best']['MOTA']:
        data['best_epoch'] = epoch_num
        data['best'] = agg
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Epoch {epoch_num:2d}: MOTA={agg['MOTA']:6.2f}% ±{agg['MOTA_std']:5.2f}  IDF1={agg['IDF1']:6.2f}% ±{agg['IDF1_std']:5.2f}  Prec={agg['Precision']:6.2f}%  Rec={agg['Recall']:6.2f}%  [CHECKPOINTED]")
else:
    print(f"✗ Epoch {epoch_num}: No results (tracker empty)")
METEOF

    echo "Epoch $EPOCH done at $(date)"
done

echo ""
echo "==========================================================="
echo "Finalizing..."
echo "==========================================================="

python - <<'FINALEOF'
import json
from pathlib import Path

CHECKPOINT_JSON = './outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val1_checkpoint.json'
FINAL_JSON = './outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val1.json'

with open(CHECKPOINT_JSON) as f:
    data = json.load(f)
data['status'] = 'complete'
with open(FINAL_JSON, 'w') as f:
    json.dump(data, f, indent=2)

epoch_results = {int(k): v for k, v in data['all_epochs'].items()}
if epoch_results:
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY (val_1)")
    print("="*100)
    for ep in sorted(epoch_results):
        a = epoch_results[ep]
        print(f"Epoch {ep:2d}:  MOTA={a['MOTA']:6.2f}% ±{a['MOTA_std']:5.2f}  IDF1={a['IDF1']:6.2f}% ±{a['IDF1_std']:5.2f}  Prec={a['Precision']:6.2f}%  Rec={a['Recall']:6.2f}%  TP={a['TP']:>7}  FP={a['FP']:>6}  FN={a['FN']:>7}  IDsw={a['IDsw']:>5}")
    if data['best_epoch'] is not None:
        b = data['best']
        print("\n" + "="*100)
        print(f"★ BEST: epoch {data['best_epoch']}  MOTA={b['MOTA']:.2f}% ±{b['MOTA_std']:.2f}  IDF1={b['IDF1']:.2f}% ±{b['IDF1_std']:.2f}  Prec={b['Precision']:.2f}%  Rec={b['Recall']:.2f}%  TP={b['TP']}  FP={b['FP']}  FN={b['FN']}  IDsw={b['IDsw']}")
        print("="*100)
print(f"\nFinal results saved to: {FINAL_JSON}")
FINALEOF

echo "==========================================================="
echo "Completed at $(date)"
echo "==========================================================="
