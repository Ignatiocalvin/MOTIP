#!/bin/bash
#SBATCH --job-name=eval_r50_7c_lw_val0
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --output=logs/eval_r50_7c_lw_val0_%j.out
#SBATCH --error=logs/eval_r50_7c_lw_val0_%j.err
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
CHECKPOINT_JSON="./outputs/${EXP}/eval_results_val0_checkpoint.json"

echo "==========================================================="
echo "Re-evaluating on val_0: $EXP"
echo "Started at $(date) on $(hostname)"
echo "Checkpointing results to: $CHECKPOINT_JSON"
echo "==========================================================="

# Initialize checkpoint file (only if it doesn't exist yet, to support resuming)
python - <<'INITEOF'
import json
from pathlib import Path
CHECKPOINT_JSON = "./outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val0_checkpoint.json"
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

    # Skip any epoch that already has 14 non-empty tracker files (already evaluated on val_0)
    NONEMPTY=0
    if [ -d "$OUT_DIR/tracker" ]; then
        for f in "$OUT_DIR/tracker"/*.txt; do
            [ -s "$f" ] && NONEMPTY=$((NONEMPTY + 1))
        done
    fi
    if [ "$NONEMPTY" -ge 14 ]; then
        echo "--- Epoch $EPOCH --- already evaluated on val_0 ($NONEMPTY files), skipping"
        continue
    fi

    # Need to re-evaluate: skip if already 14 files, otherwise resume from existing tracker
    echo ""
    if [ "$NONEMPTY" -gt 0 ]; then
        echo "--- Epoch $EPOCH --- resuming ($NONEMPTY/14 files done, evaluating remaining)"
    else
        echo "--- Epoch $EPOCH --- starting evaluation on val_0"
    fi
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
            --split val_0 \
            --output-dir "$OUT_DIR" \
            --skip-existing \
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
    else
        # Compute and checkpoint metrics for this epoch immediately
        python - <<'METEOF'
import sys
sys.path.insert(0, '.')
from evaluate_all_models import get_val_sequences, evaluate_epoch
from pathlib import Path
import json

output_dir = Path('outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0')
epoch_num = $EPOCH
CHECKPOINT_JSON = './outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val0_checkpoint.json'

# Load existing checkpoint
try:
    with open(CHECKPOINT_JSON, 'r') as f:
        data = json.load(f)
except:
    data = {'all_epochs': {}, 'best_epoch': None, 'best': None, 'status': 'in_progress'}

val_seqs = get_val_sequences('data/P-DESTRE/splits/val_0.txt')
eval_dir = output_dir / 'train' / 'eval_during_train'
epoch_dir = eval_dir / f'epoch_{epoch_num}'
tracker_dir = epoch_dir / 'tracker'

if tracker_dir.exists():
    agg = evaluate_epoch(tracker_dir, val_seqs)
    if agg:
        data['all_epochs'][str(epoch_num)] = agg
        # Update best epoch
        if not data['best'] or agg['MOTA'] > data['best']['MOTA']:
            data['best_epoch'] = epoch_num
            data['best'] = agg
        # Save checkpoint
        with open(CHECKPOINT_JSON, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Epoch {epoch_num:2d}: MOTA={agg['MOTA']:6.2f}% ±{agg['MOTA_std']:5.2f}  IDF1={agg['IDF1']:6.2f}% ±{agg['IDF1_std']:5.2f}  Prec={agg['Precision']:6.2f}%  Rec={agg['Recall']:6.2f}%  [CHECKPOINTED]")
    else:
        print(f"✗ Epoch {epoch_num}: No results (tracker empty)")
else:
    print(f"⚠ Epoch {epoch_num}: No tracker output")
METEOF
    fi
    echo "Epoch $EPOCH done at $(date)"
done

echo ""
echo "==========================================================="
echo "Finalizing checkpoint..."
echo "==========================================================="

python - <<'FINALEOF'
import json
from pathlib import Path

CHECKPOINT_JSON = './outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val0_checkpoint.json'
FINAL_JSON = './outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val0.json'

# Load checkpoint and finalize
with open(CHECKPOINT_JSON, 'r') as f:
    data = json.load(f)

data['status'] = 'complete'

# Save final results
with open(FINAL_JSON, 'w') as f:
    json.dump(data, f, indent=2)

# Display summary
epoch_results = {int(k): v for k, v in data['all_epochs'].items()}
if epoch_results:
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY (val_0)")
    print("="*100)
    for epoch_num in sorted(epoch_results.keys()):
        agg = epoch_results[epoch_num]
        print(f"Epoch {epoch_num:2d}:  MOTA={agg['MOTA']:6.2f}% ±{agg['MOTA_std']:5.2f}  IDF1={agg['IDF1']:6.2f}% ±{agg['IDF1_std']:5.2f}  Prec={agg['Precision']:6.2f}%  Rec={agg['Recall']:6.2f}%  TP={agg['TP']:>7}  FP={agg['FP']:>6}  FN={agg['FN']:>7}  IDsw={agg['IDsw']:>5}")
    
    if data['best_epoch'] is not None:
        best = data['best']
        print("\n" + "="*100)
        print(f"★ BEST: epoch {data['best_epoch']}  MOTA={best['MOTA']:.2f}% ±{best['MOTA_std']:.2f}  IDF1={best['IDF1']:.2f}% ±{best['IDF1_std']:.2f}  Prec={best['Precision']:.2f}%  Rec={best['Recall']:.2f}%  TP={best['TP']}  FP={best['FP']}  FN={best['FN']}  IDsw={best['IDsw']}")
        print("="*100 + "\n")

print(f"Checkpoint saved to: {CHECKPOINT_JSON}")
print(f"Final results saved to: {FINAL_JSON}")
FINALEOF

echo "==========================================================="
echo "Completed at $(date)"
echo "==========================================================="
