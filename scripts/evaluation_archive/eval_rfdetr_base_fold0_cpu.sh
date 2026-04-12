#!/bin/bash
#SBATCH --job-name=eval_rfdetr_fold0_cpu
#SBATCH --partition=cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/eval_rfdetr_base_fold0_cpu_%j.out
#SBATCH --error=logs/eval_rfdetr_base_fold0_cpu_%j.err
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

# Use all allocated CPUs for PyTorch/OpenMP threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Disable GPU visibility entirely
export CUDA_VISIBLE_DEVICES=""

SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rf-detr:${PYTHONPATH}"

EXP="rfdetr_large_motip_pdestre_base_fold0"
CONFIG="./outputs/${EXP}/train/config.yaml"
OUT_BASE="./outputs/${EXP}/train/eval_during_train"

echo "==========================================================="
echo "Evaluating all checkpoints for: $EXP  [CPU mode]"
echo "Started at $(date) on $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK  |  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "==========================================================="

for EPOCH in 0 1 2 3 4 5 6 7 8 9; do
    CKPT="./outputs/${EXP}/checkpoint_${EPOCH}.pth"
    OUT_DIR="${OUT_BASE}/epoch_${EPOCH}"

    if [ ! -f "$CKPT" ]; then
        echo "Checkpoint not found, skipping: $CKPT"
        continue
    fi

    # Skip if already completed successfully
    if [ -d "$OUT_DIR/tracker" ] && [ -n "$(ls -A "$OUT_DIR/tracker" 2>/dev/null)" ]; then
        echo "--- Epoch $EPOCH --- ALREADY DONE (skipping)"
        continue
    fi

    echo ""
    echo "--- Epoch $EPOCH ---"
    echo "Checkpoint: $CKPT"
    echo "Output dir: $OUT_DIR"
    mkdir -p "$OUT_DIR"

    EPOCH_OK=false
    for RETRY in 1 2 3; do
        env PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rf-detr:${PYTHONPATH}" \
        python -u -m accelerate.commands.launch --num_processes=1 --cpu \
            evaluation/evaluate_checkpoint.py \
            --checkpoint "$CKPT" \
            --config "$CONFIG" \
            --data-root ./data/ \
            --dataset P-DESTRE \
            --split val_0 \
            --output-dir "$OUT_DIR" \
            2>&1
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            EPOCH_OK=true
            break
        fi
        echo "[WARN] Epoch $EPOCH attempt $RETRY failed (exit $EXIT_CODE). Retrying in 60s..."
        sleep 60
    done
    if [ "$EPOCH_OK" = false ]; then
        echo "[ERROR] Epoch $EPOCH failed after 3 attempts, continuing to next epoch."
    fi

    echo "Epoch $EPOCH done at $(date)"
done

echo ""
echo "==========================================================="
echo "All inference done. Computing metrics..."
echo "==========================================================="

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from evaluate_all_models import get_val_sequences, evaluate_epoch
from pathlib import Path
import json

output_dir = Path('outputs/rfdetr_large_motip_pdestre_base_fold0')
val_seqs = get_val_sequences('splits/val_0.txt')
eval_dir = output_dir / 'train' / 'eval_during_train'

epoch_results = {}
for epoch_dir in sorted(eval_dir.iterdir()):
    if not epoch_dir.is_dir() or not epoch_dir.name.startswith('epoch_'):
        continue
    tracker_dir = epoch_dir / 'tracker'
    if not tracker_dir.exists():
        continue
    epoch_num = int(epoch_dir.name.split('_')[1])
    agg = evaluate_epoch(tracker_dir, val_seqs)
    if agg:
        epoch_results[epoch_num] = agg
        print(f"Epoch {epoch_num:2d}:  MOTA={agg['MOTA']:6.2f}%  IDF1={agg['IDF1']:6.2f}%  Prec={agg['Precision']:6.2f}%  Rec={agg['Recall']:6.2f}%  TP={agg['TP']:>7}  FP={agg['FP']:>6}  FN={agg['FN']:>7}  IDsw={agg['IDsw']:>5}")
    else:
        print(f"Epoch {epoch_num:2d}: No results (empty tracker)")

if epoch_results:
    best_ep, best = max(epoch_results.items(), key=lambda x: x[1]['MOTA'])
    print(f"\n★ BEST EPOCH: {best_ep}")
    print(f"  MOTA:      {best['MOTA']:.2f}% ± {best['MOTA_std']:.2f}%")
    print(f"  IDF1:      {best['IDF1']:.2f}% ± {best['IDF1_std']:.2f}%")
    print(f"  Precision: {best['Precision']:.2f}%")
    print(f"  Recall:    {best['Recall']:.2f}%")
    print(f"  TP:        {best['TP']}")
    print(f"  FP:        {best['FP']}")
    print(f"  FN:        {best['FN']}")
    print(f"  IDsw:      {best['IDsw']}")
    print(f"  GT:        {best['GT']}")
    print(f"  Pred:      {best['Pred']}")
    with open('outputs/rfdetr_large_motip_pdestre_base_fold0/eval_results.json', 'w') as f:
        json.dump({'all_epochs': {str(k): v for k, v in epoch_results.items()}, 'best_epoch': best_ep, 'best': best}, f, indent=2)
    print("\nResults saved to outputs/rfdetr_large_motip_pdestre_base_fold0/eval_results.json")
else:
    print("No evaluation results found across any epoch.")
PYEOF

echo "==========================================================="
echo "Completed at $(date)"
echo "==========================================================="
