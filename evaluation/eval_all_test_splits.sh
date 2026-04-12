#!/bin/bash
#SBATCH --job-name=eval_test_splits
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/eval_test_splits_%j.out
#SBATCH --error=logs/eval_test_splits_%j.err
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# ============================================================
# Evaluate all best-epoch checkpoints on held-out Test splits.
#
# Models trained on Train_0 → evaluated on Test_0 (15 sequences)
# Models trained on Train_1 → evaluated on Test_1 (15 sequences)
#
# Best epochs were selected on the validation split (val_0 / val_1).
# These test-set results provide unbiased generalisation estimates.
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

echo "==========================================================="
echo "Test-Split Evaluation — All Best-Epoch Checkpoints"
echo "Started at $(date) on $(hostname) — GPU: $GPU_NAME"
echo "==========================================================="

# ---- Define all models to evaluate ----
#
# Format: EXP_NAME | CHECKPOINT_EPOCH | TEST_SPLIT | CONFIG_PATH
#
# Models trained on Train_0 → Test_0
# Models trained on Train_1 → Test_1

declare -a MODELS=(
    "r50_base_motip_2concepts_fold_0|7|Test_0|outputs/r50_base_motip_2concepts_fold_0/train/config.yaml"
    "r50_motip_pdestre_3concepts_fold_0|2|Test_0|outputs/r50_motip_pdestre_3concepts_fold_0/train/config.yaml"
    "rfdetr_large_motip_pdestre_2concepts_fold0|0|Test_0|outputs/rfdetr_large_motip_pdestre_2concepts_fold0/train/config.yaml"
    "rfdetr_large_motip_pdestre_7concepts_learnable_fold0|0|Test_0|outputs/rfdetr_large_motip_pdestre_7concepts_learnable_fold0/train/config.yaml"
    "rfdetr_large_motip_pdestre_7concepts_lw_fold0|2|Test_0|outputs/rfdetr_large_motip_pdestre_7concepts_lw_fold0/train/config.yaml"
    "r50_motip_pdestre_7concepts_learnable_v2_fold_0|3|Test_1|outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/train/config.yaml"
    "r50_motip_pdestre_7concepts_learnable_fold_1|9|Test_1|outputs/r50_motip_pdestre_7concepts_learnable_fold_1/train/config.yaml"
)

TOTAL_MODELS=${#MODELS[@]}
SUCCEEDED=0
FAILED=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r EXP EPOCH SPLIT CONFIG <<< "$entry"

    CKPT="./outputs/${EXP}/checkpoint_${EPOCH}.pth"
    OUT_DIR="./outputs/${EXP}/eval_test/${SPLIT}/epoch_${EPOCH}"

    echo ""
    echo "==========================================================="
    echo "Model: $EXP"
    echo "Epoch: $EPOCH  |  Split: $SPLIT  |  Config: $CONFIG"
    echo "Checkpoint: $CKPT"
    echo "Output: $OUT_DIR"
    echo "==========================================================="

    if [ ! -f "$CKPT" ]; then
        echo "[ERROR] Checkpoint not found: $CKPT — SKIPPING"
        FAILED=$((FAILED + 1))
        continue
    fi

    if [ ! -f "$CONFIG" ]; then
        echo "[ERROR] Config not found: $CONFIG — SKIPPING"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Skip if already has non-empty tracker files (15 sequences per test split)
    NONEMPTY=0
    if [ -d "$OUT_DIR/tracker" ]; then
        for f in "$OUT_DIR/tracker"/*.txt; do
            if [ -s "$f" ]; then
                NONEMPTY=$((NONEMPTY + 1))
            fi
        done
    fi
    if [ "$NONEMPTY" -ge 15 ]; then
        echo "--- ALREADY DONE ($NONEMPTY non-empty tracker files, skipping inference) ---"
        SUCCEEDED=$((SUCCEEDED + 1))
        continue
    fi

    mkdir -p "$OUT_DIR"

    # Remove old empty tracker files if partial
    rm -rf "$OUT_DIR/tracker"

    EVAL_OK=false
    for RETRY in 1 2 3; do
        env PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rf-detr:${PYTHONPATH}" \
        python -u -m accelerate.commands.launch --num_processes=1 \
            evaluation/evaluate_checkpoint.py \
            --checkpoint "$CKPT" \
            --config "$CONFIG" \
            --data-root ./data/ \
            --dataset P-DESTRE \
            --split "$SPLIT" \
            --output-dir "$OUT_DIR" \
            2>&1
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            EVAL_OK=true
            break
        fi
        echo "[WARN] $EXP attempt $RETRY failed (exit $EXIT_CODE). Retrying in 10s..."
        sleep 10
    done

    if [ "$EVAL_OK" = true ]; then
        echo "[OK] $EXP / epoch $EPOCH / $SPLIT — inference complete at $(date)"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "[ERROR] $EXP / epoch $EPOCH / $SPLIT — FAILED after 3 attempts"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "==========================================================="
echo "Inference phase complete: $SUCCEEDED/$TOTAL_MODELS succeeded, $FAILED failed"
echo "==========================================================="
echo ""
echo "Computing tracking metrics (MOTA, IDF1, HOTA) ..."
echo "==========================================================="

# ---- Compute metrics for all evaluated models ----
python - <<'PYEOF'
import sys, json, os
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, str(Path('.') / 'TrackEval'))

from evaluate_all_models import get_val_sequences, evaluate_epoch
from trackeval.metrics.hota import HOTA

GT_DIR = Path('data/P-DESTRE/annotations')


def load_gt(seq_name):
    ann = defaultdict(list)
    with open(GT_DIR / f'{seq_name}.txt') as f:
        for line in f:
            p = line.strip().split(',')
            fid, tid = int(p[0]), int(p[1])
            if tid < 0:
                continue
            ann[fid].append((tid, float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return dict(ann)


def load_pred(path):
    pred = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            fid, tid = int(p[0]), int(p[1])
            if tid < 0:
                continue
            pred[fid].append((tid, float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return dict(pred)


def iou_box(g, p):
    ix1 = max(g[1], p[1]); iy1 = max(g[2], p[2])
    ix2 = min(g[1]+g[3], p[1]+p[3]); iy2 = min(g[2]+g[4], p[2]+p[4])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = g[3]*g[4] + p[3]*p[4] - inter
    return inter / max(1e-9, union)


def build_hota_data(gt_data, pred_data):
    gt_id_map, pr_id_map = {}, {}
    all_frames = sorted(set(gt_data) | set(pred_data))
    gt_ids_per_frame, pr_ids_per_frame, sim_per_frame = [], [], []
    for fid in all_frames:
        gt_b = gt_data.get(fid, [])
        pr_b = pred_data.get(fid, [])
        gids = []
        for g in gt_b:
            if g[0] not in gt_id_map: gt_id_map[g[0]] = len(gt_id_map)
            gids.append(gt_id_map[g[0]])
        pids = []
        for p in pr_b:
            if p[0] not in pr_id_map: pr_id_map[p[0]] = len(pr_id_map)
            pids.append(pr_id_map[p[0]])
        if gt_b and pr_b:
            sim = np.array([[iou_box(g, p) for p in pr_b] for g in gt_b], dtype=np.float32)
        else:
            sim = np.empty((len(gt_b), len(pr_b)), dtype=np.float32)
        gt_ids_per_frame.append(np.array(gids, dtype=int))
        pr_ids_per_frame.append(np.array(pids, dtype=int))
        sim_per_frame.append(sim)
    return {
        'num_timesteps': len(all_frames),
        'num_gt_ids': len(gt_id_map),
        'num_tracker_ids': len(pr_id_map),
        'num_gt_dets': sum(len(g) for g in gt_ids_per_frame),
        'num_tracker_dets': sum(len(p) for p in pr_ids_per_frame),
        'gt_ids': gt_ids_per_frame,
        'tracker_ids': pr_ids_per_frame,
        'similarity_scores': sim_per_frame,
    }


def compute_hota(tracker_dir, seqs):
    hota_metric = HOTA()
    all_seq_res = {}
    for s in seqs:
        pr_f = tracker_dir / f'{s}.txt'
        if not pr_f.exists():
            continue
        gt_data = load_gt(s)
        pred_data = load_pred(pr_f)
        data = build_hota_data(gt_data, pred_data)
        all_seq_res[s] = hota_metric.eval_sequence(data)
    if not all_seq_res:
        return None
    return {
        'HOTA': float(np.mean([np.mean(v['HOTA']) for v in all_seq_res.values()])) * 100,
        'DetA': float(np.mean([np.mean(v['DetA']) for v in all_seq_res.values()])) * 100,
        'AssA': float(np.mean([np.mean(v['AssA']) for v in all_seq_res.values()])) * 100,
        'DetRe': float(np.mean([np.mean(v['DetRe']) for v in all_seq_res.values()])) * 100,
        'DetPr': float(np.mean([np.mean(v['DetPr']) for v in all_seq_res.values()])) * 100,
        'AssRe': float(np.mean([np.mean(v['AssRe']) for v in all_seq_res.values()])) * 100,
        'AssPr': float(np.mean([np.mean(v['AssPr']) for v in all_seq_res.values()])) * 100,
        'n_seqs': len(all_seq_res),
    }


# Define models with their test split paths
MODELS = [
    {
        'name': 'R50-MOTIP-2C',
        'exp': 'r50_base_motip_2concepts_fold_0',
        'epoch': 7,
        'test_split': 'Test_0',
        'split_file': 'splits/Test_0.txt',
    },
    {
        'name': 'R50-MOTIP-3C',
        'exp': 'r50_motip_pdestre_3concepts_fold_0',
        'epoch': 2,
        'test_split': 'Test_0',
        'split_file': 'splits/Test_0.txt',
    },
    {
        'name': 'R50-MOTIP-7C-LW (F0)',
        'exp': 'r50_motip_pdestre_7concepts_learnable_v2_fold_0',
        'epoch': 3,
        'test_split': 'Test_1',
        'split_file': 'splits/Test_1.txt',
    },
    {
        'name': 'RF-DETR-MOTIP-2C',
        'exp': 'rfdetr_large_motip_pdestre_2concepts_fold0',
        'epoch': 0,
        'test_split': 'Test_0',
        'split_file': 'splits/Test_0.txt',
    },
    {
        'name': 'RF-DETR-MOTIP-7C-NLW',
        'exp': 'rfdetr_large_motip_pdestre_7concepts_learnable_fold0',
        'epoch': 0,
        'test_split': 'Test_0',
        'split_file': 'splits/Test_0.txt',
    },
    {
        'name': 'RF-DETR-MOTIP-7C-LW',
        'exp': 'rfdetr_large_motip_pdestre_7concepts_lw_fold0',
        'epoch': 2,
        'test_split': 'Test_0',
        'split_file': 'splits/Test_0.txt',
    },
    {
        'name': 'R50-MOTIP-7C-LW (F1, cross-fold)',
        'exp': 'r50_motip_pdestre_7concepts_learnable_fold_1',
        'epoch': 9,
        'test_split': 'Test_1',
        'split_file': 'splits/Test_1.txt',
    },
]

all_results = {}

print()
print(f"{'Model':<35} {'Split':>7} {'MOTA':>7} {'IDF1':>7} {'Prec':>7} {'Rec':>7} {'HOTA':>7} {'DetA':>7} {'AssA':>7} {'IDsw':>6} {'TP':>7} {'FP':>6} {'FN':>7}")
print('-' * 140)

for m in MODELS:
    tracker_dir = Path(f"outputs/{m['exp']}/eval_test/{m['test_split']}/epoch_{m['epoch']}/tracker")
    test_seqs = get_val_sequences(m['split_file'])

    if not tracker_dir.exists():
        print(f"{m['name']:<35} {'N/A':>7}  — tracker dir not found: {tracker_dir}")
        continue

    # MOT metrics
    mot = evaluate_epoch(tracker_dir, test_seqs)
    # HOTA metrics
    hota = compute_hota(tracker_dir, test_seqs)

    if mot and hota:
        print(f"{m['name']:<35} {m['test_split']:>7} {mot['MOTA']:>6.1f}% {mot['IDF1']:>6.1f}% {mot['Precision']:>6.1f}% {mot['Recall']:>6.1f}% {hota['HOTA']:>6.1f}% {hota['DetA']:>6.1f}% {hota['AssA']:>6.1f}% {mot['IDsw']:>6} {mot['TP']:>7} {mot['FP']:>6} {mot['FN']:>7}")
        all_results[m['name']] = {**mot, **hota, 'test_split': m['test_split'], 'epoch': m['epoch'], 'exp': m['exp']}
    elif mot:
        print(f"{m['name']:<35} {m['test_split']:>7} {mot['MOTA']:>6.1f}% {mot['IDF1']:>6.1f}% {mot['Precision']:>6.1f}% {mot['Recall']:>6.1f}%   —       —       —   {mot['IDsw']:>6} {mot['TP']:>7} {mot['FP']:>6} {mot['FN']:>7}")
        all_results[m['name']] = {**mot, 'test_split': m['test_split'], 'epoch': m['epoch'], 'exp': m['exp']}
    else:
        print(f"{m['name']:<35} {'N/A':>7}  — no results")

print()

# Save results to JSON
results_file = Path('evaluation_results_test_splits.json')
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {results_file}")

PYEOF

echo ""
echo "==========================================================="
echo "All done at $(date)"
echo "==========================================================="
