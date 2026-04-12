#!/bin/bash
#SBATCH --job-name=motip_7c_lw_dev
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=logs/motip_7c_lw_%j.out
#SBATCH --error=logs/motip_7c_lw_%j.err
#SBATCH --exclude=uc3n073
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# ── identity ──────────────────────────────────────────────────────────────────
EXP_NAME="rfdetr_large_motip_pdestre_7concepts_lw_fold0"
CONFIG="./configs/rfdetr_large_motip_pdestre_7concepts_learnable.yaml"
SCRIPT_PATH="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/train_rfdetr_7c_lw_dev.sh"
SCRIPT_DIR="/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP"

# ── signal handler: stop training, resubmit self ───────────────────────────
TRAINING_PID=""
resubmit() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SIGUSR1: time limit approaching — stopping and resubmitting..."
    if [ -n "$TRAINING_PID" ] && kill -0 "$TRAINING_PID" 2>/dev/null; then
        kill -TERM "$TRAINING_PID" 2>/dev/null
        for i in $(seq 1 60); do
            sleep 1
            kill -0 "$TRAINING_PID" 2>/dev/null || { echo "Training stopped."; break; }
        done
        kill -KILL "$TRAINING_PID" 2>/dev/null
    fi
    NEW_JOB=$(sbatch --parsable "$SCRIPT_PATH")
    echo "Resubmitted as job $NEW_JOB"
    exit 0
}
trap 'resubmit' SIGUSR1

# ── env setup ─────────────────────────────────────────────────────────────────
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

if command -v module &> /dev/null; then module load devel/cuda/11.8 || true; fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *)      export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" ;;
esac
export CUDA_VISIBLE_DEVICES=0
[ -d "/usr/local/cuda" ] && export CUDA_HOME="/usr/local/cuda"
export PYTHONPATH="${SCRIPT_DIR}/rf-detr:${PYTHONPATH}"

echo "======================================================="
echo "Job : $EXP_NAME  [dev_gpu_h100, self-resubmitting]"
echo "Start: $(date)  Node: $(hostname)  GPU: $GPU_NAME"
echo "======================================================="

# ── auto-detect checkpoint ────────────────────────────────────────────────────
OUTPUTS_DIR="./outputs/${EXP_NAME}"
INTRA_DIR="${OUTPUTS_DIR}/intra_epoch_checkpoints"
RESUME_ARGS=()
if ls "${INTRA_DIR}"/*.pth 2>/dev/null | head -1 | grep -q . || \
   ls "${OUTPUTS_DIR}"/checkpoint_*.pth 2>/dev/null | head -1 | grep -q .; then
    echo "[checkpoint] Found — resuming from latest (--use-previous-checkpoint True)"
    RESUME_ARGS=(--use-previous-checkpoint True)
else
    echo "[checkpoint] None found — starting fresh"
fi

# ── train ─────────────────────────────────────────────────────────────────────
echo "Starting at $(date)..."
python -u -m accelerate.commands.launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name "$EXP_NAME" \
    --config-path "$CONFIG" \
    "${RESUME_ARGS[@]}" \
    > >(while IFS= read -r line; do echo "$(date +'%Y-%m-%d %H:%M:%S') $line"; done) 2>&1 &
TRAINING_PID=$!
echo "Training PID: $TRAINING_PID"
wait $TRAINING_PID
EXIT_CODE=$?
echo "Training exited with code $EXIT_CODE at $(date)."
