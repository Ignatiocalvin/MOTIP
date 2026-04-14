#!/bin/bash
#SBATCH --job-name=motip_smoke_metrics
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/smoke_metrics_%j.out
#SBATCH --error=logs/smoke_metrics_%j.err
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# Smoke test: 2 epochs x 20 steps each, then eval on val_0
# Purpose: verify P-DESTRE metrics are computed and logged during training

# ── Environment setup ─────────────────────────────────────────────────────────
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
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
[ -d "/usr/local/cuda" ] && export CUDA_HOME="/usr/local/cuda"

echo "═══════════════════════════════════════════════════════════════"
echo "MOTIP Smoke Test — metrics integration check"
echo "GPU: $GPU_NAME   Start: $(date)   Node: $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

# ── Train ─────────────────────────────────────────────────────────────────────
accelerate launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name  r50_motip_pdestre_smoke_metrics_test \
    --config-path ./configs/r50_deformable_detr_motip_pdestre_smoke_metrics.yaml \
    > >(while IFS= read -r line; do echo "$(date +'%H:%M:%S') $line"; done) 2>&1

EXIT_CODE=$?
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Done (exit $EXIT_CODE) at $(date)"
echo "═══════════════════════════════════════════════════════════════"
