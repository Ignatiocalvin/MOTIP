#!/bin/bash
#SBATCH --job-name=motip_r50_2c_v2
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/motip_r50_2c_v2_%j.out
#SBATCH --error=logs/motip_r50_2c_v2_%j.err
#SBATCH --exclude=uc3n073
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
unset PYTHONHOME

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p logs
    exec > >(tee -a "logs/motip_r50_2c_v2_$(date +%Y%m%d_%H%M%S).log") 2>&1
fi

echo "=========================================="
echo "MOTIP R50 2-Concept v2 (corrected hyperparams)"
echo "  CONCEPT_DIM=64, SAMPLE_LENGTHS=[15], SAMPLE_INTERVALS=[4], ACCUMULATE_STEPS=2"
echo "Started at $(date) on $(hostname)"
echo "=========================================="

if command -v module &> /dev/null; then
    module load devel/cuda/11.8 || true
fi

CONDA_BASE="/home/ma/ma_ma/ma_ighidaya/miniconda3"
[ ! -d "$CONDA_BASE" ] && CONDA_BASE="$HOME/miniconda3"
[ ! -d "$CONDA_BASE" ] && CONDA_BASE="$HOME/anaconda3"
[ ! -d "$CONDA_BASE" ] && { echo "ERROR: conda not found"; exit 1; }

eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate MOTIP
export PATH="$CONDA_BASE/envs/MOTIP/bin:$PATH"

PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
if [[ "$PYTHON_PATH" != *"miniconda3/envs/MOTIP"* ]] && [[ "$PYTHON_PATH" != *"anaconda3/envs/MOTIP"* ]]; then
    echo "ERROR: Not using MOTIP environment! Python: $PYTHON_PATH"; exit 1
fi
echo "Using Python: $PYTHON_PATH"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *H100*) export TORCH_CUDA_ARCH_LIST="9.0" ;;
    *A100*) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *) export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" ;;
esac

[ -d "/usr/local/cuda" ] && export CUDA_HOME="/usr/local/cuda"
export CUDA_VISIBLE_DEVICES=0

echo "Starting training at $(date)..."

accelerate launch --num_processes=1 train.py \
    --data-root ./data/ \
    --exp-name r50_motip_pdestre_2concepts_fold_0_v2 \
    --config-path ./configs/r50_deformable_detr_motip_pdestre_2concepts_fold0_v2.yaml \
    2>&1 | while IFS= read -r line; do echo "$(date +'%Y-%m-%d %H:%M:%S') $line"; done

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
