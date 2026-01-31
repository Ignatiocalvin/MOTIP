#!/usr/bin/env bash
#SBATCH --job-name=eval_fold0
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time 00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/eval_fold0_%j.out
#SBATCH --error=logs/eval_fold0_%j.err

echo "=========================================="
echo "Evaluation Only Script"
echo "Started at $(date)"
echo "=========================================="

module load devel/cuda/11.8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME='/opt/bwhpc/common/devel/cuda/11.8'
export CUDA_VISIBLE_DEVICES=0

cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# Run evaluation on the saved checkpoint with lower thresholds
python submit_and_evaluate.py \
    --config-path ./configs/r50_deformable_detr_motip_pdestre_fast.yaml \
    --data-root ./data/ \
    --inference-model ./outputs/r50_motip_pdestre_fold_0/checkpoint_2.pth \
    --outputs-dir ./outputs/r50_motip_pdestre_fold_0 \
    --inference-dataset P-DESTRE \
    --inference-split val_0 \
    --inference-mode evaluate \
    --inference-group fold_0_epoch2 \
    --det-thresh 0.3 \
    --newborn-thresh 0.6

echo "Finished at $(date)"
