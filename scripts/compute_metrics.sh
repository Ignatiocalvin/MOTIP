#!/bin/bash
#SBATCH --job-name=motip_metrics
#SBATCH --partition=cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/logs/compute_metrics_%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/logs/compute_metrics_%j.err

set -e
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

echo "========================================"
echo "  MOTIP — Compute All Epoch Metrics"
echo "  Start: $(date)"
echo "  Node : $(hostname)"
echo "========================================"

PYTHON=/home/ma/ma_ma/ma_ighidaya/miniconda3/envs/MOTIP_fresh/bin/python

echo "[ENV] Python: $PYTHON ($(${PYTHON} --version))"

$PYTHON compute_all_metrics.py

echo ""
echo "Done: $(date)"
