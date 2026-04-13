#!/bin/bash
#SBATCH --job-name=motip_smoke_test
#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=./logs/smoke_test_%j.out
#SBATCH --error=./logs/smoke_test_%j.err

set -e

WORKDIR=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP
cd "$WORKDIR"

echo "=== MOTIP Smoke Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""

# Activate conda environment
source /home/ma/ma_ma/ma_ighidaya/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

# Ensure PyTorch CUDA ops (.so) can find libc10.so etc.
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
echo "TORCH_LIB: $TORCH_LIB"

# Verify GPU visible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}')"

echo ""
echo "=== Starting short smoke-test training (5 steps per log) ==="
echo ""

python train.py \
    --config-path ./configs/r50_test_smoke.yaml \
    --num-workers 0

echo ""
echo "=== Smoke test DONE ==="
echo "Date: $(date)"
