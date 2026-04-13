#!/bin/bash
#SBATCH --job-name=motip_smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/smoke_%j.out
#SBATCH --error=./logs/smoke_%j.err

# ================================================================== #
#  MOTIP — R50 Smoke Test                                             #
#                                                                      #
#  PREREQUISITE (run ONCE on login node before sbatch):               #
#    ./scripts/download_and_preprocess.sh                             #
#                                                                      #
#  That script creates the MOTIP conda env, installs deps,            #
#  and builds CUDA ops.                                               #
# ================================================================== #

set -e

WORKDIR=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP
CONDA_BASE=/home/ma/ma_ma/ma_ighidaya/miniconda3
ENV_NAME=MOTIP

cd "$WORKDIR"
mkdir -p logs

echo "================================================================"
echo "  MOTIP — R50 Smoke Test"
echo "================================================================"
echo "Date    : $(date)"
echo "Node    : $(hostname)"
echo "Workdir : $WORKDIR"
echo ""

# Activate env
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "[ENV] Python: $(which python)  ($(python --version))"
echo "[ENV] torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"

# Runtime setup
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo ""
python -c "import torch; print(f'[GPU] CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}')"

# Build CUDA ops if not already installed
echo ""
if python -c "import MultiScaleDeformableAttention" 2>/dev/null; then
    echo "[CUDA OPS] MultiScaleDeformableAttention already installed, skipping build."
else
    echo "[CUDA OPS] Building MultiScaleDeformableAttention CUDA extension..."
    module load devel/cuda/11.8
    export TORCH_CUDA_ARCH_LIST="8.0;9.0"
    cd "${WORKDIR}/models/ops"
    python setup.py build install
    cd "${WORKDIR}"
    echo "[CUDA OPS] Build complete."
fi

# Smoke training
echo ""
echo "================================================================"
echo "  Starting R50 MOTIP smoke training (30 steps, no concepts)"
echo "  Config: configs/r50_test_smoke.yaml"
echo "================================================================"
echo ""

python train.py \
    --config-path ./configs/r50_test_smoke.yaml \
    --num-workers 0

echo ""
echo "================================================================"
echo "  Smoke test DONE."
echo "  Date: $(date)"
echo "================================================================"
