#!/bin/bash
#SBATCH --job-name=test_full
#SBATCH --partition=gpu_h100_short 
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/logs/test_full_%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/logs/test_full_%j.err

# Full test of MOTIP_test environment
echo "=== Full MOTIP Environment Test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

export PYTHON=~/miniconda3/envs/MOTIP_test/bin/python

# Set CUDA
module load devel/cuda/11.8
export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8

cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

echo "=== Test 1: Basic imports ==="
$PYTHON << 'EOF'
import torch
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")

import torchvision
print(f"torchvision {torchvision.__version__}")

import transformers
print(f"transformers {transformers.__version__}")

import accelerate
print(f"accelerate {accelerate.__version__}")

import timm
print(f"timm {timm.__version__}")

print("Basic imports: OK")
EOF

echo ""
echo "=== Test 2: MultiScaleDeformableAttention ===" 
$PYTHON << 'EOF'
try:
    from modules import MSDeformAttn
    print("MSDeformAttn import: OK (from modules)")
except ImportError as e:
    print(f"MSDeformAttn import failed: {e}")
    
try:
    import MultiScaleDeformableAttention
    print("MultiScaleDeformableAttention SO: OK")
except ImportError as e:
    print(f"MultiScaleDeformableAttention SO failed: {e}")
EOF

echo ""
echo "=== Test 3: MOTIP model imports ==="
$PYTHON << 'EOF'
import sys
sys.path.insert(0, '/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP')

try:
    from models.deformable_detr.deformable_detr import DeformableDETR
    print("DeformableDETR: OK")
except Exception as e:
    print(f"DeformableDETR: FAILED - {e}")

try:
    from models.motip import MOTIP
    print("MOTIP: OK")
except Exception as e:
    print(f"MOTIP: FAILED - {e}")

try:
    from models.deformable_detr.deformable_transformer import DeformableTransformer
    print("DeformableTransformer: OK")
except Exception as e:
    print(f"DeformableTransformer: FAILED - {e}")
EOF

echo ""
echo "=== Test 4: Training imports ==="
$PYTHON << 'EOF'
import sys
sys.path.insert(0, '/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP')

try:
    from utils.misc import get_model, load_config
    print("utils.misc: OK")
except Exception as e:
    print(f"utils.misc: FAILED - {e}")

try:
    from data.pdestre import PDESTREDataset
    print("PDESTREDataset: OK")
except Exception as e:
    print(f"PDESTREDataset: FAILED - {e}")

try:
    from data import transforms
    print("transforms: OK")
except Exception as e:
    print(f"transforms: FAILED - {e}")
EOF

echo ""
echo "=== Test 5: Smoke test (dry run) ==="
$PYTHON train.py --config-path configs/smoke_test.yaml --exp-name test_environment --dry-run 2>&1 | head -30 || echo "Dry run not supported, proceeding..."

echo ""
echo "=== Test 6: transformers BackboneMixin ==="
$PYTHON << 'EOF'
try:
    from transformers.utils.backbone_utils import BackboneMixin
    print("BackboneMixin from transformers.utils.backbone_utils: OK")
except ImportError:
    try:
        from transformers.utils import BackboneMixin
        print("BackboneMixin from transformers.utils: OK")
    except ImportError as e:
        print(f"BackboneMixin: FAILED - {e}")
EOF

echo ""
echo "=== Summary ==="
echo "Test complete. Review output above for any failures."
