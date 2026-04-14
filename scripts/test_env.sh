#!/bin/bash
#SBATCH --job-name=test_motip_env
#SBATCH --partition=gpu_h100_short 
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/logs/test_env_%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/logs/test_env_%j.err

# Test MOTIP_test environment setup
echo "=== MOTIP Environment Test Script ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

export PIP=~/miniconda3/envs/MOTIP_test/bin/pip
export PYTHON=~/miniconda3/envs/MOTIP_test/bin/python

# Set CUDA
module load devel/cuda/11.8
export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8
echo "CUDA_HOME: $CUDA_HOME"
echo ""

echo "=== Python Version ==="
$PYTHON --version
echo ""

echo "=== Installed Packages ==="
$PIP list | head -50
echo ""

echo "=== Testing Imports ==="
$PYTHON << 'PYEOF'
import sys
print(f"Python: {sys.version}")
print()

results = []

# Test PyTorch
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_avail else 0
    results.append(f"✓ torch {torch.__version__} (CUDA: {cuda_avail}, devices: {device_count})")
    if cuda_avail:
        results.append(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    results.append(f"✗ torch: {e}")

# Test torchvision
try:
    import torchvision
    results.append(f"✓ torchvision {torchvision.__version__}")
except Exception as e:
    results.append(f"✗ torchvision: {e}")

# Test transformers
try:
    import transformers
    results.append(f"✓ transformers {transformers.__version__}")
except Exception as e:
    results.append(f"✗ transformers: {e}")

# Test accelerate
try:
    import accelerate
    results.append(f"✓ accelerate {accelerate.__version__}")
except Exception as e:
    results.append(f"✗ accelerate: {e}")

# Test timm
try:
    import timm
    results.append(f"✓ timm {timm.__version__}")
except Exception as e:
    results.append(f"✗ timm: {e}")

# Test einops
try:
    import einops
    results.append(f"✓ einops")
except Exception as e:
    results.append(f"✗ einops: {e}")

# Test other packages
for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'cv2', 'matplotlib', 'PIL', 'yaml', 'tqdm', 'motmetrics', 'lap']:
    try:
        __import__(pkg)
        results.append(f"✓ {pkg}")
    except Exception as e:
        results.append(f"✗ {pkg}: {e}")

# Test pycocotools
try:
    from pycocotools.coco import COCO
    results.append(f"✓ pycocotools")
except Exception as e:
    results.append(f"✗ pycocotools: {e}")

for r in results:
    print(r)

print()
print("=== Testing MOTIP-specific imports ===")

# Test MOTIP models
import os, sys
sys.path.insert(0, '/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP')

try:
    from models.deformable_detr import DeformableDETR
    results.append(f"✓ DeformableDETR import")
    print("✓ DeformableDETR import")
except Exception as e:
    results.append(f"✗ DeformableDETR: {e}")
    print(f"✗ DeformableDETR: {e}")

try:
    from models.motip import MOTIP
    results.append(f"✓ MOTIP import")
    print("✓ MOTIP import")
except Exception as e:
    results.append(f"✗ MOTIP: {e}")
    print(f"✗ MOTIP: {e}")

print()
print("=== Testing MultiScaleDeformableAttention CUDA Extension ===")
try:
    from models.ops import MultiScaleDeformableAttention
    print("✓ MultiScaleDeformableAttention import")
except ImportError as e:
    print(f"✗ MultiScaleDeformableAttention not built yet: {e}")
    print("  Run: cd models/ops && python setup.py build install")
except Exception as e:
    print(f"✗ MultiScaleDeformableAttention: {e}")

print()
print("=== Testing transformers BackboneMixin (potential issue with v5.x) ===")
try:
    from transformers.utils import BackboneMixin
    print("✓ BackboneMixin import from transformers.utils")
except ImportError:
    try:
        from transformers.utils.backbone_utils import BackboneMixin
        print("✓ BackboneMixin import from transformers.utils.backbone_utils")
    except ImportError as e:
        print(f"✗ BackboneMixin not found: {e}")
        print("  This may cause issues with RF-DETR integration")

PYEOF

echo ""
echo "=== Test Complete ==="
