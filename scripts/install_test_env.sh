#!/bin/bash
# Script to install MOTIP dependencies in MOTIP_test environment

set -e
export PIP=~/miniconda3/envs/MOTIP_test/bin/pip
export PYTHON=~/miniconda3/envs/MOTIP_test/bin/python

echo "=== Installing MOTIP dependencies ==="
echo "Python: $($PYTHON --version)"
echo "Pip: $($PIP --version)"

echo ""
echo "=== Step 1: Verifying PyTorch installation ==="
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || {
    echo "PyTorch not found, installing..."
    $PIP install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
}

echo ""
echo "=== Step 2: Installing core dependencies ==="
$PIP install accelerate einops wandb

echo ""
echo "=== Step 3: Installing transformers ==="
# Note: transformers 5.x may have issues with BackboneMixin
$PIP install transformers

echo ""
echo "=== Step 4: Installing computer vision packages ==="
$PIP install timm pycocotools opencv-python matplotlib pillow

echo ""
echo "=== Step 5: Installing scientific computing ==="
$PIP install numpy pandas scipy scikit-learn

echo ""
echo "=== Step 6: Installing tracking utilities ==="
$PIP install motmetrics lap tqdm pyyaml

echo ""
echo "=== Step 7: Installing TrackEval dependencies ==="
$PIP install scikit-image pytest

echo ""
echo "=== Installation Complete ==="
$PIP list | head -50

echo ""
echo "=== Testing imports ==="
$PYTHON << 'EOF'
import sys
print(f"Python: {sys.version}")

errors = []

# Test PyTorch
try:
    import torch
    print(f"✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except Exception as e:
    errors.append(f"✗ torch: {e}")

# Test torchvision
try:
    import torchvision
    print(f"✓ torchvision {torchvision.__version__}")
except Exception as e:
    errors.append(f"✗ torchvision: {e}")

# Test transformers
try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    errors.append(f"✗ transformers: {e}")

# Test accelerate
try:
    import accelerate
    print(f"✓ accelerate {accelerate.__version__}")
except Exception as e:
    errors.append(f"✗ accelerate: {e}")

# Test timm
try:
    import timm
    print(f"✓ timm {timm.__version__}")
except Exception as e:
    errors.append(f"✗ timm: {e}")

# Test einops
try:
    import einops
    print(f"✓ einops")
except Exception as e:
    errors.append(f"✗ einops: {e}")

# Test motmetrics
try:
    import motmetrics
    print(f"✓ motmetrics")
except Exception as e:
    errors.append(f"✗ motmetrics: {e}")

# Test pycocotools
try:
    from pycocotools.coco import COCO
    print(f"✓ pycocotools")
except Exception as e:
    errors.append(f"✗ pycocotools: {e}")

# Test other packages
for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'cv2', 'matplotlib', 'PIL', 'yaml', 'tqdm']:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except Exception as e:
        errors.append(f"✗ {pkg}: {e}")

print()
if errors:
    print("=== ERRORS ===")
    for e in errors:
        print(e)
else:
    print("=== All imports successful! ===")
EOF
