#!/bin/bash
# Install RF-DETR dependencies for MOTIP integration

echo "Installing RF-DETR dependencies..."

# Core dependencies
pip install transformers>=4.40.0 timm>=0.9.0 --upgrade

# Additional dependencies for RF-DETR
pip install peft fairscale einops

# Supervision for visualization (optional but useful)
pip install supervision>=0.19.0

echo ""
echo "Verifying installation..."
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import peft; print(f'peft: {peft.__version__}')"
python -c "import einops; print(f'einops installed')"

echo ""
echo "Dependencies installed successfully!"
echo "You can now use RF-DETR with MOTIP."
