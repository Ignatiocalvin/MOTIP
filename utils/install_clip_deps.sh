#!/bin/bash
# Install CLIP and additional dependencies for concept comparison

echo "Installing CLIP and dependencies..."

# Install CLIP from OpenAI
pip install git+https://github.com/openai/CLIP.git

# Install additional required packages
pip install scikit-learn matplotlib seaborn Pillow

# Install ftfy for CLIP text processing
pip install ftfy regex

echo "Installation complete!"
echo ""
echo "Usage examples:"
echo "1. Extract CLIP concepts:"
echo "   python clip_concept_extraction.py --sequence VIDEO_NAME --data-root ./data/"
echo ""
echo "2. Compare CLIP vs MOTIP:"
echo "   python clip_vs_motip_comparison.py \\"
echo "       --sequence VIDEO_NAME \\"
echo "       --motip-checkpoint ./outputs/checkpoint.pth \\"
echo "       --motip-config ./configs/r50_deformable_detr_motip_pdestre.yaml"