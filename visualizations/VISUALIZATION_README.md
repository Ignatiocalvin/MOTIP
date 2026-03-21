# MOTIP Concept Visualization

This document describes the visualization tool for comparing concept predictions between MOTIP models with different numbers of concepts.

## Overview

The `visualize_concepts.py` script loads two trained MOTIP models (2-concept and 7-concept) and visualizes their predictions side-by-side on sample images from the P-DESTRE dataset.

## Generated Visualizations

Three visualizations have been generated:

1. **visualization_comparison.png** - Sample image with multiple pedestrians
2. **visualization_image2.png** - Additional sample showing different scenarios
3. **visualization_image3.png** - Third sample for comparison

Each visualization shows:
- **Left panel**: 2-concept model (Gender, Upper Body)
  - Predicts 2 concepts: gender and upper_body
  - Uses CONCEPT_DIM=64 for ID decoder
  
- **Right panel**: 7-concept model (All Attributes)
  - Predicts all 7 concepts: gender, hairstyle, head_accessories, upper_body, lower_body, feet, accessories
  - Uses CONCEPT_DIM=224 for ID decoder

## Usage

### Basic Usage

```bash
python3 visualize_concepts.py --image <path_to_image> --output <output_path>
```

### Arguments

- `--image`: Path to input image (required)
- `--output`: Path to save visualization (default: `concept_visualization.png`)
- `--confidence`: Confidence threshold for detections (default: 0.5)
- `--checkpoint-2concepts`: Path to 2-concept model checkpoint
- `--config-2concepts`: Path to 2-concept model config
- `--checkpoint-7concepts`: Path to 7-concept model checkpoint
- `--config-7concepts`: Path to 7-concept model config

### Examples

```bash
# Use default checkpoints
python3 visualize_concepts.py \
    --image data/P-DESTRE/images/11-11-2019-1-3/img1/001554.jpg \
    --output visualization.png

# Lower confidence threshold to see more detections
python3 visualize_concepts.py \
    --image data/P-DESTRE/images/20-10-2019-1-1/img1/002296.jpg \
    --output visualization.png \
    --confidence 0.4

# Use specific checkpoints
python3 visualize_concepts.py \
    --image my_image.jpg \
    --checkpoint-2concepts outputs/r50_motip_pdestre_concepts_for_id_fold_0/checkpoint_2.pth \
    --config-2concepts outputs/r50_motip_pdestre_concepts_for_id_fold_0/train/config.yaml \
    --checkpoint-7concepts outputs/r50_motip_pdestre_7concepts_fixed_fold_0/checkpoint_2.pth \
    --config-7concepts configs/r50_deformable_detr_motip_pdestre_7concepts_fixed_fold_0.yaml \
    --output comparison.png
```

## P-DESTRE Concept Classes

The visualization uses the following concept categories:

1. **Gender**: Male, Female, Unknown
2. **Hairstyle**: Short, Long, Bald, Unknown
3. **Head Accessories**: None, Hat, Helmet, Hood, Scarf, Unknown
4. **Upper Body**: Short-sleeved shirt, Long-sleeved shirt, Sleeveless shirt, Coat, Jacket, Hoodie, Vest, T-shirt, Polo, Sweater, Dress, Suit, Unknown
5. **Lower Body**: Long pants, Short pants, Skirt, Dress, Shorts, Jeans, Leggings, Unknown
6. **Feet**: Shoes, Boots, Sandals, Barefoot, Unknown
7. **Accessories**: None, Backpack, Bag, Umbrella, Unknown

## Model Comparison

### 2-Concept Model
- **Concepts**: Gender, Upper Body
- **CONCEPT_DIM**: 64
- **Bottleneck Mode**: Hard (argmax)
- **Tracking Performance**: 
  - MOTA: 66.00%
  - IDF1: 68.96%
  - Recall: 68.29%

### 7-Concept Model (Fixed)
- **Concepts**: All 7 attributes
- **CONCEPT_DIM**: 224
- **Bottleneck Mode**: Hard (argmax)
- **Tracking Performance**:
  - MOTA: 27.27%
  - IDF1: 40.66%
  - Recall: 27.59%

## Key Findings

1. **Detection Quality**: The 7-concept model shows lower recall (27.59% vs 68.29%), meaning it misses more pedestrians
2. **Concept Predictions**: Both models successfully predict their respective concepts
3. **Practical Use**: The 2-concept model is more suitable for production due to better detection and tracking performance

## Technical Details

### Image Preprocessing
Images are preprocessed using:
- Resize: max_shorter=800, max_longer=1536
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Visualization Format
- Bounding boxes with color-coded labels
- Confidence scores for each detection
- Concept predictions displayed below each bounding box
- Side-by-side comparison for easy analysis

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- PIL (Pillow)
- MOTIP codebase with trained checkpoints

## Notes

- The script automatically uses CUDA if available
- Models are loaded in evaluation mode
- Concept labels are bounded-checked to handle any out-of-range predictions
- The visualization color palette cycles through 8 distinct colors for better distinction
