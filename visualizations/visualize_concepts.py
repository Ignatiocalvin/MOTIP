#!/usr/bin/env python3
"""
Visualize predicted bounding boxes and concepts from MOTIP models
Compares 2-concept model vs 7-concept model predictions on sample images
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from utils.misc import yaml_to_dict
from utils.nested_tensor import nested_tensor_from_tensor_list
from configs.util import load_super_config
from models.motip import build as build_motip
from models.misc import load_checkpoint
import torchvision.transforms as T


# P-DESTRE concept names
CONCEPT_NAMES = {
    'gender': ['Male', 'Female', 'Unknown'],
    'hairstyle': ['Short', 'Long', 'Bald', 'Unknown'],
    'head_accessories': ['None', 'Hat', 'Helmet', 'Hood', 'Scarf', 'Unknown'],
    'upper_body': [
        'Short-sleeved shirt', 'Long-sleeved shirt', 'Sleeveless shirt',
        'Coat', 'Jacket', 'Hoodie', 'Vest', 'T-shirt', 'Polo', 
        'Sweater', 'Dress', 'Suit', 'Unknown'
    ],
    'lower_body': [
        'Long pants', 'Short pants', 'Skirt', 'Dress', 'Shorts', 
        'Jeans', 'Leggings', 'Unknown'
    ],
    'feet': ['Shoes', 'Boots', 'Sandals', 'Barefoot', 'Unknown'],
    'accessories': ['None', 'Backpack', 'Bag', 'Umbrella', 'Unknown']
}

# Colormap for bounding boxes
COLORS = [
    (0, 114, 189), (217, 83, 25), (237, 177, 32), (126, 47, 142),
    (119, 172, 48), (77, 190, 238), (162, 20, 47), (76, 76, 76)
]


def load_model(checkpoint_path, config_path):
    """Load MOTIP model from checkpoint and config"""
    print(f"Loading config from {config_path}")
    cfg = yaml_to_dict(config_path)
    
    # Load super config if specified
    if "SUPER_CONFIG_PATH" in cfg and cfg["SUPER_CONFIG_PATH"] is not None:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    
    print(f"Building model...")
    model, _ = build_motip(config=cfg)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, cfg


def prepare_image(image_path):
    """Load and prepare image for inference"""
    img = Image.open(image_path).convert('RGB')
    
    # Apply transforms - similar to data/seq_dataset.py
    transform = T.Compose([
        T.Resize(800, max_size=1536),  # max_shorter=800, max_longer=1536
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transformed = transform(img)
    
    return img, transformed


@torch.no_grad()
def run_inference(model, image_tensor):
    """Run model inference on image"""
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    # Create nested tensor (batch size = 1)
    samples = nested_tensor_from_tensor_list([image_tensor])
    
    # Forward pass - MOTIP requires "part" parameter
    outputs = model(part="detr", frames=samples)
    
    return outputs


def visualize_predictions(
    image_path, 
    outputs_2concepts, 
    outputs_7concepts,
    cfg_2concepts,
    cfg_7concepts,
    output_path,
    confidence_threshold=0.5
):
    """Visualize predictions from both models side by side"""
    
    # Load original image
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w = img.shape[:2]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Process 2-concept model
    pred_logits_2 = outputs_2concepts['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes_2 = outputs_2concepts['pred_boxes'][0]    # [num_queries, 4]
    pred_concepts_2 = outputs_2concepts.get('pred_concepts', [])
    
    # Get detection scores and filter by confidence
    scores_2 = pred_logits_2.sigmoid().max(dim=-1)[0]
    keep_2 = scores_2 > confidence_threshold
    
    boxes_2 = pred_boxes_2[keep_2].cpu().numpy()
    scores_2 = scores_2[keep_2].cpu().numpy()
    
    # Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixel
    boxes_2_pixel = boxes_2.copy()
    boxes_2_pixel[:, 0] = (boxes_2[:, 0] - boxes_2[:, 2] / 2) * w
    boxes_2_pixel[:, 1] = (boxes_2[:, 1] - boxes_2[:, 3] / 2) * h
    boxes_2_pixel[:, 2] = (boxes_2[:, 0] + boxes_2[:, 2] / 2) * w
    boxes_2_pixel[:, 3] = (boxes_2[:, 1] + boxes_2[:, 3] / 2) * h
    
    # Extract concept predictions for 2-concept model
    concepts_2 = []
    if pred_concepts_2:
        for concept_idx, concept_pred in enumerate(pred_concepts_2):
            concept_labels = concept_pred[0][keep_2].argmax(dim=-1).cpu().numpy()
            concepts_2.append(concept_labels)
    
    # Process 7-concept model
    pred_logits_7 = outputs_7concepts['pred_logits'][0]
    pred_boxes_7 = outputs_7concepts['pred_boxes'][0]
    pred_concepts_7 = outputs_7concepts.get('pred_concepts', [])
    
    scores_7 = pred_logits_7.sigmoid().max(dim=-1)[0]
    keep_7 = scores_7 > confidence_threshold
    
    boxes_7 = pred_boxes_7[keep_7].cpu().numpy()
    scores_7 = scores_7[keep_7].cpu().numpy()
    
    boxes_7_pixel = boxes_7.copy()
    boxes_7_pixel[:, 0] = (boxes_7[:, 0] - boxes_7[:, 2] / 2) * w
    boxes_7_pixel[:, 1] = (boxes_7[:, 1] - boxes_7[:, 3] / 2) * h
    boxes_7_pixel[:, 2] = (boxes_7[:, 0] + boxes_7[:, 2] / 2) * w
    boxes_7_pixel[:, 3] = (boxes_7[:, 1] + boxes_7[:, 3] / 2) * h
    
    # Extract concept predictions for 7-concept model
    concepts_7 = []
    if pred_concepts_7:
        for concept_idx, concept_pred in enumerate(pred_concepts_7):
            concept_labels = concept_pred[0][keep_7].argmax(dim=-1).cpu().numpy()
            concepts_7.append(concept_labels)
    
    # Plot 2-concept model results
    axes[0].imshow(img)
    axes[0].set_title(f'MOTIP - 2 Concepts (Gender, Upper Body)\nDetections: {len(boxes_2_pixel)}', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    for i, (box, score) in enumerate(zip(boxes_2_pixel, scores_2)):
        x1, y1, x2, y2 = box
        color = tuple(c/255.0 for c in COLORS[i % len(COLORS)])
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                linewidth=2, edgecolor=color, facecolor='none')
        axes[0].add_patch(rect)
        
        # Prepare label with concepts
        label = f'Person {score:.2f}'
        if concepts_2:
            concept_strs = []
            concept_names_list = ['gender', 'upper_body']
            for cidx, concept_name in enumerate(concept_names_list[:len(concepts_2)]):
                concept_label = concepts_2[cidx][i]
                # Boundary check
                if concept_label < len(CONCEPT_NAMES[concept_name]):
                    concept_text = CONCEPT_NAMES[concept_name][concept_label]
                else:
                    concept_text = f"<{concept_label}>"
                concept_strs.append(f'{concept_name}: {concept_text}')
            label += '\n' + '\n'.join(concept_strs)
        
        # Draw label background
        axes[0].text(x1, y1-5, label, fontsize=9, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    # Plot 7-concept model results
    axes[1].imshow(img)
    axes[1].set_title(f'MOTIP - 7 Concepts (All Attributes)\nDetections: {len(boxes_7_pixel)}', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    for i, (box, score) in enumerate(zip(boxes_7_pixel, scores_7)):
        x1, y1, x2, y2 = box
        color = tuple(c/255.0 for c in COLORS[i % len(COLORS)])
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                linewidth=2, edgecolor=color, facecolor='none')
        axes[1].add_patch(rect)
        
        # Prepare label with concepts
        label = f'Person {score:.2f}'
        if concepts_7:
            concept_strs = []
            concept_names_list = ['gender', 'hairstyle', 'head_accessories', 
                                 'upper_body', 'lower_body', 'feet', 'accessories']
            for cidx, concept_name in enumerate(concept_names_list[:len(concepts_7)]):
                concept_label = concepts_7[cidx][i]
                # Boundary check
                if concept_label < len(CONCEPT_NAMES[concept_name]):
                    concept_text = CONCEPT_NAMES[concept_name][concept_label]
                else:
                    concept_text = f"<{concept_label}>"
                concept_strs.append(f'{concept_name[:4]}: {concept_text}')
            label += '\n' + '\n'.join(concept_strs)
        
        # Draw label background
        axes[1].text(x1, y1-5, label, fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize MOTIP concept predictions')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint-2concepts', type=str, 
                       default='outputs/r50_motip_pdestre_concepts_for_id_fold_0/checkpoint_2.pth',
                       help='Path to 2-concept model checkpoint')
    parser.add_argument('--config-2concepts', type=str,
                       default='outputs/r50_motip_pdestre_concepts_for_id_fold_0/train/config.yaml',
                       help='Path to 2-concept model config')
    parser.add_argument('--checkpoint-7concepts', type=str,
                       default='outputs/r50_motip_pdestre_7concepts_fixed_fold_0/checkpoint_2.pth',
                       help='Path to 7-concept model checkpoint')
    parser.add_argument('--config-7concepts', type=str,
                       default='configs/r50_deformable_detr_motip_pdestre_7concepts_fixed_fold_0.yaml',
                       help='Path to 7-concept model config')
    parser.add_argument('--output', type=str, default='concept_visualization.png',
                       help='Output visualization path')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Verify image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    print("="*60)
    print("MOTIP Concept Visualization")
    print("="*60)
    
    # Load models
    print("\n[1/4] Loading 2-concept model...")
    model_2concepts, cfg_2concepts = load_model(args.checkpoint_2concepts, args.config_2concepts)
    
    print("\n[2/4] Loading 7-concept model...")
    model_7concepts, cfg_7concepts = load_model(args.checkpoint_7concepts, args.config_7concepts)
    
    # Prepare image
    print(f"\n[3/4] Processing image: {args.image}")
    img_pil, img_tensor = prepare_image(args.image)
    
    # Run inference
    print("\n[4/4] Running inference...")
    print("  - 2-concept model...")
    outputs_2concepts = run_inference(model_2concepts, img_tensor)
    
    print("  - 7-concept model...")
    outputs_7concepts = run_inference(model_7concepts, img_tensor)
    
    # Visualize results
    print(f"\nGenerating visualization...")
    visualize_predictions(
        args.image,
        outputs_2concepts,
        outputs_7concepts,
        cfg_2concepts,
        cfg_7concepts,
        args.output,
        args.confidence
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
