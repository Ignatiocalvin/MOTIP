#!/usr/bin/env python3
"""
Visualize tracking results from both 2-concept and 7-concept models on a video sequence.
Shows 10 consecutive frames with bounding boxes, track IDs, and concept predictions.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torchvision.transforms as T
import json

from utils.misc import yaml_to_dict
from utils.nested_tensor import nested_tensor_from_tensor_list
from configs.util import load_super_config
from models.motip import build as build_motip
from models.misc import load_checkpoint
from models.runtime_tracker import RuntimeTracker


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

# Colormap for different track IDs (20 distinct colors)
TRACK_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
]


def load_model_and_tracker(checkpoint_path, config_path, sequence_hw):
    """Load MOTIP model and create runtime tracker"""
    print(f"Loading config from {config_path}")
    cfg = yaml_to_dict(config_path)
    
    if "SUPER_CONFIG_PATH" in cfg and cfg["SUPER_CONFIG_PATH"] is not None:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    
    print(f"Building model...")
    model, _ = build_motip(config=cfg)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Get concept bottleneck mode
    concept_bottleneck_mode = cfg.get("MOTIP", {}).get("CONCEPT_BOTTLENECK_MODE", "hard")
    
    # Create tracker
    tracker = RuntimeTracker(
        model=model,
        sequence_hw=sequence_hw,
        det_thresh=0.5,
        concept_bottleneck_mode=concept_bottleneck_mode
    )
    
    return model, tracker, cfg


def load_and_preprocess_image(image_path):
    """Load and preprocess image for inference"""
    img = Image.open(image_path).convert('RGB')
    
    transform = T.Compose([
        T.Resize(800, max_size=1536),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return img, transform(img)


def track_sequence(tracker, image_paths, num_frames=10):
    """Run tracker on sequence and collect results"""
    results = []
    
    for frame_idx, img_path in enumerate(image_paths[:num_frames]):
        print(f"  Processing frame {frame_idx + 1}/{num_frames}...", end='\r')
        
        # Load and preprocess image
        img_pil, img_tensor = load_and_preprocess_image(img_path)
        
        # Convert to nested tensor for tracker
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        nested_img = nested_tensor_from_tensor_list([img_tensor])
        
        # Run tracker update
        tracker.update(nested_img)
        
        # Get tracking results
        track_results = tracker.get_track_results()
        
        results.append({
            'image': img_pil,
            'tracks': track_results.copy() if track_results else []
        })
    
    print(f"\n  Tracked {len(results)} frames")
    return results


def create_tracking_visualization(results_2concept, results_7concept, output_path, 
                                  concept_names_2, concept_names_7):
    """Create visualization grid showing 10 frames side-by-side"""
    
    num_frames = len(results_2concept)
    fig = plt.figure(figsize=(20, num_frames * 3))
    
    for frame_idx in range(num_frames):
        # Get results for this frame
        result_2 = results_2concept[frame_idx]
        result_7 = results_7concept[frame_idx]
        
        img = np.array(result_2['image'])
        h, w = img.shape[:2]
        
        # 2-concept visualization (left)
        ax = plt.subplot(num_frames, 2, frame_idx * 2 + 1)
        ax.imshow(img)
        
        # Get track data
        tracks_2 = result_2['tracks']
        num_tracks_2 = len(tracks_2.get('id', [])) if tracks_2 else 0
        
        ax.set_title(f'Frame {frame_idx + 1} - 2 Concepts | {num_tracks_2} tracks', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Draw tracks for 2-concept model
        if tracks_2 and 'id' in tracks_2 and len(tracks_2['id']) > 0:
            bboxes = tracks_2['bbox'].cpu().numpy()
            track_ids = tracks_2['id'].cpu().numpy()
            concepts = tracks_2.get('concepts', None)
            
            for i in range(len(track_ids)):
                track_id = int(track_ids[i])
                bbox = bboxes[i]  # [x, y, w, h]
                
                # Get color for this track ID
                color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
                color_rgb = tuple(int(color[j:j+2], 16)/255.0 for j in (1, 3, 5))
                
                # Draw bounding box (convert from xywh to xyxy)
                x, y, w_box, h_box = bbox
                rect = patches.Rectangle((x, y), w_box, h_box,
                                        linewidth=2, edgecolor=color_rgb, facecolor='none')
                ax.add_patch(rect)
                
                # Prepare label
                label = f'ID:{track_id}'
                if concepts is not None and len(concepts) > 0:
                    concept_vals = concepts[i].cpu().numpy()
                    concept_strs = []
                    for cidx, concept_name in enumerate(concept_names_2):
                        if cidx < len(concept_vals):
                            concept_label = int(concept_vals[cidx])
                            if concept_label < len(CONCEPT_NAMES[concept_name]):
                                concept_text = CONCEPT_NAMES[concept_name][concept_label][:10]
                            else:
                                concept_text = f"<{concept_label}>"
                            concept_strs.append(f'{concept_name[:3]}:{concept_text}')
                    if concept_strs:
                        label += '\n' + '\n'.join(concept_strs)
                
                # Draw label
                ax.text(x, y-3, label, fontsize=7, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color_rgb, alpha=0.8))
        
        # 7-concept visualization (right)
        ax = plt.subplot(num_frames, 2, frame_idx * 2 + 2)
        ax.imshow(img)
        
        tracks_7 = result_7['tracks']
        num_tracks_7 = len(tracks_7.get('id', [])) if tracks_7 else 0
        
        ax.set_title(f'Frame {frame_idx + 1} - 7 Concepts | {num_tracks_7} tracks', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Draw tracks for 7-concept model
        if tracks_7 and 'id' in tracks_7 and len(tracks_7['id']) > 0:
            bboxes = tracks_7['bbox'].cpu().numpy()
            track_ids = tracks_7['id'].cpu().numpy()
            concepts = tracks_7.get('concepts', None)
            
            for i in range(len(track_ids)):
                track_id = int(track_ids[i])
                bbox = bboxes[i]
                
                # Get color for this track ID
                color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
                color_rgb = tuple(int(color[j:j+2], 16)/255.0 for j in (1, 3, 5))
                
                # Draw bounding box
                x, y, w_box, h_box = bbox
                rect = patches.Rectangle((x, y), w_box, h_box,
                                        linewidth=2, edgecolor=color_rgb, facecolor='none')
                ax.add_patch(rect)
                
                # Prepare label
                label = f'ID:{track_id}'
                if concepts is not None and len(concepts) > 0:
                    concept_vals = concepts[i].cpu().numpy()
                    concept_strs = []
                    for cidx, concept_name in enumerate(concept_names_7[:min(4, len(concept_names_7))]):
                        if cidx < len(concept_vals):
                            concept_label = int(concept_vals[cidx])
                            if concept_label < len(CONCEPT_NAMES[concept_name]):
                                concept_text = CONCEPT_NAMES[concept_name][concept_label][:8]
                            else:
                                concept_text = f"<{concept_label}>"
                            concept_strs.append(f'{concept_name[:3]}:{concept_text}')
                    if len(concept_names_7) > 4:
                        concept_strs.append('...')
                    if concept_strs:
                        label += '\n' + '\n'.join(concept_strs)
                
                # Draw label
                ax.text(x, y-3, label, fontsize=6, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color_rgb, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved tracking visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize MOTIP tracking on video sequence')
    parser.add_argument('--sequence', type=str, 
                       default='11-11-2019-1-7',
                       help='Sequence name (e.g., 11-11-2019-1-7)')
    parser.add_argument('--start-frame', type=int, default=1,
                       help='Starting frame number')
    parser.add_argument('--num-frames', type=int, default=10,
                       help='Number of frames to visualize')
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
    parser.add_argument('--output', type=str, default='tracking_sequence_visualization.png',
                       help='Output visualization path')
    
    args = parser.parse_args()
    
    # Get sequence paths
    sequence_dir = f'data/P-DESTRE/images/{args.sequence}/img1'
    if not os.path.exists(sequence_dir):
        print(f"Error: Sequence directory not found: {sequence_dir}")
        sys.exit(1)
    
    # Get image paths
    image_paths = sorted([
        os.path.join(sequence_dir, f) 
        for f in os.listdir(sequence_dir) 
        if f.endswith('.jpg')
    ])
    
    # Select frames
    start_idx = args.start_frame - 1
    image_paths = image_paths[start_idx:start_idx + args.num_frames]
    
    if len(image_paths) < args.num_frames:
        print(f"Warning: Only {len(image_paths)} frames available")
    
    # Get sequence dimensions
    test_img = Image.open(image_paths[0])
    sequence_hw = (test_img.height, test_img.width)
    
    print("="*60)
    print("MOTIP Tracking Sequence Visualization")
    print("="*60)
    print(f"Sequence: {args.sequence}")
    print(f"Frames: {args.start_frame} to {args.start_frame + len(image_paths) - 1}")
    print(f"Resolution: {sequence_hw[1]}x{sequence_hw[0]}")
    
    # Load 2-concept model and tracker
    print("\n[1/4] Loading 2-concept model and tracker...")
    model_2, tracker_2, cfg_2 = load_model_and_tracker(
        args.checkpoint_2concepts, 
        args.config_2concepts,
        sequence_hw
    )
    concept_names_2 = ['gender', 'upper_body']
    
    # Load 7-concept model and tracker
    print("\n[2/4] Loading 7-concept model and tracker...")
    model_7, tracker_7, cfg_7 = load_model_and_tracker(
        args.checkpoint_7concepts,
        args.config_7concepts,
        sequence_hw
    )
    concept_names_7 = ['gender', 'hairstyle', 'head_accessories', 
                       'upper_body', 'lower_body', 'feet', 'accessories']
    
    # Track with 2-concept model
    print(f"\n[3/4] Running 2-concept tracker on {len(image_paths)} frames...")
    results_2concept = track_sequence(tracker_2, image_paths, len(image_paths))
    
    # Track with 7-concept model
    print(f"\n[4/4] Running 7-concept tracker on {len(image_paths)} frames...")
    results_7concept = track_sequence(tracker_7, image_paths, len(image_paths))
    
    # Create visualization
    print("\nGenerating visualization...")
    create_tracking_visualization(
        results_2concept,
        results_7concept,
        args.output,
        concept_names_2,
        concept_names_7
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Tracking Summary")
    print("="*60)
    
    # Count unique IDs
    ids_2 = set()
    ids_7 = set()
    total_detections_2 = 0
    total_detections_7 = 0
    
    for r in results_2concept:
        if r['tracks'] and 'id' in r['tracks']:
            track_ids = r['tracks']['id'].cpu().numpy()
            ids_2.update(track_ids.tolist())
            total_detections_2 += len(track_ids)
    
    for r in results_7concept:
        if r['tracks'] and 'id' in r['tracks']:
            track_ids = r['tracks']['id'].cpu().numpy()
            ids_7.update(track_ids.tolist())
            total_detections_7 += len(track_ids)
    
    print(f"2-Concept Model:")
    print(f"  - Unique track IDs: {len(ids_2)}")
    print(f"  - Total detections: {total_detections_2}")
    print(f"  - Avg detections per frame: {total_detections_2/len(results_2concept):.1f}")
    
    print(f"\n7-Concept Model:")
    print(f"  - Unique track IDs: {len(ids_7)}")
    print(f"  - Total detections: {total_detections_7}")
    print(f"  - Avg detections per frame: {total_detections_7/len(results_7concept):.1f}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
