"""
Visualization script for MOTIP tracking predictions with concept annotations.
Compares R50 and RF-DETR models on P-DESTRE dataset.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import colorsys
import argparse
from collections import defaultdict

# Concept definitions for 7-concept models
CONCEPT_DEFINITIONS = {
    0: {
        "name": "gender",
        "labels": {0: "M", 1: "F", 2: "?"}  # 0=Male, 1=Female, 2=Unknown
    },
    1: {
        "name": "hair",
        "labels": {0: "Bald", 1: "Short", 2: "Med", 3: "Long", 4: "HTail", 5: "?"}
    },
    2: {
        "name": "head_acc",  
        "labels": {0: "Hat", 1: "Scarf", 2: "Neck", 3: "NoSee", 4: "?"}
    },
    3: {
        "name": "upper",
        "labels": {0: "T-Shirt", 1: "Shirt", 2: "Jumper", 3: "Hoodie", 4: "Jacket", 
                   5: "Vest", 6: "Blouse", 7: "Dress", 8: "Coat", 9: "Top", 
                   10: "Tanktop", 11: "Other", 12: "?"}
    },
    4: {
        "name": "lower",
        "labels": {0: "Jeans", 1: "Trous", 2: "Shorts", 3: "Skirt", 4: "Trackp",
                   5: "Leggin", 6: "Dress", 7: "Jumps", 8: "Other", 9: "?"}
    },
    5: {
        "name": "feet",
        "labels": {0: "Sport", 1: "Casual", 2: "Form", 3: "Boot", 4: "Sandal", 5: "Other", 6: "?"}
    },
    6: {
        "name": "acc",
        "labels": {0: "Bag", 1: "BPack", 2: "HBag", 3: "Cart", 4: "WChair", 5: "Phone", 6: "Other", 7: "?"}
    }
}


# Golden-ratio color generator: maps any track_id to a vivid, maximally-distinct color.
# Uses multiplicative hashing so consecutive IDs land far apart in hue space.
_GOLDEN_RATIO = 0.6180339887


def track_color(track_id):
    """Return a vivid BGR color for a track_id, spread across the full hue wheel."""
    hue = (track_id * _GOLDEN_RATIO) % 1.0
    # Alternate saturation/value in two independent patterns for extra separation
    saturation = 0.90 if (track_id % 3) != 1 else 0.55
    value = 0.95 if (track_id % 5) != 2 else 0.70
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


def generate_colors(n):
    """Kept for API compatibility; returns per-index colors."""
    return [track_color(i) for i in range(n)]


def parse_predictions(pred_file):
    """
    Parse prediction file.
    Format: frame_id, track_id, x, y, w, h, conf, ?, ?, ?, c0, c1, c2, c3, c4, c5, c6
    """
    predictions = defaultdict(list)
    
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 17:  # Need at least 17 columns for 7 concepts
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            
            # 7 concepts start at index 10
            concepts = [int(parts[i]) for i in range(10, 17)]
            
            predictions[frame_id].append({
                'track_id': track_id,
                'bbox': (x, y, w, h),
                'conf': conf,
                'concepts': concepts
            })
    
    return predictions


# Which concepts to show and their display names
_SHOWN_CONCEPTS = [
    (0, "Gender"),
    (1, "Hair"),
    (3, "Upper"),
    (4, "Lower"),
    (6, "Acc"),
]


def format_concept_lines(concepts):
    """Return list of 'Attribute: Value' strings for each shown concept."""
    lines = []
    for idx, display_name in _SHOWN_CONCEPTS:
        c_val = concepts[idx]
        label = CONCEPT_DEFINITIONS[idx]["labels"].get(c_val, "?")
        lines.append(f"{display_name}: {label}")
    return lines


def draw_predictions(frame, predictions, colors, show_concepts=True):
    """Draw predictions on frame."""
    overlay = frame.copy()
    
    for pred in predictions:
        track_id = pred['track_id']
        x, y, w, h = pred['bbox']
        conf = pred['conf']
        concepts = pred['concepts']
        
        # Get color for this track — use hash-based assignment for max spread
        color = track_color(track_id)
        
        # Draw bounding box
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        
        # Draw track ID and confidence
        label = f"ID:{track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Draw concept lines below bbox
        if show_concepts:
            concept_lines = format_concept_lines(concepts)
            font_scale_concept = 1.1
            concept_thickness = 2
            pad = 8
            line_spacing = 6
            # Measure all lines to find box width
            sizes = [cv2.getTextSize(ln, font, font_scale_concept, concept_thickness) for ln in concept_lines]
            box_w = max(s[0][0] for s in sizes) + pad * 2
            line_h = sizes[0][0][1]  # all same font size
            box_h = len(concept_lines) * (line_h + line_spacing) + pad
            # Colored semi-transparent background
            bg_y1, bg_y2 = y2, y2 + box_h
            cv2.rectangle(overlay, (x1, bg_y1), (x1 + box_w, bg_y2), color, -1)
            cv2.rectangle(overlay, (x1, bg_y1), (x1 + box_w, bg_y2), (0, 0, 0), 1)
            for li, ln in enumerate(concept_lines):
                text_y = bg_y1 + pad + line_h + li * (line_h + line_spacing)
                cv2.putText(overlay, ln, (x1 + pad, text_y), font,
                            font_scale_concept, (255, 255, 255), concept_thickness)
    
    # Blend overlay
    alpha = 0.85
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame


def visualize_sequence(image_dir, pred_file, output_dir, model_name, 
                       start_frame=1, num_frames=30, show_concepts=True):
    """Generate visualizations for a sequence of frames."""
    
    # Parse predictions
    predictions = parse_predictions(pred_file)
    
    # Generate colors for tracks
    max_track_id = 0
    for frame_preds in predictions.values():
        for pred in frame_preds:
            max_track_id = max(max_track_id, pred['track_id'])
    colors = generate_colors(max_track_id + 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process frames
    processed = 0
    for frame_idx in range(start_frame, start_frame + num_frames):
        img_path = os.path.join(image_dir, 'img1', f'{frame_idx:06d}.jpg')
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
        
        # Get predictions for this frame
        frame_preds = predictions.get(frame_idx, [])
        
        # Draw predictions
        frame = draw_predictions(frame, frame_preds, colors, show_concepts)
        
        # Add frame info
        info_text = f"{model_name} | Frame {frame_idx} | {len(frame_preds)} detections"
        cv2.putText(frame, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add legend
        legend_y = 80
        legend_text = "Concepts shown: Gender, Hair, Upper, Lower, Acc"
        cv2.putText(frame, legend_text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, legend_text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Save frame
        output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.jpg')
        cv2.imwrite(output_path, frame)
        processed += 1
        print(f"Saved: {output_path}")
    
    print(f"\nProcessed {processed} frames for {model_name}")
    return processed


def create_frame_grid(frame_dir, frame_ids, output_path, model_name,
                      cell_w=960, cell_h=540, gap=8, label_h=48):
    """
    Compose a 2x2 grid of specific frames.
    Each cell is resized to (cell_w x cell_h) with a frame-number label strip.
    """
    assert len(frame_ids) == 4, "Need exactly 4 frame IDs for a 2x2 grid"
    cells = []
    for fid in frame_ids:
        img_path = os.path.join(frame_dir, f'frame_{fid:06d}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        img = cv2.resize(img, (cell_w, cell_h))
        cells.append(img)

    row0 = np.hstack([cells[0], np.zeros((cell_h, gap, 3), np.uint8), cells[1]])
    row1 = np.hstack([cells[2], np.zeros((cell_h, gap, 3), np.uint8), cells[3]])
    grid_gap = np.zeros((gap, row0.shape[1], 3), np.uint8)
    grid = np.vstack([row0, grid_gap, row1])

    cv2.imwrite(output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved grid: {output_path}")


def create_comparison_video(r50_dir, rfdetr_dir, output_path, fps=5):
    """Create a side-by-side comparison video."""
    
    # Get all frames
    r50_frames = sorted(Path(r50_dir).glob('frame_*.jpg'))
    rfdetr_frames = sorted(Path(rfdetr_dir).glob('frame_*.jpg'))
    
    if not r50_frames or not rfdetr_frames:
        print("No frames found for video creation")
        return
    
    # Read first frame to get dimensions
    sample = cv2.imread(str(r50_frames[0]))
    h, w = sample.shape[:2]
    
    # Create video writer (side by side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
    
    for r50_f, rfdetr_f in zip(r50_frames, rfdetr_frames):
        r50_img = cv2.imread(str(r50_f))
        rfdetr_img = cv2.imread(str(rfdetr_f))
        
        # Resize if needed
        r50_img = cv2.resize(r50_img, (w, h))
        rfdetr_img = cv2.resize(rfdetr_img, (w, h))
        
        # Concatenate side by side
        combined = np.hstack([r50_img, rfdetr_img])
        out.write(combined)
    
    out.release()
    print(f"Created comparison video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize MOTIP tracking predictions')
    parser.add_argument('--sequence', type=str, default='14-11-2019-2-2',
                        help='Sequence name to visualize')
    parser.add_argument('--start_frame', type=int, default=1480,
                        help='Starting frame number')
    parser.add_argument('--num_frames', type=int, default=30,
                        help='Number of frames to visualize')
    parser.add_argument('--create_video', action='store_true',
                        help='Create comparison video')
    args = parser.parse_args()
    
    # Base paths
    base_dir = Path('/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP')
    data_dir = base_dir / 'data' / 'P-DESTRE'
    
    # Model prediction paths
    r50_pred_dir = base_dir / 'outputs' / 'r50_motip_pdestre_7concepts_learnable_v2_fold_0' / 'eval' / 'PDESTRE_Test_0' / 'checkpoint_3' / 'tracker'
    rfdetr_pred_dir = base_dir / 'outputs' / 'rfdetr_large_motip_pdestre_7concepts_learnable_fold0' / 'eval' / 'PDESTRE_Test_0' / 'checkpoint_0' / 'tracker'
    
    # Image directory
    image_dir = data_dir / 'images' / args.sequence
    
    # Output directories
    output_base = base_dir / 'visualizations' / f'{args.sequence}_frames_{args.start_frame}-{args.start_frame + args.num_frames - 1}'
    r50_output = output_base / 'r50_7concepts_lw_v2'
    rfdetr_output = output_base / 'rfdetr_7concepts_lw'
    
    # Prediction files
    r50_pred_file = r50_pred_dir / f'{args.sequence}.txt'
    rfdetr_pred_file = rfdetr_pred_dir / f'{args.sequence}.txt'
    
    print("="*80)
    print("MOTIP Tracking Visualization")
    print("="*80)
    print(f"Sequence: {args.sequence}")
    print(f"Frames: {args.start_frame} to {args.start_frame + args.num_frames - 1}")
    print(f"R50 predictions: {r50_pred_file}")
    print(f"RF-DETR predictions: {rfdetr_pred_file}")
    print("="*80)
    
    # Check files exist
    if not r50_pred_file.exists():
        print(f"ERROR: R50 prediction file not found: {r50_pred_file}")
        return
    if not rfdetr_pred_file.exists():
        print(f"ERROR: RF-DETR prediction file not found: {rfdetr_pred_file}")
        return
    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}")
        return
    
    # Generate visualizations
    print("\n[1/2] Generating R50 visualizations...")
    visualize_sequence(
        str(image_dir), str(r50_pred_file), str(r50_output),
        "R50 7-Concept Learnable v2",
        args.start_frame, args.num_frames
    )
    
    print("\n[2/2] Generating RF-DETR visualizations...")
    visualize_sequence(
        str(image_dir), str(rfdetr_pred_file), str(rfdetr_output),
        "RF-DETR 7-Concept Learnable",
        args.start_frame, args.num_frames
    )
    
    # 2x2 grids for specific frames
    grid_frames = [1480, 1481, 1485, 1490]
    print(f"\n[3/4] Creating 2x2 grid for R50 (frames {grid_frames})...")
    create_frame_grid(
        str(r50_output), grid_frames,
        str(output_base / 'grid_r50.jpg'),
        "R50 7-Concept Learnable v2"
    )
    print(f"\n[4/4] Creating 2x2 grid for RF-DETR (frames {grid_frames})...")
    create_frame_grid(
        str(rfdetr_output), grid_frames,
        str(output_base / 'grid_rfdetr.jpg'),
        "RF-DETR 7-Concept Learnable"
    )

    # Create comparison video if requested
    if args.create_video:
        print("\n[5/5] Creating comparison video...")
        video_path = output_base / 'comparison.mp4'
        create_comparison_video(str(r50_output), str(rfdetr_output), str(video_path))

    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"R50 output:    {r50_output}")
    print(f"RF-DETR output:{rfdetr_output}")
    print(f"R50 grid:      {output_base / 'grid_r50.jpg'}")
    print(f"RF-DETR grid:  {output_base / 'grid_rfdetr.jpg'}")
    print("="*80)


if __name__ == '__main__':
    main()
