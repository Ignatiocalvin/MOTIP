#!/usr/bin/env python3
"""
Visualize Tracker Outputs on Video Frames
Draws bounding boxes and IDs from tracker predictions on actual frames
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

def parse_mot_file(mot_file, parse_concepts=False):
    """Parse MOT format tracker output file.
    
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, ...
    For tracker: may have gender (col 10) and upper_body (col 11)
    For GT: has many concept columns
    """
    tracks = defaultdict(list)
    
    with open(mot_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
                
            frame_id = int(parts[0])
            track_id = int(parts[1])
            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            conf = float(parts[6])
            
            concepts = {}
            if parse_concepts and len(parts) >= 12:
                # Tracker format: gender at col 10, upper_body at col 11
                concepts['gender'] = int(parts[10])
                concepts['upper_body'] = int(parts[11])
            elif parse_concepts and len(parts) >= 11:
                # GT format: gender at col 10 (different encoding)
                concepts['gender'] = int(parts[10])
                # Upper body might be encoded differently in GT, skip for now
            
            tracks[frame_id].append({
                'id': track_id,
                'bbox': bbox,
                'conf': conf,
                'concepts': concepts
            })
    
    return tracks

def get_first_frame_from_gt(gt_file):
    """Get the first frame number from ground truth annotations."""
    if not gt_file.exists():
        return None
    
    with open(gt_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            return int(first_line.split(',')[0])
    return None

def get_color_for_id(track_id, seed=42):
    """Generate consistent color for each track ID."""
    random.seed(track_id + seed)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def draw_tracks_on_frame(frame, tracks, show_conf=True, is_gt=False, show_concepts=True):
    """Draw bounding boxes and IDs on frame."""
    for track in tracks:
        bbox = track['bbox']
        track_id = track['id']
        conf = track.get('conf', 1.0)
        concepts = track.get('concepts', {})
        
        # Get bbox coordinates
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        if is_gt:
            # Ground truth: green with thicker lines
            color = (0, 255, 0)
            thickness = 3
            label_prefix = "GT"
        else:
            # Predictions: colored by ID with normal lines
            color = get_color_for_id(track_id)
            thickness = 2
            label_prefix = "PRED"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        if is_gt:
            label = f"{label_prefix}:{track_id}"
        else:
            if show_conf:
                label = f"{label_prefix}:{track_id} ({conf:.2f})"
            else:
                label = f"{label_prefix}:{track_id}"
        
        # Add concepts to label
        if show_concepts and concepts:
            gender_map = {0: 'F', 1: 'M'}
            upper_map = {0: 'Dark', 1: 'Light'}
            concept_str = []
            if 'gender' in concepts:
                concept_str.append(gender_map.get(concepts['gender'], '?'))
            if 'upper_body' in concepts:
                concept_str.append(upper_map.get(concepts['upper_body'], '?'))
            if concept_str:
                label += f" [{'/'.join(concept_str)}]"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 5 if not is_gt else y2 + label_size[1] + 10
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0], label_y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, label_y - 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if is_gt else (255, 255, 255), 1)
    
    return frame

def visualize_sequence(tracker_file, images_dir, gt_dir, output_dir, 
                       max_frames=None, skip_frames=1, show_conf=True, show_gt=True, show_concepts=True):
    """Visualize tracking results for one sequence."""
    
    seq_name = tracker_file.stem
    
    # Parse tracker output with concepts
    print(f"Processing: {seq_name}")
    tracks = parse_mot_file(tracker_file, parse_concepts=show_concepts)
    
    if not tracks:
        print(f"  ⚠ No tracks found")
        return 0
    
    # Get and parse ground truth with concepts
    gt_file = gt_dir / f"{seq_name}.txt"
    gt_tracks = {}
    if show_gt and gt_file.exists():
        gt_tracks = parse_mot_file(gt_file, parse_concepts=show_concepts)
    
    first_frame = get_first_frame_from_gt(gt_file)
    
    if first_frame is None:
        print(f"  ⚠ Could not determine first frame from GT")
        return 0
    
    # Get frame numbers from tracker
    frame_ids = sorted(tracks.keys())
    if max_frames:
        frame_ids = frame_ids[:max_frames]
    
    print(f"  Frames: {len(frame_ids)}, First GT frame: {first_frame}")
    
    # Create output directory
    seq_output_dir = output_dir / seq_name
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    processed = 0
    for i, frame_id in enumerate(frame_ids):
        if i % skip_frames != 0:
            continue
        
        # Map frame_id to image filename
        # image_number = frame_id - first_frame + 1
        img_number = frame_id - first_frame + 1
        
        if img_number < 1:
            continue
        
        # Try to find the image
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = images_dir / f"{img_number:06d}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            if processed == 0:
                print(f"  ⚠ Image not found for frame {frame_id} (img {img_number:06d})")
            continue
        
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Draw ground truth first (underneath)
        if show_gt and frame_id in gt_tracks:
            frame = draw_tracks_on_frame(frame, gt_tracks[frame_id], show_conf=False, is_gt=True, show_concepts=show_concepts)
        
        # Draw predictions on top
        frame = draw_tracks_on_frame(frame, tracks[frame_id], show_conf, is_gt=False, show_concepts=show_concepts)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Pred: {len(tracks[frame_id])} | GT: {len(gt_tracks.get(frame_id, []))}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save annotated frame
        output_path = seq_output_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        processed += 1
    
    print(f"  ✓ Saved {processed} frames")
    return processed

def main():
    parser = argparse.ArgumentParser(description='Visualize tracking results on video frames')
    parser.add_argument('--tracker-dir', type=str, required=True,
                       help='Directory containing tracker output files')
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing sequence images')
    parser.add_argument('--gt-dir', type=str, required=True,
                       help='Directory containing ground truth annotations')
    parser.add_argument('--output-dir', type=str, default='../outputs/tracking_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Specific sequence to visualize (default: all)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process per sequence')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (default: 1 = all frames)')
    parser.add_argument('--no-conf', action='store_true',
                       help='Hide confidence scores in labels')
    parser.add_argument('--no-gt', action='store_true',
                       help='Hide ground truth boxes')
    parser.add_argument('--no-concepts', action='store_true',
                       help='Hide concept predictions')
    
    args = parser.parse_args()
    
    tracker_dir = Path(args.tracker_dir)
    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not tracker_dir.exists():
        print(f"❌ Tracker directory not found: {tracker_dir}")
        return
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    if not gt_dir.exists():
        print(f"❌ Ground truth directory not found: {gt_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Tracking Visualization on Video Frames")
    print("=" * 70)
    print(f"Tracker: {tracker_dir}")
    print(f"Images: {images_dir}")
    print(f"GT: {gt_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Get tracker files
    if args.sequence:
        tracker_files = [tracker_dir / f"{args.sequence}.txt"]
        if not tracker_files[0].exists():
            print(f"❌ Sequence not found: {args.sequence}")
            return
    else:
        tracker_files = sorted(tracker_dir.glob("*.txt"))
    
    if not tracker_files:
        print(f"❌ No tracker files found in {tracker_dir}")
        return
    
    print(f"Found {len(tracker_files)} sequence(s)")
    print()
    
    # Process each sequence
    total_frames = 0
    for tracker_file in tracker_files:
        seq_name = tracker_file.stem
        
        # Find sequence images (check img1 subdirectory)
        seq_images_dir = images_dir / seq_name / "img1"
        if not seq_images_dir.exists():
            seq_images_dir = images_dir / seq_name
            if not seq_images_dir.exists():
                print(f"⚠ Images not found for: {seq_name}")
                continue
        
        frames = visualize_sequence(
            tracker_file, seq_images_dir, gt_dir, output_dir,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames,
            show_conf=not args.no_conf,
            show_gt=not args.no_gt,
            show_concepts=not args.no_concepts
        )
        
        total_frames += frames
    
    print()
    print("=" * 70)
    print(f"✅ Complete! Processed {total_frames} frames")
    print(f"   Output: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
