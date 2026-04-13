#!/usr/bin/env python3
"""
Precompute SAM Segmentation Masks for MOTIP Concept Bottleneck

This script generates precomputed SAM masks for each object in each frame.
The masks are used by the SAM concept bottleneck model for spatial feature pooling.

Usage:
    python precompute_sam_masks.py --dataset DanceTrack --split train
    python precompute_sam_masks.py --dataset P-DESTRE --split Train_0

Output structure:
    {SAVE_ROOT}/{sequence_name}/{frame_id:08d}/{track_id}.png

Each mask is a binary PNG (255=foreground, 0=background) at original image resolution.
"""

import os
import cv2
import torch
import argparse
import numpy as np
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor


# =========================================================
# DEFAULTS
# =========================================================
DEFAULT_SAM_CHECKPOINT = "./pretrains/sam_vit_b_01ec64.pth"
DEFAULT_MODEL_TYPE = "vit_b"
DEFAULT_SAVE_ROOT = "./precomputed_sam_masks"
DEFAULT_DEBUG_OVERLAYS = False

# Mask selection strategy
BEST_MASK_MODE = "score"  # "score" or "fill"

# Restrict final mask to its bbox
CLIP_MASK_TO_BOX = True

# Skip masks that already exist
SKIP_EXISTING = True


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute SAM masks for MOTIP")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name: DanceTrack, P-DESTRE, etc.")
    parser.add_argument("--split", type=str, required=True,
                        help="Dataset split: train, val, Train_0, etc.")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory containing datasets")
    parser.add_argument("--sam-checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT,
                        help="Path to SAM checkpoint")
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE,
                        help="SAM model type: vit_b, vit_l, vit_h")
    parser.add_argument("--save-root", type=str, default=DEFAULT_SAVE_ROOT,
                        help="Output directory for masks")
    parser.add_argument("--debug-overlays", action="store_true",
                        help="Save debug overlay visualizations")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Limit number of sequences (for testing)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit frames per sequence (for testing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run SAM on")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_sequences(root_dir):
    """List all sequence directories."""
    if not os.path.isdir(root_dir):
        return []
    seqs = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    seqs = sorted(seqs)
    return seqs


def load_gt_by_frame(gt_path):
    """
    Load ground truth annotations grouped by frame.
    
    Returns:
        gt_by_frame[frame_id] = [
            {"track_id": int, "bbox_xywh": [x, y, w, h]},
            ...
        ]
    """
    gt_by_frame = defaultdict(list)

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            gt_by_frame[frame_id].append({
                "track_id": track_id,
                "bbox_xywh": [x, y, w, h],
            })

    return gt_by_frame


def choose_best_mask(masks, scores, box_area, mode="score"):
    """
    Choose best mask from SAM's multi-mask output.
    
    Args:
        masks: [K, H, W] predicted masks
        scores: [K] confidence scores
        box_area: area of the bounding box
        mode: "score" (highest confidence) or "fill" (best fill ratio)
    """
    if mode == "score":
        best_idx = int(np.argmax(scores))
    elif mode == "fill":
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        fill_ratio = areas / (box_area + 1e-6)
        best_idx = int(np.argmax(fill_ratio))
    else:
        raise ValueError(f"Unknown BEST_MASK_MODE: {mode}")

    return best_idx, masks[best_idx]


def get_color(i):
    """Get a color for visualization."""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255),
    ]
    return colors[i % len(colors)]


def get_dataset_structure(dataset, split, data_root):
    """
    Get paths for different dataset structures.
    
    Returns:
        list of (sequence_dir, img_folder_name, gt_path) tuples
    """
    if dataset == "DanceTrack":
        split_dir = os.path.join(data_root, "DanceTrack", split)
        sequences = []
        for seq_name in list_sequences(split_dir):
            seq_dir = os.path.join(split_dir, seq_name)
            img_folder = "img1"
            gt_path = os.path.join(seq_dir, "gt", "gt.txt")
            sequences.append((seq_dir, img_folder, gt_path, seq_name))
        return sequences
    
    elif dataset == "P-DESTRE":
        # P-DESTRE structure: data_root/P-DESTRE/{split}/
        split_dir = os.path.join(data_root, "P-DESTRE", split)
        sequences = []
        for seq_name in list_sequences(split_dir):
            seq_dir = os.path.join(split_dir, seq_name)
            img_folder = "img1"
            gt_path = os.path.join(seq_dir, "gt", "gt.txt")
            sequences.append((seq_dir, img_folder, gt_path, seq_name))
        return sequences
    
    else:
        # Generic MOT structure
        split_dir = os.path.join(data_root, dataset, split)
        sequences = []
        for seq_name in list_sequences(split_dir):
            seq_dir = os.path.join(split_dir, seq_name)
            img_folder = "img1"
            gt_path = os.path.join(seq_dir, "gt", "gt.txt")
            sequences.append((seq_dir, img_folder, gt_path, seq_name))
        return sequences


def process_sequence(seq_dir, img_folder, gt_path, seq_name, predictor, save_root, 
                     debug_overlays=False, debug_root=None, max_frames=None):
    """Process a single sequence and generate SAM masks."""
    img_dir = os.path.join(seq_dir, img_folder)

    if not os.path.isdir(img_dir):
        print(f"[SKIP] Missing images folder: {img_dir}")
        return

    if not os.path.isfile(gt_path):
        print(f"[SKIP] Missing gt.txt: {gt_path}")
        return

    print(f"\n=== Processing sequence: {seq_name} ===")

    gt_by_frame = load_gt_by_frame(gt_path)
    frame_ids = sorted(gt_by_frame.keys())

    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    print(f"Frames to process: {len(frame_ids)}")

    seq_save_dir = os.path.join(save_root, seq_name)
    ensure_dir(seq_save_dir)

    seq_debug_dir = os.path.join(debug_root, seq_name) if debug_overlays else None
    if debug_overlays:
        ensure_dir(seq_debug_dir)

    for frame_id in frame_ids:
        # Try different image name patterns
        img_name = None
        for pattern in [f"{frame_id:08d}.jpg", f"{frame_id:06d}.jpg", f"{frame_id:08d}.png", f"{frame_id:06d}.png"]:
            if os.path.exists(os.path.join(img_dir, pattern)):
                img_name = pattern
                break
        
        if img_name is None:
            print(f"[WARN] Image not found for frame {frame_id}")
            continue

        img_path = os.path.join(img_dir, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]

        predictor.set_image(img_rgb)

        anns = gt_by_frame[frame_id]

        frame_save_dir = os.path.join(seq_save_dir, f"{frame_id:08d}")
        ensure_dir(frame_save_dir)

        debug_canvas = np.zeros_like(img_bgr) if debug_overlays else None

        for i, ann in enumerate(anns):
            track_id = ann["track_id"]
            x, y, w, h = ann["bbox_xywh"]

            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(w_img - 1, int(round(x + w)))
            y2 = min(h_img - 1, int(round(y + h)))

            if x2 <= x1 or y2 <= y1:
                continue

            mask_save_path = os.path.join(frame_save_dir, f"{track_id}.png")

            if SKIP_EXISTING and os.path.exists(mask_save_path):
                continue

            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            box_area = max((x2 - x1) * (y2 - y1), 1.0)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            masks, scores, _ = predictor.predict(
                box=box,
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )

            best_idx, best_mask = choose_best_mask(masks, scores, box_area, mode=BEST_MASK_MODE)
            best_mask = best_mask.copy()

            if CLIP_MASK_TO_BOX:
                clipped_mask = np.zeros_like(best_mask, dtype=bool)
                clipped_mask[y1:y2 + 1, x1:x2 + 1] = best_mask[y1:y2 + 1, x1:x2 + 1]
                best_mask = clipped_mask

            mask_u8 = (best_mask.astype(np.uint8) * 255)
            cv2.imwrite(mask_save_path, mask_u8)

            if debug_overlays:
                color = get_color(i)
                debug_canvas[best_mask] = color
                cv2.rectangle(debug_canvas, (x1, y1), (x2, y2), color, 2)

        if debug_overlays:
            overlay = cv2.addWeighted(img_bgr, 0.45, debug_canvas, 0.55, 0)
            overlay_save_path = os.path.join(seq_debug_dir, f"{frame_id:08d}_overlay.jpg")
            cv2.imwrite(overlay_save_path, overlay)

    print(f"  Completed {len(frame_ids)} frames")


def main():
    args = parse_args()

    ensure_dir(args.save_root)
    debug_root = os.path.join(args.save_root, "_debug_overlays") if args.debug_overlays else None
    if args.debug_overlays:
        ensure_dir(debug_root)

    print(f"Loading SAM ({args.model_type}) on device: {args.device}")
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    print("SAM loaded successfully.")

    sequences = get_dataset_structure(args.dataset, args.split, args.data_root)
    
    if args.max_sequences is not None:
        sequences = sequences[:args.max_sequences]

    print(f"Found {len(sequences)} sequences to process.")

    for seq_dir, img_folder, gt_path, seq_name in sequences:
        process_sequence(
            seq_dir=seq_dir,
            img_folder=img_folder,
            gt_path=gt_path,
            seq_name=seq_name,
            predictor=predictor,
            save_root=args.save_root,
            debug_overlays=args.debug_overlays,
            debug_root=debug_root,
            max_frames=args.max_frames,
        )

    print("\nDone!")
    print(f"Masks saved to: {args.save_root}")
    if args.debug_overlays:
        print(f"Debug overlays saved to: {debug_root}")


if __name__ == "__main__":
    main()
