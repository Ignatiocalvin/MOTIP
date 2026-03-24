#!/usr/bin/env python3
"""
Compute tracking metrics (HOTA, MOTA, IDF1) for P-DESTRE results.

This script:
1. Reads P-DESTRE ground truth annotations
2. Compares with tracker predictions
3. Computes standard MOT metrics using motmetrics library
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import motmetrics as mm
    HAS_MOTMETRICS = True
except ImportError:
    HAS_MOTMETRICS = False
    print("Warning: motmetrics not available. Install with: pip install motmetrics")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data" / "P-DESTRE"
SPLITS_DIR = BASE_DIR / "splits"
VIS_DIR = BASE_DIR / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

# Model configurations
MODELS = {
    "R50 MOTIP (No Concepts)": {
        "prefix": "r50_motip_pdestre_fold",
        "color": "#3498db",
    },
    "R50 MOTIP (2 Concepts)": {
        "prefix": "r50_motip_pdestre_concepts_for_id_fold",
        "color": "#2ecc71",
    },
    "R50 MOTIP (7 Concepts)": {
        "prefix": "r50_motip_pdestre_7concepts_for_id_fold",
        "color": "#e74c3c",
    },
    "RF-DETR Base (No Concepts)": {
        "prefix": "rfdetr_base_motip_pdestre_base_fold",  # Note: no underscore before fold number
        "color": "#9b59b6",
        "use_underscore": False,
    },
    "RF-DETR Large (2 Concepts)": {
        "prefix": "rfdetr_large_motip_pdestre_2concepts_fold",  # Note: no underscore before fold number
        "color": "#e67e22",
        "use_underscore": False,
    },
}

NUM_FOLDS = 3


def get_val_sequences(fold_idx: int) -> list:
    """Get validation sequence names for a fold (used during training evaluation)."""
    val_file = SPLITS_DIR / f"val_{fold_idx}.txt"
    if not val_file.exists():
        return []
    
    with open(val_file, 'r') as f:
        sequences = [line.strip().replace('.txt', '') for line in f if line.strip()]
    return sequences


def get_test_sequences(fold_idx: int) -> list:
    """Get test sequence names for a fold."""
    test_file = SPLITS_DIR / f"Test_{fold_idx}.txt"
    if not test_file.exists():
        return []
    
    with open(test_file, 'r') as f:
        sequences = [line.strip().replace('.txt', '') for line in f if line.strip()]
    return sequences


def load_gt_annotations(sequence_name: str) -> dict:
    """
    Load ground truth annotations for a sequence.
    
    P-DESTRE format: frame_id, track_id, x, y, w, h, -1, -1, -1, -1, concepts...
    
    Returns: {frame_id: [(track_id, x, y, w, h), ...]}
    """
    gt_file = DATA_DIR / "annotations" / f"{sequence_name}.txt"
    
    annotations = defaultdict(list)
    
    if not gt_file.exists():
        return annotations
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            
            # Skip invalid track IDs
            if track_id < 0:
                continue
            
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            
            # Skip invalid bounding boxes
            if w <= 0 or h <= 0:
                continue
            
            annotations[frame_id].append((track_id, x, y, w, h))
    
    return annotations


def load_tracker_results(tracker_file: Path) -> dict:
    """
    Load tracker predictions.
    
    MOT format: frame_id, track_id, x, y, w, h, conf, -1, -1, -1, [concept_probs...]
    
    Returns: {frame_id: [(track_id, x, y, w, h), ...]}
    """
    predictions = defaultdict(list)
    
    if not tracker_file.exists():
        return predictions
    
    with open(tracker_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            
            # Skip invalid track IDs
            if track_id < 0:
                continue
            
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            
            # Skip invalid bounding boxes
            if w <= 0 or h <= 0:
                continue
            
            predictions[frame_id].append((track_id, x, y, w, h))
    
    return predictions


def compute_iou(box1, box2):
    """Compute IoU between two boxes (x, y, w, h format)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def compute_sequence_metrics_motmetrics(gt_data: dict, pred_data: dict) -> dict:
    """Compute metrics using motmetrics library."""
    if not HAS_MOTMETRICS:
        return {}
    
    acc = mm.MOTAccumulator(auto_id=True)
    
    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
    
    for frame_id in all_frames:
        gt_boxes = gt_data.get(frame_id, [])
        pred_boxes = pred_data.get(frame_id, [])
        
        gt_ids = [box[0] for box in gt_boxes]
        pred_ids = [box[0] for box in pred_boxes]
        
        # Compute distance matrix (1 - IoU)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            dist_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou = compute_iou(gt_box[1:], pred_box[1:])
                    dist_matrix[i, j] = 1 - iou
            
            # Mark pairs with IoU < 0.5 as infinite distance
            dist_matrix[dist_matrix > 0.5] = np.nan
        else:
            dist_matrix = np.empty((len(gt_boxes), len(pred_boxes)))
            dist_matrix[:] = np.nan
        
        acc.update(gt_ids, pred_ids, dist_matrix)
    
    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'precision', 'recall', 
                                        'num_switches', 'num_false_positives', 
                                        'num_misses', 'num_objects', 'num_predictions'], 
                         name='seq')
    
    return {
        'MOTA': float(summary['mota'].values[0]) * 100,
        'MOTP': float(summary['motp'].values[0]) * 100,
        'IDF1': float(summary['idf1'].values[0]) * 100,
        'Precision': float(summary['precision'].values[0]) * 100,
        'Recall': float(summary['recall'].values[0]) * 100,
        'IDsw': int(summary['num_switches'].values[0]),
        'FP': int(summary['num_false_positives'].values[0]),
        'FN': int(summary['num_misses'].values[0]),
        'GT': int(summary['num_objects'].values[0]),
        'Pred': int(summary['num_predictions'].values[0]),
    }


def compute_sequence_metrics_manual(gt_data: dict, pred_data: dict, iou_thresh: float = 0.5) -> dict:
    """
    Compute basic MOT metrics manually (fallback without motmetrics).
    """
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Track ID associations across frames for ID switches
    prev_matches = {}  # gt_id -> pred_id
    id_switches = 0
    
    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
    
    for frame_id in all_frames:
        gt_boxes = gt_data.get(frame_id, [])
        pred_boxes = pred_data.get(frame_id, [])
        
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = compute_iou(gt_box[1:], pred_box[1:])
        
        # Greedy matching
        matched_gt = set()
        matched_pred = set()
        current_matches = {}
        
        # Sort by IoU descending
        indices = np.unravel_index(np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape)
        
        for i, j in zip(indices[0], indices[1]):
            if i in matched_gt or j in matched_pred:
                continue
            if iou_matrix[i, j] >= iou_thresh:
                matched_gt.add(i)
                matched_pred.add(j)
                
                gt_id = gt_boxes[i][0]
                pred_id = pred_boxes[j][0]
                current_matches[gt_id] = pred_id
                
                # Check for ID switch
                if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                    id_switches += 1
        
        prev_matches.update(current_matches)
        
        tp = len(matched_gt)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Compute metrics
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    
    # MOTA = 1 - (FN + FP + IDsw) / GT
    mota = 1 - (total_fn + total_fp + id_switches) / (total_gt + 1e-10)
    mota = max(-1, min(1, mota)) * 100  # Clip to [-100, 100]
    
    # IDF1 approximation (simplified)
    idf1 = 2 * total_tp / (total_gt + total_pred + 1e-10) * 100
    
    return {
        'MOTA': mota,
        'IDF1': idf1,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'IDsw': id_switches,
        'FP': total_fp,
        'FN': total_fn,
        'GT': total_gt,
        'Pred': total_pred,
        'TP': total_tp,
    }


def find_tracker_dir(output_dir: Path) -> Path:
    """Find tracker results directory."""
    # Check for standalone evaluate results first (newest)
    eval_tracker = output_dir / "evaluate" / "tracker"
    if eval_tracker.exists() and list(eval_tracker.glob("*.txt")):
        return eval_tracker
    
    # Check for eval_during_train results (use latest epoch)
    eval_during_train = output_dir / "train" / "eval_during_train"
    if eval_during_train.exists():
        epochs = sorted(eval_during_train.glob("epoch_*"), 
                       key=lambda x: int(x.name.split("_")[1]))
        if epochs:
            latest_epoch = epochs[-1]
            tracker_path = latest_epoch / "tracker"
            if tracker_path.exists() and list(tracker_path.glob("*.txt")):
                return tracker_path
    
    # Check other common locations
    for path in [output_dir / "tracker"]:
        if path.exists() and list(path.glob("*.txt")):
            return path
    
    return None


def evaluate_model_fold(model_name: str, fold_idx: int, tracker_dir: Path) -> dict:
    """Evaluate a single model fold."""
    # Use validation sequences since eval_during_train uses validation set
    val_sequences = get_val_sequences(fold_idx)
    
    all_metrics = []
    
    for seq_name in val_sequences:
        tracker_file = tracker_dir / f"{seq_name}.txt"
        if not tracker_file.exists():
            continue
        
        gt_data = load_gt_annotations(seq_name)
        pred_data = load_tracker_results(tracker_file)
        
        if not gt_data or not pred_data:
            continue
        
        if HAS_MOTMETRICS:
            metrics = compute_sequence_metrics_motmetrics(gt_data, pred_data)
        else:
            metrics = compute_sequence_metrics_manual(gt_data, pred_data)
        
        if metrics:
            metrics['sequence'] = seq_name
            all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    # Aggregate metrics
    aggregated = {
        'num_sequences': len(all_metrics),
        'MOTA_mean': np.mean([m['MOTA'] for m in all_metrics]),
        'MOTA_std': np.std([m['MOTA'] for m in all_metrics]),
        'IDF1_mean': np.mean([m['IDF1'] for m in all_metrics]),
        'IDF1_std': np.std([m['IDF1'] for m in all_metrics]),
        'Precision_mean': np.mean([m['Precision'] for m in all_metrics]),
        'Recall_mean': np.mean([m['Recall'] for m in all_metrics]),
        'IDsw_total': sum([m['IDsw'] for m in all_metrics]),
        'per_sequence': all_metrics,
    }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Compute tracking metrics for P-DESTRE")
    parser.add_argument("--model", type=str, default=None, help="Specific model to evaluate")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold to evaluate")
    args = parser.parse_args()
    
    print("="*80)
    print("🔬 P-DESTRE Tracking Metrics Evaluation")
    print("="*80)
    
    if not HAS_MOTMETRICS:
        print("\n⚠️  motmetrics not installed. Using manual computation (less accurate).")
        print("   Install with: pip install motmetrics\n")
    
    all_results = {}
    
    for model_name, config in MODELS.items():
        if args.model and args.model not in model_name:
            continue
        
        print(f"\n📊 {model_name}")
        print("-" * 50)
        
        model_results = {"folds": {}}
        
        for fold_idx in range(NUM_FOLDS):
            if args.fold is not None and args.fold != fold_idx:
                continue
            
            # Handle different fold naming conventions
            if config.get('use_underscore', True):
                output_dir = OUTPUTS_DIR / f"{config['prefix']}_{fold_idx}"
            else:
                output_dir = OUTPUTS_DIR / f"{config['prefix']}{fold_idx}"
            
            if not output_dir.exists():
                print(f"  Fold {fold_idx}: Output directory not found")
                continue
            
            tracker_dir = find_tracker_dir(output_dir)
            
            if not tracker_dir:
                print(f"  Fold {fold_idx}: No tracker results found")
                continue
            
            print(f"  Fold {fold_idx}: Evaluating {tracker_dir}...")
            
            metrics = evaluate_model_fold(model_name, fold_idx, tracker_dir)
            
            if metrics:
                model_results["folds"][fold_idx] = metrics
                print(f"    MOTA: {metrics['MOTA_mean']:.2f}% ± {metrics['MOTA_std']:.2f}")
                print(f"    IDF1: {metrics['IDF1_mean']:.2f}% ± {metrics['IDF1_std']:.2f}")
                print(f"    ID Switches: {metrics['IDsw_total']}")
                print(f"    Sequences: {metrics['num_sequences']}")
        
        if model_results["folds"]:
            all_results[model_name] = model_results
    
    # Print summary
    print("\n" + "="*80)
    print("📋 TRACKING METRICS SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for model_name, model_data in all_results.items():
        folds = model_data["folds"]
        if not folds:
            continue
        
        mota_values = [f['MOTA_mean'] for f in folds.values()]
        idf1_values = [f['IDF1_mean'] for f in folds.values()]
        idsw_values = [f['IDsw_total'] for f in folds.values()]
        
        summary = {
            'model': model_name,
            'MOTA': f"{np.mean(mota_values):.2f} ± {np.std(mota_values):.2f}",
            'IDF1': f"{np.mean(idf1_values):.2f} ± {np.std(idf1_values):.2f}",
            'IDsw': f"{np.mean(idsw_values):.0f}",
            'mota_mean': np.mean(mota_values),
            'idf1_mean': np.mean(idf1_values),
        }
        summary_data.append(summary)
        
        print(f"\n{model_name}")
        print(f"  MOTA: {summary['MOTA']}%")
        print(f"  IDF1: {summary['IDF1']}%")
        print(f"  Avg ID Switches per fold: {summary['IDsw']}")
    
    # Save results
    results_file = VIS_DIR / "tracking_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Create visualization
    if HAS_MATPLOTLIB and summary_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        models = [s['model'].replace(' ', '\n') for s in summary_data]
        mota_vals = [s['mota_mean'] for s in summary_data]
        idf1_vals = [s['idf1_mean'] for s in summary_data]
        colors = [MODELS[s['model']]['color'] for s in summary_data]
        
        # MOTA plot
        ax = axes[0]
        bars = ax.bar(range(len(models)), mota_vals, color=colors, alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, mota_vals):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel('MOTA (%)', fontsize=12)
        ax.set_title('Multi-Object Tracking Accuracy (MOTA)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # IDF1 plot
        ax = axes[1]
        bars = ax.bar(range(len(models)), idf1_vals, color=colors, alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, idf1_vals):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel('IDF1 (%)', fontsize=12)
        ax.set_title('ID F1 Score (IDF1)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(VIS_DIR / "tracking_metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Visualization saved to: {VIS_DIR / 'tracking_metrics_comparison.png'}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
