#!/usr/bin/env python3
"""Quick script to compute tracking metrics for 7-concept learnable model."""

import sys
from pathlib import Path
import numpy as np
import motmetrics as mm
from collections import defaultdict


def load_gt_annotations(sequence_name):
    """Load ground truth annotations."""
    gt_file = Path('data/P-DESTRE/annotations') / f'{sequence_name}.txt'
    annotations = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            # Skip track_id < 0 (invalid annotations)
            if track_id < 0:
                continue
            x, y, w, h = map(float, parts[2:6])
            annotations[frame_id].append((track_id, x, y, w, h))
    return dict(annotations)


def load_tracker_results(tracker_file):
    """Load tracker predictions."""
    predictions = defaultdict(list)
    with open(tracker_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            # Skip track_id < 0 (invalid predictions)
            if track_id < 0:
                continue
            x, y, w, h = map(float, parts[2:6])
            predictions[frame_id].append((track_id, x, y, w, h))
    return dict(predictions)


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_sequence_metrics(gt_data, pred_data):
    """Compute metrics for a sequence using motmetrics."""
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
    
    for frame_id in all_frames:
        gt_boxes = gt_data.get(frame_id, [])
        pred_boxes = pred_data.get(frame_id, [])
        gt_ids = [box[0] for box in gt_boxes]
        pred_ids = [box[0] for box in pred_boxes]
        
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            dist_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou = compute_iou(gt_box[1:], pred_box[1:])
                    dist_matrix[i, j] = 1 - iou
            dist_matrix[dist_matrix > 0.5] = np.nan
        else:
            dist_matrix = np.empty((len(gt_boxes), len(pred_boxes)))
            dist_matrix[:] = np.nan
        
        acc.update(gt_ids, pred_ids, dist_matrix)
    
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'precision', 'recall', 
                                        'num_switches', 'num_false_positives', 
                                        'num_misses', 'num_objects', 'num_predictions'], 
                         name='seq')
    
    return {
        'MOTA': float(summary['mota'].values[0]) * 100,
        'IDF1': float(summary['idf1'].values[0]) * 100,
        'Precision': float(summary['precision'].values[0]) * 100,
        'Recall': float(summary['recall'].values[0]) * 100,
        'IDsw': int(summary['num_switches'].values[0]),
        'FP': int(summary['num_false_positives'].values[0]),
        'FN': int(summary['num_misses'].values[0]),
        'GT': int(summary['num_objects'].values[0]),
        'Pred': int(summary['num_predictions'].values[0]),
    }


def main():
    # Validation sequences for fold 0
    val_seqs = [
        '13-11-2019-2-5', '13-11-2019-2-2', '18-07-2019-1-1', '11-11-2019-1-2', 
        '13-11-2019-2-4', '13-11-2019-4-4', '13-11-2019-3-4', '12-11-2019-4-3',
        '14-11-2019-1-3', '13-11-2019-3-3', '12-11-2019-4-5', '13-11-2019-1-2',
        '18-07-2019-1-2', '22-10-2019-1-1'
    ]
    
    tracker_dir = Path('outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/train/eval_during_train/epoch_2/tracker')
    
    print("="*80)
    print("7-Concept Learnable V2 - Tracking Metrics Evaluation")
    print("="*80)
    
    all_metrics = []
    for seq_name in val_seqs:
        tracker_file = tracker_dir / f'{seq_name}.txt'
        if not tracker_file.exists():
            print(f'⚠️  Missing: {seq_name}')
            continue
        
        try:
            gt_data = load_gt_annotations(seq_name)
            pred_data = load_tracker_results(tracker_file)
            metrics = compute_sequence_metrics(gt_data, pred_data)
            all_metrics.append(metrics)
            print(f'✓ {seq_name}: MOTA={metrics["MOTA"]:.2f}%, IDF1={metrics["IDF1"]:.2f}%, IDsw={metrics["IDsw"]}')
        except Exception as e:
            print(f'✗ {seq_name}: Error - {e}')
            continue
    
    if not all_metrics:
        print("\n❌ No metrics computed!")
        return
    
    print("\n" + "="*80)
    print("AGGREGATE METRICS (7-Concept Learnable V2)")
    print("="*80)
    print(f'MOTA:       {np.mean([m["MOTA"] for m in all_metrics]):.2f}% ± {np.std([m["MOTA"] for m in all_metrics]):.2f}%')
    print(f'IDF1:       {np.mean([m["IDF1"] for m in all_metrics]):.2f}% ± {np.std([m["IDF1"] for m in all_metrics]):.2f}%')
    print(f'Precision:  {np.mean([m["Precision"] for m in all_metrics]):.2f}%')
    print(f'Recall:     {np.mean([m["Recall"] for m in all_metrics]):.2f}%')
    print(f'ID Switches: {sum([m["IDsw"] for m in all_metrics])}')
    print(f'FP:         {sum([m["FP"] for m in all_metrics])}')
    print(f'FN (Misses): {sum([m["FN"] for m in all_metrics])}')
    print(f'GT Objects:  {sum([m["GT"] for m in all_metrics])}')
    print(f'Predictions: {sum([m["Pred"] for m in all_metrics])}')
    print("="*80)


if __name__ == "__main__":
    main()
