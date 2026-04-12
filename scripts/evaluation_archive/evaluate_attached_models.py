#!/usr/bin/env python3
"""Evaluate tracking metrics for all attached model outputs across all available epochs."""

import sys
import json
from pathlib import Path
import numpy as np
import motmetrics as mm
from collections import defaultdict


def load_gt_annotations(sequence_name, data_dir='data/P-DESTRE/annotations'):
    gt_file = Path(data_dir) / f'{sequence_name}.txt'
    annotations = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            if track_id < 0:
                continue
            x, y, w, h = map(float, parts[2:6])
            annotations[frame_id].append((track_id, x, y, w, h))
    return dict(annotations)


def load_tracker_results(tracker_file):
    predictions = defaultdict(list)
    with open(tracker_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            if track_id < 0:
                continue
            x, y, w, h = map(float, parts[2:6])
            predictions[frame_id].append((track_id, x, y, w, h))
    return dict(predictions)


def compute_sequence_metrics(gt_data, pred_data):
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
                    x1, y1, w1, h1 = gt_box[1:]
                    x2, y2, w2, h2 = pred_box[1:]
                    inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
                    inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    union_area = w1 * h1 + w2 * h2 - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
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
        'MOTP': float(summary['motp'].values[0]),
        'Precision': float(summary['precision'].values[0]) * 100,
        'Recall': float(summary['recall'].values[0]) * 100,
        'IDsw': int(summary['num_switches'].values[0]),
        'FP': int(summary['num_false_positives'].values[0]),
        'FN': int(summary['num_misses'].values[0]),
        'GT': int(summary['num_objects'].values[0]),
        'Pred': int(summary['num_predictions'].values[0]),
    }


def get_val_sequences(val_split_file):
    seqs = []
    with open(val_split_file) as f:
        for line in f:
            line = line.strip()
            if line:
                seqs.append(line.replace('.txt', ''))
    return seqs


def evaluate_epoch(tracker_dir, val_seqs, data_dir='data/P-DESTRE/annotations'):
    all_metrics = []
    for seq_name in val_seqs:
        tracker_file = tracker_dir / f'{seq_name}.txt'
        if not tracker_file.exists():
            continue
        gt_data = load_gt_annotations(seq_name, data_dir)
        pred_data = load_tracker_results(tracker_file)
        metrics = compute_sequence_metrics(gt_data, pred_data)
        all_metrics.append(metrics)

    if not all_metrics:
        return None

    return {
        'MOTA': float(np.mean([m['MOTA'] for m in all_metrics])),
        'MOTA_std': float(np.std([m['MOTA'] for m in all_metrics])),
        'IDF1': float(np.mean([m['IDF1'] for m in all_metrics])),
        'IDF1_std': float(np.std([m['IDF1'] for m in all_metrics])),
        'Precision': float(np.mean([m['Precision'] for m in all_metrics])),
        'Precision_std': float(np.std([m['Precision'] for m in all_metrics])),
        'Recall': float(np.mean([m['Recall'] for m in all_metrics])),
        'Recall_std': float(np.std([m['Recall'] for m in all_metrics])),
        'IDsw': int(sum(m['IDsw'] for m in all_metrics)),
        'FP': int(sum(m['FP'] for m in all_metrics)),
        'FN': int(sum(m['FN'] for m in all_metrics)),
        'GT': int(sum(m['GT'] for m in all_metrics)),
        'Pred': int(sum(m['Pred'] for m in all_metrics)),
        'TP': int(sum(m['Pred'] for m in all_metrics) - sum(m['FP'] for m in all_metrics)),
        'n_seqs': len(all_metrics),
    }


# Define models to evaluate
MODELS = [
    {
        'name': 'R50 2-Concept (Fold 0)',
        'dir': 'r50_base_motip_2concepts_fold_0',
        'tracker_base': 'outputs/r50_base_motip_2concepts_fold_0/train/eval_during_train',
        'val_split': 'splits/val_0.txt',
    },
    {
        'name': 'R50 3-Concept (Fold 0)',
        'dir': 'r50_motip_pdestre_3concepts_fold_0',
        'tracker_base': 'outputs/r50_motip_pdestre_3concepts_fold_0/train/eval_during_train',
        'val_split': 'splits/val_0.txt',
    },
    {
        'name': 'R50 7-Concept Learnable v2 (Fold 0)',
        'dir': 'r50_motip_pdestre_7concepts_learnable_v2_fold_0',
        'tracker_base': 'outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/train/eval_during_train',
        'val_split': 'splits/val_1.txt',
    },
    {
        'name': 'R50 7-Concept Learnable (Fold 1)',
        'dir': 'r50_motip_pdestre_7concepts_learnable_fold_1',
        'tracker_base': 'outputs/r50_motip_pdestre_7concepts_learnable_fold_1/train/eval_during_train',
        'val_split': 'splits/val_1.txt',
    },
    {
        'name': 'RF-DETR Large 2-Concept (Fold 0)',
        'dir': 'rfdetr_large_motip_pdestre_2concepts_fold0',
        'tracker_base': 'outputs/rfdetr_large_motip_pdestre_2concepts_fold0/eval_during_train',
        'val_split': 'splits/val_0.txt',
    },
    {
        'name': 'RF-DETR 7-Concept NO Learnable Weights (Fold 0)',
        'dir': 'rfdetr_large_motip_pdestre_7concepts_learnable_fold0',
        'tracker_base': 'outputs/rfdetr_large_motip_pdestre_7concepts_learnable_fold0/train/eval_during_train',
        'val_split': 'splits/val_0.txt',
    },
    {
        'name': 'RF-DETR 7-Concept WITH Learnable Weights (Fold 0)',
        'dir': 'rfdetr_large_motip_pdestre_7concepts_lw_fold0',
        'tracker_base': 'outputs/rfdetr_large_motip_pdestre_7concepts_lw_fold0/train/eval_during_train',
        'val_split': 'splits/val_0.txt',
    },
]


def main():
    results = {}
    for model in MODELS:
        name = model['name']
        tracker_base = Path(model['tracker_base'])
        val_seqs = get_val_sequences(model['val_split'])

        # Find available epochs
        epochs = []
        if tracker_base.exists():
            for ep_dir in sorted(tracker_base.iterdir()):
                if ep_dir.is_dir() and ep_dir.name.startswith('epoch_'):
                    epoch_num = int(ep_dir.name.replace('epoch_', ''))
                    epochs.append(epoch_num)

        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"  Dir: {model['dir']}")
        print(f"  Val split: {model['val_split']} ({len(val_seqs)} seqs)")
        print(f"  Available epochs: {sorted(epochs)}")
        print(f"{'='*60}")

        model_results = {}
        best_epoch = None
        best_mota = -999

        for epoch in sorted(epochs):
            tracker_dir = tracker_base / f'epoch_{epoch}' / 'tracker'
            print(f"  Evaluating epoch {epoch}...", end=' ', flush=True)
            metrics = evaluate_epoch(tracker_dir, val_seqs)
            if metrics is None:
                print("SKIPPED (no valid results)")
                continue
            model_results[epoch] = metrics
            print(f"MOTA={metrics['MOTA']:.2f}%, IDF1={metrics['IDF1']:.2f}%, "
                  f"Prec={metrics['Precision']:.2f}%, Rec={metrics['Recall']:.2f}%")

            if metrics['MOTA'] > best_mota:
                best_mota = metrics['MOTA']
                best_epoch = epoch

        results[name] = {
            'epoch_results': model_results,
            'best_epoch': best_epoch,
            'best_mota': best_mota,
            'val_split': model['val_split'],
            'n_val_seqs': len(val_seqs),
        }

    # Save results
    out_file = 'evaluation_results_attached.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
