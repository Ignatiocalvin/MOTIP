#!/usr/bin/env python3
"""Evaluate tracking metrics for all models across all available epochs.
Finds the best epoch (by MOTA) for each model and prints a summary."""

import sys
from pathlib import Path
import numpy as np
import motmetrics as mm
from collections import defaultdict
import json


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
                    inter_x1 = max(x1, x2)
                    inter_y1 = max(y1, y2)
                    inter_x2 = min(x1 + w1, x2 + w2)
                    inter_y2 = min(y1 + h1, y2 + h2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
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
    """Evaluate one epoch's tracker outputs against GT."""
    all_metrics = []
    per_seq = {}
    for seq_name in val_seqs:
        tracker_file = tracker_dir / f'{seq_name}.txt'
        if not tracker_file.exists():
            continue
        try:
            gt_data = load_gt_annotations(seq_name, data_dir)
            pred_data = load_tracker_results(tracker_file)
            metrics = compute_sequence_metrics(gt_data, pred_data)
            all_metrics.append(metrics)
            per_seq[seq_name] = metrics
        except Exception as e:
            print(f'  Error on {seq_name}: {e}')
            continue

    if not all_metrics:
        return None

    agg = {
        'MOTA': np.mean([m['MOTA'] for m in all_metrics]),
        'MOTA_std': np.std([m['MOTA'] for m in all_metrics]),
        'IDF1': np.mean([m['IDF1'] for m in all_metrics]),
        'IDF1_std': np.std([m['IDF1'] for m in all_metrics]),
        'Precision': np.mean([m['Precision'] for m in all_metrics]),
        'Recall': np.mean([m['Recall'] for m in all_metrics]),
        'IDsw': sum(m['IDsw'] for m in all_metrics),
        'FP': sum(m['FP'] for m in all_metrics),
        'FN': sum(m['FN'] for m in all_metrics),
        'GT': sum(m['GT'] for m in all_metrics),
        'Pred': sum(m['Pred'] for m in all_metrics),
        'TP': sum(m['Pred'] for m in all_metrics) - sum(m['FP'] for m in all_metrics),
        'n_seqs': len(all_metrics),
    }
    return agg


# Define all models to evaluate
MODELS = [
    {
        'name': 'R50 Base (No Concepts) Fold 0',
        'output_dir': 'r50_motip_pdestre_fold_0',
        'val_split': 'splits/val_0.txt',
        'description': 'ResNet-50, 2 concepts, fold_0',
    },
    {
        'name': 'R50 Base (No Concepts) Fold 1',
        'output_dir': 'r50_motip_pdestre_fold_1',
        'val_split': 'splits/val_1.txt',
        'description': 'ResNet-50, 2 concepts, fold_1',
    },
    {
        'name': 'R50 Base (No Concepts) Fold 2',
        'output_dir': 'r50_motip_pdestre_fold_2',
        'val_split': 'splits/val_1.txt',  # config says val_1
        'description': 'ResNet-50, 2 concepts, fold_2 (train_1/val_1)',
    },
    {
        'name': 'R50 Base (No Concepts) Fold 3',
        'output_dir': 'r50_motip_pdestre_fold_3',
        'val_split': 'splits/val_3.txt',
        'description': 'ResNet-50, 2 concepts, fold_3',
    },
    {
        'name': 'R50 2-Concept Fold 0',
        'output_dir': 'r50_base_motip_2concepts_fold_0',
        'val_split': 'splits/val_0.txt',
        'description': 'ResNet-50, 2 concepts (gender+upper_body), fold_0',
    },
    {
        'name': 'R50 3-Concept Fold 0 (still running)',
        'output_dir': 'r50_motip_pdestre_3concepts_fold_0',
        'val_split': 'splits/val_0.txt',
        'description': 'ResNet-50, 3 concepts, fold_0, still training',
    },
    {
        'name': 'R50 7-Concept Learnable v2 Fold 0',
        'output_dir': 'r50_motip_pdestre_7concepts_learnable_v2_fold_0',
        'val_split': 'splits/val_1.txt',  # config says val_1
        'description': 'ResNet-50, 7 concepts learnable, Train_1/val_1',
    },
    {
        'name': 'R50 7-Concept Learnable Fold 1',
        'output_dir': 'r50_motip_pdestre_7concepts_learnable_fold_1',
        'val_split': 'splits/val_1.txt',
        'description': 'ResNet-50, 7 concepts learnable, fold_1',
    },
    {
        'name': 'RF-DETR Base (No Concepts) Fold 0',
        'output_dir': 'rfdetr_large_motip_pdestre_base_fold0',
        'val_split': 'splits/val_0.txt',
        'description': 'RF-DETR/DINOv2, no concepts, fold_0',
    },
    {
        'name': 'RF-DETR Large 2-Concepts Fold 0',
        'output_dir': 'rfdetr_large_motip_pdestre_2concepts_fold0',
        'val_split': 'splits/val_0.txt',
        'description': 'RF-DETR/DINOv2, 2 concepts, fold_0',
    },
    {
        'name': 'RF-DETR 7-Concept Learnable Fold 0 (still running)',
        'output_dir': 'rfdetr_large_motip_pdestre_7concepts_learnable_fold0',
        'val_split': 'splits/val_0.txt',
        'description': 'RF-DETR/DINOv2, 7 concepts learnable, fold_0, still training',
    },
]


def main():
    base_dir = Path('.')
    outputs_dir = base_dir / 'outputs'

    results = {}

    for model in MODELS:
        name = model['name']
        output_dir = outputs_dir / model['output_dir']

        print(f"\n{'='*80}")
        print(f"MODEL: {name}")
        print(f"  Dir: {model['output_dir']}")
        print(f"  Description: {model['description']}")
        print(f"{'='*80}")

        val_seqs = get_val_sequences(base_dir / model['val_split'])
        print(f"  Val sequences: {len(val_seqs)}")

        # Also check root-level tracker/ directory
        root_tracker = output_dir / 'tracker'

        epoch_results = {}

        # Check both possible eval_during_train locations
        eval_dirs_to_check = [
            output_dir / 'train' / 'eval_during_train',
            output_dir / 'eval_during_train',
        ]
        eval_dir = None
        for ed in eval_dirs_to_check:
            if ed.exists():
                eval_dir = ed
                break

        # Evaluate all available epochs
        if eval_dir is not None and eval_dir.exists():
            epoch_dirs = sorted(eval_dir.iterdir())
            for epoch_dir in epoch_dirs:
                if not epoch_dir.is_dir() or not epoch_dir.name.startswith('epoch_'):
                    continue
                tracker_dir = epoch_dir / 'tracker'
                if not tracker_dir.exists():
                    continue
                epoch_num = int(epoch_dir.name.split('_')[1])
                print(f"  Evaluating epoch {epoch_num}...", end=' ')
                agg = evaluate_epoch(tracker_dir, val_seqs)
                if agg:
                    epoch_results[epoch_num] = agg
                    print(f"MOTA: {agg['MOTA']:.2f}% | IDF1: {agg['IDF1']:.2f}% | Prec: {agg['Precision']:.2f}% | Rec: {agg['Recall']:.2f}% ({agg['n_seqs']} seqs)")
                else:
                    print("No results")

        # Also check root tracker/ if exists and no epoch results
        if root_tracker.exists() and root_tracker.is_dir():
            txt_files = list(root_tracker.glob('*.txt'))
            if txt_files:
                print(f"  Evaluating root tracker/ ({len(txt_files)} files)...", end=' ')
                agg = evaluate_epoch(root_tracker, val_seqs)
                if agg:
                    epoch_results['root'] = agg
                    print(f"MOTA: {agg['MOTA']:.2f}% | IDF1: {agg['IDF1']:.2f}% | ({agg['n_seqs']} seqs)")
                else:
                    print("No results")

        if not epoch_results:
            print("  ⚠️  No evaluation results found!")
            continue

        # Find best epoch by MOTA
        best_epoch = max(
            [(k, v) for k, v in epoch_results.items() if k != 'root'],
            key=lambda x: x[1]['MOTA'],
            default=(None, None)
        )

        if best_epoch[0] is not None:
            be, bm = best_epoch
            print(f"\n  ★ BEST EPOCH: {be}")
            print(f"    MOTA:      {bm['MOTA']:.2f}% ± {bm['MOTA_std']:.2f}%")
            print(f"    IDF1:      {bm['IDF1']:.2f}% ± {bm['IDF1_std']:.2f}%")
            print(f"    Precision: {bm['Precision']:.2f}%")
            print(f"    Recall:    {bm['Recall']:.2f}%")
            print(f"    TP:        {bm['TP']}")
            print(f"    FP:        {bm['FP']}")
            print(f"    FN:        {bm['FN']}")
            print(f"    ID Sw:     {bm['IDsw']}")
            print(f"    GT:        {bm['GT']}")
            print(f"    Pred:      {bm['Pred']}")

        results[name] = {
            'epoch_results': {str(k): v for k, v in epoch_results.items()},
            'best_epoch': best_epoch[0],
            'best_metrics': best_epoch[1],
            'config': model,
        }

    # Print summary table
    print(f"\n\n{'='*120}")
    print("SUMMARY: Best Epoch Results for All Models")
    print(f"{'='*120}")
    print(f"{'Model':<50s} {'Epoch':>5s} {'MOTA%':>8s} {'IDF1%':>8s} {'Prec%':>8s} {'Rec%':>8s} {'IDsw':>6s} {'TP':>8s} {'FP':>8s} {'FN':>8s} {'GT':>8s} {'Pred':>8s}")
    print('-'*120)

    for name, data in results.items():
        if data['best_metrics'] is None:
            print(f"{name:<50s}  {'N/A':>5s}")
            continue
        m = data['best_metrics']
        be = data['best_epoch']
        print(f"{name:<50s} {str(be):>5s} {m['MOTA']:>7.2f}% {m['IDF1']:>7.2f}% {m['Precision']:>7.2f}% {m['Recall']:>7.2f}% {m['IDsw']:>6d} {m['TP']:>8d} {m['FP']:>8d} {m['FN']:>8d} {m['GT']:>8d} {m['Pred']:>8d}")

    # Save JSON for later use
    with open('evaluation_results_all.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to evaluation_results_all.json")


if __name__ == '__main__':
    main()
