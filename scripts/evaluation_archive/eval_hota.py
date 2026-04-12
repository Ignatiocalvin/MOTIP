#!/usr/bin/env python3
"""Compute HOTA, DetA, AssA for all MOTIP models using TrackEval."""
import sys
import os
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'TrackEval'))
from trackeval.metrics.hota import HOTA

GT_DIR = Path('data/P-DESTRE/annotations')

MODELS = [
    {
        'name': 'Base R50 (Fold 0)',
        'tracker_dir': 'outputs/r50_motip_pdestre_fold_0/train/eval_during_train/epoch_2/tracker',
        'val_split': 'splits/val_1.txt',  # tracker files cover val_1 seqs (trained on Train_0)
    },
    {
        'name': 'R50 2-Concept (Fold 0)',
        'tracker_dir': 'outputs/r50_base_motip_2concepts_fold_0/train/eval_during_train/epoch_9/tracker',
        'val_split': 'splits/val_0.txt',
    },
    {
        'name': 'R50 3-Concept (Fold 0)',
        'tracker_dir': 'outputs/r50_motip_pdestre_3concepts_fold_0/train/eval_during_train/epoch_0/tracker',
        'val_split': 'splits/val_0.txt',
    },
    {
        'name': 'R50 7-Concept Learnable (Fold 0)',
        'tracker_dir': 'outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/train/eval_during_train/epoch_3/tracker',
        'val_split': 'splits/val_1.txt',  # tracker files cover val_1 seqs; epoch_3 is best (MOTA=50.22%)
    },
    {
        'name': 'RF-DETR Large 2-Concept (Fold 0)',
        'tracker_dir': 'outputs/rfdetr_large_motip_pdestre_2concepts_fold0/eval_during_train/epoch_0/tracker',
        'val_split': 'splits/val_0.txt',
    },
    {
        'name': 'R50 7-Concept Learnable (Fold 1)',
        'tracker_dir': 'outputs/r50_motip_pdestre_7concepts_learnable_fold_1/train/eval_during_train/epoch_9/tracker',
        'val_split': 'splits/val_1.txt',
    },
]


def load_gt(seq_name):
    ann = defaultdict(list)
    with open(GT_DIR / f'{seq_name}.txt') as f:
        for line in f:
            p = line.strip().split(',')
            fid, tid = int(p[0]), int(p[1])
            if tid < 0:
                continue
            ann[fid].append((tid, float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return dict(ann)


def load_pred(path):
    pred = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            fid, tid = int(p[0]), int(p[1])
            if tid < 0:
                continue
            pred[fid].append((tid, float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return dict(pred)


def iou_box(g, p):
    ix1 = max(g[1], p[1])
    iy1 = max(g[2], p[2])
    ix2 = min(g[1] + g[3], p[1] + p[3])
    iy2 = min(g[2] + g[4], p[2] + p[4])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = g[3] * g[4] + p[3] * p[4] - inter
    return inter / max(1e-9, union)


def build_hota_data(gt_data, pred_data):
    """Build the data dict expected by HOTA.eval_sequence."""
    gt_id_map = {}
    pr_id_map = {}

    all_frames = sorted(set(gt_data) | set(pred_data))
    gt_ids_per_frame = []
    pr_ids_per_frame = []
    sim_per_frame = []

    for fid in all_frames:
        gt_b = gt_data.get(fid, [])
        pr_b = pred_data.get(fid, [])

        # remap IDs to contiguous 0-indexed
        gids = []
        for g in gt_b:
            if g[0] not in gt_id_map:
                gt_id_map[g[0]] = len(gt_id_map)
            gids.append(gt_id_map[g[0]])

        pids = []
        for p in pr_b:
            if p[0] not in pr_id_map:
                pr_id_map[p[0]] = len(pr_id_map)
            pids.append(pr_id_map[p[0]])

        if gt_b and pr_b:
            sim = np.array([[iou_box(g, p) for p in pr_b] for g in gt_b], dtype=np.float32)
        else:
            sim = np.empty((len(gt_b), len(pr_b)), dtype=np.float32)

        gt_ids_per_frame.append(np.array(gids, dtype=int))
        pr_ids_per_frame.append(np.array(pids, dtype=int))
        sim_per_frame.append(sim)

    return {
        'num_timesteps': len(all_frames),
        'num_gt_ids': len(gt_id_map),
        'num_tracker_ids': len(pr_id_map),
        'num_gt_dets': sum(len(g) for g in gt_ids_per_frame),
        'num_tracker_dets': sum(len(p) for p in pr_ids_per_frame),
        'gt_ids': gt_ids_per_frame,
        'tracker_ids': pr_ids_per_frame,
        'similarity_scores': sim_per_frame,
    }


def evaluate_model(model):
    name = model['name']
    tracker_dir = Path(model['tracker_dir'])
    seqs = [l.strip().replace('.txt', '') for l in open(model['val_split']) if l.strip()]

    if not tracker_dir.exists():
        print(f'  ⚠️  Tracker dir not found: {tracker_dir}')
        return None

    hota_metric = HOTA()
    all_seq_res = {}
    n_seqs = 0

    for s in seqs:
        pr_f = tracker_dir / f'{s}.txt'
        if not pr_f.exists():
            continue
        gt_data = load_gt(s)
        pred_data = load_pred(pr_f)
        data = build_hota_data(gt_data, pred_data)
        all_seq_res[s] = hota_metric.eval_sequence(data)
        n_seqs += 1

    if not all_seq_res:
        print(f'  ⚠️  No sequences evaluated for {name}')
        return None

    combined = hota_metric.combine_sequences(all_seq_res)
    # Aggregate: average per-sequence HOTA values over alpha thresholds
    # (consistent with per-sequence averaging reported in METRICS_COMPARISON.md)
    return {
        'HOTA': float(np.mean([np.mean(v['HOTA']) for v in all_seq_res.values()])) * 100,
        'DetA': float(np.mean([np.mean(v['DetA']) for v in all_seq_res.values()])) * 100,
        'AssA': float(np.mean([np.mean(v['AssA']) for v in all_seq_res.values()])) * 100,
        'DetRe': float(np.mean([np.mean(v['DetRe']) for v in all_seq_res.values()])) * 100,
        'DetPr': float(np.mean([np.mean(v['DetPr']) for v in all_seq_res.values()])) * 100,
        'AssRe': float(np.mean([np.mean(v['AssRe']) for v in all_seq_res.values()])) * 100,
        'AssPr': float(np.mean([np.mean(v['AssPr']) for v in all_seq_res.values()])) * 100,
        'n_seqs': n_seqs,
    }


def main():
    print('=' * 70)
    print('HOTA Evaluation — All Models')
    print('=' * 70)

    results = {}
    for model in MODELS:
        print(f'\n[{model["name"]}]')
        r = evaluate_model(model)
        if r:
            print(f'  HOTA:  {r["HOTA"]:.2f}%')
            print(f'  DetA:  {r["DetA"]:.2f}%  (DetRe={r["DetRe"]:.2f}%, DetPr={r["DetPr"]:.2f}%)')
            print(f'  AssA:  {r["AssA"]:.2f}%  (AssRe={r["AssRe"]:.2f}%, AssPr={r["AssPr"]:.2f}%)')
            print(f'  Seqs:  {r["n_seqs"]}')
            results[model['name']] = r

    print('\n' + '=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)
    print(f'{"Model":<40} {"HOTA":>7} {"DetA":>7} {"AssA":>7}')
    print('-' * 70)
    for name, r in results.items():
        print(f'{name:<40} {r["HOTA"]:>6.2f}% {r["DetA"]:>6.2f}% {r["AssA"]:>6.2f}%')
    print('=' * 70)


if __name__ == '__main__':
    main()
