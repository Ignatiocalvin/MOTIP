#!/usr/bin/env python3
"""Compute MOTA/IDF1/Prec/Rec metrics for R50-MOTIP-7C-LW on val_1 split (all 9 epochs)."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import motmetrics as mm

EXP = "r50_motip_pdestre_7concepts_learnable_v2_fold_0"
VAL_SPLIT = "val_1"
BASE = Path(f"outputs/{EXP}/train/eval_during_train")
GT_DIR = Path("data/P-DESTRE/annotations")
SPLIT_FILE = Path(f"data/P-DESTRE/splits/{VAL_SPLIT}.txt")

val_seqs = [l.strip().replace('.txt', '') for l in SPLIT_FILE.read_text().splitlines() if l.strip()]
print(f"Val sequences ({len(val_seqs)}): {val_seqs}", flush=True)


def load_gt(seq):
    p = GT_DIR / f"{seq}.txt"
    data = {}
    for line in p.read_text().splitlines():
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        frame, tid, x, y, w, h = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        if w <= 0 or h <= 0:
            continue
        data.setdefault(frame, []).append((tid, x, y, w, h))
    return data


def load_pred(seq, epoch):
    p = BASE / f"epoch_{epoch}" / "tracker" / f"{seq}.txt"
    if not p.exists():
        print(f"  WARNING: Missing tracker file for epoch {epoch}, seq {seq}", flush=True)
        return {}
    data = {}
    for line in p.read_text().splitlines():
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        frame, tid, x, y, w, h = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        data.setdefault(frame, []).append((tid, x, y, w, h))
    return data


def compute_epoch(epoch):
    print(f"  Computing epoch {epoch}...", flush=True)
    acc = mm.MOTAccumulator(auto_id=True)
    for seq in val_seqs:
        gt = load_gt(seq)
        pred = load_pred(seq, epoch)
        all_frames = sorted(set(gt.keys()) | set(pred.keys()))
        for frame in all_frames:
            gt_objs = gt.get(frame, [])
            pred_objs = pred.get(frame, [])
            gt_ids = [str(o[0]) for o in gt_objs]
            pred_ids = [str(o[0]) for o in pred_objs]
            gt_boxes = np.array([[o[1], o[2], o[3], o[4]] for o in gt_objs]) if gt_objs else np.empty((0, 4))
            pred_boxes = np.array([[o[1], o[2], o[3], o[4]] for o in pred_objs]) if pred_objs else np.empty((0, 4))
            if len(gt_ids) == 0 and len(pred_ids) == 0:
                continue
            dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            acc.update(gt_ids, pred_ids, dists)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_matches', 'num_false_positives', 'num_misses', 'num_switches', 'idf1', 'precision', 'recall', 'mota'], name='acc')
    r = summary.iloc[0]
    tp = int(r['num_matches'])
    fp = int(r['num_false_positives'])
    fn = int(r['num_misses'])
    idsw = int(r['num_switches'])
    mota = r['mota'] * 100
    idf1 = r['idf1'] * 100
    prec = r['precision'] * 100
    rec = r['recall'] * 100
    return mota, idf1, prec, rec, tp, fp, fn, idsw


results = []
header = f"{'Ep':>3}  {'MOTA':>7}  {'IDF1':>7}  {'Prec':>7}  {'Rec':>7}  {'TP':>8}  {'FP':>7}  {'FN':>8}  {'IDsw':>7}"
print(header, flush=True)

for ep in range(9):
    mota, idf1, prec, rec, tp, fp, fn, idsw = compute_epoch(ep)
    row = f"{ep:>3}  {mota:>7.2f}  {idf1:>7.2f}  {prec:>7.2f}  {rec:>7.2f}  {tp:>8}  {fp:>7}  {fn:>8}  {idsw:>7}"
    print(row, flush=True)
    results.append(row)

print("\nDone.", flush=True)
