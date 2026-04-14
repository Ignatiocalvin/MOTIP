#!/usr/bin/env python3
"""
Compute MOTA / IDF1 / Prec / Rec for every experiment in outputs/,
then write an updated visualizations/EPOCH_METRICS_ALL_MODELS.md.

Supports two modes:
  - P-DESTRE: compute metrics from per-sequence tracker .txt files using motmetrics
  - DanceTrack: read pre-computed metrics from TrackEval pedestrian_summary.txt

GT format  : frame, tid, x, y, w, h, ...   (tid==-1 → ignore)
Pred format: frame, tid, x, y, w, h, ...
"""

import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from collections import defaultdict
import numpy as np

# motmetrics uses np.asfarray which was removed in NumPy 2.0 — patch before import
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm

# Import run_hota from sibling module via explicit path (no __init__.py in evaluation/)
import importlib.util as _ilu
_hota_spec = _ilu.spec_from_file_location(
    "compute_hota_pdestre",
    str(Path(__file__).parent / "compute_hota_pdestre.py"),
)
_hota_mod = _ilu.module_from_spec(_hota_spec)
_hota_spec.loader.exec_module(_hota_mod)
run_hota = _hota_mod.run_hota

# ── paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
GT_DIR = ROOT / "data" / "P-DESTRE" / "annotations"
SPLIT  = ROOT / "data" / "P-DESTRE" / "splits"
OUT_MD = Path(__file__).parent / "EPOCH_METRICS_ALL_MODELS.md"
CACHE  = Path(__file__).parent / ".metrics_cache.json"


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))

# ── P-DESTRE experiments: (output_folder, val_split, display_label) ──────────
EXPERIMENTS = [
    # R50 baseline
    ("r50_motip_pdestre_base_fold0",
     "val_0",
     "R50-MOTIP Base (fold 0)"),

    # R50 2 concepts (original completed run, SAMPLE_LENGTHS=5, CONCEPT_DIM=256)
    ("r50_base_motip_2concepts_fold_0",
     "val_0",
     "R50-MOTIP 2 Concepts (fold 0, orig)"),

    # R50 2 concepts v2 (SAMPLE_LENGTHS=15, CONCEPT_DIM=64, still training)
    ("r50_motip_pdestre_2concepts_fold_0_v2",
     "val_0",
     "R50-MOTIP 2 Concepts (fold 0, v2)"),

    # R50 3 concepts
    ("r50_motip_pdestre_3concepts_fold_0",
     "val_0",
     "R50-MOTIP 3 Concepts (fold 0)"),

    # R50 7 concepts learnable fold 0 (val_1)
    ("r50_motip_pdestre_7concepts_learnable_v2_fold_0",
     "val_1",
     "R50-MOTIP 7C-LW (fold 0)"),

    # R50 7 concepts learnable fold 1 (only epoch_9 eval)
    ("r50_motip_pdestre_7concepts_learnable_fold_1",
     "val_1",
     "R50-MOTIP 7C-LW (fold 1)"),

    # RF-DETR base (first completed run)
    ("rfdetr_large_motip_pdestre_base_fold0",
     "val_0",
     "RF-DETR-L Base (fold 0)"),

    # RF-DETR base v2
    ("rfdetr_large_motip_pdestre_base_fold0_v2",
     "val_0",
     "RF-DETR-L Base (fold 0, v2)"),

    # RF-DETR base v3
    ("rfdetr_large_motip_pdestre_base_fold0_v3",
     "val_0",
     "RF-DETR-L Base (fold 0, v3)"),

    # RF-DETR 2 concepts
    ("rfdetr_large_motip_pdestre_2concepts_fold0",
     "val_0",
     "RF-DETR-L 2 Concepts (fold 0)"),

    # RF-DETR 7C learnable (no-LW variant stored as 7concepts_lw)
    ("rfdetr_large_motip_pdestre_7concepts_lw_fold0",
     "val_0",
     "RF-DETR-L 7C-LW (fold 0)"),

    # RF-DETR 7C learnable fold0 (learnable)
    ("rfdetr_large_motip_pdestre_7concepts_learnable_fold0",
     "val_0",
     "RF-DETR-L 7C-Learnable (fold 0)"),

    # RF-DETR base (small) fold0
    ("rfdetr_base_motip_pdestre_base_fold0",
     "val_0",
     "RF-DETR-B Base (fold 0)"),
]

# ── DanceTrack experiments: (output_folder, eval_during_train_subpath, display_label) ─
# These read pre-computed TrackEval pedestrian_summary.txt instead of motmetrics.
DANCETRACK_EXPERIMENTS = [
    # R50 + SAM concepts, sample_length=5 (completed)
    ("r50_deformable_detr_motip_sam_concepts_sl5_dancetrack",
     "train/eval_during_train",
     "R50-MOTIP + SAM Concepts SL5 (DanceTrack)"),

    # R50 + SAM concepts, sample_length=15 (default)
    ("r50_deformable_detr_motip_sam_concepts_dancetrack",
     "train/eval_during_train",
     "R50-MOTIP + SAM Concepts (DanceTrack)"),

    # RF-DETR-B + SAM concepts
    ("rfdetr_base_motip_sam_concept_dancetrack/rfdetr_base_motip_sam_concept_dancetrack",
     "train/eval_during_train",
     "RF-DETR-B + SAM Concepts (DanceTrack)"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_gt(seq_name: str):
    """Return {frame_id: [(tid, x, y, w, h), ...]} from GT annotation."""
    p = GT_DIR / f"{seq_name}.txt"
    if not p.exists():
        return {}
    data = defaultdict(list)
    for line in p.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        fid, tid = int(parts[0]), int(parts[1])
        if tid < 0:
            continue
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        if w <= 0 or h <= 0:
            continue
        data[fid].append((tid, x, y, w, h))
    return dict(data)


def load_pred(pred_file: Path):
    """Return {frame_id: [(tid, x, y, w, h), ...]} from tracker output."""
    if not pred_file.exists():
        return {}
    data = defaultdict(list)
    for line in pred_file.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        fid, tid = int(parts[0]), int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        if w <= 0 or h <= 0:
            continue
        data[fid].append((tid, x, y, w, h))
    return dict(data)


def compute_metrics(tracker_dir: Path, val_seqs: list):
    """Compute aggregate MOT metrics over val_seqs using motmetrics."""
    acc = mm.MOTAccumulator(auto_id=True)

    for seq in val_seqs:
        gt = load_gt(seq)
        pred = load_pred(tracker_dir / f"{seq}.txt")

        all_frames = sorted(set(gt.keys()) | set(pred.keys()))
        for fid in all_frames:
            gt_rows  = gt.get(fid, [])
            pr_rows  = pred.get(fid, [])

            gt_ids   = [r[0] for r in gt_rows]
            gt_boxes = np.array([[r[1], r[2], r[3], r[4]] for r in gt_rows], dtype=float) if gt_rows else np.empty((0, 4))
            pr_ids   = [r[0] for r in pr_rows]
            pr_boxes = np.array([[r[1], r[2], r[3], r[4]] for r in pr_rows], dtype=float) if pr_rows else np.empty((0, 4))

            if len(gt_rows) == 0 and len(pr_rows) == 0:
                continue

            # IoU distance matrix
            if len(gt_rows) > 0 and len(pr_rows) > 0:
                dist = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
            else:
                dist = mm.distances.iou_matrix(
                    gt_boxes if len(gt_rows) > 0 else np.empty((0, 4)),
                    pr_boxes if len(pr_rows) > 0 else np.empty((0, 4)),
                    max_iou=0.5,
                )

            acc.update(gt_ids, pr_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        "num_frames", "mota", "idf1", "precision", "recall",
        "num_matches", "num_misses", "num_false_positives", "num_switches",
    ], name="agg")

    r = summary.iloc[0]
    return {
        "mota":  float(r["mota"])  * 100,
        "idf1":  float(r["idf1"])  * 100,
        "prec":  float(r["precision"]) * 100,
        "rec":   float(r["recall"]) * 100,
        "tp":    int(r["num_matches"]),
        "fn":    int(r["num_misses"]),
        "fp":    int(r["num_false_positives"]),
        "idsw":  int(r["num_switches"]),
    }


def fmt(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


# ── main ─────────────────────────────────────────────────────────────────────

def find_eval_base(exp_path: Path):
    """Find eval_during_train dir — may be under train/ or directly in exp_path."""
    for subpath in ["train/eval_during_train", "eval_during_train"]:
        candidate = exp_path / subpath
        if candidate.exists():
            return candidate
    return None


def parse_dancetrack_summary(summary_file: Path):
    """Parse TrackEval pedestrian_summary.txt → dict of metrics."""
    lines = summary_file.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    headers = lines[0].split()
    values = lines[1].split()
    d = dict(zip(headers, values))
    return {
        "hota":  float(d.get("HOTA", 0)),
        "deta":  float(d.get("DetA", 0)),
        "assa":  float(d.get("AssA", 0)),
        "mota":  float(d.get("MOTA", 0)),
        "idf1":  float(d.get("IDF1", 0)),
        "motp":  float(d.get("MOTP", 0)),
    }


def main():
    pdestre_sections = []
    dancetrack_sections = []
    cache = load_cache()

    # ── P-DESTRE experiments ──────────────────────────────────────────────────
    for (exp_folder, val_split, label) in EXPERIMENTS:
        exp_path = ROOT / "outputs" / exp_folder
        eval_base = find_eval_base(exp_path)

        if eval_base is None:
            print(f"[SKIP] {exp_folder} — no eval_during_train dir")
            continue

        # load val sequences
        split_file = SPLIT / f"{val_split}.txt"
        val_seqs = [l.strip().replace(".txt", "") for l in split_file.read_text().splitlines() if l.strip()]

        # find all evaluated epochs
        epoch_dirs = sorted(
            [d for d in eval_base.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
            key=lambda d: int(d.name.split("_")[1]),
        )

        if not epoch_dirs:
            print(f"[SKIP] {exp_folder} — no epoch dirs")
            continue

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  path: outputs/{exp_folder}")
        print(f"  val : {val_split}  ({len(val_seqs)} seqs)  |  epochs: {len(epoch_dirs)}")
        print(f"{'='*60}")
        print(f"  {'Epoch':>6}  {'HOTA':>7}  {'DetA':>7}  {'AssA':>7}  {'MOTA':>7}  {'IDF1':>7}  {'Prec':>7}  {'Rec':>7}  {'IDSW':>6}")
        print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")

        rows = []
        best_mota = -999
        best_ep   = -1

        for ep_dir in epoch_dirs:
            ep_num = int(ep_dir.name.split("_")[1])
            tracker_dir = ep_dir / "tracker"

            if not tracker_dir.exists():
                print(f"  {ep_num:>6}  (no tracker dir)")
                continue

            # count non-empty tracker files for this val split
            present = sum(1 for s in val_seqs if (tracker_dir / f"{s}.txt").exists())
            if present == 0:
                print(f"  {ep_num:>6}  (no tracker files for {val_split})")
                continue

            try:
                cache_key = f"{exp_folder}/{val_split}/epoch_{ep_num}"
                if cache_key in cache and "hota" in cache[cache_key]:
                    m = cache[cache_key]
                    print(f"  {ep_num:>6}  {fmt(m.get('hota')):>7}  {fmt(m.get('deta')):>7}  "
                          f"{fmt(m.get('assa')):>7}  {fmt(m['mota']):>7}  {fmt(m['idf1']):>7}  "
                          f"{fmt(m['prec']):>7}  {fmt(m['rec']):>7}  {m['idsw']:>6}  (cached)")
                else:
                    m = compute_metrics(tracker_dir, val_seqs)
                    try:
                        hota_m = run_hota(tracker_dir, val_split)
                        # Only add HOTA/DetA/AssA keys; do NOT overwrite mota/idf1
                        for k in ("hota", "deta", "assa", "clear_mota", "clear_idf1"):
                            if k in hota_m:
                                m[k] = hota_m[k]
                    except Exception as he:
                        print(f"  {ep_num:>6}  HOTA error: {he}")
                        m["hota"] = m["deta"] = m["assa"] = None
                    cache[cache_key] = m
                    save_cache(cache)
                    print(f"  {ep_num:>6}  {fmt(m.get('hota')):>7}  {fmt(m.get('deta')):>7}  "
                          f"{fmt(m.get('assa')):>7}  {fmt(m['mota']):>7}  {fmt(m['idf1']):>7}  "
                          f"{fmt(m['prec']):>7}  {fmt(m['rec']):>7}  {m['idsw']:>6}")
            except Exception as e:
                print(f"  {ep_num:>6}  ERROR: {e}")
                continue

            rows.append((ep_num, m))
            if m["mota"] > best_mota:
                best_mota = m["mota"]
                best_ep   = ep_num

        if rows:
            best_by_hota = max(rows, key=lambda r: r[1].get("hota") or -999)
            print(f"\n  ★ Best (MOTA): Epoch {best_ep}  MOTA={best_mota:.2f}%")
            print(f"  ★ Best (HOTA): Epoch {best_by_hota[0]}  HOTA={fmt(best_by_hota[1].get('hota'))}%")

        pdestre_sections.append((label, exp_folder, val_split, rows))

    # ── DanceTrack experiments ────────────────────────────────────────────────
    for (exp_folder, eval_subpath, label) in DANCETRACK_EXPERIMENTS:
        exp_path = ROOT / "outputs" / exp_folder
        eval_base = exp_path / eval_subpath

        if not eval_base.exists():
            print(f"[SKIP] {exp_folder} — no {eval_subpath} dir")
            continue

        epoch_dirs = sorted(
            [d for d in eval_base.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
            key=lambda d: int(d.name.split("_")[1]),
        )

        if not epoch_dirs:
            print(f"[SKIP] {exp_folder} — no epoch dirs")
            continue

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  path: outputs/{exp_folder}")
        print(f"  epochs: {len(epoch_dirs)}")
        print(f"{'='*60}")
        print(f"  {'Epoch':>6}  {'HOTA':>7}  {'DetA':>7}  {'AssA':>7}  {'MOTA':>7}  {'IDF1':>7}")
        print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

        rows = []
        best_hota = -999
        best_ep   = -1

        for ep_dir in epoch_dirs:
            ep_num = int(ep_dir.name.split("_")[1])
            summary_file = ep_dir / "tracker" / "pedestrian_summary.txt"

            if not summary_file.exists():
                print(f"  {ep_num:>6}  (no pedestrian_summary.txt)")
                continue

            try:
                m = parse_dancetrack_summary(summary_file)
                if m is None:
                    print(f"  {ep_num:>6}  (empty summary)")
                    continue
            except Exception as e:
                print(f"  {ep_num:>6}  ERROR: {e}")
                continue

            print(f"  {ep_num:>6}  {fmt(m['hota']):>7}  {fmt(m['deta']):>7}  "
                  f"{fmt(m['assa']):>7}  {fmt(m['mota']):>7}  {fmt(m['idf1']):>7}")

            rows.append((ep_num, m))
            if m["hota"] > best_hota:
                best_hota = m["hota"]
                best_ep   = ep_num

        if rows:
            print(f"\n  ★ Best: Epoch {best_ep}  HOTA={best_hota:.2f}%")

        dancetrack_sections.append((label, exp_folder, rows))

    # ── write markdown ────────────────────────────────────────────────────────
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Epoch Metrics — All Models",
        "",
        "> Auto-generated by `evaluation/compute_all_metrics.py`.",
        "> P-DESTRE HOTA/DetA/AssA via TrackEval (IoU ≥ 0.5). MOTA/IDF1/Prec/Rec via motmetrics.",
        "> DanceTrack metrics from TrackEval (pedestrian_summary.txt).",
        "",
    ]

    # ── P-DESTRE sections ─────────────────────────────────────────────────────
    for (label, exp_folder, val_split, rows) in pdestre_sections:
        lines.append(f"## {label}")
        lines.append(f"**Path:** `outputs/{exp_folder}`  ")
        lines.append(f"**Val split:** `{val_split}`")
        lines.append("")
        lines.append("| Epoch | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | IDF1 (%) | Prec (%) | Rec (%) | IDSW |")
        lines.append("|------:|---------:|---------:|---------:|---------:|---------:|---------:|--------:|-----:|")

        best_mota = -999
        best_ep   = -1
        for ep_num, m in rows:
            lines.append(
                f"| {ep_num} "
                f"| {fmt(m.get('hota'))} "
                f"| {fmt(m.get('deta'))} "
                f"| {fmt(m.get('assa'))} "
                f"| {fmt(m['mota'])} "
                f"| {fmt(m['idf1'])} "
                f"| {fmt(m['prec'])} "
                f"| {fmt(m['rec'])} "
                f"| {m['idsw']} |"
            )
            if m["mota"] > best_mota:
                best_mota = m["mota"]
                best_ep   = ep_num

        if rows:
            best_by_hota = max(rows, key=lambda r: r[1].get("hota") or -999)
            lines.append("")
            lines.append(f"**★ Best (MOTA): Epoch {best_ep} — MOTA = {best_mota:.2f}%**  ")
            lines.append(f"**★ Best (HOTA): Epoch {best_by_hota[0]} — HOTA = {fmt(best_by_hota[1].get('hota'))}%**")
        else:
            lines.append("")
            lines.append("_No evaluated epochs found._")
        lines.append("")

    # ── DanceTrack sections ───────────────────────────────────────────────────
    if dancetrack_sections:
        lines.append("---")
        lines.append("")
        lines.append("# DanceTrack Experiments")
        lines.append("")
        lines.append("> Metrics from TrackEval `pedestrian_summary.txt`.")
        lines.append("")

    for (label, exp_folder, rows) in dancetrack_sections:
        lines.append(f"## {label}")
        lines.append(f"**Path:** `outputs/{exp_folder}`  ")
        lines.append(f"**Val split:** `val`")
        lines.append("")
        lines.append("| Epoch | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | IDF1 (%) |")
        lines.append("|------:|---------:|---------:|---------:|---------:|---------:|")

        best_hota = -999
        best_ep   = -1
        for ep_num, m in rows:
            lines.append(
                f"| {ep_num} "
                f"| {fmt(m['hota'])} "
                f"| {fmt(m['deta'])} "
                f"| {fmt(m['assa'])} "
                f"| {fmt(m['mota'])} "
                f"| {fmt(m['idf1'])} |"
            )
            if m["hota"] > best_hota:
                best_hota = m["hota"]
                best_ep   = ep_num

        if rows:
            lines.append("")
            lines.append(f"**★ Best: Epoch {best_ep} — HOTA = {best_hota:.2f}%**")
        else:
            lines.append("")
            lines.append("_No evaluated epochs found._")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"\n\n✅ Written: {OUT_MD}")


if __name__ == "__main__":
    main()
