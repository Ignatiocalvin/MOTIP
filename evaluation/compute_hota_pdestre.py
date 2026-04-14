"""compute_hota_pdestre.py

Compute HOTA / DetA / AssA for P-DESTRE tracker output files.
Uses the TrackEval library bundled at MOTIP/TrackEval/.

P-DESTRE annotation format: frame,id,x,y,h,w,conf,...
MOT Challenge (TrackEval) format: frame,id,x,y,w,h,conf,class,vis

Columns 4 & 5 are swapped when writing MOT-format files.

Usage (single tracker directory):
    python evaluation/compute_hota_pdestre.py \\
        --tracker_dir outputs/r50_motip_pdestre_base_fold0/train/eval_during_train/epoch_2/tracker \\
        --split val_0

Usage (batch — all PDESTRE experiments in outputs/):
    python evaluation/compute_hota_pdestre.py --all --split val_0
"""

import sys
import os
import argparse
import tempfile
import shutil
from pathlib import Path
from configparser import ConfigParser

# ── project root on path ──────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
MOTIP_ROOT  = SCRIPT_DIR.parent
TRACKEVAL   = MOTIP_ROOT / "TrackEval"

sys.path.insert(0, str(MOTIP_ROOT))
sys.path.insert(0, str(TRACKEVAL))

import trackeval  # noqa: E402  (must come after sys.path update)

# ── P-DESTRE paths ─────────────────────────────────────────────────────────────
GT_DIR    = MOTIP_ROOT / "data" / "P-DESTRE" / "annotations"
SPLIT_DIR = MOTIP_ROOT / "data" / "P-DESTRE" / "splits"
IMG_DIR   = MOTIP_ROOT / "data" / "P-DESTRE" / "images"

FRAME_RATE  = 30
# Fallback resolution if PIL / image files are unavailable
DEFAULT_W   = 3840
DEFAULT_H   = 2160


# ── helpers ───────────────────────────────────────────────────────────────────

def get_sequences(split: str) -> list:
    """Return sorted list of sequence names for a P-DESTRE split."""
    split_file = SPLIT_DIR / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    seqs = []
    for line in split_file.read_text().splitlines():
        seq = line.strip().replace(".txt", "")
        if seq:
            seqs.append(seq)
    return sorted(set(seqs))


def get_seq_length(seq_name: str) -> int:
    """
    Return the sequence length (total number of frames).
    Prefers counting image files (most accurate), falls back to max GT frame id.
    """
    img_dir = IMG_DIR / seq_name / "img1"
    if img_dir.exists():
        count = sum(1 for f in img_dir.iterdir() if f.suffix == ".jpg")
        if count > 0:
            return count
    # Fallback: max frame id in GT annotation
    gt_file = GT_DIR / f"{seq_name}.txt"
    max_frame = 0
    for line in gt_file.read_text().splitlines():
        parts = line.strip().split(",")
        if parts:
            try:
                f = int(parts[0])
                if f > max_frame:
                    max_frame = f
            except ValueError:
                pass
    return max_frame


def get_seq_resolution(seq_name: str):
    """Return (width, height) for the sequence, reading the first image."""
    img_dir = IMG_DIR / seq_name / "img1"
    if img_dir.exists():
        imgs = sorted(img_dir.iterdir())
        if imgs:
            try:
                from PIL import Image
                with Image.open(imgs[0]) as im:
                    return im.size  # (width, height)
            except Exception:
                pass
    return DEFAULT_W, DEFAULT_H


def write_seqinfo(seq_dir: Path, seq_name: str, seq_length: int,
                  width: int, height: int):
    """Write seqinfo.ini into seq_dir."""
    seq_dir.mkdir(parents=True, exist_ok=True)
    cfg = ConfigParser()
    cfg["Sequence"] = {
        "name":      seq_name,
        "imDir":     "img1",
        "frameRate": str(FRAME_RATE),
        "seqLength": str(seq_length),
        "imWidth":   str(width),
        "imHeight":  str(height),
        "imExt":     ".jpg",
    }
    with (seq_dir / "seqinfo.ini").open("w") as fh:
        cfg.write(fh)


def convert_gt_to_mot(seq_name: str, out_path: Path):
    """
    Convert P-DESTRE GT annotation → MOT-format file.

    P-DESTRE columns: frame, id, x, y, h, w, conf, ...
    MOT-format:       frame, id, x, y, w, h, conf, class, visibility
    """
    gt_file = GT_DIR / f"{seq_name}.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"GT annotation not found: {gt_file}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in gt_file.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        try:
            fid = int(parts[0])
            tid = int(parts[1])
        except ValueError:
            continue
        if tid < 0:
            continue  # Skip distractors / crowd regions
        x = float(parts[2])
        y = float(parts[3])
        h = float(parts[4])  # P-DESTRE: h comes before w
        w = float(parts[5])  # P-DESTRE: w comes after h
        if w <= 0 or h <= 0:
            continue
        # conf: column 6 (1 for valid GT tracks, -1 for distractors already filtered)
        try:
            conf = float(parts[6]) if len(parts) > 6 and parts[6] not in ("-1", "") else 1.0
        except (ValueError, IndexError):
            conf = 1.0
        conf = max(conf, 0.0)
        # MOT format: frame, id, x, y, w, h, conf, class(1=pedestrian), visibility
        lines.append(f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},1,1")

    out_path.write_text("\n".join(lines))


def convert_tracker_to_mot(tracker_file: Path, out_path: Path):
    """
    Convert P-DESTRE tracker output → MOT-format file.

    Tracker columns: frame, id, x, y, h, w, conf, -1, -1, -1[,]
    MOT-format:      frame, id, x, y, w, h, conf, -1, -1, -1
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not tracker_file.exists():
        out_path.write_text("")
        return
    lines = []
    for line in tracker_file.read_text().splitlines():
        parts = line.strip().rstrip(",").split(",")
        if len(parts) < 6:
            continue
        try:
            fid = int(parts[0])
            tid = int(parts[1])
        except ValueError:
            continue
        x = float(parts[2])
        y = float(parts[3])
        h = float(parts[4])  # tracker: h comes before w
        w = float(parts[5])  # tracker: w comes after h
        if w <= 0 or h <= 0:
            continue
        try:
            conf = float(parts[6]) if len(parts) > 6 else 1.0
        except (ValueError, IndexError):
            conf = 1.0
        # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
        lines.append(f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1")
    out_path.write_text("\n".join(lines))


# ── main evaluation function ──────────────────────────────────────────────────

def run_hota(tracker_dir: Path, split: str,
             work_dir: Path = None, tracker_name: str = "model",
             keep_work_dir: bool = False) -> dict:
    """
    Prepare GT + tracker files, run TrackEval, and return HOTA/DetA/AssA/MOTA/IDF1.

    Returns dict with keys: hota, deta, assa, mota, idf1 (all as floats, percentages).
    """
    cleanup = (work_dir is None) and not keep_work_dir
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="trackeval_pdestre_"))

    try:
        gt_folder  = work_dir / "gt"
        trk_folder = work_dir / "trackers"

        sequences   = get_sequences(split)
        seq_lengths = {}

        for seq in sequences:
            # ── GT ──────────────────────────────────────────────────────────
            gt_seq_dir = gt_folder / seq
            convert_gt_to_mot(seq, gt_seq_dir / "gt" / "gt.txt")

            seq_len      = get_seq_length(seq)
            width, height = get_seq_resolution(seq)
            seq_lengths[seq] = seq_len

            write_seqinfo(gt_seq_dir, seq, seq_len, width, height)

            # ── Tracker ──────────────────────────────────────────────────────
            trk_seq_file = trk_folder / tracker_name / "data" / f"{seq}.txt"
            convert_tracker_to_mot(tracker_dir / f"{seq}.txt", trk_seq_file)

        # ── TrackEval config ─────────────────────────────────────────────────
        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config.update({
            "USE_PARALLEL":          False,
            "PRINT_RESULTS":         False,
            "PRINT_CONFIG":          False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY":        True,
            "OUTPUT_DETAILED":       False,
            "PLOT_CURVES":           False,
            "TIME_PROGRESS":         False,
        })

        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update({
            "GT_FOLDER":       str(gt_folder),
            "TRACKERS_FOLDER": str(trk_folder),
            "TRACKERS_TO_EVAL": [tracker_name],
            "SEQ_INFO":        seq_lengths,
            "DO_PREPROC":      False,
            "TRACKER_SUB_FOLDER": "data",
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK":       "PDESTRE",
            "SPLIT_TO_EVAL":   split,
            "GT_LOC_FORMAT":   "{gt_folder}/{seq}/gt/gt.txt",
            "SKIP_SPLIT_FOL":  True,
            "PRINT_CONFIG":    False,
        })

        metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5,
                          "PRINT_CONFIG": False}

        evaluator    = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        for metric_cls in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                           trackeval.metrics.Identity]:
            if metric_cls.get_name() in metrics_config["METRICS"]:
                metrics_list.append(metric_cls(metrics_config))

        results, _ = evaluator.evaluate(dataset_list, metrics_list)

        # ── Extract combined results ─────────────────────────────────────────
        combined = results["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]

        hota_res  = combined["HOTA"]
        clear_res = combined["CLEAR"]
        id_res    = combined["Identity"]

        import numpy as np
        return {
            # HOTA, DetA, AssA are arrays across alpha thresholds → mean
            "hota": float(np.mean(hota_res["HOTA"])) * 100,
            "deta": float(np.mean(hota_res["DetA"])) * 100,
            "assa": float(np.mean(hota_res["AssA"])) * 100,
            # Use distinct keys so caller's motmetrics MOTA/IDF1 are not overwritten
            "clear_mota": float(clear_res["MOTA"]) * 100,
            "clear_idf1": float(id_res["IDF1"]) * 100,
        }

    finally:
        if cleanup and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute HOTA/DetA/AssA for a P-DESTRE tracker output directory."
    )
    parser.add_argument(
        "--tracker_dir", type=str, default=None,
        help="Path to tracker directory containing <seq>.txt files",
    )
    parser.add_argument(
        "--split", type=str, default="val_0",
        help='P-DESTRE split name, e.g. "val_0", "Test_0"',
    )
    parser.add_argument(
        "--work_dir", type=str, default=None,
        help="Working directory for converted files (default: auto temp dir)",
    )
    parser.add_argument(
        "--keep_work_dir", action="store_true",
        help="Keep the working directory after evaluation (useful for debugging)",
    )
    parser.add_argument(
        "--all", dest="run_all", action="store_true",
        help="Evaluate all PDESTRE experiments found in outputs/",
    )
    args = parser.parse_args()

    if args.run_all:
        # Find all tracker directories for the given split
        outputs_dir = MOTIP_ROOT / "outputs"
        experiments = [
            d for d in outputs_dir.iterdir()
            if d.is_dir() and "pdestre" in d.name.lower() and "dancetrack" not in d.name.lower()
        ]
        if not experiments:
            print("No P-DESTRE experiments found in outputs/")
            sys.exit(1)

        print(f"Found {len(experiments)} P-DESTRE experiments\n")
        for exp in sorted(experiments):
            # Find epoch tracker dirs that have the given split
            eval_base = None
            for candidate in ["train/eval_during_train", "eval_during_train",
                               f"eval/PDESTRE_{args.split}"]:
                p = exp / candidate
                if p.exists():
                    eval_base = p
                    break
            if eval_base is None:
                print(f"  {exp.name}: no eval directory found, skipping")
                continue

            # Collect all epoch dirs that have tracker files for the split
            epoch_dirs = sorted(eval_base.glob("*/tracker")) if "eval_during_train" in str(eval_base) \
                else [eval_base / "tracker"]

            for tracker_dir in sorted(epoch_dirs):
                if not tracker_dir.exists():
                    continue
                txt_files = list(tracker_dir.glob("*.txt"))
                if len(txt_files) < 3:
                    continue  # Skip incomplete runs
                try:
                    res = run_hota(tracker_dir, args.split,
                                   work_dir=Path(args.work_dir) if args.work_dir else None,
                                   keep_work_dir=args.keep_work_dir)
                    label = f"{exp.name}/{tracker_dir.parent.name}"
                    print(f"  {label}")
                    print(f"    HOTA={res['hota']:.2f}  DetA={res['deta']:.2f}  "
                          f"AssA={res['assa']:.2f}  MOTA={res['mota']:.2f}  IDF1={res['idf1']:.2f}")
                except Exception as e:
                    print(f"  {exp.name}/{tracker_dir.parent.name}: ERROR — {e}")
        return

    # ── Single-directory mode ─────────────────────────────────────────────────
    if args.tracker_dir is None:
        parser.error("Provide --tracker_dir or use --all")

    tracker_dir = Path(args.tracker_dir)
    if not tracker_dir.exists():
        print(f"Tracker directory not found: {tracker_dir}")
        sys.exit(1)

    work_dir     = Path(args.work_dir) if args.work_dir else None
    tracker_name = tracker_dir.parent.name  # e.g. "epoch_2"

    print(f"Tracker : {tracker_dir}")
    print(f"Split   : {args.split}")

    results = run_hota(tracker_dir, args.split,
                       work_dir=work_dir, tracker_name=tracker_name,
                       keep_work_dir=args.keep_work_dir)

    print(f"\nResults (split={args.split}):")
    print(f"  HOTA : {results['hota']:.2f}")
    print(f"  DetA : {results['deta']:.2f}")
    print(f"  AssA : {results['assa']:.2f}")
    print(f"  MOTA : {results['mota']:.2f}")
    print(f"  IDF1 : {results['idf1']:.2f}")


if __name__ == "__main__":
    main()
