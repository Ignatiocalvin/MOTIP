#!/usr/bin/env python3
"""
Download DanceTrack dataset from Hugging Face to ./data/DanceTrack/

Usage:
    # From MOTIP root directory:
    python scripts/download_dancetrack.py
    python scripts/download_dancetrack.py --splits train val test
    
    # Or with custom output:
    python scripts/download_dancetrack.py --output-dir /path/to/data/DanceTrack

The dataset is downloaded from: https://huggingface.co/datasets/noahcao/dancetrack

Expected output structure:
    ./data/DanceTrack/
    ├── train/
    │   ├── dancetrack0001/
    │   │   ├── img1/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── gt/
    │   │   │   └── gt.txt
    │   │   └── seqinfo.ini
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
"""

import os
import sys
import shutil
import argparse
import zipfile

# Detect MOTIP root directory (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIP_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else os.getcwd()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download DanceTrack dataset from Hugging Face",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(MOTIP_ROOT, "data", "DanceTrack"),
        help="Output directory",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to download",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token (if dataset requires authentication)",
    )
    return parser.parse_args()


def check_existing(output_dir: str, splits: list) -> dict:
    """
    Check which splits are already fully downloaded.
    Returns dict: {split: True/False}
    """
    status = {}
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            status[split] = False
            continue
        # Check if there's at least one valid sequence
        sequences = [
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
            and os.path.exists(os.path.join(split_dir, d, "seqinfo.ini"))
        ]
        status[split] = len(sequences) > 0
    return status


def reorganize_structure(downloaded_dir: str, output_dir: str, splits: list):
    """
    After snapshot_download, reorganize the file structure if needed.
    The HF repo structure may differ from MOTIP's expected structure.
    Handles common cases:
      - Files already in {split}/dancetrackXXXX/ → just copy
      - Files in dancetrack/{split}/dancetrackXXXX/ → flatten one level
    """
    os.makedirs(output_dir, exist_ok=True)

    # First: check if it already has the correct structure
    for split in splits:
        direct_split = os.path.join(downloaded_dir, split)
        if os.path.isdir(direct_split):
            seqs = [d for d in os.listdir(direct_split) if os.path.isdir(os.path.join(direct_split, d))]
            if seqs:
                print(f"  Structure for {split}: already at top level, copying...")
                target_split = os.path.join(output_dir, split)
                if direct_split != target_split:
                    shutil.copytree(direct_split, target_split, dirs_exist_ok=True)
                continue

        # Check nested dancetrack/split/ structure
        nested_split = os.path.join(downloaded_dir, "dancetrack", split)
        if os.path.isdir(nested_split):
            print(f"  Structure for {split}: nested at dancetrack/{split}/, copying to output...")
            target_split = os.path.join(output_dir, split)
            shutil.copytree(nested_split, target_split, dirs_exist_ok=True)
            continue

        print(f"  WARNING: Could not find {split} data in downloaded directory.")


def extract_archives(directory: str):
    """Extract any zip/tar archives found in the directory."""
    import tarfile
    for root, dirs, files in os.walk(directory):
        for f in files:
            fpath = os.path.join(root, f)
            if f.endswith(".zip"):
                print(f"  Extracting {fpath}...")
                with zipfile.ZipFile(fpath, "r") as zf:
                    zf.extractall(root)
                os.remove(fpath)
            elif f.endswith((".tar.gz", ".tgz", ".tar")):
                print(f"  Extracting {fpath}...")
                with tarfile.open(fpath) as tf:
                    tf.extractall(root)
                os.remove(fpath)


def verify_structure(output_dir: str, splits: list) -> bool:
    """Verify the output directory has the expected MOT structure."""
    all_ok = True
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  MISSING: {split_dir}")
            all_ok = False
            continue

        sequences = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
            and os.path.exists(os.path.join(split_dir, d, "seqinfo.ini"))
        ])

        if not sequences:
            print(f"  MISSING sequences in {split_dir}")
            all_ok = False
        else:
            # Check one sequence for proper structure
            sample_seq = os.path.join(split_dir, sequences[0])
            has_imgs = os.path.isdir(os.path.join(sample_seq, "img1"))
            has_gt = os.path.exists(os.path.join(sample_seq, "gt", "gt.txt")) if split != "test" else True
            if not has_imgs:
                print(f"  MISSING img1/ in {sample_seq}")
                all_ok = False
            else:
                n_imgs = len([f for f in os.listdir(os.path.join(sample_seq, "img1")) if f.endswith(".jpg")])
                print(f"  OK: {split}: {len(sequences)} sequences — sample '{sequences[0]}': {n_imgs} frames{'' if has_gt else ' (no gt.txt)'}")

    return all_ok


def generate_seqmaps(output_dir: str, splits: list):
    """Copy official {split}_seqmap.txt files (from scripts/) to the dataset directory.
    
    The seqmap files are stored as static assets in the same directory as this
    script, ensuring the exact official split assignments are always used.
    """
    for split in splits:
        src = os.path.join(SCRIPT_DIR, f"{split}_seqmap.txt")
        dst = os.path.join(output_dir, f"{split}_seqmap.txt")
        
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found — skipping seqmap for {split}")
            continue
        
        shutil.copy2(src, dst)
        print(f"  Copied {split}_seqmap.txt")


def main():
    args = parse_args()

    print("=" * 60)
    print("DanceTrack Dataset Download")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print()

    # Check what's already downloaded
    existing = check_existing(args.output_dir, args.splits)
    splits_to_download = [s for s in args.splits if not existing[s]]

    if not splits_to_download:
        print("All requested splits already exist. Verifying structure...")
        verify_structure(args.output_dir, args.splits)
        # Ensure seqmap files exist (may be missing from older downloads)
        print()
        print("Ensuring seqmap files exist...")
        generate_seqmaps(args.output_dir, args.splits)
        return

    for s, ok in existing.items():
        if ok:
            print(f"  {s}: already exists, skipping")
    print(f"  Downloading: {splits_to_download}")
    print()

    # Try importing huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    print("Downloading from Hugging Face (noahcao/dancetrack)...")
    print("This may take a while — DanceTrack is ~20GB for all splits.")
    print()

    try:
        # Download the full dataset repository
        downloaded_dir = snapshot_download(
            repo_id="noahcao/dancetrack",
            repo_type="dataset",
            local_dir=args.output_dir,
            token=args.token,
            cache_dir=args.cache_dir,
            ignore_patterns=["*.parquet"],   # skip parquet if raw files are present
        )
        print(f"Download complete: {downloaded_dir}")
    except Exception as e:
        print(f"snapshot_download failed: {e}")
        print()
        print("Trying alternative: load_dataset + extract...")
        try:
            _download_via_datasets(args.output_dir, args.splits, args.token)
        except Exception as e2:
            print(f"Alternative also failed: {e2}")
            print()
            print("Manual download instructions:")
            print("  1. Download from: https://huggingface.co/datasets/noahcao/dancetrack")
            print("  2. Or: pip install huggingface_hub && huggingface-cli download noahcao/dancetrack --repo-type dataset")
            print(f"  3. Extract to: {args.output_dir}/")
            print(f"     Structure: {args.output_dir}/train/dancetrackXXXX/img1/")
            sys.exit(1)

    # Extract archives if any
    print("Checking for archives to extract...")
    extract_archives(args.output_dir)

    # Reorganize if needed
    print("Verifying directory structure...")
    reorganize_structure(args.output_dir, args.output_dir, args.splits)

    # Final verification
    print()
    print("Final structure check:")
    ok = verify_structure(args.output_dir, args.splits)
    
    # Generate seqmap files (required by TrackEval)
    print()
    print("Generating seqmap files...")
    generate_seqmaps(args.output_dir, args.splits)
    
    print()
    if ok:
        print(f"Dataset ready at: {args.output_dir}")
        print()
        print("To use in training, the config DATA_ROOT should be:")
        parent = os.path.dirname(args.output_dir)
        print(f"  DATA_ROOT: {parent}/")
        print("  (DanceTrack sub-directory is added automatically)")
    else:
        print("WARNING: Some parts of the dataset may be missing or malformed.")
        print("Please check the output directory and retry if necessary.")


def _download_via_datasets(output_dir: str, splits: list, token: str):
    """Fallback: download with datasets library and write raw files."""
    from datasets import load_dataset
    from PIL import Image
    import io

    print("Downloading via 'datasets' library (slower, writes image files)...")
    dataset = load_dataset("noahcao/dancetrack", token=token)

    for split in splits:
        if split not in dataset:
            print(f"  Skipping {split} (not in dataset)")
            continue
        print(f"  Processing split: {split}")
        split_data = dataset[split]

        # Each example should have: sequence_name, frame_id, image (PIL), gt, seqinfo, etc.
        for example in split_data:
            seq_name = example.get("sequence_name", example.get("video_id", "unknown"))
            frame_id = example.get("frame_id", 0)

            seq_dir = os.path.join(output_dir, split, seq_name)
            img_dir = os.path.join(seq_dir, "img1")
            os.makedirs(img_dir, exist_ok=True)

            # Write image
            img = example.get("image")
            if img is not None:
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img))
                img_path = os.path.join(img_dir, f"{frame_id:08d}.jpg")
                img.save(img_path, "JPEG", quality=95)

            # Write seqinfo.ini if present
            seqinfo = example.get("seqinfo")
            if seqinfo:
                seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
                if not os.path.exists(seqinfo_path):
                    with open(seqinfo_path, "w") as f:
                        f.write(seqinfo)

        print(f"  Done {split}")


if __name__ == "__main__":
    main()
