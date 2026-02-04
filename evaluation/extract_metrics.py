#!/usr/bin/env python3
"""
Extract evaluation metrics from P-DESTRE tracking results

This script parses the TrackEval output and creates a standardized results.json file
that can be used by the visualization tools.

Usage:
    python extract_metrics.py --fold 0
    python extract_metrics.py --exp-prefix r50_motip_pdestre --fold 0
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def parse_trackeval_log(log_file: Path) -> dict:
    """Parse TrackEval log file to extract metrics"""
    metrics = {}
    
    if not log_file.exists():
        return metrics
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Look for metric patterns in the log
    # Common patterns from TrackEval output
    patterns = {
        'HOTA': r'HOTA\s*[:\|]\s*([0-9.]+)',
        'DetA': r'DetA\s*[:\|]\s*([0-9.]+)',
        'AssA': r'AssA\s*[:\|]\s*([0-9.]+)',
        'DetRe': r'DetRe\s*[:\|]\s*([0-9.]+)',
        'DetPr': r'DetPr\s*[:\|]\s*([0-9.]+)',
        'AssRe': r'AssRe\s*[:\|]\s*([0-9.]+)',
        'AssPr': r'AssPr\s*[:\|]\s*([0-9.]+)',
        'LocA': r'LocA\s*[:\|]\s*([0-9.]+)',
        'MOTA': r'MOTA\s*[:\|]\s*([0-9.]+)',
        'MOTP': r'MOTP\s*[:\|]\s*([0-9.]+)',
        'IDF1': r'IDF1\s*[:\|]\s*([0-9.]+)',
        'IDR': r'IDR\s*[:\|]\s*([0-9.]+)',
        'IDP': r'IDP\s*[:\|]\s*([0-9.]+)',
    }
    
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                metrics[metric_name] = float(match.group(1))
            except ValueError:
                pass
    
    return metrics


def find_latest_checkpoint_results(fold_dir: Path) -> Path:
    """Find the latest checkpoint evaluation results"""
    
    evaluate_dir = fold_dir / "evaluate"
    if not evaluate_dir.exists():
        return None
    
    # Look for checkpoint directories
    checkpoint_dirs = list(evaluate_dir.glob("*/P-DESTRE/*/checkpoint_*"))
    
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number and get the latest
    checkpoint_dirs.sort(key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else -1)
    
    return checkpoint_dirs[-1] if checkpoint_dirs else None


def parse_tracker_results(tracker_dir: Path) -> dict:
    """Parse individual tracking result files"""
    
    if not tracker_dir.exists():
        return {}
    
    # Count tracking statistics
    total_tracks = 0
    total_frames = 0
    
    for tracker_file in tracker_dir.glob("*.txt"):
        try:
            with open(tracker_file, 'r') as f:
                lines = f.readlines()
                
            # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
            unique_ids = set()
            frames = set()
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    frames.add(int(parts[0]))
                    unique_ids.add(int(parts[1]))
            
            total_tracks += len(unique_ids)
            total_frames += len(frames)
            
        except Exception:
            pass
    
    return {
        'total_tracks': total_tracks,
        'total_frames': total_frames,
        'num_sequences': len(list(tracker_dir.glob("*.txt")))
    }


def extract_metrics_from_fold(fold_dir: Path) -> dict:
    """Extract all available metrics from a fold directory"""
    
    print(f"Processing: {fold_dir.name}")
    
    metrics = {}
    
    # Find latest checkpoint results
    checkpoint_dir = find_latest_checkpoint_results(fold_dir)
    
    if checkpoint_dir is None:
        print(f"  ⚠ No evaluation results found")
        return metrics
    
    print(f"  Using: {checkpoint_dir.relative_to(fold_dir)}")
    
    # Parse log file
    log_file = checkpoint_dir / "log.txt"
    if log_file.exists():
        log_metrics = parse_trackeval_log(log_file)
        metrics.update(log_metrics)
        print(f"  ✓ Extracted {len(log_metrics)} metrics from log")
    
    # Parse tracker results
    tracker_dir = checkpoint_dir / "tracker"
    if tracker_dir.exists():
        tracker_stats = parse_tracker_results(tracker_dir)
        metrics.update(tracker_stats)
        print(f"  ✓ Parsed {tracker_stats.get('num_sequences', 0)} tracking sequences")
    
    # Check for existing results.json
    for results_file in [fold_dir / "results.json", checkpoint_dir / "results.json"]:
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    existing_metrics = json.load(f)
                metrics.update(existing_metrics)
                print(f"  ✓ Loaded existing results from {results_file.name}")
            except Exception as e:
                print(f"  ⚠ Could not parse {results_file}: {e}")
    
    return metrics


def save_metrics(metrics: dict, output_file: Path):
    """Save metrics to JSON file"""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    
    print(f"  ✓ Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract evaluation metrics from P-DESTRE tracking results'
    )
    
    parser.add_argument('--exp-prefix', type=str, default='r50_motip_pdestre',
                       help='Experiment name prefix')
    parser.add_argument('--fold', type=int, required=False,
                       help='Fold number to process')
    parser.add_argument('--outputs-dir', type=str, default='../outputs',
                       help='Directory containing fold outputs')
    parser.add_argument('--all-folds', action='store_true',
                       help='Process all available folds')
    
    args = parser.parse_args()
    
    # Validate: need either --fold or --all-folds
    if args.fold is None and not args.all_folds:
        parser.error('Either --fold or --all-folds is required')
    
    outputs_dir = Path(args.outputs_dir)
    
    print("="*70)
    print("Extracting Evaluation Metrics")
    print("="*70 + "\n")
    
    if args.all_folds:
        # Process all folds
        fold_dirs = sorted(outputs_dir.glob(f"{args.exp_prefix}_fold_*"))
        fold_dirs = [d for d in fold_dirs if "no_concepts" not in d.name]
        
        print(f"Found {len(fold_dirs)} folds\n")
        
        for fold_dir in fold_dirs:
            metrics = extract_metrics_from_fold(fold_dir)
            
            if metrics:
                output_file = fold_dir / "results.json"
                save_metrics(metrics, output_file)
            
            print()
    
    else:
        # Process single fold
        fold_dir = outputs_dir / f"{args.exp_prefix}_fold_{args.fold}"
        
        if not fold_dir.exists():
            print(f"❌ Fold directory not found: {fold_dir}")
            return 1
        
        metrics = extract_metrics_from_fold(fold_dir)
        
        if metrics:
            output_file = fold_dir / "results.json"
            save_metrics(metrics, output_file)
            
            print("\nExtracted metrics:")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:8.2f}")
                else:
                    print(f"  {key:20s}: {value}")
        else:
            print("⚠ No metrics could be extracted")
            print("\nMake sure evaluation has been run (requires GPU):")
            print(f"  python submit_and_evaluate.py --config-path configs/r50_deformable_detr_motip_pdestre_fast.yaml --inference-model {fold_dir}/checkpoint_2.pth --inference-group fold_{args.fold} --inference-dataset P-DESTRE --inference-split val_{args.fold} --outputs-dir {fold_dir}")
    
    print("\n" + "="*70)
    print("✅ Extraction complete!")
    print("\nNext steps:")
    print("  1. Extract metrics for all folds: python extract_metrics.py --all-folds")
    print("  2. Generate visualizations: ./visualize_all_folds.sh")
    print("="*70 + "\n")


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
