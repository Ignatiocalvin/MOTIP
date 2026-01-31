#!/usr/bin/env python3
"""
Aggregate Cross-Validation Results for P-DESTRE

This script reads the results from all 10 folds and computes:
- Mean and standard deviation of each metric
- Per-fold breakdown

Usage:
    python aggregate_crossval_results.py
    python aggregate_crossval_results.py --exp-prefix pdestre_crossval
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np


def find_result_files(output_dir: Path, exp_prefix: str) -> dict:
    """Find result files for all folds."""
    results = {}
    
    for fold in range(10):
        fold_dir = output_dir / f"{exp_prefix}_fold_{fold}"
        
        # Look for common result file patterns
        possible_files = [
            fold_dir / "results.json",
            fold_dir / "eval_results.json",
            fold_dir / "metrics.json",
            fold_dir / "test_results.json",
        ]
        
        for result_file in possible_files:
            if result_file.exists():
                results[fold] = result_file
                break
        
        # Also check for log files that might contain metrics
        if fold not in results:
            log_files = list(fold_dir.glob("*.log")) if fold_dir.exists() else []
            if log_files:
                print(f"  Fold {fold}: No JSON results, but found logs: {[f.name for f in log_files]}")
    
    return results


def parse_tracking_metrics(result_file: Path) -> dict:
    """Parse tracking metrics from a result file."""
    metrics = {}
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Common tracking metrics
        metric_keys = [
            'HOTA', 'MOTA', 'MOTP', 'IDF1',
            'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr',
            'LocA', 'OWTA',
            'Dets', 'GT_Dets', 'IDs', 'GT_IDs',
            'IDsw', 'Frag',
            # Concept accuracy metrics
            'concept_accuracy', 'gender_accuracy', 'clothing_accuracy',
        ]
        
        for key in metric_keys:
            if key in data:
                metrics[key] = data[key]
            # Also check nested structures
            elif 'HOTA' in data and key in data.get('HOTA', {}):
                metrics[key] = data['HOTA'][key]
            elif 'CLEAR' in data and key in data.get('CLEAR', {}):
                metrics[key] = data['CLEAR'][key]
            elif 'Identity' in data and key in data.get('Identity', {}):
                metrics[key] = data['Identity'][key]
                
    except json.JSONDecodeError:
        print(f"  Warning: Could not parse JSON from {result_file}")
    except Exception as e:
        print(f"  Warning: Error reading {result_file}: {e}")
    
    return metrics


def aggregate_metrics(all_metrics: dict) -> dict:
    """Compute mean and std for each metric across folds."""
    aggregated = {}
    
    # Collect all metric keys
    all_keys = set()
    for fold_metrics in all_metrics.values():
        all_keys.update(fold_metrics.keys())
    
    for key in sorted(all_keys):
        values = []
        for fold, metrics in all_metrics.items():
            if key in metrics:
                values.append(metrics[key])
        
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n_folds': len(values),
                'values': values,
            }
    
    return aggregated


def print_results(aggregated: dict, all_metrics: dict):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("P-DESTRE 10-Fold Cross-Validation Results")
    print("=" * 70)
    
    # Print per-fold results
    print("\nPer-Fold Results:")
    print("-" * 70)
    
    folds = sorted(all_metrics.keys())
    if not folds:
        print("  No results found!")
        return
    
    # Get common metrics
    common_keys = ['HOTA', 'MOTA', 'IDF1', 'DetA', 'AssA']
    available_keys = [k for k in common_keys if k in aggregated]
    
    if available_keys:
        # Header
        header = "Fold  " + "  ".join(f"{k:>8}" for k in available_keys)
        print(header)
        print("-" * len(header))
        
        # Per-fold values
        for fold in folds:
            values = []
            for key in available_keys:
                if key in all_metrics[fold]:
                    values.append(f"{all_metrics[fold][key]:>8.2f}")
                else:
                    values.append(f"{'N/A':>8}")
            print(f"  {fold}   " + "  ".join(values))
    
    # Print aggregated results
    print("\n" + "-" * 70)
    print("Aggregated Results (Mean ± Std):")
    print("-" * 70)
    
    for key, stats in sorted(aggregated.items()):
        if stats['n_folds'] >= 5:  # Only show metrics with enough data
            print(f"  {key:20s}: {stats['mean']:7.2f} ± {stats['std']:5.2f}  "
                  f"[{stats['min']:.2f} - {stats['max']:.2f}] (n={stats['n_folds']})")
    
    print("\n" + "=" * 70)


def save_results(aggregated: dict, all_metrics: dict, output_file: Path):
    """Save results to JSON file."""
    results = {
        'aggregated': aggregated,
        'per_fold': {str(k): v for k, v in all_metrics.items()},
        'n_folds': len(all_metrics),
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate P-DESTRE cross-validation results')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory containing fold outputs')
    parser.add_argument('--exp-prefix', type=str, default='pdestre_crossval',
                        help='Experiment name prefix')
    parser.add_argument('--save', type=str, default='crossval_results.json',
                        help='Output file for aggregated results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("Searching for result files...")
    result_files = find_result_files(output_dir, args.exp_prefix)
    
    if not result_files:
        print(f"\nNo result files found in {output_dir} with prefix '{args.exp_prefix}'")
        print("\nExpected directory structure:")
        for fold in range(10):
            print(f"  {output_dir}/{args.exp_prefix}_fold_{fold}/results.json")
        print("\nMake sure all training jobs have completed.")
        return
    
    print(f"\nFound results for {len(result_files)} folds: {sorted(result_files.keys())}")
    
    # Parse metrics from each fold
    all_metrics = {}
    for fold, result_file in result_files.items():
        print(f"  Parsing fold {fold}: {result_file}")
        metrics = parse_tracking_metrics(result_file)
        if metrics:
            all_metrics[fold] = metrics
    
    if not all_metrics:
        print("\nNo metrics could be parsed from result files.")
        return
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)
    
    # Print results
    print_results(aggregated, all_metrics)
    
    # Save results
    save_file = output_dir / args.save
    save_results(aggregated, all_metrics, save_file)


if __name__ == '__main__':
    main()
