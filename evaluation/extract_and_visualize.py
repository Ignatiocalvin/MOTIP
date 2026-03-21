#!/usr/bin/env python3
"""
Extract Training Metrics and Create Visualizations for Model Comparison

This script extracts the final training metrics from log files for all MOTIP models
and creates comprehensive visualizations comparing them.

Models compared:
- Base MOTIP (no concepts for ID): r50_motip_pdestre_fold_{0,1,2}
- Base MOTIP (2 concepts for ID): r50_motip_pdestre_concepts_for_id_fold_{0,1,2}  
- Base MOTIP (7 concepts for ID): r50_motip_pdestre_7concepts_for_id_fold_{0,1,2}
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [14, 10]
    plt.rcParams['font.size'] = 12
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
VIS_DIR = BASE_DIR / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

# Model configurations - All models with training logs
MODELS = {
    "R50 MOTIP\n(2 Concepts)": {
        "prefix": "r50_motip_pdestre_concepts_for_id_fold",
        "color": "#2ecc71",  # Green
        "marker": "s",
        "description": "ResNet-50 MOTIP with 2 concepts (gender, upper_body) for ID"
    },
    "R50 MOTIP\n(7 Concepts)": {
        "prefix": "r50_motip_pdestre_7concepts_for_id_fold",
        "color": "#e74c3c",  # Red
        "marker": "^",
        "description": "ResNet-50 MOTIP with all 7 concepts for ID"
    },
    "RF-DETR Large\n(No Concepts)": {
        "prefix": "rfdetr_large_motip_pdestre_no_concepts_fold",
        "color": "#9b59b6",  # Purple
        "marker": "D",
        "description": "RF-DETR Large without concept-based ID"
    },
}

NUM_FOLDS = 3
CONCEPT_NAMES = ["gender", "hairstyle", "head_accessories", "upper_body", "lower_body", "feet", "accessories"]


def extract_final_metrics_from_log(log_path: Path) -> dict:
    """
    Extract the final metrics from a training log.
    
    The log format has running averages in parentheses:
    concept_acc = 192.7068 (192.1541)
    
    We want the averaged value in parentheses from the last epoch.
    """
    metrics = {}
    
    if not log_path.exists():
        return metrics
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Find all lines with final epoch metrics (Epoch: 2 for 3 epochs)
        # Get the last metrics line for each epoch
        lines = content.split('\n')
        
        epoch_metrics = {}
        current_epoch = -1
        
        for line in lines:
            # Check for epoch marker
            epoch_match = re.search(r'\[Epoch: (\d+)\]', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Extract metrics from lines with [Metrics] tag
            if '[Metrics]' in line and current_epoch >= 0:
                # Extract running average values (in parentheses)
                
                # Total loss - get the averaged value
                loss_match = re.search(r'loss = [\d.]+ \(([\d.]+)\)', line)
                if loss_match:
                    epoch_metrics[current_epoch] = epoch_metrics.get(current_epoch, {})
                    epoch_metrics[current_epoch]['loss'] = float(loss_match.group(1))
                
                # Concept accuracy
                concept_acc_match = re.search(r'concept_acc = [\d.]+ \(([\d.]+)\)', line)
                if concept_acc_match:
                    epoch_metrics[current_epoch]['concept_acc'] = float(concept_acc_match.group(1))
                
                # ID loss
                id_loss_match = re.search(r'id_loss = [\d.]+ \(([\d.]+)\)', line)
                if id_loss_match:
                    epoch_metrics[current_epoch]['id_loss'] = float(id_loss_match.group(1))
                
                # DETR loss
                detr_loss_match = re.search(r'detr_loss = [\d.]+ \(([\d.]+)\)', line)
                if detr_loss_match:
                    epoch_metrics[current_epoch]['detr_loss'] = float(detr_loss_match.group(1))
                
                # Per-concept accuracies
                for concept in CONCEPT_NAMES:
                    pattern = rf'{concept}_accuracy = [\d.]+ \(([\d.]+)\)'
                    match = re.search(pattern, line)
                    if match:
                        epoch_metrics[current_epoch][f'{concept}_accuracy'] = float(match.group(1))
        
        # Get metrics from the last epoch
        if epoch_metrics:
            last_epoch = max(epoch_metrics.keys())
            metrics = epoch_metrics[last_epoch]
            metrics['last_epoch'] = last_epoch
            
            # Store all epoch data for training curves
            metrics['epoch_data'] = {e: epoch_metrics[e] for e in sorted(epoch_metrics.keys())}
        
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
    
    return metrics


def load_all_model_results():
    """Load results for all models and folds."""
    results = {}
    
    for model_name, config in MODELS.items():
        print(f"\n📊 Loading: {model_name.replace(chr(10), ' ')}")
        model_results = {
            "config": config,
            "folds": {},
        }
        
        for fold_idx in range(NUM_FOLDS):
            fold_dir = OUTPUTS_DIR / f"{config['prefix']}_{fold_idx}"
            train_log = fold_dir / "train" / "log.txt"
            
            if train_log.exists():
                fold_metrics = extract_final_metrics_from_log(train_log)
                if fold_metrics:
                    model_results["folds"][fold_idx] = fold_metrics
                    print(f"  ✓ Fold {fold_idx}: Found {len(fold_metrics)} metrics (epoch {fold_metrics.get('last_epoch', '?')})")
                else:
                    print(f"  ⚠ Fold {fold_idx}: No metrics found")
            else:
                # Check for direct log.txt in folder
                alt_log = fold_dir / "log.txt"
                if alt_log.exists():
                    fold_metrics = extract_final_metrics_from_log(alt_log)
                    if fold_metrics:
                        model_results["folds"][fold_idx] = fold_metrics
                        print(f"  ✓ Fold {fold_idx}: Found {len(fold_metrics)} metrics")
                else:
                    print(f"  ✗ Fold {fold_idx}: Log not found at {train_log}")
        
        results[model_name] = model_results
    
    return results


def compute_aggregate_stats(results: dict) -> dict:
    """Compute mean and std across folds for each model."""
    stats = {}
    
    for model_name, model_data in results.items():
        folds = model_data["folds"]
        if not folds:
            continue
        
        model_stats = {}
        
        # Collect all metric values across folds
        metric_values = defaultdict(list)
        for fold_idx, fold_metrics in folds.items():
            for key, value in fold_metrics.items():
                if key not in ['last_epoch', 'epoch_data'] and isinstance(value, (int, float)):
                    metric_values[key].append(value)
        
        # Compute mean and std
        for metric, values in metric_values.items():
            model_stats[f"{metric}_mean"] = np.mean(values)
            model_stats[f"{metric}_std"] = np.std(values)
        
        model_stats["n_folds"] = len(folds)
        stats[model_name] = model_stats
    
    return stats


def create_bar_comparison(results: dict, stats: dict, metric: str, ylabel: str, title: str, output_name: str):
    """Create a bar chart comparing a metric across models."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.6
    
    means = []
    stds = []
    colors = []
    
    for model_name in model_names:
        config = results[model_name]["config"]
        colors.append(config["color"])
        
        if model_name in stats and f"{metric}_mean" in stats[model_name]:
            means.append(stats[model_name][f"{metric}_mean"])
            stds.append(stats[model_name][f"{metric}_std"])
        else:
            means.append(0)
            stds.append(0)
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f} ± {std:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Model Configuration', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('\n', ' ') for name in model_names], fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / output_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_name}")


def create_concept_accuracy_comparison(results: dict, stats: dict):
    """Create grouped bar chart for per-concept accuracies."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(CONCEPT_NAMES))
    n_models = len(results)
    width = 0.8 / n_models
    
    for i, (model_name, model_data) in enumerate(results.items()):
        config = model_data["config"]
        
        means = []
        stds = []
        
        for concept in CONCEPT_NAMES:
            metric = f"{concept}_accuracy"
            if model_name in stats and f"{metric}_mean" in stats[model_name]:
                means.append(stats[model_name][f"{metric}_mean"])
                stds.append(stats[model_name][f"{metric}_std"])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, 
                     yerr=stds, capsize=3,
                     label=model_name.replace('\n', ' '),
                     color=config["color"],
                     alpha=0.85,
                     edgecolor='black',
                     linewidth=0.5)
    
    ax.set_xlabel('Concept Type', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Per-Concept Prediction Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in CONCEPT_NAMES], fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal reference lines
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / "concept_accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: concept_accuracy_comparison.png")


def create_fold_heatmap(results: dict, metric: str, title: str, output_name: str):
    """Create a heatmap showing metric values across models and folds."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    data = []
    
    for model_name in model_names:
        row = []
        for fold_idx in range(NUM_FOLDS):
            folds = results[model_name]["folds"]
            if fold_idx in folds and metric in folds[fold_idx]:
                row.append(folds[fold_idx][metric])
            else:
                row.append(np.nan)
        data.append(row)
    
    data = np.array(data)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(metric.replace('_', ' ').title(), rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(NUM_FOLDS))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([f'Fold {i}' for i in range(NUM_FOLDS)], fontsize=11)
    ax.set_yticklabels([name.replace('\n', ' ') for name in model_names], fontsize=11)
    
    # Add value annotations
    for i in range(len(model_names)):
        for j in range(NUM_FOLDS):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / output_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_name}")


def create_summary_table(results: dict, stats: dict):
    """Create and save a summary table."""
    if not HAS_MATPLOTLIB:
        return
    
    # Prepare data
    metrics_to_show = ['concept_acc', 'loss', 'id_loss']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Build table data
    columns = ['Model', '# Folds', 'Concept Acc', 'Total Loss', 'ID Loss']
    rows = []
    
    for model_name in results.keys():
        row = [model_name.replace('\n', ' ')]
        
        if model_name in stats:
            row.append(str(stats[model_name].get('n_folds', 0)))
            
            for metric in metrics_to_show:
                mean = stats[model_name].get(f'{metric}_mean', 0)
                std = stats[model_name].get(f'{metric}_std', 0)
                row.append(f'{mean:.2f} ± {std:.2f}')
        else:
            row.extend(['N/A'] * (len(columns) - 1))
        
        rows.append(row)
    
    # Create table
    table = ax.table(cellText=rows,
                    colLabels=columns,
                    loc='center',
                    cellLoc='center',
                    colColours=['#4a90d9'] * len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    plt.title('Model Comparison Summary (Mean ± Std across Folds)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(VIS_DIR / "summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: summary_table.png")
    
    # Also save as JSON
    json_data = {
        model_name.replace('\n', ' '): {
            k: v for k, v in model_stats.items()
        }
        for model_name, model_stats in stats.items()
    }
    
    with open(VIS_DIR / "metrics_summary.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    print("  ✓ Saved: metrics_summary.json")


def create_all_visualizations(results: dict, stats: dict):
    """Create all comparison visualizations."""
    print("\n📊 Creating visualizations...")
    
    # 1. Concept Accuracy Comparison
    create_bar_comparison(results, stats, 
                         'concept_acc', 
                         'Concept Accuracy (%)',
                         'Overall Concept Accuracy Comparison\n(Higher is Better)',
                         'overall_concept_accuracy.png')
    
    # 2. Total Loss Comparison
    create_bar_comparison(results, stats,
                         'loss',
                         'Total Loss',
                         'Training Loss Comparison\n(Lower is Better)',
                         'total_loss_comparison.png')
    
    # 3. ID Loss Comparison
    create_bar_comparison(results, stats,
                         'id_loss',
                         'ID Loss',
                         'ID Loss Comparison\n(Lower is Better)',
                         'id_loss_comparison.png')
    
    # 4. Per-Concept Accuracy Comparison
    create_concept_accuracy_comparison(results, stats)
    
    # 5. Fold-wise Heatmaps
    create_fold_heatmap(results, 'concept_acc', 
                       'Concept Accuracy Across Folds', 
                       'concept_accuracy_heatmap.png')
    
    create_fold_heatmap(results, 'loss',
                       'Training Loss Across Folds',
                       'loss_heatmap.png')
    
    # 6. Summary Table
    create_summary_table(results, stats)


def print_summary(results: dict, stats: dict):
    """Print a text summary of the results."""
    print("\n" + "="*80)
    print("📈 RESULTS SUMMARY")
    print("="*80)
    
    for model_name in results.keys():
        clean_name = model_name.replace('\n', ' ')
        print(f"\n{clean_name}")
        print("-" * len(clean_name))
        
        if model_name in stats:
            s = stats[model_name]
            print(f"  Folds analyzed: {s.get('n_folds', 0)}")
            
            if 'concept_acc_mean' in s:
                print(f"  Concept Accuracy: {s['concept_acc_mean']:.2f} ± {s['concept_acc_std']:.2f}")
            
            if 'loss_mean' in s:
                print(f"  Total Loss: {s['loss_mean']:.2f} ± {s['loss_std']:.2f}")
            
            if 'id_loss_mean' in s:
                print(f"  ID Loss: {s['id_loss_mean']:.4f} ± {s['id_loss_std']:.4f}")
            
            # Print per-concept accuracies
            print("  Per-Concept Accuracies:")
            for concept in CONCEPT_NAMES:
                metric = f"{concept}_accuracy"
                if f"{metric}_mean" in s:
                    print(f"    {concept:20s}: {s[f'{metric}_mean']:.2f} ± {s[f'{metric}_std']:.2f}")
        else:
            print("  No data available")
    
    print("\n" + "="*80)


def main():
    print("="*80)
    print("🔬 MOTIP Model Comparison - Metric Extraction & Visualization")
    print("="*80)
    
    print(f"\nOutputs directory: {OUTPUTS_DIR}")
    print(f"Visualizations will be saved to: {VIS_DIR}")
    
    # Load all results
    results = load_all_model_results()
    
    # Compute aggregate statistics
    stats = compute_aggregate_stats(results)
    
    # Print summary
    print_summary(results, stats)
    
    # Create visualizations
    if HAS_MATPLOTLIB:
        create_all_visualizations(results, stats)
        print(f"\n✅ All visualizations saved to: {VIS_DIR}")
    else:
        print("\n⚠️ Matplotlib not available - skipping visualizations")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
