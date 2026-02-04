#!/usr/bin/env python3
"""
Comprehensive Visualization Script for MOTIP Training Results

After all folds have finished training, this script generates visualizations for:
1. Detection performance (AP, AR, precision, recall)
2. Tracking performance (MOTA, MOTP, HOTA, IDF1, ID switches)
3. Concept prediction performance (accuracy, confusion matrices, per-concept metrics)

Usage:
    python visualize_results.py --exp-prefix r50_motip_pdestre
    python visualize_results.py --exp-prefix r50_motip_pdestre --output-dir ./visualizations
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Any


class MOTIPResultsVisualizer:
    """Comprehensive visualization for MOTIP training results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        sns.set_context("paper", font_scale=1.3)
        
        print(f"Visualization output directory: {self.output_dir}")
    
    def load_fold_results(self, fold_dir: Path) -> Dict[str, Any]:
        """Load results from a single fold directory"""
        results = {}
        
        # Look for TrackEval results
        eval_dirs = list(fold_dir.glob("evaluate/*/pedestrian_summary.txt"))
        if eval_dirs:
            # Parse TrackEval summary
            results.update(self.parse_trackeval_summary(eval_dirs[-1]))
        
        # Look for JSON results
        json_files = [
            fold_dir / "evaluate" / "results.json",
            fold_dir / "results.json",
            fold_dir / "eval_results.json",
        ]
        
        for json_file in json_files:
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        results.update(data)
                except Exception as e:
                    print(f"Warning: Could not parse {json_file}: {e}")
        
        # Look for concept prediction results
        concept_file = fold_dir / "concept_results.json"
        if concept_file.exists():
            try:
                with open(concept_file, 'r') as f:
                    results['concepts'] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not parse {concept_file}: {e}")
        
        return results
    
    def parse_trackeval_summary(self, summary_file: Path) -> Dict[str, float]:
        """Parse TrackEval pedestrian_summary.txt file"""
        metrics = {}
        
        try:
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0]
                    try:
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
                    except ValueError:
                        pass
        
        except Exception as e:
            print(f"Warning: Could not parse TrackEval summary {summary_file}: {e}")
        
        return metrics
    
    def aggregate_folds(self, all_fold_results: Dict[int, Dict]) -> Dict[str, Dict]:
        """Aggregate metrics across all folds"""
        aggregated = {}
        
        # Collect all metric keys
        all_keys = set()
        for fold_results in all_fold_results.values():
            all_keys.update(fold_results.keys())
        
        # Remove non-numeric keys
        all_keys = {k for k in all_keys if k != 'concepts'}
        
        for key in sorted(all_keys):
            values = []
            for fold, results in all_fold_results.items():
                if key in results and isinstance(results[key], (int, float)):
                    values.append(results[key])
            
            if values:
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values,
                    'n_folds': len(values),
                }
        
        return aggregated
    
    def plot_tracking_metrics(self, aggregated: Dict, save_name: str = "tracking_metrics.png"):
        """Plot tracking performance metrics"""
        
        # Define tracking metrics of interest
        tracking_metrics = {
            'HOTA': 'HOTA (Higher Order Tracking Accuracy)',
            'MOTA': 'MOTA (Multiple Object Tracking Accuracy)',
            'MOTP': 'MOTP (Multiple Object Tracking Precision)',
            'IDF1': 'IDF1 (ID F1 Score)',
            'DetA': 'DetA (Detection Accuracy)',
            'AssA': 'AssA (Association Accuracy)',
        }
        
        # Filter available metrics
        available_metrics = {k: v for k, v in tracking_metrics.items() if k in aggregated}
        
        if not available_metrics:
            print("Warning: No tracking metrics found")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tracking Performance Metrics Across Folds', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (metric_key, metric_name) in enumerate(available_metrics.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            data = aggregated[metric_key]
            
            # Bar plot with error bars
            ax.bar(0, data['mean'], yerr=data['std'], capsize=10, 
                   color='steelblue', alpha=0.7, width=0.4)
            
            # Scatter individual fold values
            fold_values = data['values']
            x_positions = np.random.normal(0, 0.05, len(fold_values))
            ax.scatter(x_positions, fold_values, color='darkred', s=50, 
                      alpha=0.6, zorder=3, label='Individual Folds')
            
            # Add mean line
            ax.axhline(y=data['mean'], color='green', linestyle='--', 
                      linewidth=2, label=f"Mean: {data['mean']:.2f}")
            
            # Formatting
            ax.set_ylabel(metric_key, fontsize=12, fontweight='bold')
            ax.set_title(metric_name, fontsize=11)
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add text with statistics
            stats_text = f"μ={data['mean']:.2f}\nσ={data['std']:.2f}\nn={data['n_folds']}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if idx == 0:
                ax.legend(loc='lower right', fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Tracking metrics saved to: {save_path}")
    
    def plot_detection_metrics(self, aggregated: Dict, save_name: str = "detection_metrics.png"):
        """Plot detection performance metrics"""
        
        detection_metrics = {
            'DetPr': 'Detection Precision',
            'DetRe': 'Detection Recall',
            'DetA': 'Detection Accuracy',
            'LocA': 'Localization Accuracy',
        }
        
        available_metrics = {k: v for k, v in detection_metrics.items() if k in aggregated}
        
        if not available_metrics:
            print("Warning: No detection metrics found")
            return
        
        # Create radar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Detection Performance Analysis', fontsize=16, fontweight='bold')
        
        # Subplot 1: Bar chart with error bars
        metric_names = list(available_metrics.keys())
        means = [aggregated[m]['mean'] for m in metric_names]
        stds = [aggregated[m]['std'] for m in metric_names]
        
        x_pos = np.arange(len(metric_names))
        ax1.bar(x_pos, means, yerr=stds, capsize=5, color='teal', alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metric_names, rotation=0, ha='center')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Detection Metrics (Mean ± Std)', fontsize=13)
        ax1.set_ylim(0, max(means) * 1.2)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax1.text(i, m + s + 0.02, f'{m:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Subplot 2: Boxplot showing distribution across folds
        fold_data = [aggregated[m]['values'] for m in metric_names]
        bp = ax2.boxplot(fold_data, labels=metric_names, patch_artist=True,
                         showmeans=True, meanline=True)
        
        # Customize boxplot
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Detection Metrics Distribution Across Folds', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Detection metrics saved to: {save_path}")
    
    def plot_concept_performance(self, all_fold_results: Dict, 
                                save_name: str = "concept_performance.png"):
        """Plot concept prediction performance"""
        
        # Extract concept metrics if available
        concept_accuracies = []
        for fold, results in all_fold_results.items():
            if 'concepts' in results:
                concept_data = results['concepts']
                if 'accuracy' in concept_data:
                    concept_accuracies.append(concept_data['accuracy'])
        
        if not concept_accuracies:
            print("Warning: No concept prediction metrics found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Concept Prediction Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall concept accuracy across folds
        ax1 = axes[0, 0]
        fold_ids = list(range(len(concept_accuracies)))
        ax1.plot(fold_ids, concept_accuracies, marker='o', linewidth=2, 
                markersize=8, color='purple', label='Accuracy')
        ax1.axhline(y=np.mean(concept_accuracies), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(concept_accuracies):.2f}%')
        ax1.fill_between(fold_ids, 
                        np.mean(concept_accuracies) - np.std(concept_accuracies),
                        np.mean(concept_accuracies) + np.std(concept_accuracies),
                        alpha=0.2, color='red')
        ax1.set_xlabel('Fold', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Concept Accuracy Across Folds', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of concept accuracies
        ax2 = axes[0, 1]
        ax2.hist(concept_accuracies, bins=10, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(concept_accuracies), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(concept_accuracies):.2f}%')
        ax2.set_xlabel('Accuracy (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Concept Accuracies', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Per-concept breakdown if available
        ax3 = axes[1, 0]
        # Aggregate per-concept metrics
        concept_types = defaultdict(list)
        for fold, results in all_fold_results.items():
            if 'concepts' in results and 'per_concept' in results['concepts']:
                for concept_name, acc in results['concepts']['per_concept'].items():
                    concept_types[concept_name].append(acc)
        
        if concept_types:
            concept_names = list(concept_types.keys())
            concept_means = [np.mean(concept_types[c]) for c in concept_names]
            concept_stds = [np.std(concept_types[c]) for c in concept_names]
            
            x_pos = np.arange(len(concept_names))
            ax3.bar(x_pos, concept_means, yerr=concept_stds, capsize=5, 
                   color='orange', alpha=0.7)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(concept_names, rotation=45, ha='right')
            ax3.set_ylabel('Accuracy (%)', fontsize=12)
            ax3.set_title('Per-Concept Accuracy', fontsize=13)
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'No per-concept data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Concept Prediction Summary
        {'='*40}
        
        Mean Accuracy:     {np.mean(concept_accuracies):.2f}%
        Std Deviation:     {np.std(concept_accuracies):.2f}%
        Min Accuracy:      {np.min(concept_accuracies):.2f}%
        Max Accuracy:      {np.max(concept_accuracies):.2f}%
        Number of Folds:   {len(concept_accuracies)}
        
        {'='*40}
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Concept performance saved to: {save_path}")
    
    def plot_correlation_matrix(self, aggregated: Dict, 
                               save_name: str = "metric_correlations.png"):
        """Plot correlation matrix between different metrics"""
        
        # Prepare data for correlation
        metric_keys = ['HOTA', 'MOTA', 'IDF1', 'DetA', 'AssA', 'DetPr', 'DetRe']
        available_keys = [k for k in metric_keys if k in aggregated and aggregated[k]['n_folds'] > 1]
        
        if len(available_keys) < 2:
            print("Warning: Not enough metrics for correlation analysis")
            return
        
        # Create dataframe with fold values
        data_dict = {}
        min_folds = min(aggregated[k]['n_folds'] for k in available_keys)
        
        for key in available_keys:
            data_dict[key] = aggregated[key]['values'][:min_folds]
        
        df = pd.DataFrame(data_dict)
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Correlation Matrix of Performance Metrics', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Correlation matrix saved to: {save_path}")
    
    def plot_comparative_overview(self, aggregated: Dict,
                                 save_name: str = "comparative_overview.png"):
        """Create a comprehensive comparative overview"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('MOTIP Training Results: Comprehensive Overview', 
                    fontsize=18, fontweight='bold')
        
        # Main tracking metrics
        ax1 = fig.add_subplot(gs[0, :2])
        main_metrics = ['HOTA', 'MOTA', 'IDF1', 'DetA', 'AssA']
        available = [m for m in main_metrics if m in aggregated]
        
        if available:
            means = [aggregated[m]['mean'] for m in available]
            stds = [aggregated[m]['std'] for m in available]
            x_pos = np.arange(len(available))
            
            bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, 
                          color=plt.cm.viridis(np.linspace(0.2, 0.8, len(available))),
                          alpha=0.8)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(available, fontsize=11, fontweight='bold')
            ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax1.set_title('Key Performance Metrics', fontsize=14)
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (m, s) in enumerate(zip(means, stds)):
                ax1.text(i, m + s + 2, f'{m:.1f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        # Detection metrics
        ax2 = fig.add_subplot(gs[0, 2])
        det_metrics = ['DetPr', 'DetRe']
        available_det = [m for m in det_metrics if m in aggregated]
        
        if available_det:
            values = [[aggregated[m]['mean']] for m in available_det]
            ax2.bar(range(len(available_det)), [v[0] for v in values], 
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
            ax2.set_xticks(range(len(available_det)))
            ax2.set_xticklabels(available_det, fontsize=10)
            ax2.set_ylabel('Score', fontsize=11)
            ax2.set_title('Detection\nPrecision & Recall', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # ID metrics over folds
        ax3 = fig.add_subplot(gs[1, :])
        if 'IDF1' in aggregated and aggregated['IDF1']['n_folds'] > 0:
            fold_indices = range(aggregated['IDF1']['n_folds'])
            idf1_values = aggregated['IDF1']['values']
            
            ax3.plot(fold_indices, idf1_values, marker='o', linewidth=2, 
                    markersize=8, color='#FF6B6B', label='IDF1')
            
            if 'MOTA' in aggregated:
                mota_values = aggregated['MOTA']['values'][:len(idf1_values)]
                ax3.plot(fold_indices, mota_values, marker='s', linewidth=2, 
                        markersize=8, color='#4ECDC4', label='MOTA')
            
            ax3.axhline(y=np.mean(idf1_values), color='#FF6B6B', 
                       linestyle='--', alpha=0.5)
            ax3.set_xlabel('Fold', fontsize=12)
            ax3.set_ylabel('Score', fontsize=12)
            ax3.set_title('Performance Across Folds', fontsize=14)
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
        
        # Summary statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        table_metrics = ['HOTA', 'MOTA', 'IDF1', 'DetA', 'AssA']
        
        for metric in table_metrics:
            if metric in aggregated:
                data = aggregated[metric]
                table_data.append([
                    metric,
                    f"{data['mean']:.2f}",
                    f"{data['std']:.2f}",
                    f"{data['min']:.2f}",
                    f"{data['max']:.2f}",
                    f"{data['n_folds']}"
                ])
        
        if table_data:
            table = ax4.table(cellText=table_data,
                            colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max', 'N'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.2, 0.8, 0.6])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header
            for i in range(6):
                table[(0, i)].set_facecolor('#4ECDC4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(table_data) + 1):
                for j in range(6):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#F0F0F0')
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparative overview saved to: {save_path}")
    
    def generate_summary_report(self, aggregated: Dict, all_fold_results: Dict,
                               save_name: str = "summary_report.txt"):
        """Generate text summary report"""
        
        report_path = self.output_dir / save_name
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MOTIP TRAINING RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Number of Folds: {len(all_fold_results)}\n\n")
            
            # Tracking metrics
            f.write("-"*70 + "\n")
            f.write("TRACKING PERFORMANCE METRICS\n")
            f.write("-"*70 + "\n")
            
            tracking_keys = ['HOTA', 'MOTA', 'MOTP', 'IDF1', 'DetA', 'AssA']
            for key in tracking_keys:
                if key in aggregated:
                    data = aggregated[key]
                    f.write(f"{key:10s}: {data['mean']:6.2f} ± {data['std']:5.2f} "
                           f"[{data['min']:.2f} - {data['max']:.2f}] (n={data['n_folds']})\n")
            
            # Detection metrics
            f.write("\n" + "-"*70 + "\n")
            f.write("DETECTION METRICS\n")
            f.write("-"*70 + "\n")
            
            detection_keys = ['DetPr', 'DetRe', 'LocA']
            for key in detection_keys:
                if key in aggregated:
                    data = aggregated[key]
                    f.write(f"{key:10s}: {data['mean']:6.2f} ± {data['std']:5.2f} "
                           f"[{data['min']:.2f} - {data['max']:.2f}] (n={data['n_folds']})\n")
            
            # Per-fold breakdown
            f.write("\n" + "-"*70 + "\n")
            f.write("PER-FOLD RESULTS\n")
            f.write("-"*70 + "\n\n")
            
            main_metrics = ['HOTA', 'MOTA', 'IDF1']
            available = [m for m in main_metrics if m in aggregated]
            
            if available:
                # Header
                header = f"{'Fold':>6}  " + "  ".join([f"{m:>8}" for m in available])
                f.write(header + "\n")
                f.write("-"*len(header) + "\n")
                
                # Values
                for fold in sorted(all_fold_results.keys()):
                    values = []
                    for metric in available:
                        if metric in all_fold_results[fold]:
                            values.append(f"{all_fold_results[fold][metric]:8.2f}")
                        else:
                            values.append(f"{'N/A':>8}")
                    f.write(f"{fold:6d}  " + "  ".join(values) + "\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive visualizations for MOTIP training results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_results.py --exp-prefix r50_motip_pdestre
  python visualize_results.py --exp-prefix standard_motip_pdestre --output-dir ./viz
        """
    )
    
    parser.add_argument('--exp-prefix', type=str, required=True,
                       help='Experiment name prefix (e.g., r50_motip_pdestre)')
    parser.add_argument('--outputs-dir', type=str, default='../outputs',
                       help='Directory containing fold output directories')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save visualizations (default: outputs/exp_prefix_visualizations)')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                       help='Specific folds to visualize (default: all found folds)')
    
    args = parser.parse_args()
    
    # Setup paths
    outputs_dir = Path(args.outputs_dir)
    
    if args.output_dir:
        viz_dir = Path(args.output_dir)
    else:
        viz_dir = outputs_dir / f"{args.exp_prefix}_visualizations"
    
    # Create visualizer
    visualizer = MOTIPResultsVisualizer(viz_dir)
    
    print(f"\nSearching for fold results in: {outputs_dir}")
    print(f"Experiment prefix: {args.exp_prefix}\n")
    
    # Find all fold directories
    all_fold_results = {}
    
    if args.folds:
        fold_numbers = args.folds
    else:
        # Auto-detect folds
        fold_dirs = list(outputs_dir.glob(f"{args.exp_prefix}_fold_*"))
        fold_numbers = []
        for fold_dir in fold_dirs:
            try:
                fold_num = int(fold_dir.name.split('_fold_')[-1])
                fold_numbers.append(fold_num)
            except ValueError:
                pass
        fold_numbers = sorted(fold_numbers)
    
    print(f"Looking for folds: {fold_numbers}")
    
    # Load results from each fold
    for fold_num in fold_numbers:
        fold_dir = outputs_dir / f"{args.exp_prefix}_fold_{fold_num}"
        
        if not fold_dir.exists():
            print(f"  Fold {fold_num}: Directory not found - {fold_dir}")
            continue
        
        print(f"  Loading Fold {fold_num}...")
        results = visualizer.load_fold_results(fold_dir)
        
        if results:
            all_fold_results[fold_num] = results
            print(f"    ✓ Loaded {len(results)} metrics")
        else:
            print(f"    ⚠ No results found")
    
    if not all_fold_results:
        print("\n❌ No results found! Make sure:")
        print("  1. Training has completed for at least some folds")
        print("  2. Evaluation has been run (check for 'evaluate' directories)")
        print("  3. The --exp-prefix matches your experiment name")
        return
    
    print(f"\n✓ Successfully loaded results from {len(all_fold_results)} folds\n")
    
    # Aggregate metrics
    print("Aggregating metrics across folds...")
    aggregated = visualizer.aggregate_folds(all_fold_results)
    print(f"  ✓ Aggregated {len(aggregated)} metrics\n")
    
    # Generate all visualizations
    print("Generating visualizations...")
    print("-" * 50)
    
    visualizer.plot_tracking_metrics(aggregated)
    visualizer.plot_detection_metrics(aggregated)
    visualizer.plot_concept_performance(all_fold_results)
    visualizer.plot_correlation_matrix(aggregated)
    visualizer.plot_comparative_overview(aggregated)
    visualizer.generate_summary_report(aggregated, all_fold_results)
    
    print("-" * 50)
    print(f"\n✅ All visualizations saved to: {viz_dir}\n")
    print("Generated files:")
    for file in sorted(viz_dir.glob("*")):
        print(f"  - {file.name}")
    print()


if __name__ == '__main__':
    main()
