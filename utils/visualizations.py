"""
MOTIP Training Visualizations and Analysis Tools
Provides comprehensive visualization capabilities for concept bottleneck model training
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
from collections import defaultdict
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class MOTIPTrainingVisualizer:
    """Comprehensive visualization system for MOTIP training analysis"""
    
    def __init__(self, output_dir="./outputs/visualizations/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Initialize tracking dictionaries
        self.metrics_history = defaultdict(list)
        self.concept_predictions = []
        self.concept_targets = []
        
    def log_training_metrics(self, epoch, metrics_dict):
        """Log training metrics for visualization"""
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics_history[f"{key}_epoch"].append((epoch, value))
    
    def log_batch_metrics(self, step, metrics_dict):
        """Log batch-level metrics"""
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics_history[f"{key}_step"].append((step, value))
    
    def plot_training_curves(self, save_path=None):
        """Plot comprehensive training curves"""
        if save_path is None:
            save_path = self.output_dir / "training_curves.png"
        
        # Determine number of subplots needed
        loss_metrics = [k for k in self.metrics_history.keys() if 'loss' in k and 'epoch' in k]
        acc_metrics = [k for k in self.metrics_history.keys() if 'accuracy' in k and 'epoch' in k]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MOTIP Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        for metric in loss_metrics:
            if self.metrics_history[metric]:
                epochs, values = zip(*self.metrics_history[metric])
                ax1.plot(epochs, values, label=metric.replace('_epoch', ''), marker='o', markersize=3)
        ax1.set_title('Training Losses')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2 = axes[0, 1]
        for metric in acc_metrics:
            if self.metrics_history[metric]:
                epochs, values = zip(*self.metrics_history[metric])
                ax2.plot(epochs, values, label=metric.replace('_epoch', ''), marker='s', markersize=3)
        ax2.set_title('Training Accuracies')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Concept vs Detection Loss Comparison
        ax3 = axes[1, 0]
        concept_loss = [v for k, v in self.metrics_history.items() if 'loss_concepts_epoch' in k]
        detection_loss = [v for k, v in self.metrics_history.items() if 'loss_ce_epoch' in k]
        
        if concept_loss and detection_loss:
            epochs1, concept_vals = zip(*concept_loss[0]) if concept_loss else ([], [])
            epochs2, detection_vals = zip(*detection_loss[0]) if detection_loss else ([], [])
            
            ax3.plot(epochs1, concept_vals, label='Concept Loss', color='blue', linewidth=2)
            ax3_twin = ax3.twinx()
            ax3_twin.plot(epochs2, detection_vals, label='Detection Loss', color='red', linewidth=2)
            
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Concept Loss', color='blue')
            ax3_twin.set_ylabel('Detection Loss', color='red')
            ax3.set_title('Concept vs Detection Learning')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning rate schedule
        ax4 = axes[1, 1]
        lr_history = [v for k, v in self.metrics_history.items() if 'learning_rate' in k]
        if lr_history:
            steps, lr_vals = zip(*lr_history[0])
            ax4.semilogy(steps, lr_vals, color='green', linewidth=2)
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Learning Rate (log scale)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {save_path}")
    
    def plot_concept_confusion_matrix(self, predictions, targets, save_path=None):
        """Plot concept prediction confusion matrix"""
        if save_path is None:
            save_path = self.output_dir / "concept_confusion_matrix.png"
        
        # Filter out unknown labels
        valid_mask = targets != 2
        valid_preds = predictions[valid_mask] if len(predictions) > 0 else np.array([])
        valid_targets = targets[valid_mask] if len(targets) > 0 else np.array([])
        
        if len(valid_preds) == 0:
            print("No valid concept predictions to plot confusion matrix")
            return
        
        # Create confusion matrix
        cm = confusion_matrix(valid_targets, valid_preds, labels=[0, 1])
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Male', 'Female'],
                    yticklabels=['Male', 'Female'],
                    square=True, linewidths=0.5)
        
        plt.title('Concept Prediction Confusion Matrix\n(Unknown labels excluded)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Gender', fontsize=12)
        plt.ylabel('Actual Gender', fontsize=12)
        
        # Add accuracy information
        accuracy = np.trace(cm) / np.sum(cm) * 100
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.1f}%', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_concept_distribution_over_time(self, concept_history, save_path=None):
        """Plot concept distribution evolution during training"""
        if save_path is None:
            save_path = self.output_dir / "concept_distribution_timeline.png"
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Prepare data
        steps = list(concept_history.keys())
        male_counts = [concept_history[step].get(0, 0) for step in steps]
        female_counts = [concept_history[step].get(1, 0) for step in steps]
        unknown_counts = [concept_history[step].get(2, 0) for step in steps]
        
        # Plot 1: Absolute counts
        ax1.plot(steps, male_counts, label='Male', color='blue', linewidth=2, marker='o')
        ax1.plot(steps, female_counts, label='Female', color='red', linewidth=2, marker='s')
        ax1.plot(steps, unknown_counts, label='Unknown', color='gray', linewidth=2, marker='^')
        
        ax1.set_title('Concept Predictions Over Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Proportions
        total_counts = np.array(male_counts) + np.array(female_counts) + np.array(unknown_counts)
        male_prop = np.array(male_counts) / (total_counts + 1e-8)
        female_prop = np.array(female_counts) / (total_counts + 1e-8)
        unknown_prop = np.array(unknown_counts) / (total_counts + 1e-8)
        
        ax2.stackplot(steps, male_prop, female_prop, unknown_prop,
                     labels=['Male %', 'Female %', 'Unknown %'],
                     colors=['lightblue', 'lightcoral', 'lightgray'],
                     alpha=0.8)
        
        ax2.set_title('Concept Prediction Proportions', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Proportion')
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Concept distribution timeline saved to: {save_path}")
    
    def plot_tracking_performance_analysis(self, tracking_metrics, save_path=None):
        """Analyze tracking performance metrics"""
        if save_path is None:
            save_path = self.output_dir / "tracking_performance.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Tracking Performance Analysis', fontsize=16, fontweight='bold')
        
        # ID accuracy over time
        if 'id_accuracy' in tracking_metrics:
            steps, accuracies = zip(*tracking_metrics['id_accuracy'])
            axes[0, 0].plot(steps, accuracies, color='green', linewidth=2, marker='o')
            axes[0, 0].set_title('ID Tracking Accuracy')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # ID switches over time
        if 'id_switches' in tracking_metrics:
            steps, switches = zip(*tracking_metrics['id_switches'])
            axes[0, 1].plot(steps, switches, color='red', linewidth=2, marker='s')
            axes[0, 1].set_title('ID Switches per Batch')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Switch Count')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Detection confidence distribution
        if 'detection_confidences' in tracking_metrics:
            conf_data = tracking_metrics['detection_confidences']
            axes[1, 0].hist(conf_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].set_title('Detection Confidence Distribution')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(np.mean(conf_data), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(conf_data):.3f}')
            axes[1, 0].legend()
        
        # Concept confidence vs accuracy correlation
        if 'concept_confidences' in tracking_metrics and 'concept_accuracies' in tracking_metrics:
            conf_data = tracking_metrics['concept_confidences']
            acc_data = tracking_metrics['concept_accuracies']
            
            axes[1, 1].scatter(conf_data, acc_data, alpha=0.6, color='purple')
            axes[1, 1].set_title('Concept Confidence vs Accuracy')
            axes[1, 1].set_xlabel('Concept Confidence')
            axes[1, 1].set_ylabel('Concept Accuracy')
            
            # Add trend line
            if len(conf_data) > 1 and len(acc_data) > 1:
                z = np.polyfit(conf_data, acc_data, 1)
                p = np.poly1d(z)
                axes[1, 1].plot(conf_data, p(conf_data), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Tracking performance analysis saved to: {save_path}")
    
    def create_interactive_dashboard(self, metrics_data):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves', 
                          'Concept Distribution', 'ID Tracking Performance',
                          'Detection Confidence', 'Learning Progress'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add interactive plots here...
        # (Implementation would continue with Plotly interactive visualizations)
        
        dashboard_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(dashboard_path))
        
        print(f"Interactive dashboard saved to: {dashboard_path}")
    
    def generate_training_report(self, final_metrics):
        """Generate comprehensive training report"""
        report_path = self.output_dir / "training_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MOTIP Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .header {{ color: #333; border-bottom: 2px solid #333; }}
                .section {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1 class="header">MOTIP Concept Bottleneck Training Report</h1>
            
            <div class="section">
                <h2>Final Performance Metrics</h2>
                <div class="metric">Concept Accuracy: {final_metrics.get('concept_accuracy', 'N/A'):.2f}%</div>
                <div class="metric">Detection Accuracy: {final_metrics.get('detection_accuracy', 'N/A'):.2f}%</div>
                <div class="metric">ID Tracking Accuracy: {final_metrics.get('id_accuracy', 'N/A'):.2f}%</div>
                <div class="metric">Total Training Time: {final_metrics.get('training_time', 'N/A'):.1f} hours</div>
            </div>
            
            <div class="section">
                <h2>Training Curves</h2>
                <img src="training_curves.png" alt="Training Curves">
            </div>
            
            <div class="section">
                <h2>Concept Analysis</h2>
                <img src="concept_confusion_matrix.png" alt="Confusion Matrix">
                <img src="concept_distribution_timeline.png" alt="Concept Distribution">
            </div>
            
            <div class="section">
                <h2>Tracking Performance</h2>
                <img src="tracking_performance.png" alt="Tracking Performance">
            </div>
            
            <div class="section">
                <h2>Model Configuration</h2>
                <pre>{json.dumps(final_metrics.get('config', {}), indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Training report generated: {report_path}")


def create_training_visualizer(output_dir="./outputs/visualizations/"):
    """Factory function to create training visualizer"""
    return MOTIPTrainingVisualizer(output_dir)


# Integration functions for training loop
def visualize_training_batch(visualizer, step, outputs, targets, indices):
    """Function to call during training to log batch metrics"""
    # Extract concept predictions and targets for visualization
    if 'pred_concepts' in outputs:
        # Get matched predictions
        idx = visualizer._get_src_permutation_idx(indices) if hasattr(visualizer, '_get_src_permutation_idx') else indices
        
        try:
            concept_preds = outputs['pred_concepts'][idx]
            concept_targets = torch.cat([t["concepts"][J] for t, (_, J) in zip(targets, indices)])
            
            # Log for later visualization
            pred_concepts = torch.argmax(concept_preds, dim=-1).cpu().numpy()
            target_concepts = concept_targets.cpu().numpy()
            
            # Store predictions for confusion matrix
            visualizer.concept_predictions.extend(pred_concepts)
            visualizer.concept_targets.extend(target_concepts)
            
        except Exception as e:
            print(f"Visualization logging error: {e}")


def generate_final_report(visualizer, final_metrics):
    """Generate final training report with all visualizations"""
    print("Generating comprehensive training report...")
    
    # Generate all plots
    visualizer.plot_training_curves()
    
    if visualizer.concept_predictions and visualizer.concept_targets:
        visualizer.plot_concept_confusion_matrix(
            np.array(visualizer.concept_predictions),
            np.array(visualizer.concept_targets)
        )
    
    # Generate report
    visualizer.generate_training_report(final_metrics)
    
    print(f"All visualizations saved to: {visualizer.output_dir}")


# Example usage function
def example_usage():
    """Example of how to use the visualization system"""
    
    # Create visualizer
    visualizer = create_training_visualizer()
    
    # During training loop:
    # for epoch in range(num_epochs):
    #     for batch_idx, (data, targets) in enumerate(dataloader):
    #         outputs = model(data)
    #         loss, indices = criterion(outputs, targets)
    #         
    #         # Log metrics
    #         visualizer.log_batch_metrics(step, {
    #             'loss_total': loss.item(),
    #             'loss_concepts': criterion.concept_loss.item(),
    #             'concept_accuracy': accuracy
    #         })
    #         
    #         # Visualize batch (every N steps)
    #         if batch_idx % 100 == 0:
    #             visualize_training_batch(visualizer, step, outputs, targets, indices)
    
    # At end of training:
    # generate_final_report(visualizer, final_metrics)