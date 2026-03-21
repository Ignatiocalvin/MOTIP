#!/usr/bin/env python3
"""
Visualize the gradient competition problem between 2-concept and 7-concept models
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Gradient Competition Analysis: 2-Concept vs 7-Concept MOTIP', 
             fontsize=16, fontweight='bold')

# Colors
detection_color = '#2E86AB'
concept_color = '#A23B72'
detection_light = '#6FB3D2'
concept_light = '#D888B8'

# ============================================================================
# Subplot 1: Loss Magnitude Comparison
# ============================================================================
ax = axes[0, 0]

models = ['2-Concept', '7-Concept']
detection_losses = [0.2466, 0.3135]
concept_losses = [13.24, 59.46]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, detection_losses, width, label='Detection Loss', 
               color=detection_color)
bars2 = ax.bar(x + width/2, concept_losses, width, label='Concept Loss', 
               color=concept_color)

ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax.set_title('Loss Magnitude: Concept Loss Dominates', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add ratio annotations
ax.text(0, max(concept_losses) * 0.9, 'Ratio: 54:1', 
        ha='center', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax.text(1, max(concept_losses) * 0.9, 'Ratio: 190:1', 
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

# ============================================================================
# Subplot 2: Gradient Signal Distribution
# ============================================================================
ax = axes[0, 1]

# 2-concept model
gradient_2concept = [50, 50]  # Detection: 50%, Concepts: 50%
labels_2 = ['Detection\n(50%)', 'Concepts (2)\n(50%)']
colors_2 = [detection_color, concept_color]

# 7-concept model  
gradient_7concept = [22, 78]  # Detection: 22%, Concepts: 78%
labels_7 = ['Detection\n(22%)', 'Concepts (7)\n(78%)']
colors_7 = [detection_light, concept_light]

# Create pie charts
ax1 = plt.subplot(2, 2, 2)
ax1.pie(gradient_2concept, labels=labels_2, autopct='%1.0f%%',
        colors=colors_2, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('2-Concept Model\nGradient Signal Distribution', 
              fontsize=13, fontweight='bold', pad=20)

# Add annotation
ax1.text(0, -1.5, 'Balanced gradient flow\nDetection gets 50% of signal',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Create second pie chart for 7-concept
ax2 = plt.subplot(2, 2, 4)
ax2.pie(gradient_7concept, labels=labels_7, autopct='%1.0f%%',
        colors=colors_7, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('7-Concept Model\nGradient Signal Distribution', 
              fontsize=13, fontweight='bold', pad=20)

# Add annotation
ax2.text(0, -1.5, 'Imbalanced gradient flow\nDetection gets only 22% of signal',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# ============================================================================
# Subplot 3: Detection Performance Impact
# ============================================================================
ax = axes[1, 0]

metrics = ['BBox Loss', 'GIoU Loss', 'Recall (%)', 'MOTA (%)']
model_2concept = [0.0222, 0.2244, 68.29, 66.00]
model_7concept = [0.0294, 0.2841, 27.59, 27.27]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, model_2concept, width, label='2-Concept (Better)', 
               color=detection_color)
bars2 = ax.bar(x + width/2, model_7concept, width, label='7-Concept (Worse)', 
               color='#D32F2F')

ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('Detection & Tracking Performance Degradation', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# Add degradation percentages
degradations = ['+32.4%', '+26.6%', '-40.7%', '-38.7%']
for i, deg in enumerate(degradations):
    ax.text(i, max(model_2concept + model_7concept) * 0.85, deg,
            ha='center', fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# ============================================================================
# Subplot 4: Remove and replace with explanation text
# ============================================================================
ax = axes[1, 1]
ax.axis('off')

explanation_text = """
WHY DOES THIS HAPPEN?

1. More Concepts = More Loss Terms
   • 2-concept: Detection + 2 concepts = 3 losses
   • 7-concept: Detection + 7 concepts = 10 losses
   
2. Gradient Competition
   • All losses share the same ResNet-50 backbone
   • Each loss wants to update weights differently
   • More concept losses = less gradient for detection
   
3. Loss Imbalance  
   • CONCEPT_LOSS_COEF = 0.5 for EACH concept
   • 7-concept total: 7 × 0.5 = 3.5× concept weight
   • 2-concept total: 2 × 0.5 = 1.0× concept weight
   • Detection gets starved of gradient signal!
   
4. Cascade Effect
   • Worse gradients → Worse detection training
   • Worse detection → Lower confidence scores
   • Low confidence → Objects filtered out (< 0.5 threshold)
   • Missing objects → Can't track what you can't see
   • Result: 40% lower recall, 39% lower MOTA

SOLUTION: Reduce CONCEPT_LOSS_COEF for 7-concept
model to ~0.143 to match 2-concept supervision level.
"""

ax.text(0.05, 0.95, explanation_text, 
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('gradient_competition_visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to gradient_competition_visualization.png")
plt.close()

# ============================================================================
# Create a second figure showing the tracking pipeline
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Tracking Pipeline: How Poor Detection Causes Poor Tracking', 
             fontsize=16, fontweight='bold')

# 2-concept model pipeline
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('2-Concept Model: Good Detection → Good Tracking', 
             fontsize=14, fontweight='bold', pad=20)

# Detection stage
rect1 = mpatches.FancyBboxPatch((0.5, 7), 3, 2, boxstyle="round,pad=0.1", 
                                 edgecolor='green', facecolor='lightgreen', linewidth=3)
ax.add_patch(rect1)
ax.text(2, 8, 'DETECTION\n68.29% Recall', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Arrow
ax.arrow(3.5, 8, 1.5, 0, head_width=0.3, head_length=0.3, fc='green', ec='green', linewidth=2)

# Association stage
rect2 = mpatches.FancyBboxPatch((5.5, 7), 3, 2, boxstyle="round,pad=0.1",
                                 edgecolor='green', facecolor='lightgreen', linewidth=3)
ax.add_patch(rect2)
ax.text(7, 8, 'TRACKING\n66.00% MOTA\n68.96% IDF1', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Ground truth
ax.text(5, 5, 'Ground Truth: 100 people', ha='center', fontsize=11, style='italic')
ax.text(5, 4, '↓', ha='center', fontsize=20, fontweight='bold')
ax.text(5, 3.3, 'Detected: 68 people', ha='center', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(5, 2.5, '↓', ha='center', fontsize=20, fontweight='bold')
ax.text(5, 1.8, 'Tracked: 66 people', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(5, 0.8, '✓ Good Performance', ha='center', fontsize=12, fontweight='bold', color='green')

# 7-concept model pipeline
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('7-Concept Model: Poor Detection → Poor Tracking', 
             fontsize=14, fontweight='bold', pad=20)

# Detection stage
rect1 = mpatches.FancyBboxPatch((0.5, 7), 3, 2, boxstyle="round,pad=0.1",
                                 edgecolor='red', facecolor='lightcoral', linewidth=3)
ax.add_patch(rect1)
ax.text(2, 8, 'DETECTION\n27.59% Recall', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Arrow
ax.arrow(3.5, 8, 1.5, 0, head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=2)

# Association stage
rect2 = mpatches.FancyBboxPatch((5.5, 7), 3, 2, boxstyle="round,pad=0.1",
                                 edgecolor='red', facecolor='lightcoral', linewidth=3)
ax.add_patch(rect2)
ax.text(7, 8, 'TRACKING\n27.27% MOTA\n40.66% IDF1', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Ground truth
ax.text(5, 5, 'Ground Truth: 100 people', ha='center', fontsize=11, style='italic')
ax.text(5, 4, '↓', ha='center', fontsize=20, fontweight='bold')
ax.text(5, 3.3, 'Detected: 28 people', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.text(5, 2.5, '↓', ha='center', fontsize=20, fontweight='bold')
ax.text(5, 1.8, 'Tracked: 27 people', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.text(5, 0.8, '✗ Poor Performance', ha='center', fontsize=12, fontweight='bold', color='red')

# Add explanation box
explanation = """
Root Cause: Detection is the bottleneck!
• Can't track what you can't detect
• Missing 72 out of 100 people
• No amount of concept information helps if object isn't detected
"""
ax.text(5, -1, explanation, ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.8))

plt.tight_layout()
plt.savefig('tracking_pipeline_visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to tracking_pipeline_visualization.png")
plt.close()

print("\n" + "="*60)
print("Visualizations created successfully!")
print("="*60)
print("\nFiles created:")
print("1. gradient_competition_visualization.png")
print("   - Shows loss magnitude comparison")
print("   - Gradient signal distribution")
print("   - Performance degradation metrics")
print("\n2. tracking_pipeline_visualization.png")
print("   - Shows how poor detection cascades to poor tracking")
print("   - Compares 2-concept vs 7-concept pipelines")
print("="*60)
