# Concept Prediction and Ground Truth Logging Implementation

## Overview
Added comprehensive logging functionality to show concept predictions vs ground truth during MOTIP training with concept bottleneck models.

## Key Features

### 1. Enhanced Loss Function (`loss_concepts`)
- **Location**: `models/deformable_detr/deformable_detr.py`, line ~320
- **Features**:
  - Computes cross-entropy loss for concept classification
  - Ignores "Unknown" labels (label=2) during training
  - Calculates concept accuracy for logging
  - Returns both loss and accuracy metrics

### 2. Detailed Concept Logging (`log_concept_predictions`)
- **Location**: `models/deformable_detr/deformable_detr.py`, line ~431
- **Features**:
  - Shows concept prediction accuracy
  - Displays sample predictions vs ground truth with ✓/✗ indicators
  - Shows distribution of predictions and ground truth
  - Human-readable labels (Male/Female/Unknown)
  - Configurable number of samples to display

### 3. Automatic Logging Integration
- **Location**: `models/deformable_detr/deformable_detr.py`, SetCriterion.forward()
- **Features**:
  - Automatically calls logging every 50 batches
  - Prevents log spam while providing regular updates
  - Integrated into the loss computation pipeline

### 4. Training Metrics Integration
- **Location**: `train.py`, line ~448
- **Features**:
  - Adds concept accuracy to training metrics
  - Displayed alongside other losses in training logs
  - Tracked as "concept_acc" metric

## Configuration Updates

### Memory Optimization for GPU Training:
```yaml
# Reduced settings to prevent OOM
BATCH_SIZE: 1                    # Was 2
DETR_NUM_TRAIN_FRAMES: 2        # Was 4
NUM_WORKERS: 2                  # Was 6
PREFETCH_FACTOR: 2              # Was 6
NUM_CONCEPTS: 3                 # Was 312 (for gender: Male/Female/Unknown)
AUG_MAX_SIZE: 800              # Was 1440
AUG_RESIZE_SCALES: [400, 480, 512, 544, 576, 608]  # Reduced from larger scales
```

### SLURM Configuration:
```bash
#SBATCH --mem=32G               # Increased from 9G
#SBATCH --partition=gpu_a100_il # Changed to A100 partition
```

## Expected Output

### Training Logs Will Show:
```
[Epoch: 0] [50/XXX] loss: 2.45 | detr_loss: 1.89 | loss_concepts: 0.56 | concept_acc: 67.3

=== Concept Predictions (Batch 50) ===
  [Concepts] Accuracy: 67.3% (11/16)
  [Concepts] Sample predictions:
    ✓ Pred: Male | GT: Male
    ✗ Pred: Female | GT: Male
    ✓ Pred: Female | GT: Female
    ✗ Pred: Male | GT: Female
    ✓ Pred: Male | GT: Male
  [Concepts] Distribution - Pred: Male:7 Female:9 | GT: Male:8 Female:8
==================================================
```

### Key Metrics:
- **loss_concepts**: Cross-entropy loss for concept classification
- **concept_acc**: Accuracy percentage for concept predictions
- **Sample predictions**: Individual examples with match indicators
- **Distribution**: Count of each concept class in predictions vs ground truth

## Usage Notes

1. **Gender Labels**: 
   - 0 = Male
   - 1 = Female  
   - 2 = Unknown (ignored during training)

2. **Logging Frequency**: 
   - Detailed logs every 50 batches
   - Metrics logged every batch
   - Configurable via `self._log_counter % 50`

3. **Memory Considerations**:
   - Reduced batch size and image resolution
   - Fewer data loading workers
   - Should prevent OOM errors on 32GB RAM

4. **Debugging**:
   - Use `test_concept_logging.py` to test logging functionality
   - Check that concept targets are properly formatted
   - Verify matcher indices are correct

## Files Modified:
1. `models/deformable_detr/deformable_detr.py` - Core concept logging
2. `train.py` - Metrics integration  
3. `configs/r50_deformable_detr_motip_dancetrack.yaml` - Memory optimization
4. `train.sh` - SLURM resource allocation
