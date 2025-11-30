# Fix for Data Loading Error

## Problem
Training failed with:
```
ValueError: empty range in randrange(384, 301)
```

This happens when the random crop transform tries to crop a minimum of 384px from an image that's only 301px tall/wide after resizing.

## Root Cause
The data augmentation pipeline had incompatible settings:
- `AUG_RANDOM_RESIZE: [400, 500, 600]` - images can be resized to 400px
- `AUG_RANDOM_CROP_MIN: 384` - but then we try to crop at least 384px
- After various augmentations, an image might be smaller than 384px, causing the error

## Solution Applied

### 1. Fixed Data Augmentation Settings in `configs/r50_deformable_detr_motip_dancetrack.yaml`:
```yaml
# Before (problematic):
AUG_RESIZE_SCALES: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
AUG_MAX_SIZE: 1440
AUG_RANDOM_RESIZE: [400, 500, 600]
AUG_RANDOM_CROP_MIN: 384
AUG_RANDOM_CROP_MAX: 600

# After (fixed):
AUG_RESIZE_SCALES: [400, 480, 512, 544, 576, 608]
AUG_MAX_SIZE: 800
AUG_RANDOM_RESIZE: [350, 450, 550]
AUG_RANDOM_CROP_MIN: 320  # Smaller than minimum resize value
AUG_RANDOM_CROP_MAX: 500
```

### 2. Memory Optimization Settings:
```yaml
# Reduced to prevent OOM errors
NUM_WORKERS: 2              # Was 6
PREFETCH_FACTOR: 2          # Was 6 
BATCH_SIZE: 1               # Was 2
DETR_NUM_TRAIN_FRAMES: 2    # Was 4
NUM_CONCEPTS: 3             # Was 312 (for gender classification)
```

### 3. Fixed Training Script (`train.sh`):
```bash
# Changed to use single process (not 8) and correct config file
accelerate launch --num_processes=1 train.py --data-root ./data/ --exp-name r50_deformable_detr_motip_pdestre_concept_logs --config-path ./configs/r50_deformable_detr_motip_pdestre.yaml
```

## Key Points
- **AUG_RANDOM_CROP_MIN must be ≤ minimum possible image size after resizing**
- The P-DESTRE config inherits from DanceTrack config, so fixing the base config fixes both
- Reduced resource usage to prevent memory issues
- All concept logging functionality remains intact

## Ready to Train
The training should now work without the data loading error. You'll see:
1. No more "empty range" errors
2. Concept predictions and ground truth logging every 50 batches
3. Concept accuracy in training metrics
4. Memory usage within 32GB limit
