# MOTIP with Multi-Concept Prediction and Concept Bottleneck Integration

This document outlines the modifications made to the original MOTIP repository to add **multi-concept prediction** (Concept Bottleneck Model integration) and **concept-based identity prediction** for multi-object tracking.

## Overview

The original MOTIP (Multi-Object Tracking with Identity Prediction) has been extended to include:
1. **Multi-Concept Prediction** - Predicts multiple person attributes simultaneously (gender, hairstyle, clothing, accessories)
2. **Concept Bottleneck Integration** - Uses predicted concepts to enhance identity prediction
3. **P-DESTRE Dataset Support** - Full integration with the P-DESTRE dataset and its annotation format
4. **RF-DETR Integration** - Support for RF-DETR backbone (DINOv2-based) as an alternative to Deformable DETR

### Supported Concepts (P-DESTRE Dataset)
| Attributes | Classes | Unknown Label |
|---------|---------|---------------|
| Gender | 3 (Male, Female, Unknown) | 2 | # OK
| Hairstyle | 6 (Bald, Short, Medium, Long, Horse Tail, Unknown) | 5 | # No
| Head Accessories | 5 (Hat, Scarf, Neckless, Cannot see, Unknown) | 4 | # No
| Upper Body | 13 (T-Shirt, Blouse, Sweater, Coat, ..., Unknown) | 12 | # Reasonable
| Lower Body | 10 (Jeans, Leggins, Pants, Shorts, ..., Unknown) | 9 |  # Reasonable
| Feet | 7 (Sport Shoe, Classic Shoe, High Heels, ..., Unknown) | 6 | # No
| Accessories | 8 (Bag, Backpack, Rolling Bag, Umbrella, ..., Unknown) | 7 | # Unsure

## Key Modifications

### 1. Core Model Architecture Changes

#### A. DeformableDETR Model (`models/deformable_detr/deformable_detr.py`)

**Multi-Concept Prediction Architecture:**
- **Lines 42-75**: Added multi-concept support with `concept_classes` parameter
  ```python
  # concept_classes: list of class counts per concept [3, 6, 5, 13, 10, 7, 8]
  self.concept_embeds = nn.ModuleList()  # Separate MLP head for each concept type
  for n_classes in concept_classes:
      self.concept_embeds.append(MLP(hidden_dim, hidden_dim, n_classes, 3))
  ```

- **Lines 125-133**: Concept embed bias initialization with per-concept handling

- **Lines 136-160**: Concept head cloning for auxiliary losses (`with_box_refine` support)

- **Lines 215-248**: Forward pass multi-concept prediction loop
  ```python
  for concept_idx, concept_embed in enumerate(self.concept_embeds):
      outputs_concepts[concept_idx].append(concept_embed[lvl](hs[lvl]))
  ```

- **Lines 251-262**: Output dictionary with `pred_concepts` as list of per-concept predictions

- **Lines 280-303**: `_set_aux_loss_multi_concept` method for auxiliary loss handling

- **Lines 408-475**: `loss_concepts` method with per-concept cross-entropy and unknown label filtering
  ```python
  def loss_concepts(self, outputs, targets, indices, num_boxes, **kwargs):
      """Multi-concept classification loss with per-concept unknown filtering."""
      for concept_idx, (concept_name, n_classes, unknown_label) in enumerate(self.concept_classes):
          # Filter out Unknown labels during training
          valid_mask = (target_concepts[:, concept_idx] != unknown_label)
          loss = F.cross_entropy(src_logits_valid, target_valid)
  ```

- **Lines 535-590**: `debug_concept_targets` with multi-concept debugging

- **Lines 592-680**: `log_concept_predictions` with per-concept accuracy logging

- **Lines 860-940**: `build()` function with multi-concept support and per-concept loss weights

#### B. Loss Function Enhancements
- **Multi-concept loss**: Cross-entropy loss per concept type
- **Unknown label handling**: Per-concept unknown label filtering (configurable via `concept_classes`)
- **Per-concept weight_dict**: Separate loss weights like `loss_gender`, `loss_upper_body`
- **Multi-layer supervision**: Concept losses computed for all decoder layers

### 2. Concept-Based Identity Prediction (Concept Bottleneck Integration)

#### A. IDDecoder Model (`models/motip/id_decoder.py`)

**Key additions for concept bottleneck integration:**
- **Lines 24-27**: New constructor parameters for concept integration
  ```python
  concept_classes: Optional[List[tuple]] = None,  # [(name, n_classes, unknown_label), ...]
  concept_dim: int = 0,  # Dimension of concept embeddings (0 = disabled)
  ```

- **Lines 45-56**: Concept embedding layer setup
  ```python
  # Total embedding dimension: features + concepts + id
  self.total_embed_dim = self.feature_dim + self.concept_dim + self.id_dim
  
  # Concept embedding layers: convert one-hot concept labels to embeddings
  total_concept_classes = sum(n_classes for _, n_classes, _ in self.concept_classes)
  self.concept_to_embed = nn.Linear(total_concept_classes, self.concept_dim, bias=False)
  ```

- **Lines 202-225**: Forward pass concept embedding integration
  ```python
  # Concatenate: [features, concept_embeds, id_embeds]
  trajectory_embeds = torch.cat([trajectory_features, trajectory_concept_embeds, trajectory_id_embeds], dim=-1)
  ```

- **Lines 360-390**: `concept_labels_to_embed` method for one-hot encoding and projection

#### B. MOTIP Builder (`models/motip/__init__.py`)
- **Lines 172-190**: Reads `CONCEPT_DIM` from config and passes to IDDecoder

### 3. Data Pipeline Modifications

#### A. Data Transforms (`data/transforms.py`)

**Concept field preservation in augmentations:**
- **Lines 265-270**: Added concepts to field filtering in `MultiRandomCrop`
  ```python
  _need_to_select_fields = ["bbox", "category", "id", "visibility"]
  if "concepts" in _annotation:
      _need_to_select_fields.append("concepts")
  ```

#### B. P-DESTRE Dataset (`data/pdestre.py`) - **NEW FILE**

Complete P-DESTRE dataset implementation with:
- **Lines 1-50**: Dataset class inheriting from DanceTrack
- **Lines 50-100**: Split file loading (`Train_0.txt`, `Test_0.txt`, `val_0.txt`)
- **Lines 100-175**: Sequence info construction with image path validation
- **Lines 175-230**: Custom image path methods for P-DESTRE folder structure
- **Lines 230-255**: `_init_annotations` with 7-concept 2D tensor support
  ```python
  "concepts": torch.empty(size=(0, 7), dtype=torch.int64),  # 2D tensor for 7 concepts
  ```
- **Lines 255-331**: Annotation loading from P-DESTRE format (25 columns)
  ```python
  # Parse all 7 concepts from P-DESTRE columns
  gender = int(items[10])
  hairstyle = int(items[16])
  head_accessories = int(items[20])
  upper_body = int(items[21])
  lower_body = int(items[22])
  feet = int(items[23])
  accessories = int(items[24])
  ```

#### C. Joint Dataset (`data/joint_dataset.py`)
- **Lines 17-26**: Added P-DESTRE to `dataset_classes` registry
  ```python
  dataset_classes = {
      "DanceTrack": DanceTrack,
      "P-DESTRE": PDESTRE,  # Must match config string exactly
      ...
  }
  ```
- **Lines 70-95**: Path correction hotfix for P-DESTRE image paths

#### D. Data Utilities (`data/util.py`)
- **Lines 36-47**: Updated `is_legal` to validate concepts tensor
- **Lines 65-100**: Updated `append_annotation` for multi-concept support
  ```python
  if isinstance(concepts, (list, tuple)):
      concepts_tensor = torch.tensor([concepts], dtype=torch.int64)  # 2D shape
  ```

#### E. Training Pipeline (`train.py`)
- **Lines 6-28**: RF-DETR path setup for optional backbone integration
- **Concept target handling**: Concepts passed through to DETR targets

### 4. Runtime Tracker with Concept Support

#### A. Runtime Tracker (`models/runtime_tracker.py`)

**Key modifications for inference with concepts:**
- **Lines 85-90**: Added `trajectory_concepts` tensor field
  ```python
  self.trajectory_concepts = torch.zeros(
      (0, 0, 0), dtype=torch.int64, device=distributed_device(),
  )
  ```

- **Lines 252-253**: Passing concepts to seq_info for ID decoder
- **Lines 364-420**: **Fixed dimension mismatch bug** in `_update_trajectory_infos`
  ```python
  # Handle dimension mismatch when trajectory count changes
  if self.trajectory_concepts.shape[1] != _N:
      if _N > _N_concepts:
          _pad = torch.zeros((_T_concepts, _N - _N_concepts, num_concepts), ...)
          self.trajectory_concepts = torch.cat([self.trajectory_concepts, _pad], dim=1)
  ```
- **Lines 430-435**: Filtering concepts when filtering inactive tracks

### 5. Configuration System

#### A. Available Configurations

| Config File | Description |
|-------------|-------------|
| `r50_deformable_detr_motip_pdestre_concepts_for_id.yaml` | **Concept-based ID prediction** - uses concepts to enhance ID matching |
| `r50_deformable_detr_motip_pdestre_fast.yaml` | Fast training with all 7 concepts, larger intervals |
| `r50_deformable_detr_motip_pdestre_standard.yaml` | Standard training config |
| `rfdetr_medium_motip_pdestre.yaml` | RF-DETR backbone with DINOv2 |

#### B. Key Configuration Parameters

```yaml
MOTIP:
  N_CONCEPTS: 7  # Number of concept types (not total classes)
  CONCEPT_DIM: 64  # NEW: Enables concept embeddings in IDDecoder (0 = disabled)
  CONCEPT_CLASSES:  # Multi-concept definition: [name, num_classes, unknown_label]
    - ["gender", 3, 2]
    - ["hairstyle", 6, 5]
    - ["head_accessories", 5, 4]
    - ["upper_body", 13, 12]
    - ["lower_body", 10, 9]
    - ["feet", 7, 6]
    - ["accessories", 8, 7]
  DETR_LOSSES: [labels, boxes, cardinality, concepts]
  CONCEPT_LOSS_COEF: 0.5

# Auto-resume from checkpoint (NEW)
USE_PREVIOUS_CHECKPOINT: True

# Intra-epoch checkpointing (for long epochs)
SAVE_CHECKPOINT_EVERY_N_STEPS: 5000
RESUME_FROM_STEP: 0
```

#### C. RF-DETR Configuration (`rfdetr_medium_motip_pdestre.yaml`)

```yaml
DETR_FRAMEWORK: rf_detr  # Switch from deformable_detr

RFDETR:
  ENCODER: dinov2_windowed_small  # DINOv2 backbone
  RESOLUTION: 588
  HIDDEN_DIM: 256
  DEC_LAYERS: 4
  PATCH_SIZE: 14
  TWO_STAGE: True
  LOAD_DINOV2_WEIGHTS: True  # Auto-download DINOv2 weights
```

### 6. Training Scripts

| Script | Description |
|--------|-------------|
| `train_fast.sh` | Fast training with concept-for-ID config |
| `train_standard_motip.sh` | Standard MOTIP training |
| `train_rfdetr.sh` | RF-DETR backbone training |
| `eval_checkpoint0.sh` | Evaluate checkpoint_0 |
| `evaluate_checkpoint.py` | General checkpoint evaluation script |

## How to Run Training

### 1. Environment Setup
```bash
cd /path/to/MOTIP
conda activate MOTIP

# Verify RF-DETR is available (optional, for RF-DETR training)
ls rf-detr/  # Should contain rfdetr package
```

### 2. Data Structure
```
data/
└── P-DESTRE/
    ├── images/
    │   └── <sequence_name>/
    │       └── img1/
    │           └── 000001.jpg, ...
    ├── annotations/
    │   └── <sequence_name>.txt  # 25-column P-DESTRE format
    └── splits/
        ├── Train_0.txt, Train_1.txt, ...
        ├── Test_0.txt, Test_1.txt, ...
        └── val_0.txt, val_1.txt, ...
```

### 3. Training Commands

**Standard MOTIP with Concepts:**
```bash
./train_fast.sh
# or
accelerate launch --num_processes=1 train.py \
    --data-root ./data/ \
    --config-path ./configs/r50_deformable_detr_motip_pdestre_concepts_for_id.yaml \
    --exp-name my_experiment
```

**RF-DETR Backbone:**
```bash
./train_rfdetr.sh
```

### 4. Checkpoint Evaluation
```bash
# Evaluate specific checkpoint
python evaluate_checkpoint.py \
    --checkpoint outputs/<exp_name>/train/checkpoint_0.pth \
    --dataset P-DESTRE \
    --split Test_0

# Quick evaluation of checkpoint_0
./eval_checkpoint0.sh
```

### 5. Resume Training
Training automatically resumes from the last checkpoint when:
- `USE_PREVIOUS_CHECKPOINT: True` is set in config
- Checkpoint files exist in the output directory

For mid-epoch resume:
```yaml
RESUME_FROM_STEP: 5000  # Resume from step 5000
```

## Output Structure
```
outputs/
└── <exp_name>/
    ├── train/
    │   ├── log.txt
    │   ├── checkpoint_0.pth, checkpoint_1.pth, ...
    │   ├── step_checkpoint_5000.pth  # Intra-epoch checkpoint
    │   └── eval_during_train/
    └── multi_last_checkpoints/
```

## File Structure Changes

```
MOTIP/
├── train.py                          # ✓ RF-DETR path setup, concept handling
├── train_fast.sh                     # ✓ Fast training script
├── train_rfdetr.sh                   # ✓ RF-DETR training script  
├── evaluate_checkpoint.py            # ✓ NEW: Standalone evaluation
├── eval_checkpoint0.sh               # ✓ NEW: Quick checkpoint_0 eval
├── models/
│   ├── deformable_detr/
│   │   └── deformable_detr.py        # ✓ Multi-concept prediction heads
│   ├── motip/
│   │   ├── __init__.py               # ✓ Concept dim passing to IDDecoder
│   │   └── id_decoder.py             # ✓ Concept bottleneck integration
│   └── runtime_tracker.py            # ✓ Concept tracking + dimension fix
├── data/
│   ├── pdestre.py                    # ✓ NEW: P-DESTRE dataset
│   ├── joint_dataset.py              # ✓ P-DESTRE registration + path fix
│   ├── transforms.py                 # ✓ Concept field preservation
│   └── util.py                       # ✓ Multi-concept append/validation
├── configs/
│   ├── r50_deformable_detr_motip_pdestre_concepts_for_id.yaml  # ✓ Concept-for-ID
│   ├── r50_deformable_detr_motip_pdestre_fast.yaml             # ✓ Fast training
│   ├── r50_deformable_detr_motip_pdestre_standard.yaml         # ✓ Standard
│   └── rfdetr_medium_motip_pdestre.yaml                        # ✓ RF-DETR config
└── rf-detr/                          # ✓ RF-DETR submodule (optional)
```

## Technical Implementation Details

### Multi-Concept Prediction Architecture
```python
# DeformableDETR with multi-concept heads
self.concept_embeds = nn.ModuleList()
for n_classes in concept_classes:  # [3, 6, 5, 13, 10, 7, 8]
    self.concept_embeds.append(MLP(hidden_dim, hidden_dim, n_classes, 3))

# Output: list of predictions per concept type
out['pred_concepts'] = [concept_preds[i][-1] for i in range(num_concepts)]
```

### Concept Bottleneck for ID Prediction
```python
# IDDecoder with concept integration
# 1. Convert concept labels to one-hot
# 2. Project to concept embedding space
# 3. Concatenate with visual features and ID embeddings
trajectory_embeds = torch.cat([
    trajectory_features,      # Visual features (256-dim)
    trajectory_concept_embeds,  # Concept embeddings (concept_dim)
    trajectory_id_embeds        # ID embeddings (id_dim)
], dim=-1)
# Total: feature_dim + concept_dim + id_dim
```

### P-DESTRE Annotation Format
```
# 25-column format:
# Frame,ID,x,y,h,w,conf,world_x,world_y,world_z,Gender,...,Hairstyle,...,Accessories
# Column indices: gender=10, hairstyle=16, head_accessories=20, upper_body=21, 
#                 lower_body=22, feet=23, accessories=24
```

### Data Structure
```python
# Enhanced annotation format with multi-concept support
annotation = {
    "bbox": tensor,      # Bounding boxes (N, 4)
    "category": tensor,  # Category labels (N,)
    "id": tensor,        # Track IDs (N,)
    "visibility": tensor,  # Visibility scores (N,)
    "concepts": tensor,  # NEW: Multi-concept labels (N, 7)
    # concepts[:, 0] = gender, concepts[:, 1] = hairstyle, etc.
}
```

## Bug Fixes

### 1. Runtime Tracker Dimension Mismatch (Fixed)
**Issue**: `RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 52 but got size 50`

**Cause**: `trajectory_concepts` tensor dimension mismatch when trajectory count changes between frames.

**Fix** ([runtime_tracker.py](models/runtime_tracker.py#L402-L420)):
```python
if self.trajectory_concepts.shape[1] != _N:
    if _N > _N_concepts:
        _pad = torch.zeros((_T_concepts, _N - _N_concepts, num_concepts), ...)
        self.trajectory_concepts = torch.cat([self.trajectory_concepts, _pad], dim=1)
```

### 2. P-DESTRE Dataset Registration
**Issue**: `KeyError: 'PDESTRE'` when loading dataset

**Fix**: Use `"P-DESTRE"` (with hyphen) in config to match `dataset_classes` registry.

## Compatibility

- ✅ Backward compatible with original MOTIP (set `N_CONCEPTS: 0`)
- ✅ Works with existing Deformable DETR pretrained models
- ✅ Optional RF-DETR backbone support
- ✅ Supports all original training configurations
- ✅ Auto-resume from checkpoints

## Future Extensions

The multi-concept framework can be extended to:
- Additional person attributes (age, pose, etc.)
- Other datasets with soft/rich annotations
- Concept-guided re-identification
- Temporal concept consistency constraints

---

*This implementation demonstrates Concept Bottleneck Model integration for multi-object tracking, combining visual features with semantic concept predictions to enhance identity association.*
