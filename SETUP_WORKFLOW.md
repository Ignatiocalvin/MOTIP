# MOTIP RF-DETR Training Setup Workflow

Complete workflow from cloning the repository to running training on cloud GPU.

---

## 1. Clone the Repository

```bash
# Clone MOTIP repository
git clone <your-motip-repo-url> MOTIP
cd MOTIP
```

---

## 2. Set Up Python Environment

```bash
# Create virtual environment (Python 3.8-3.12)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR: venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip
```

---

## 3. Install Core Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
# For CUDA 12.1+:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install accelerate wandb pillow opencv-python scipy pyyaml tqdm einops timm

# Install transformers (needed for RF-DETR)
pip install transformers

# For evaluation (optional)
pip install motmetrics lap
```

---

## 4. Download and Preprocess Dataset

```bash
# Run the automated setup script
# This will:
# - Download P-DESTRE dataset
# - Extract frames from videos
# - Clone RF-DETR repository
# - Apply compatibility fixes
# - Build CUDA operators
bash scripts/download_and_preprocess.sh
```

**Expected output:**
- `data/P-DESTRE/images/` - extracted frames
- `data/P-DESTRE/annotations/` - annotation files
- `data/P-DESTRE/splits/` - train/val/test splits
- `../rf-detr/` - RF-DETR repository with fixes
- CUDA operators built successfully

---

## 5. Verify Code Fixes (Already in Repository)

The following fixes are already committed to the repository:

### ✅ ColorJitter Compatibility (`data/transforms.py`)
- Handles `get_params` vs `_get_params` API change in torchvision

### ✅ Missing Image Handling (`data/joint_dataset.py`)  
- Gracefully skips missing image files without crashing

### ✅ RF-DETR Debug Logs Removed (`models/rfdetr/rfdetr_motip.py`)
- Cleaned up excessive debug output

### ✅ Training Logging Frequency (`train.py`)
- Metrics display every 50 steps (was 20,000)

### ✅ RF-DETR Transformers Compatibility
- Automatically applied by `download_and_preprocess.sh`
- Fixes for transformers v5.x API changes

**If cloning fresh:** All fixes are already applied - no manual changes needed!

---

## 6. (Optional) Filter Dataset for Faster Training

If you want to train on a subset (recommended for testing):

```bash
# Edit the filter ratio in the script
nano data/P-DESTRE/filter_splits.py
# Change: KEEP_RATIO = 0.5  # Use 50% of sequences
# Change: KEEP_RATIO = 0.1  # Use 10% of sequences

# Run the filter
python data/P-DESTRE/filter_splits.py
```

**This will:**
- Backup original split files (`.original` suffix)
- Randomly sample X% of sequences
- Update all train/val/test splits

**To restore originals later:**
```bash
cd splits
rm *.txt
for f in *.original; do mv "$f" "${f%.original}"; done
```

---

## 7. Configure Training

### 7.1. Check Configuration Files

Main config: `configs/rfdetr_medium_motip_pdestre.yaml`

**Key settings:**
```yaml
# Training
EPOCHS: 12
BATCH_SIZE: 2
LR: 0.0002

# Dataset
NUM_CONCEPTS: 7  # IMPORTANT: Must be 7 for P-DESTRE

# Model
BACKBONE: rfdetr_medium
DETR_NUM_QUERIES: 300
```

### 7.2. Verify Splits Exist

```bash
ls splits/Train_*.txt splits/Test_*.txt splits/val_*.txt
# Should see 10 files for each (30 total for 10-fold CV)
```

---

## 8. Run Training

### 8.1. Single Fold (for testing)

```bash
# Test with fold 0 only
bash train_rfdetr.sh
```

The script will:
- Use fold 0 by default (modify line 152 in `train_rfdetr.sh` to change)
- Save to `outputs/rfdetr_motip_pdestre_fold_0/`
- Log metrics every 50 steps
- Save checkpoints periodically

### 8.2. All 10 Folds (full training)

```bash
# Edit train_rfdetr.sh line 152
nano train_rfdetr.sh
# Change: for FOLD in {0..0}
# To:     for FOLD in {0..9}

# Run full 10-fold cross-validation
bash train_rfdetr.sh
```

**Time estimate:** ~12-48 hours per fold depending on GPU.

---

## 9. Monitor Training

### 9.1. Check Training Progress

**Logs appear every 50 steps:**
```
[Epoch: 0] [50/16283] [tps: 2.5s] [eta: 11:23:45] loss: 5.234 lr: 0.0001 ...
[Epoch: 0] [100/16283] [tps: 2.4s] [eta: 11:18:32] loss: 4.987 lr: 0.0001 ...
```

**Key metrics:**
- `loss` - Total loss (should decrease)
- `detr_loss` - Detection loss
- `id_loss` - Identity/tracking loss (if not `only_detr`)
- `concept_acc` - Concept prediction accuracy
- `lr` - Current learning rate
- `tps` - Time per step (seconds)
- `eta` - Estimated time remaining

### 9.2. Check Output Files

```bash
# Checkpoints
ls outputs/rfdetr_motip_pdestre_fold_0/checkpoint_*.pth

# Logs
tail -f outputs/rfdetr_motip_pdestre_fold_0/train/logs.txt

# Wandb (if enabled)
# Check: https://wandb.ai/your-project
```

---

## 10. Troubleshooting

### Issue: "FileNotFoundError: image not found"
**Solution:** Run the dataset filter or preprocessing again:
```bash
cd data/P-DESTRE
python preprocess_pdestre.py --force
```

### Issue: "ModuleNotFoundError: No module named 'rfdetr'"
**Solution:** RF-DETR must be in parent directory:
```bash
ls ../rf-detr/  # Should exist
# If not, re-run: bash scripts/download_and_preprocess.sh
```

### Issue: ImportError from transformers
**Solution:** The compatibility fixes should handle this. Verify:
```bash
ls ../rf-detr/rfdetr/models/backbone/dinov2_with_windowed_attn.py
# Should show the fixed file
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size in config:
```yaml
BATCH_SIZE: 1  # Instead of 2
```

### Issue: Training too slow
**Solution:** 
1. Use filtered dataset (10-30% of sequences)
2. Check GPU utilization: `nvidia-smi`
3. Reduce `DETR_NUM_CHECKPOINT_FRAMES` in config

---

## 11. After Training

### 11.1. Evaluate Model

```bash
# Run evaluation on test set
cd evaluation
bash evaluate_fold.sh 0  # For fold 0
```

### 11.2. Visualize Results

```bash
# Generate visualizations
bash generate_all_visualizations.sh
```

### 11.3. Extract Metrics

```bash
# Get summary metrics
python extract_metrics.py --fold 0
```

---

## Quick Start (TL;DR)

```bash
# 1. Clone and setup
git clone <repo-url> MOTIP && cd MOTIP
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision accelerate wandb pillow opencv-python scipy pyyaml tqdm einops timm transformers

# 2. Download and preprocess (includes RF-DETR setup + fixes)
bash scripts/download_and_preprocess.sh

# 3. (Optional) Filter dataset for faster training
python data/P-DESTRE/filter_splits.py  # Edit KEEP_RATIO first

# 4. Start training
bash train_rfdetr.sh
```

---

## Important Notes

1. **GPU Memory:** Training requires ~16GB VRAM with batch_size=2
2. **Dataset Size:** Full P-DESTRE is large (~33k frames). Consider filtering to 10-30% for initial experiments
3. **Checkpoints:** Save every N epochs (default: every 3 epochs)
4. **Wandb:** Set `USE_WANDB: true` in config for online monitoring
5. **Debug Mode:** Training metrics log every 50 steps (was 20,000)

---

## File Checklist

All fixes are already in the repository:

- ✅ `data/transforms.py` - ColorJitter compatibility
- ✅ `data/joint_dataset.py` - Missing image handling  
- ✅ `models/rfdetr/rfdetr_motip.py` - Debug logs removed
- ✅ `train.py` - Logging interval = 50 steps
- ✅ `rf-detr/` compatibility - Applied automatically by setup script

**No manual file updates needed after cloning!**

---

## Expected Timeline

| Task | Time |
|------|------|
| Setup environment | 10-15 min |
| Download dataset | 20-30 min |
| Preprocess (extract frames) | 15-30 min |
| Build CUDA operators | 2-5 min |
| Training (1 fold, 12 epochs) | 12-48 hours |
| Training (10 folds) | 5-20 days |

**Recommendation:** Start with filtered dataset (10-30%) for initial testing, then scale up.
