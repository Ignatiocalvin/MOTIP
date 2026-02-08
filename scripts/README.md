# Scripts Directory

Utility scripts for MOTIP dataset setup and model management.

## Scripts

### `download_and_preprocess.sh`
Downloads and prepares the P-DESTRE dataset for training.

**Usage:**
```bash
cd /path/to/MOTIP
./scripts/download_and_preprocess.sh
```

**What it does:**
- Downloads P-DESTRE dataset from the official source
- Extracts the dataset to `data/P-DESTRE/`
- Removes corrupted sequences (22-10-2019-1-2, 13-11-2019-4-3)
- Extracts frames from videos using `preprocess_each.py`
- Verifies dataset structure

### `slim_outputs.sh`
Creates a compressed archive of model checkpoints for cloud GPU inference.

**Usage:**
```bash
cd /path/to/MOTIP
./scripts/slim_outputs.sh
```

**What it does:**
- Keeps only the latest checkpoint from each training fold
- Preserves results.json and evaluation metrics
- Removes large training artifacts (intra-epoch checkpoints, logs, visualizations)
- Creates `outputs_for_inference.tar.gz` (~5GB instead of ~152GB)
- Generates `checkpoint_inventory.txt` as backup reference

**Output:**
- `outputs_slim/` - Slim directory with essential files only
- `outputs_for_inference.tar.gz` - Compressed archive ready for download
- `checkpoint_inventory.txt` - List of all original checkpoints

## Notes

- All scripts automatically detect the MOTIP root directory
- Can be run from any location
- Training scripts remain in the root directory for easy access
