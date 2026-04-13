# Scripts Directory

Utility scripts for MOTIP environment setup, data preparation, and model management.

## Quick Start

```bash
# Full setup: environment + data download + SAM mask precomputation
./scripts/download_and_preprocess.sh

# Only setup environment (no data download)
./scripts/download_and_preprocess.sh --env-only

# Only download data (assumes environment exists)
./scripts/download_and_preprocess.sh --data-only

# Setup + download, but skip SAM mask precomputation
./scripts/download_and_preprocess.sh --skip-masks
```

## Scripts

### `download_and_preprocess.sh`
Main setup script that handles environment creation, data downloading, and SAM mask precomputation.

**Usage:**
```bash
cd /path/to/MOTIP
./scripts/download_and_preprocess.sh [OPTIONS]
```

**Options:**
- `--env-only` - Only setup conda environment
- `--data-only` - Only download datasets  
- `--masks-only` - Only precompute SAM masks
- `--skip-masks` - Skip SAM mask precomputation

**Environment Variables:**
- `ENV_NAME` - Conda environment name (default: `MOTIP_fresh`)
- `DATA_ROOT` - Data directory (default: `./data`)
- `MASK_ROOT` - SAM masks directory (default: `./precomputed_sam_masks`)
- `SAM_CKPT` - SAM checkpoint path (default: `./pretrains/sam_vit_b_01ec64.pth`)

**What it does:**
1. Creates conda environment with Python 3.12
2. Installs Python dependencies from `requirements.txt`
3. Builds CUDA operators (if GPU available)
4. Downloads DanceTrack dataset from Hugging Face
5. Precomputes SAM segmentation masks for training

---

### `download_dancetrack.py`
Downloads the DanceTrack dataset from Hugging Face.

**Usage:**
```bash
python scripts/download_dancetrack.py
python scripts/download_dancetrack.py --splits train val test
python scripts/download_dancetrack.py --output-dir /custom/path/DanceTrack
```

**Arguments:**
- `--output-dir` - Output directory (default: `./data/DanceTrack`)
- `--splits` - Splits to download: `train`, `val`, `test` (default: all)
- `--cache-dir` - HuggingFace cache directory
- `--token` - HuggingFace token (if needed)

---

### `precompute_sam_masks.py`
Precomputes SAM segmentation masks for the SAM Concept Bottleneck training mode.

**Usage:**
```bash
# Default: DanceTrack train split
python scripts/precompute_sam_masks.py

# DanceTrack val split
python scripts/precompute_sam_masks.py --split val

# P-DESTRE dataset
python scripts/precompute_sam_masks.py --dataset P-DESTRE --split Train_0

# Testing with limited data
python scripts/precompute_sam_masks.py --max-sequences 2 --max-frames 10 --debug-overlays
```

**Arguments:**
- `--dataset` - Dataset name (default: `DanceTrack`)
- `--split` - Dataset split (default: `train`)
- `--data-root` - Root directory containing datasets (default: `./data`)
- `--sam-checkpoint` - Path to SAM checkpoint (default: `./pretrains/sam_vit_b_01ec64.pth`)
- `--model-type` - SAM model type: `vit_b`, `vit_l`, `vit_h` (default: `vit_b`)
- `--save-root` - Output directory for masks (default: `./precomputed_sam_masks`)
- `--debug-overlays` - Save debug overlay visualizations
- `--max-sequences` - Limit number of sequences (for testing)
- `--max-frames` - Limit frames per sequence (for testing)
- `--device` - Device to run SAM on (default: `cuda`)

**Output Structure:**
```
precomputed_sam_masks/
├── dancetrack0001/
│   ├── 00000001/
│   │   ├── 1.png      # Track ID 1 mask
│   │   ├── 2.png      # Track ID 2 mask
│   │   └── ...
│   ├── 00000002/
│   └── ...
└── dancetrack0002/
    └── ...
```

**SAM Checkpoint:**
Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

---

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

## Directory Structure

After running the full setup, your MOTIP directory will look like:

```
MOTIP/
├── data/
│   └── DanceTrack/
│       ├── train/
│       ├── val/
│       └── test/
├── precomputed_sam_masks/
│   └── dancetrack0001/
│       └── 00000001/
│           └── {track_id}.png
├── pretrains/
│   └── sam_vit_b_01ec64.pth
└── scripts/
    ├── download_and_preprocess.sh
    ├── download_dancetrack.py
    ├── precompute_sam_masks.py
    └── ...
```

## Notes

- All scripts automatically detect the MOTIP root directory
- Can be run from any location (MOTIP root or scripts/ folder)
- SAM mask precomputation requires a GPU
- Training scripts remain in the root directory for easy access
