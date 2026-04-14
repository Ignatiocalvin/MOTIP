# MOTIP — Complete Reproduction Guide

> **Multi-Object Tracking with Identity Prediction (MOTIP)** on the P-DESTRE dataset, with concept bottleneck extensions and RF-DETR backbone support.

This guide walks through every step needed to reproduce the experiments: downloading the dataset, setting up the environment, running training, and evaluating results.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start (3 Steps)](#2-quick-start-3-steps)
3. [What the Setup Script Does](#3-what-the-setup-script-does)
4. [Repository Structure](#4-repository-structure)
5. [Configuration Overview](#5-configuration-overview)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Visualization](#8-visualization)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

- Access to **bwUniCluster** (or any SLURM cluster with NVIDIA H100 / A100 GPUs)
- `miniconda3` or `anaconda3` installed in your home directory
- SSH access for file transfers
- ~200 GB free disk space (dataset + weights + outputs)

---

## 2. Quick Start (3 Steps)

### Step 1: Download P-DESTRE manually

The P-DESTRE dataset (~30 GB) must be downloaded manually from Google Drive because the original download link was invalidated by the dataset owners.

Open this link in your browser and download the archive:

> **https://drive.google.com/file/d/1yan3gA59xzMdKIPsf6UAfgwTAJQLQUeg/view?usp=drive_link**

### Step 2: Upload and extract on the cluster

From your local machine, upload the downloaded file:

```bash
scp dataset.tar <user>@bwunicluster.scc.kit.edu:/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/data/
```

Then on the cluster, extract it:

```bash
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/data
tar -xf dataset.tar
```

This creates `data/P-DESTRE/` with `annotation/` and `videos/` subdirectories.

### Step 3: Run the setup script

The setup script handles **everything else** automatically:

```bash
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP
./scripts/download_and_preprocess.sh
```

This will:
1. **Create conda environment** (`MOTIP`) and install all Python dependencies
2. **Preprocess P-DESTRE** — rename folders, remove bad sequences, extract video frames to JPEG
3. **Download pretrained weights** — R50 Deformable DETR, RF-DETR Large, SAM ViT-B
4. **Download DanceTrack** dataset from HuggingFace
5. **Clone RF-DETR** repository and apply compatibility fixes
6. **Build CUDA extension** (if run on a GPU node; otherwise deferred to first training job)

The script is idempotent — it skips any step that was already completed, so it's safe to re-run.

> **Note on CUDA extension**: If you run the script on a login node (no GPU), the CUDA extension cannot be built there. It will be built automatically when you submit your first training job. Alternatively, build it manually on a GPU node:
> ```bash
> srun --partition=gpu_h100 --gres=gpu:1 --time=00:30:00 --mem=8G --pty bash
> conda activate MOTIP
> module load devel/cuda/11.8
> export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8
> cd models/ops && python setup.py build install && cd ../..
> exit
> ```

### Verify with a smoke test

After the script finishes, submit a quick 2-epoch training job to validate the full pipeline:

```bash
sbatch scripts/smoke_test.sh
```

Check the result:

```bash
squeue -u $USER                # Check job status
cat logs/smoke_test_*.out      # View output when done
```

If it completes without errors and creates `outputs/smoke_test/`, everything is working.

---

## 3. What the Setup Script Does

For reference, here is what `scripts/download_and_preprocess.sh` does in detail:

### P-DESTRE preprocessing

The raw P-DESTRE download contains MP4 videos and annotations. The script converts it into the format MOTIP expects:

1. **Renames** `annotation/` → `annotations/` (the raw tar uses the singular form)
2. **Removes** known problematic sequences (`22-10-2019-1-2`, `13-11-2019-4-3`) that have corrupted data
3. **Extracts frames** from each MP4 video into `images/{sequence}/img1/000001.jpg, 000002.jpg, ...` using `data/P-DESTRE/preprocess_pdestre.py`
4. **Deletes** the MP4 videos after extraction to reclaim ~30 GB of space

After preprocessing, the P-DESTRE directory looks like:

```
data/P-DESTRE/
├── images/                          # Extracted JPEG frames
│   ├── 13-11-2019-4-2/
│   │   └── img1/
│   │       ├── 000001.jpg           (6-digit zero-padded, 1-indexed)
│   │       ├── 000002.jpg
│   │       └── ...
│   └── ...                          (~70 sequences)
├── annotations/                     # Per-sequence annotation files
│   ├── 13-11-2019-4-2.txt
│   └── ...
└── splits/                          # 10-fold cross-validation
    ├── Train_0.txt ... Train_9.txt
    ├── val_0.txt   ... val_9.txt
    └── Test_0.txt  ... Test_9.txt
```

**Image resolution**: 3840 × 2160 @ 30 fps

**Annotation format** (each line):
```
frame_id, track_id, x, y, w, h, -1, -1, -1, -1, gender, hairstyle, head_acc, upper_body, lower_body, feet, accessories, ...
```

### Pretrained weights

Downloaded to `pretrains/`:

| File | Size | Used by |
|------|------|---------|
| `r50_deformable_detr_coco.pth` | 467 MB | R50 Deformable DETR experiments |
| `rf-detr-large.pth` | 1.5 GB | RF-DETR DINOv2-Large experiments |
| `sam_vit_b_01ec64.pth` | 358 MB | SAM mask experiments |

### DanceTrack dataset

Downloaded from HuggingFace to `data/DanceTrack/` with train/val/test splits. Used for SAM mask experiments.

### Script options

```bash
./scripts/download_and_preprocess.sh             # Full setup (default)
./scripts/download_and_preprocess.sh --env-only  # Only create env + install pip packages
./scripts/download_and_preprocess.sh --skip-env  # Skip env creation, do everything else
./scripts/download_and_preprocess.sh --help      # Show usage
```

You can also override the conda environment name:

```bash
ENV_NAME=my_env ./scripts/download_and_preprocess.sh
```

---

## 4. Repository Structure

```
MOTIP/
├── train.py                    # Main training script (uses Accelerate)
├── train_r50.sh                # SLURM script: R50 Deformable DETR training
├── train_rf-detr.sh            # SLURM script: RF-DETR training
├── requirements.txt            # Python dependencies
├── runtime_option.py           # Command-line argument definitions
│
├── configs/                    # YAML experiment configurations
│   ├── r50_deformable_detr_motip_pdestre_*.yaml
│   ├── rfdetr_large_motip_pdestre_*.yaml
│   └── smoke_test*.yaml
│
├── data/                       # Dataset loaders
│   ├── P-DESTRE/               # Dataset files (images + annotations)
│   ├── pdestre.py              # P-DESTRE dataset class
│   ├── dancetrack.py           # Base dataset class
│   └── joint_dataset.py        # Multi-dataset wrapper
│
├── models/                     # Model architectures
│   ├── motip/                  # MOTIP model (tracker + ID decoder + concepts)
│   ├── deformable_detr/        # Deformable DETR detector
│   ├── rfdetr/                 # RF-DETR detector wrapper
│   └── ops/                    # CUDA extensions (MultiScaleDeformableAttention)
│
├── splits/                     # 10-fold cross-validation split files
├── pretrains/                  # Pretrained backbone weights
├── rf-detr/                    # RF-DETR submodule (cloned separately)
├── outputs/                    # Training outputs (checkpoints, logs, metrics)
├── logs/                       # SLURM job output files
│
├── evaluation/                 # Evaluation & visualization tools
│   ├── submit_and_evaluate.py  # Core inference + metric computation
│   ├── evaluate_fold.sh        # SLURM evaluation submission
│   ├── extract_metrics.py      # Parse evaluation logs → JSON
│   ├── visualize_results.py    # Generate metric charts
│   └── generate_all_visualizations.sh
│
├── scripts/                    # Utility scripts
│   ├── smoke_test.sh           # Quick validation (R50)
│   ├── smoke_test_rfdetr.sh    # Quick validation (RF-DETR)
│   └── ...
│
└── TrackEval/                  # External tracking evaluation library
```

---

## 5. Configuration Overview

All experiments are defined by YAML config files in `configs/`. Configs use inheritance — a child config specifies `SUPER_CONFIG_PATH` to inherit from a base.

### Concept attributes (P-DESTRE)

The model can predict 0 to 7 semantic person attributes alongside tracking:

| # | Attribute | Classes | Description |
|---|-----------|---------|-------------|
| 1 | Gender | 3 | Male, Female, Unknown |
| 2 | Hairstyle | 6 | Bald, Short, Medium, Long, Horse Tail, Unknown |
| 3 | Head Accessories | 5 | Hat, Scarf, Neckless, Cannot see, Unknown |
| 4 | Upper Body | 13 | T-Shirt, Blouse, Sweater, Coat, Suit, Dress, etc. |
| 5 | Lower Body | 10 | Jeans, Trousers, Shorts, Skirt, etc. |
| 6 | Feet | 7 | Sport Shoe, Classic Shoe, High Heels, etc. |
| 7 | Accessories | 8 | Bag, Backpack, Rolling Bag, Umbrella, etc. |

### Experiment matrix

#### R50 Deformable DETR backbone

| Config file | Concepts | Description |
|-------------|----------|-------------|
| `r50_deformable_detr_motip_pdestre_base_fold0.yaml` | 0 | Baseline (detection + tracking only) |
| `r50_deformable_detr_motip_pdestre_2concepts_fold0_v2.yaml` | 2 | Gender + Upper Body |
| `r50_deformable_detr_motip_pdestre_3concepts.yaml` | 3 | Gender + Upper Body + Lower Body |
| `r50_motip_pdestre_7concepts_lw.yaml` | 7 | All concepts, learnable task weights |
| `r50_deformable_detr_motip_pdestre_sam_fold0.yaml` | — | R50 with SAM object masks |

#### RF-DETR (DINOv2-Large) backbone

| Config file | Concepts | Description |
|-------------|----------|-------------|
| `rfdetr_large_motip_pdestre_base_fold0_v4.yaml` | 0 | Baseline |
| `rfdetr_large_motip_pdestre_7concepts_learnable.yaml` | 7 | All concepts, learnable weights |
| `rfdetr_large_motip_pdestre_7concepts_nolw_fold0.yaml` | 7 | All concepts, fixed loss coefficient |
| `rfdetr_large_motip_pdestre_sam_fold0.yaml` | — | RF-DETR with SAM object masks |

### Key config parameters

```yaml
# Base training settings
EPOCHS: 10
BATCH_SIZE: 1
ACCUMULATE_STEPS: 2          # Effective batch size = 2
LR: 1e-4
WEIGHT_DECAY: 1e-4

# Concept configuration
NUM_CONCEPTS: 7              # 0=none, 2, 3, or 7
CONCEPT_LOSS_COEF: 0.5       # Fixed loss weight (ignored if learnable)
USE_LEARNABLE_TASK_WEIGHTS: True  # Learn loss balance automatically

# Inference thresholds
DET_THRESH: 0.3
NEWBORN_THRESH: 0.6
ID_THRESH: 0.2
MISS_TOLERANCE: 30

# Cross-validation
DATASET_SPLITS: [Train_0]    # Training split
INFERENCE_SPLIT: val_0       # Validation split
```

---

## 6. Training

### Option A: Using SLURM scripts (recommended)

#### R50 Deformable DETR

Edit the configuration section at the top of `train_r50.sh`:

```bash
NUM_CONCEPTS=2               # 0=base, 2=2concepts, 3=3concepts
FOLD=0                       # 0-9
RESUME_MODE="auto"           # "none", "auto", or "manual:/path/to/checkpoint.pth"
```

Submit:

```bash
sbatch train_r50.sh
```

#### RF-DETR

Edit the configuration section at the top of `train_rf-detr.sh`:

```bash
NUM_CONCEPTS=7               # 0=base, 7=7concepts
USE_LEARNABLE_WEIGHTS=true   # true or false
FOLD=0                       # 0-9
CONFIG_VERSION="v4"          # "v4" for base config
RESUME_MODE="auto"           # "none", "auto", or "manual:/path/to/checkpoint.pth"
```

Submit:

```bash
sbatch train_rf-detr.sh
```

### Option B: Direct command (for debugging)

```bash
python train.py \
    --config-path configs/r50_deformable_detr_motip_pdestre_2concepts_fold0_v2.yaml \
    --exp-name r50_motip_pdestre_2concepts_fold_0 \
    --detr-pretrain ./pretrains/r50_deformable_detr_coco.pth
```

### SLURM resource settings

Both scripts request:
- **Partition**: `gpu_h100`
- **GPUs**: 1× H100
- **CPUs**: 8
- **Memory**: 16 GB
- **Time limit**: 72 hours
- **Auto-resubmit**: On SIGUSR1 (handles preemption)

### Monitoring training

```bash
# Check job status
squeue -u $USER

# Watch training log (live)
tail -f outputs/<experiment_name>/train/log.txt

# Check SLURM output
cat logs/motip_r50_<job_id>.out
```

### Output structure

After training, each experiment creates:

```
outputs/<experiment_name>/
├── checkpoint_0.pth          # Checkpoints (keeps last 4)
├── checkpoint_1.pth
├── ...
└── train/
    ├── log.txt               # Full training log
    ├── config.yaml           # Saved config (for reproducibility)
    ├── events.out.tfevents.* # TensorBoard logs
    └── eval_during_train/    # Per-epoch validation results
        ├── epoch_0/
        ├── epoch_1/
        └── ...
```

### Resuming interrupted training

Training scripts support automatic resume. With `RESUME_MODE="auto"`, the script finds the latest checkpoint in the output directory and continues from there. No manual intervention needed after a job timeout or preemption.

---

## 7. Evaluation

After training completes, evaluate tracking performance on the validation or test split.

### Step 1: Submit evaluation job

```bash
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP

# Evaluate fold 0 on the validation split
sbatch evaluation/evaluate_fold.sh 0
```

> **Note**: Edit `evaluation/evaluate_fold.sh` to match your experiment name, config path, and checkpoint path before submitting. The default evaluates `r50_motip_pdestre_fold_0`.

### Step 2: Run evaluation directly (alternative)

```bash
python evaluation/submit_and_evaluate.py \
    --config-path configs/r50_deformable_detr_motip_pdestre_2concepts_fold0_v2.yaml \
    --inference-model outputs/r50_motip_pdestre_2concepts_fold_0/checkpoint_9.pth \
    --inference-group fold_0 \
    --inference-dataset P-DESTRE \
    --inference-split val_0 \
    --outputs-dir outputs/r50_motip_pdestre_2concepts_fold_0
```

### Step 3: Extract metrics

```bash
# Single fold
python evaluation/extract_metrics.py --exp-prefix r50_motip_pdestre_2concepts --fold 0

# All completed folds
python evaluation/extract_metrics.py --exp-prefix r50_motip_pdestre_2concepts --all-folds
```

### Metrics computed

| Category | Metrics |
|----------|---------|
| **Tracking** | HOTA, MOTA, MOTP, IDF1 |
| **Detection** | DetA, DetRe, DetPr |
| **Identity** | AssA, AssRe, AssPr |
| **Concept accuracy** | Per-attribute accuracy (gender, upper_body, etc.) |

---

## 8. Visualization

### Generate all charts at once

```bash
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/evaluation
./generate_all_visualizations.sh r50_motip_pdestre_2concepts
```

This runs metric extraction + chart generation for all folds and saves output to `outputs/<prefix>_visualizations/`.

### Generate charts manually

```bash
python evaluation/visualize_results.py --exp-prefix r50_motip_pdestre_2concepts_fold_0
```

### What you get

- Tracking metric bar charts (HOTA, MOTA, IDF1 per fold)
- Detection performance plots (DetA, DetRe, DetPr)
- Concept prediction accuracy per attribute
- Cross-fold summary statistics (mean ± std)
- Summary report text file

---

## 9. Troubleshooting

### CUDA extension build fails

The build **must run on a compute node** (not a login node). Start an interactive session first:

```bash
srun --partition=gpu_h100 --gres=gpu:1 --time=00:30:00 --mem=8G --pty bash
conda activate MOTIP

# Ensure CUDA module is loaded
module load devel/cuda/11.8
export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8

# Verify nvcc is accessible
$CUDA_HOME/bin/nvcc --version

# Clean rebuild
rm -rf models/ops/build/
cd models/ops && python setup.py build install && cd ../..
```

### Out of memory (OOM)

- Default settings use `BATCH_SIZE: 1` with `ACCUMULATE_STEPS: 2` (effective batch size 2)
- If OOM persists, reduce `SAMPLE_LENGTHS` in the config (e.g., from `[15]` to `[10]`)

### "CUDA not available" during evaluation

- Evaluation requires a GPU. Always use `sbatch evaluation/evaluate_fold.sh` — do not run directly on a login node.

### RF-DETR import errors

- Verify the `rf-detr/` directory exists (it is a separate clone, not a submodule)
- `train.py` adds it to `sys.path` automatically

### Training job gets preempted

- Both `train_r50.sh` and `train_rf-detr.sh` handle SLURM preemption via `--signal=B:SIGUSR1@120`
- With `RESUME_MODE="auto"`, the job resubmits itself and resumes from the latest checkpoint

### Transformers version compatibility

- If you see errors related to `BackboneMixin` imports, ensure `transformers >= 5.0` is installed
- The codebase includes a compatibility fix in `models/deformable_detr/dinov2_with_windowed_attn.py`

### Missing evaluation results

1. Check SLURM logs: `cat logs/eval_fold_*.out`
2. Verify the checkpoint exists: `ls outputs/<exp_name>/checkpoint_*.pth`
3. Ensure the correct config and split are specified in the evaluation command

---

## Quick Reference

```bash
# === FULL SETUP (one command after P-DESTRE is extracted in data/) ===
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP
./scripts/download_and_preprocess.sh

# === BUILD CUDA EXTENSION (if not done by script — requires GPU node) ===
srun --partition=gpu_h100 --gres=gpu:1 --time=00:30:00 --mem=8G --pty bash
conda activate MOTIP
module load devel/cuda/11.8 && export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8
cd models/ops && python setup.py build install && cd ../..
exit  # back to login node

# === SMOKE TEST ===
sbatch scripts/smoke_test.sh            # R50 (2 epochs, ~5 min)
sbatch scripts/smoke_test_rfdetr.sh     # RF-DETR (2 epochs, ~5 min)

# === TRAINING ===
# Edit NUM_CONCEPTS and FOLD, then:
sbatch train_r50.sh                     # R50 backbone
sbatch train_rf-detr.sh                 # RF-DETR backbone

# === EVALUATION ===
sbatch evaluation/evaluate_fold.sh 0    # Evaluate fold 0

# === VISUALIZATION ===
cd evaluation/
./generate_all_visualizations.sh r50_motip_pdestre_2concepts
```
