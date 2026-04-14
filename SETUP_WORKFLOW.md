# MOTIP Setup Workflow — Quick Reference

> **For the full reproduction guide, see [README.md](README.md).**

This file is a command-only cheatsheet. If anything is unclear, README.md has detailed explanations for every step.

---

## 1. Prerequisites

- Access to **bwUniCluster** with `miniconda3` in your home directory
- P-DESTRE downloaded from Google Drive (see README.md § 2 for the link)
- ~200 GB free disk space

---

## 2. Upload & Extract P-DESTRE

P-DESTRE must be downloaded manually from Google Drive. After downloading `dataset.tar`:

```bash
# Upload from your local machine:
scp dataset.tar <user>@bwunicluster.scc.kit.edu:/pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/data/

# On the cluster, extract:
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/data
tar -xf dataset.tar
```

---

## 3. Run the Setup Script

From the repository root, run the single setup script. It creates the conda
environment, preprocesses P-DESTRE, downloads pretrained weights, clones
RF-DETR, and builds the CUDA extension (if on a GPU node).

```bash
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP
./scripts/download_and_preprocess.sh
```

Options:
```bash
./scripts/download_and_preprocess.sh --env-only   # Only conda env + pip install
./scripts/download_and_preprocess.sh --skip-env   # Skip env, do everything else
```

> The script is **idempotent** — safe to re-run; it skips already-completed steps.

---

## 4. Build CUDA Extension (if needed)

The setup script attempts to build the CUDA extension automatically. If it ran
on a login node (no GPU), build it manually once on a GPU node:

```bash
srun --partition=gpu_h100 --gres=gpu:1 --time=00:30:00 --mem=8G --pty bash
conda activate MOTIP
module load devel/cuda/11.8
export CUDA_HOME=/opt/bwhpc/common/devel/cuda/11.8
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP/models/ops
python setup.py build install
exit
```

Alternatively, it is built automatically when the first training job starts.

---

## 5. Validate with Smoke Test

```bash
sbatch scripts/smoke_test.sh            # R50 backbone (2 epochs, ~15 min)
sbatch scripts/smoke_test_rfdetr.sh     # RF-DETR backbone (2 epochs, ~15 min)

# Check result:
squeue -u $USER
cat logs/smoke_test_<job_id>.out
```

Expected: no errors, `outputs/smoke_test/checkpoint_0.pth` and `checkpoint_1.pth` created.

---

## 6. Training

Edit the configuration block at the top of the training script, then submit:

```bash
# == R50 Deformable DETR ==
# Edit NUM_CONCEPTS (0/2/3) and FOLD (0-9) at the top of train_r50.sh
sbatch train_r50.sh

# == RF-DETR (DINOv2-Large) ==
# Edit NUM_CONCEPTS (0/7), USE_LEARNABLE_WEIGHTS, and FOLD at the top of train_rf-detr.sh
sbatch train_rf-detr.sh
```

Monitor:
```bash
squeue -u $USER
tail -f outputs/<experiment_name>/train/log.txt
cat logs/motip_r50_<job_id>.out
```

Training auto-resumes on preemption (`RESUME_MODE="auto"` is the default).

---

## 7. Evaluation

```bash
# Evaluate a checkpoint on a specific split:
sbatch evaluation/eval_checkpoint.sh \
    outputs/<exp_name>/checkpoint_9.pth \
    val_0
```

The script auto-discovers the config from `outputs/<exp_name>/train/config.yaml`.

---

## 8. Quick Reference

```bash
# Full setup (after P-DESTRE is extracted in data/):
./scripts/download_and_preprocess.sh

# Smoke test:
sbatch scripts/smoke_test.sh

# Train R50 (edit config vars at top first):
sbatch train_r50.sh

# Train RF-DETR (edit config vars at top first):
sbatch train_rf-detr.sh

# Evaluate a checkpoint:
sbatch evaluation/eval_checkpoint.sh outputs/<exp>/checkpoint_9.pth val_0

# Check jobs:
squeue -u $USER
```

---

## 9. Repository Structure

```
MOTIP/
├── train.py                    # Main training script
├── train_r50.sh                # SLURM: R50 training
├── train_rf-detr.sh            # SLURM: RF-DETR training
├── requirements.txt            # Python dependencies
├── configs/                    # YAML experiment configs
├── data/                       # Dataset loaders + P-DESTRE files
│   └── P-DESTRE/images/        # Extracted frames (after preprocessing)
├── models/                     # Model architectures + CUDA ops
├── rf-detr/                    # RF-DETR repo (cloned by setup script)
├── pretrains/                  # Pretrained weights
├── splits/                     # Cross-validation split files
├── outputs/                    # Training outputs (checkpoints, logs)
├── logs/                       # SLURM job output files
├── evaluation/                 # Evaluation scripts
│   ├── eval_checkpoint.sh      # Evaluate a single checkpoint
│   ├── submit_and_evaluate.py  # Core inference + metrics
│   └── compute_all_metrics.py  # Aggregate metrics across runs
└── scripts/                    # Utility scripts
    ├── download_and_preprocess.sh
    ├── smoke_test.sh
    └── smoke_test_rfdetr.sh
```

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA extension build fails | Must build on a GPU node (not login node); see § 4 |
| `No module named 'rfdetr'` | `rf-detr/` dir missing — re-run setup script |
| `ImportError` from transformers | Compatibility fix applied by setup script; verify `rf-detr/` exists |
| Out of memory | Reduce `SAMPLE_LENGTHS: [10]` in the config (default is 15) |
| Job preempted | Resubmit — `RESUME_MODE="auto"` resumes from latest checkpoint |
| Smoke test shows TIMEOUT | Normal for `gpu_h100_short`; training and epoch 0 eval completed fine |

For more detail on any of these, see [README.md § 9 Troubleshooting](README.md#9-troubleshooting).
