# MOTIP Evaluation & Visualization

Tools for evaluating tracking performance and visualizing results across cross-validation folds.

## Quick Start

```bash
# 1. Run evaluation on GPU node for each completed fold
cd /pfs/work9/workspace/scratch/ma_ighidaya-thesis_ignatio/MOTIP
sbatch evaluate_fold.sh 0  # Replace 0 with fold number (0-9)

# 2. After evaluations complete, generate visualizations
cd evaluation/
./generate_all_visualizations.sh r50_motip_pdestre

# 3. View results
cd ../outputs/r50_motip_pdestre_visualizations/
ls -lh
```

## Files

**Evaluation:**
- `../evaluate_fold.sh` - SLURM job to run TrackEval on GPU node

**Visualization:**
- `extract_metrics.py` - Parse TrackEval logs → JSON
- `visualize_results.py` - Generate charts from JSON
- `generate_all_visualizations.sh` - Complete workflow (extract + visualize)
- `visualize_all_folds.sh` - Quick wrapper

## What You Get

- Tracking metrics charts (HOTA, MOTA, MOTP, IDF1)
- Detection performance (DetA, DetRe, DetPr)
- Concept prediction accuracy (gender, upper_body)
- Correlation analysis
- Summary statistics (mean ± std)

## Workflow

1. **Train** → Multiple folds complete training
2. **Evaluate** → `sbatch evaluate_fold.sh N` computes tracking metrics
3. **Extract** → `extract_metrics.py` parses logs into results.json
4. **Visualize** → `visualize_results.py` creates charts
5. **Report** → Use charts and summary_report.txt in your thesis

## Examples

```bash
# Evaluate single fold
sbatch evaluate_fold.sh 0

# Extract metrics from one fold
python extract_metrics.py --exp-prefix r50_motip_pdestre --fold 0

# Extract from all folds
python extract_metrics.py --exp-prefix r50_motip_pdestre --all-folds

# Generate all visualizations
./generate_all_visualizations.sh r50_motip_pdestre

# Custom visualization
python visualize_results.py --exp-prefix r50_motip_pdestre_fold_0
```

## Troubleshooting

**"No evaluation results found"**
→ Run `sbatch evaluate_fold.sh N` first on GPU node

**"CUDA not available"**
→ Evaluation needs GPU. Use `sbatch evaluate_fold.sh` not direct python call

**"Missing results.json"**
→ Run `extract_metrics.py` before visualizing

**No metrics in log.txt**
→ Evaluation may have failed or not completed. Check SLURM output files in `logs/`
