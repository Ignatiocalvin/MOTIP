#!/usr/bin/env python3
"""
Extract and display results from the evaluation checkpoint at any point.
Can be run while the job is in progress or after it completes.
"""

import json
import sys
from pathlib import Path

CHECKPOINT_FILE = Path("./outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val0_checkpoint.json")
FINAL_FILE = Path("./outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val0.json")

def load_data():
    """Load from checkpoint or final results file."""
    # Prefer final file if it exists
    if FINAL_FILE.exists():
        with open(FINAL_FILE, 'r') as f:
            return json.load(f), "final"
    elif CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f), "checkpoint"
    else:
        print("❌ No results file found")
        return None, None

def main():
    data, source = load_data()
    if data is None:
        sys.exit(1)
    
    completed = len(data.get('all_epochs', {}))
    status = data.get('status', 'unknown')
    
    print(f"\n{'='*120}")
    print(f"EVALUATION RESULTS [{source.upper()}] | Completed: {completed}/9 | Status: {status.upper()}")
    print(f"{'='*120}\n")
    
    if data.get('all_epochs'):
        # Print header
        print(f"{'Epoch':<8} {'MOTA':<20} {'MOTA_std':<12} {'IDF1':<20} {'IDF1_std':<12} {'Prec':<10} {'Rec':<10} {'TP':<10} {'FP':<10} {'FN':<10} {'IDsw':<8}")
        print(f"{'-'*120}")
        
        for epoch_num in sorted([int(k) for k in data['all_epochs'].keys()]):
            agg = data['all_epochs'][str(epoch_num)]
            mark = "★" if epoch_num == data.get('best_epoch') else " "
            print(f"{epoch_num}{mark:<7} {agg['MOTA']:6.2f}% ±{agg['MOTA_std']:5.2f}   {agg['IDF1']:6.2f}% ±{agg['IDF1_std']:5.2f}   {agg['Precision']:6.2f}%  {agg['Recall']:6.2f}%  {agg['TP']:>8}  {agg['FP']:>8}  {agg['FN']:>8}  {agg['IDsw']:>6}")
    
    # Print best epoch
    if data.get('best'):
        best = data['best']
        print(f"\n{'='*120}")
        print(f"★ BEST EPOCH: {data['best_epoch']}")
        print(f"  MOTA: {best['MOTA']:.2f}% ±{best['MOTA_std']:.2f}")
        print(f"  IDF1: {best['IDF1']:.2f}% ±{best['IDF1_std']:.2f}")
        print(f"  Precision: {best['Precision']:.2f}%  |  Recall: {best['Recall']:.2f}%")
        print(f"  TP: {best['TP']}  |  FP: {best['FP']}  |  FN: {best['FN']}  |  IDsw: {best['IDsw']}")
        print(f"{'='*120}\n")

if __name__ == "__main__":
    main()
