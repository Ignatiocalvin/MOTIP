#!/usr/bin/env python3
"""
Monitor checkpoint progress for evaluation job in real-time.
Run this in a separate terminal while the evaluation job is running.
"""

import json
import time
from pathlib import Path
from datetime import datetime

CHECKPOINT_FILE = Path("./outputs/r50_motip_pdestre_7concepts_learnable_v2_fold_0/eval_results_val0_checkpoint.json")

def load_checkpoint():
    """Load current checkpoint state."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def format_metrics(agg):
    """Format aggregated metrics nicely."""
    if not agg:
        return "No data"
    return f"MOTA={agg['MOTA']:6.2f}% ±{agg['MOTA_std']:5.2f}  IDF1={agg['IDF1']:6.2f}% ±{agg['IDF1_std']:5.2f}  Prec={agg['Precision']:6.2f}%  Rec={agg['Recall']:6.2f}%"

def print_status():
    """Print current checkpoint status."""
    data = load_checkpoint()
    
    if data is None:
        print("❌ Checkpoint file not found yet (job may not have started)")
        return
    
    completed = len(data.get('all_epochs', {}))
    status = data.get('status', 'unknown')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*120}")
    print(f"CHECKPOINT STATUS [{timestamp}] | Completed: {completed}/9 | Status: {status.upper()}")
    print(f"{'='*120}")
    
    if data.get('all_epochs'):
        for epoch_num in sorted([int(k) for k in data['all_epochs'].keys()]):
            agg = data['all_epochs'][str(epoch_num)]
            mark = "✓" if epoch_num < 9 else ("★" if epoch_num == data.get('best_epoch') else "✓")
            print(f"  Epoch {epoch_num:2d} {mark}  {format_metrics(agg)}")
    
    if data.get('best'):
        best = data['best']
        print(f"\n  {'─'*115}")
        print(f"  CURRENT BEST: Epoch {data['best_epoch']}  {format_metrics(best)}")
    
    print(f"{'='*120}\n")

if __name__ == "__main__":
    print("📊 Starting checkpoint monitor (Ctrl+C to stop)...")
    print(f"   Watching: {CHECKPOINT_FILE}")
    
    try:
        last_completed = -1
        while True:
            data = load_checkpoint()
            current_completed = len(data.get('all_epochs', {})) if data else 0
            
            # Print whenever status changes
            if current_completed != last_completed or (data and data.get('status') == 'complete'):
                print_status()
                last_completed = current_completed
                
                if data and data.get('status') == 'complete':
                    print("✨ Evaluation COMPLETE! 🎉")
                    break
            
            time.sleep(5)  # Check every 5 seconds
    
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped by user")
