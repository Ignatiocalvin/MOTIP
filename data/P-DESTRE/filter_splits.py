#!/usr/bin/env python3
"""
Filter P-DESTRE split files to use only a random subset of sequences.
This reduces dataset size for faster training.
"""

import os
import random
import shutil

# Configuration
KEEP_RATIO = 0.5  # Keep 50% of sequences (adjust as needed: 0.1 = 10%, 0.3 = 30%, etc.)
SPLITS_DIR = "/workspace/MOTIP/splits"  # Adjust path for cloud GPU
BACKUP_SUFFIX = ".original"

def filter_split_file(split_path, keep_ratio, seed=42):
    """Filter a split file to keep only a random subset of sequences."""
    # Backup original file
    backup_path = split_path + BACKUP_SUFFIX
    if not os.path.exists(backup_path):
        shutil.copy2(split_path, backup_path)
        print(f"Backed up: {split_path} -> {backup_path}")
    
    # Read all sequences
    with open(backup_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    original_count = len(sequences)
    
    # Randomly select subset
    random.seed(seed)
    keep_count = max(1, int(original_count * keep_ratio))
    selected = sorted(random.sample(sequences, keep_count))
    
    # Write filtered file
    with open(split_path, 'w') as f:
        for seq in selected:
            f.write(seq + '\n')
    
    print(f"{os.path.basename(split_path)}: {original_count} -> {keep_count} sequences ({keep_ratio*100:.0f}%)")
    return original_count, keep_count

def main():
    print(f"Filtering P-DESTRE splits to {KEEP_RATIO*100:.0f}% of sequences...")
    print(f"Splits directory: {SPLITS_DIR}")
    print()
    
    if not os.path.exists(SPLITS_DIR):
        print(f"ERROR: Splits directory not found: {SPLITS_DIR}")
        print("Please update SPLITS_DIR in this script to match your setup.")
        return
    
    # Find all split files
    split_files = []
    for fname in os.listdir(SPLITS_DIR):
        if fname.endswith('.txt') and not fname.endswith(BACKUP_SUFFIX):
            if any(fname.startswith(prefix) for prefix in ['Train_', 'Test_', 'val_']):
                split_files.append(os.path.join(SPLITS_DIR, fname))
    
    if not split_files:
        print("No split files found!")
        return
    
    split_files.sort()
    
    total_original = 0
    total_kept = 0
    
    # Process each split file with same seed for consistency
    for split_path in split_files:
        orig, kept = filter_split_file(split_path, KEEP_RATIO, seed=42)
        total_original += orig
        total_kept += kept
    
    print()
    print(f"Total: {total_original} -> {total_kept} sequences")
    print(f"Reduction: {(1 - total_kept/total_original)*100:.1f}%")
    print()
    print(f"Original files backed up with '{BACKUP_SUFFIX}' suffix")
    print("To restore originals: rm splits/*.txt && rename 's/\\.original$//' splits/*.original")

if __name__ == '__main__':
    main()
