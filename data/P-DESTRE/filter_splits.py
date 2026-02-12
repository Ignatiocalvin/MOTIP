#!/usr/bin/env python3
"""
Optimize P-DESTRE dataset storage by intelligently removing image folders.
- Removes problematic sequences (known issues)
- Removes sequences with very few images
- Keeps most high-image-count sequences but removes some for space
- Automatically updates split files and deletes corresponding annotations
- Target: ~40% reduction in image folders
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# Configuration
IMAGES_DIR = "/workspace/MOTIP/data/P-DESTRE/images"  # Adjust for your setup
ANNOTATIONS_DIR = "/workspace/MOTIP/data/P-DESTRE/annotations"
SPLITS_DIR = "/workspace/MOTIP/splits"
BACKUP_SUFFIX = ".original"
TARGET_REDUCTION = 0.40  # Remove 40% of sequences

# Known problematic sequences (missing images, corrupted, etc.)
PROBLEMATIC_SEQUENCES = [
    "14-11-2019-1-2",  # Known to have missing frame 000067.jpg
]

def count_images_in_sequence(seq_path):
    """Count number of images in a sequence folder."""
    img_folder = os.path.join(seq_path, "img1")
    if not os.path.exists(img_folder):
        return 0
    try:
        return len([f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))])
    except Exception:
        return 0

def analyze_dataset(images_dir):
    """Analyze all sequences and return statistics."""
    sequences = {}
    
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        return sequences
    
    for seq_name in os.listdir(images_dir):
        seq_path = os.path.join(images_dir, seq_name)
        if os.path.isdir(seq_path):
            img_count = count_images_in_sequence(seq_path)
            sequences[seq_name] = {
                'path': seq_path,
                'count': img_count,
                'problematic': seq_name in PROBLEMATIC_SEQUENCES
            }
    
    return sequences

def select_sequences_to_keep(sequences, target_reduction):
    """
    Select which sequences to keep based on:
    1. Remove all problematic sequences
    2. Remove sequences with very few images (bottom 20%)
    3. Keep most high-count sequences but remove some large ones for space
    """
    if not sequences:
        return set()
    
    # Sort by image count
    sorted_seqs = sorted(sequences.items(), key=lambda x: x[1]['count'])
    total_seqs = len(sorted_seqs)
    target_keep = int(total_seqs * (1 - target_reduction))
    
    print(f"\nDataset Analysis:")
    print(f"  Total sequences: {total_seqs}")
    print(f"  Target to keep: {target_keep} ({(1-target_reduction)*100:.0f}%)")
    print(f"  Target to remove: {total_seqs - target_keep} ({target_reduction*100:.0f}%)")
    
    # Calculate statistics
    counts = [s[1]['count'] for s in sorted_seqs]
    min_count, max_count = min(counts), max(counts)
    avg_count = sum(counts) / len(counts)
    
    print(f"\nImage counts per sequence:")
    print(f"  Min: {min_count}, Max: {max_count}, Avg: {avg_count:.0f}")
    
    # Identify sequences to remove
    to_remove = set()
    to_keep = set()
    
    # 1. Remove all problematic sequences first
    problematic_count = 0
    for seq_name, info in sequences.items():
        if info['problematic']:
            to_remove.add(seq_name)
            problematic_count += 1
    
    print(f"\nRemoving {problematic_count} problematic sequences")
    
    # 2. Remove bottom 20% (smallest sequences)
    bottom_20_percent = int(total_seqs * 0.20)
    small_seqs_removed = 0
    for seq_name, info in sorted_seqs[:bottom_20_percent]:
        if seq_name not in to_remove:
            to_remove.add(seq_name)
            small_seqs_removed += 1
    
    print(f"Removing {small_seqs_removed} sequences with fewest images (bottom 20%)")
    
    # 3. Calculate how many more to remove from larger sequences
    remaining_to_remove = (total_seqs - target_keep) - len(to_remove)
    
    if remaining_to_remove > 0:
        # Remove some from the upper half to save space
        # Take from 60-80th percentile (medium-large but not the largest)
        start_idx = int(total_seqs * 0.60)
        end_idx = int(total_seqs * 0.85)
        candidates = [s[0] for s in sorted_seqs[start_idx:end_idx] if s[0] not in to_remove]
        
        # Take the needed amount
        import random
        random.seed(42)
        additional_remove = random.sample(candidates, min(remaining_to_remove, len(candidates)))
        to_remove.update(additional_remove)
        print(f"Removing {len(additional_remove)} medium-large sequences to reach target")
    
    # Everything not removed is kept
    to_keep = set(sequences.keys()) - to_remove
    
    print(f"\nFinal counts:")
    print(f"  Keeping: {len(to_keep)} sequences")
    print(f"  Removing: {len(to_remove)} sequences")
    
    return to_keep

def delete_sequences(sequences, to_keep, dry_run=True):
    """Delete sequence folders that are not in to_keep set."""
    deleted = []
    total_size_freed = 0
    
    for seq_name, info in sequences.items():
        if seq_name not in to_keep:
            seq_path = info['path']
            
            # Calculate folder size
            try:
                folder_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(seq_path)
                    for filename in filenames
                )
                total_size_freed += folder_size
            except Exception:
                folder_size = 0
            
            if not dry_run:
                try:
                    shutil.rmtree(seq_path)
                    deleted.append((seq_name, info['count'], folder_size))
                    print(f"  Deleted: {seq_name} ({info['count']} images, {folder_size/1024/1024:.1f} MB)")
                except Exception as e:
                    print(f"  ERROR deleting {seq_name}: {e}")
            else:
                deleted.append((seq_name, info['count'], folder_size))
                print(f"  Would delete: {seq_name} ({info['count']} images, {folder_size/1024/1024:.1f} MB)")
    
    return deleted, total_size_freed

def delete_annotations(annotations_dir, to_keep, dry_run=True):
    """Delete annotation files for sequences that are not in to_keep set."""
    if not os.path.exists(annotations_dir):
        print(f"  No annotations directory found at {annotations_dir}")
        return []
    
    deleted_annotations = []
    
    for annotation_file in os.listdir(annotations_dir):
        if annotation_file.endswith('.txt'):
            seq_name = annotation_file[:-4]  # Remove .txt extension
            
            if seq_name not in to_keep:
                annotation_path = os.path.join(annotations_dir, annotation_file)
                
                if not dry_run:
                    try:
                        os.remove(annotation_path)
                        deleted_annotations.append(seq_name)
                        print(f"  Deleted annotation: {annotation_file}")
                    except Exception as e:
                        print(f"  ERROR deleting annotation {annotation_file}: {e}")
                else:
                    deleted_annotations.append(seq_name)
                    print(f"  Would delete annotation: {annotation_file}")
    
    return deleted_annotations

def update_split_files(splits_dir, to_keep, dry_run=True):
    """Update split files to remove deleted sequences."""
    if not os.path.exists(splits_dir):
        print(f"ERROR: Splits directory not found: {splits_dir}")
        return
    
    split_files = [
        f for f in os.listdir(splits_dir)
        if f.endswith('.txt') and not f.endswith(BACKUP_SUFFIX)
        and any(f.startswith(p) for p in ['Train_', 'Test_', 'val_'])
    ]
    annotations-dir', default=ANNOTATIONS_DIR,
                       help='Path to annotations directory')
    parser.add_argument('--
    for split_file in sorted(split_files):
        split_path = os.path.join(splits_dir, split_file)
        backup_path = split_path + BACKUP_SUFFIX
        
        # Backup if not exists
        if not dry_run and not os.path.exists(backup_path):
            shutil.copy2(split_path, backup_path)
        
        # Read and filter
        with open(split_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        
        filtered = [seq for seq in sequences if seq in to_keep]
        
        if not dry_run:
            with open(split_path, 'w') as f:
                for seq in filtered:
                    f.write(seq + '\n')
        
        print(f"  {split_file}: {len(sequences)} -> {len(filtered)} sequences")

def main():
    parser = argparse.ArgumentParser(description='Optimize P-DESTRE dataset storage')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--images-dir', default=IMAGES_DIR,
                       help='Path to images directory')
    parser.add_argument('--splits-dir', default=SPLITS_DIR,
                       help='Path to splits directory')
    parser.add_argument('--target-reduction', type=float, default=TARGET_REDUCTION,
                       help='Target reduction ratio (0.4 = 40%%)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("P-DESTRE Dataset Storage Optimizer")
    print("="*70)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be deleted")
    else:
        print("\n⚠️  LIVE MODE - Files will be permanently deleted!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    print(f"\nAnalyzing dataset at: {args.images_dir}")
    sequences = analyze_dataset(args.images_dir)
    
    if not sequences:
        print("No sequences found!")
        return
    
    # Select which to keep
    to_keep = select_sequences_to_keep(sequences, args.target_reduction)
    
    # Delete sequences
    print(f"\n{'DRY RUN: Would delete' if args.dry_run else 'Deleting'} sequences...")
    deleted, size_freed = delete_sequences(sequences, to_keep, args.dry_run)
    
    prDelete corresponding annotation files
    print(f"\n{'DRY RUN: Would delete' if args.dry_run else 'Deleting'} annotation files...")
    deleted_annotations = delete_annotations(args.annotations_dir, to_keep, args.dry_run)
    print(f"  {'Would delete' if args.dry_run else 'Deleted'} {len(deleted_annotations)} annotation files")
    
    # int(f"\nTotal space {'that would be' if args.dry_run else ''} freed: {size_freed/1024/1024/1024:.2f} GB")
    
    # Update split files
    print(f"\n{'DRY RUN: Would update' if args.dry_run else 'Updating'} split files...")
    update_split_files(args.splits_dir, to_keep, args.dry_run)
    
    if args.dry_run:
        print("\n" + "="*70)
        print("DRY RUN COMPLETE - No changes made")
        print("Run without --dry-run to actually delete sequences")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print(f"Backup split files saved with '{BACKUP_SUFFIX}' suffix")
        print('To restore: for f in splits/*.original; do mv "$f" "${f%.original}"; done')
        print("="*70)

if __name__ == '__main__':
    main()
