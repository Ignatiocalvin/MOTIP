#!/usr/bin/env python3
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Standalone evaluation script for evaluating a specific checkpoint

import os
import sys
import argparse
import torch

# Add project root and rf-detr to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if MOTIP_ROOT not in sys.path:
    sys.path.insert(0, MOTIP_ROOT)
RFDETR_PATH = os.path.join(MOTIP_ROOT, 'rf-detr')
if os.path.exists(RFDETR_PATH) and RFDETR_PATH not in sys.path:
    sys.path.insert(0, RFDETR_PATH)

from accelerate import Accelerator
from accelerate.state import PartialState

from utils.misc import yaml_to_dict
from configs.util import load_super_config
from log.logger import Logger
from models.motip import build as build_motip
from models.misc import load_checkpoint
from evaluation.submit_and_evaluate import submit_and_evaluate_one_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint on a dataset')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to checkpoint file (e.g., outputs/exp_name/checkpoint_0.pth)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data-root', type=str, default='./data/',
                        help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, default='PDESTRE',
                        help='Dataset name (e.g., PDESTRE)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split (e.g., test, val)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated from checkpoint path)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip sequences whose tracker output file already exists')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Validate config exists
    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)
    
    # Load config
    cfg = yaml_to_dict(args.config)
    
    # Load super config if specified
    if "SUPER_CONFIG_PATH" in cfg and cfg["SUPER_CONFIG_PATH"] is not None:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    
    # Override config with command line arguments
    cfg["DATA_ROOT"] = args.data_root
    cfg["INFERENCE_DATASET"] = args.dataset
    cfg["INFERENCE_SPLIT"] = args.split
    
    # Generate output directory
    if args.output_dir is None:
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(checkpoint_dir, "eval", f"{args.dataset}_{args.split}", checkpoint_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[INFO] Evaluating checkpoint: {args.checkpoint}")
    print(f"[INFO] Dataset: {args.dataset} / {args.split}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Config: {args.config}")
    
    # Init Accelerator
    accelerator = Accelerator()
    state = PartialState()
    
    # Init Logger
    logger = Logger(
        logdir=args.output_dir,
        use_wandb=False,
        config=cfg,
    )
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset} / {args.split}")
    
    # Build model
    logger.info("Building model...")
    model, _ = build_motip(config=cfg)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model=model, path=args.checkpoint)
    logger.success(f"Checkpoint loaded successfully")
    
    # Prepare with accelerator
    model = accelerator.prepare(model)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    eval_metrics = submit_and_evaluate_one_model(
        is_evaluate=True,
        accelerator=accelerator,
        state=state,
        logger=logger,
        model=model,
        data_root=cfg["DATA_ROOT"],
        dataset=cfg["INFERENCE_DATASET"],
        data_split=cfg["INFERENCE_SPLIT"],
        outputs_dir=args.output_dir,
        skip_existing=args.skip_existing,
        image_max_longer=cfg.get("INFERENCE_MAX_LONGER", 1333),
        size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
        miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        det_thresh=cfg.get("DET_THRESH", 0.5),
        newborn_thresh=cfg.get("NEWBORN_THRESH", 0.5),
        id_thresh=cfg.get("ID_THRESH", 0.1),
        area_thresh=cfg.get("AREA_THRESH", 0),
        inference_only_detr=cfg.get("INFERENCE_ONLY_DETR", cfg.get("ONLY_DETR", False)),
        concept_bottleneck_mode=cfg.get("MOTIP", {}).get("CONCEPT_BOTTLENECK_MODE", "hard"),
    )
    
    # Sync and log metrics
    eval_metrics.sync()
    logger.metrics(
        log=f"[Evaluation Results] ",
        metrics=eval_metrics,
        fmt="{global_average:.4f}",
        statistic="global_average",
    )
    
    logger.success("Evaluation completed!")
    print(f"\n[SUCCESS] Results saved to: {args.output_dir}")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
