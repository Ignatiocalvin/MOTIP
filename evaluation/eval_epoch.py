#!/usr/bin/env python3
"""
Standalone script to re-evaluate a specific epoch checkpoint.
Usage:
    accelerate launch eval_epoch.py \
        --config-path ./configs/r50_deformable_detr_motip_pdestre_base_fold0.yaml \
        --checkpoint ./outputs/r50_motip_pdestre_base_fold0/checkpoint_0.pth \
        --epoch 0

This loads the config, builds the model, loads the checkpoint,
and runs submit_and_evaluate_one_model, writing tracker files to
  outputs/<exp>/train/eval_during_train/epoch_<N>/
"""

import os
import sys
import argparse

# Add RF-DETR to Python path (rf-detr/ lives one level up, in MOTIP root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RFDETR_PATH = os.path.join(MOTIP_ROOT, 'rf-detr')
if os.path.exists(RFDETR_PATH) and RFDETR_PATH not in sys.path:
    sys.path.insert(0, RFDETR_PATH)

from accelerate import Accelerator
from accelerate.state import PartialState

from models.motip import build as build_motip
from models.misc import load_checkpoint
from utils.misc import yaml_to_dict
from configs.util import load_super_config, update_config
from log.logger import Logger
from evaluation.submit_and_evaluate import submit_and_evaluate_one_model


def main():
    parser = argparse.ArgumentParser("Evaluate a single epoch checkpoint")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--outputs-dir", type=str, default=None,
                        help="Override outputs dir. Default: auto from config.")
    args = parser.parse_args()

    # Load config
    config = yaml_to_dict(args.config_path)
    if "SUPER_CONFIG_PATH" in config and config["SUPER_CONFIG_PATH"] is not None:
        config = load_super_config(config)
    config = update_config(config=config, args=None)

    # Determine outputs dir
    if args.outputs_dir:
        outputs_dir = args.outputs_dir
    else:
        outputs_dir = config.get("OUTPUTS_DIR") or os.path.join(
            "./outputs/", config["EXP_NAME"]
        )

    eval_dir = os.path.join(outputs_dir, "train", "eval_during_train", f"epoch_{args.epoch}")
    os.makedirs(eval_dir, exist_ok=True)

    # Init accelerator
    accelerator = Accelerator()
    state = PartialState()

    # Logger
    logger = Logger(logdir=eval_dir, use_wandb=False, config=config)

    # Build model and load checkpoint
    model, _ = build_motip(config=config)
    load_checkpoint(model, path=args.checkpoint)
    model = accelerator.prepare(model)

    logger.info(f"Evaluating checkpoint {args.checkpoint} for epoch {args.epoch}")

    eval_metrics = submit_and_evaluate_one_model(
        is_evaluate=True,
        accelerator=accelerator,
        state=state,
        logger=logger,
        model=model,
        data_root=config["DATA_ROOT"],
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=eval_dir,
        image_max_longer=config["INFERENCE_MAX_LONGER"],
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
        miss_tolerance=config["MISS_TOLERANCE"],
        use_sigmoid=config["USE_FOCAL_LOSS"] if "USE_FOCAL_LOSS" in config else False,
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        det_thresh=config["DET_THRESH"],
        newborn_thresh=config["NEWBORN_THRESH"],
        id_thresh=config["ID_THRESH"],
        area_thresh=config.get("AREA_THRESH", 0),
        inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
        else config["ONLY_DETR"],
        concept_bottleneck_mode=config.get("MOTIP", {}).get("CONCEPT_BOTTLENECK_MODE", "hard"),
    )

    if eval_metrics is not None:
        eval_metrics.sync()
        logger.metrics(
            log=f"[Eval epoch: {args.epoch}] ",
            metrics=eval_metrics,
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=0,
            prefix="epoch",
            x_axis_step=args.epoch,
            x_axis_name="epoch",
        )

    logger.success(f"Done evaluating epoch {args.epoch}.")


if __name__ == "__main__":
    main()
