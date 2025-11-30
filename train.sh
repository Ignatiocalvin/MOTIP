#!/usr/bin/env bash
#SBATCH --job-name=mk21
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time 00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/make_%j_concept_id_logs.out
#SBATCH --error=logs/make_%j_concept_id_logs.err
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

echo "Started at $(date)"

# Make sure we're not in any virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

# Source conda - you need to find your miniconda3 path
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

TORCH_CUDA_ARCH_LIST="8.0" 
CUDA_HOME='/opt/bwhpc/common/devel/cuda/11.8'  
accelerate launch --num_processes=1 train.py --data-root ./data/ --exp-name r50_deformable_detr_motip_pdestre_concept_and_ID_logs --config-path ./configs/r50_deformable_detr_motip_pdestre.yaml

# JOBID 2191314 - short (1 epoch, sample length 5)
# JOBID 2191406 - medium (3 epochs, sample length 15)
# JOBID 2191379 - long (6 epochs, sample length 30)

# JOBID 2224063 - long (5 epochs, 15 sample lengths) with added concepts
# JOBID 2229082 - long (4 epochs, 20 sample lengths) with added concepts