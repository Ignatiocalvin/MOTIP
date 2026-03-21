#!/usr/bin/env bash
# ============================================================================
# RF-DETR Model Training Script - All Variants
# ============================================================================
# This script trains all RF-DETR model variants for comparison with R50 models:
#   1. Base (0 concepts) - Pure detection + ID tracking
#   2. 2-Concept - Gender + Upper Body (optimal for R50)
#   3. 3-Concept - Gender + Upper Body + Lower Body
#   4. 7-Concept Manual - All concepts with fixed CONCEPT_LOSS_COEF=0.5
#   5. 7-Concept Learnable - All concepts with uncertainty-based weighting
#
# Usage:
#   ./train_rfdetr_all.sh [model] [fold]
#   ./train_rfdetr_all.sh                    # Train all models, fold 0
#   ./train_rfdetr_all.sh base 0             # Train base model, fold 0
#   ./train_rfdetr_all.sh 2concept 0         # Train 2-concept model, fold 0
#   ./train_rfdetr_all.sh 3concept 0         # Train 3-concept model, fold 0
#   ./train_rfdetr_all.sh 7concept_manual 0  # Train 7-concept manual, fold 0
#   ./train_rfdetr_all.sh 7concept_learnable 0  # Train 7-concept learnable, fold 0
# ============================================================================

set -e  # Exit on error

# Default values
MODEL=${1:-"all"}
FOLD=${2:-0}

# Setup logging
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/rfdetr_${MODEL}_fold${FOLD}_${TIMESTAMP}.log"
echo "Logging to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "RF-DETR Training Script"
echo "Model: $MODEL"
echo "Fold: $FOLD"
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "=============================================="

# ============================================
# Environment Setup
# ============================================

# Load CUDA module (HPC only)
if command -v module &> /dev/null; then
    module load devel/cuda/11.8 2>/dev/null || echo "CUDA module not available"
fi

# Deactivate any existing virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

# Source conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate base
fi

# Set CUDA environment
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4070 Ti SUPER
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi
export CUDA_VISIBLE_DEVICES=0

# Add RF-DETR to Python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/rf-detr:${PYTHONPATH}"

# Verify GPU availability
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: CUDA not available!"
    exit 1
fi

# ============================================
# Build CUDA operators if needed
# ============================================
python -c "import MultiScaleDeformableAttention" 2>/dev/null || {
    echo "Building CUDA operators..."
    cd models/ops && python setup.py build install && cd ../..
}

# ============================================
# Model Configurations
# ============================================

declare -A CONFIGS
CONFIGS["base"]="configs/rfdetr_large_motip_pdestre_base.yaml"
CONFIGS["2concept"]="configs/rfdetr_large_motip_pdestre_2concepts.yaml"
CONFIGS["3concept"]="configs/rfdetr_large_motip_pdestre_3concepts.yaml"
CONFIGS["7concept_manual"]="configs/rfdetr_large_motip_pdestre_7concepts_manual.yaml"
CONFIGS["7concept_learnable"]="configs/rfdetr_large_motip_pdestre_7concepts_learnable.yaml"

declare -A EXP_NAMES
EXP_NAMES["base"]="rfdetr_large_motip_pdestre_base_fold"
EXP_NAMES["2concept"]="rfdetr_large_motip_pdestre_2concepts_fold"
EXP_NAMES["3concept"]="rfdetr_large_motip_pdestre_3concepts_fold"
EXP_NAMES["7concept_manual"]="rfdetr_large_motip_pdestre_7concepts_manual_fold"
EXP_NAMES["7concept_learnable"]="rfdetr_large_motip_pdestre_7concepts_learnable_fold"

# ============================================
# Training Function
# ============================================

train_model() {
    local model_type=$1
    local fold_idx=$2
    
    local config_file=${CONFIGS[$model_type]}
    local exp_name="${EXP_NAMES[$model_type]}${fold_idx}"
    
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file not found: $config_file"
        return 1
    fi
    
    echo "=============================================="
    echo "Training: $model_type (Fold $fold_idx)"
    echo "Config: $config_file"
    echo "Experiment: $exp_name"
    echo "=============================================="
    
    # Create temporary config with fold-specific settings
    local fold_config="/tmp/rfdetr_${model_type}_fold${fold_idx}.yaml"
    cat > "$fold_config" << EOF
# Temporary config for $model_type fold ${fold_idx}
SUPER_CONFIG_PATH: ./${config_file}

# Override dataset splits for this fold
DATASETS: [P-DESTRE]
DATASET_SPLITS: [Train_${fold_idx}]
INFERENCE_DATASET: P-DESTRE
INFERENCE_SPLIT: val_${fold_idx}
EOF
    
    # Check for existing checkpoint (for resuming)
    local checkpoint_dir="./outputs/${exp_name}"
    local resume_arg=""
    
    if [ -d "$checkpoint_dir" ]; then
        # Find latest checkpoint
        local latest_ckpt=$(ls -t ${checkpoint_dir}/checkpoint_*.pth 2>/dev/null | head -1)
        if [ -n "$latest_ckpt" ]; then
            # Check if model uses learnable weights (can't resume due to optimizer structure)
            if [[ "$model_type" == *"learnable"* ]]; then
                echo "Learnable model - starting fresh (optimizer structure differs)"
            else
                echo "Found checkpoint: $latest_ckpt"
                resume_arg="--resume-model $latest_ckpt"
            fi
        fi
    fi
    
    # Launch training
    env PYTHONPATH="${SCRIPT_DIR}/rf-detr:${PYTHONPATH}" \
    accelerate launch --num_processes=1 train.py \
        --data-root ./data/ \
        --exp-name "$exp_name" \
        --config-path "$fold_config" \
        $resume_arg
    
    # Cleanup
    rm -f "$fold_config"
    
    echo "Completed: $model_type (Fold $fold_idx) at $(date)"
}

# ============================================
# Main Execution
# ============================================

if [ "$MODEL" == "all" ]; then
    echo "Training ALL RF-DETR models for fold $FOLD..."
    
    # Train in order: base → 2concept → 3concept → 7concept_manual → 7concept_learnable
    for model in "base" "2concept" "3concept" "7concept_manual" "7concept_learnable"; do
        train_model "$model" "$FOLD"
        echo ""
    done
else
    # Train specific model
    if [ -z "${CONFIGS[$MODEL]}" ]; then
        echo "ERROR: Unknown model type: $MODEL"
        echo "Available models: base, 2concept, 3concept, 7concept_manual, 7concept_learnable, all"
        exit 1
    fi
    
    train_model "$MODEL" "$FOLD"
fi

echo "=============================================="
echo "All training completed at $(date)"
echo "=============================================="

# ============================================
# Print Summary
# ============================================

echo ""
echo "=============================================="
echo "TRAINING SUMMARY"
echo "=============================================="
echo ""
echo "Models trained:"
if [ "$MODEL" == "all" ]; then
    for model in "base" "2concept" "3concept" "7concept_manual" "7concept_learnable"; do
        exp_name="${EXP_NAMES[$model]}${FOLD}"
        if [ -d "outputs/$exp_name" ]; then
            ckpts=$(ls outputs/$exp_name/checkpoint_*.pth 2>/dev/null | wc -l)
            echo "  ✓ $model: $ckpts checkpoints"
        else
            echo "  ✗ $model: not found"
        fi
    done
else
    exp_name="${EXP_NAMES[$MODEL]}${FOLD}"
    if [ -d "outputs/$exp_name" ]; then
        ckpts=$(ls outputs/$exp_name/checkpoint_*.pth 2>/dev/null | wc -l)
        echo "  ✓ $MODEL: $ckpts checkpoints"
    else
        echo "  ✗ $MODEL: not found"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Run evaluation: python evaluation/compute_tracking_metrics.py"
echo "  2. Compare with R50 models in visualizations/METRICS_COMPARISON.md"
echo ""
