# Copyright (c) Ruopeng Gao. All Rights Reserved.

from .motip import MOTIP
from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder


def build_rfdetr_args(config: dict) -> Args:
    """Build arguments for RF-DETR model."""
    args = Args()
    
    # Basic settings
    args.device = config["DEVICE"]
    args.num_classes = config["NUM_CLASSES"]
    args.num_queries = config.get("DETR_NUM_QUERIES", 300)
    
    # RF-DETR Medium configuration (default)
    rfdetr_config = config.get("RFDETR", {})
    args.encoder = rfdetr_config.get("ENCODER", "dinov2_windowed_small")
    args.resolution = rfdetr_config.get("RESOLUTION", 576)
    args.hidden_dim = rfdetr_config.get("HIDDEN_DIM", 256)
    args.dec_layers = rfdetr_config.get("DEC_LAYERS", 4)
    args.patch_size = rfdetr_config.get("PATCH_SIZE", 16)
    args.num_windows = rfdetr_config.get("NUM_WINDOWS", 2)
    args.positional_encoding_size = rfdetr_config.get("POSITIONAL_ENCODING_SIZE", 36)
    args.out_feature_indexes = rfdetr_config.get("OUT_FEATURE_INDEXES", [3, 6, 9, 12])
    args.projector_scale = rfdetr_config.get("PROJECTOR_SCALE", ["P4"])
    
    # Transformer settings
    args.sa_nheads = rfdetr_config.get("SA_NHEADS", 8)
    args.ca_nheads = rfdetr_config.get("CA_NHEADS", 16)
    args.dec_n_points = rfdetr_config.get("DEC_N_POINTS", 2)
    args.vit_encoder_num_layers = max(args.out_feature_indexes) + 1
    
    # Transformer decoder settings (required by build_transformer)
    args.dropout = rfdetr_config.get("DROPOUT", 0.0)
    args.dim_feedforward = rfdetr_config.get("DIM_FEEDFORWARD", 1024)
    args.decoder_norm = rfdetr_config.get("DECODER_NORM", "LN")  # LN or Identity
    
    # Training settings
    args.aux_loss = config.get("DETR_AUX_LOSS", True)
    args.two_stage = rfdetr_config.get("TWO_STAGE", True)
    args.lite_refpoint_refine = rfdetr_config.get("LITE_REFPOINT_REFINE", True)
    args.bbox_reparam = rfdetr_config.get("BBOX_REPARAM", True)
    args.group_detr = rfdetr_config.get("GROUP_DETR", 1)  # Use 1 for tracking (stability)
    args.ia_bce_loss = rfdetr_config.get("IA_BCE_LOSS", True)
    
    # Loss coefficients
    args.cls_loss_coef = config.get("DETR_CLS_LOSS_COEF", 1.0)
    args.bbox_loss_coef = config.get("DETR_BBOX_LOSS_COEF", 5.0)
    args.giou_loss_coef = config.get("DETR_GIOU_LOSS_COEF", 2.0)
    args.focal_alpha = config.get("DETR_FOCAL_ALPHA", 0.25)
    args.set_cost_class = config.get("DETR_SET_COST_CLASS", 2.0)
    args.set_cost_bbox = config.get("DETR_SET_COST_BBOX", 5.0)
    args.set_cost_giou = config.get("DETR_SET_COST_GIOU", 2.0)
    
    # Backbone settings
    args.layer_norm = rfdetr_config.get("LAYER_NORM", True)
    args.rms_norm = rfdetr_config.get("RMS_NORM", False)
    args.freeze_encoder = rfdetr_config.get("FREEZE_ENCODER", False)
    args.backbone_lora = rfdetr_config.get("BACKBONE_LORA", False)
    args.gradient_checkpointing = rfdetr_config.get("GRADIENT_CHECKPOINTING", False)
    args.load_dinov2_weights = rfdetr_config.get("LOAD_DINOV2_WEIGHTS", True)
    args.force_no_pretrain = rfdetr_config.get("FORCE_NO_PRETRAIN", False)
    args.pretrained_encoder = rfdetr_config.get("PRETRAINED_ENCODER", None)
    args.window_block_indexes = rfdetr_config.get("WINDOW_BLOCK_INDEXES", None)
    args.drop_path = rfdetr_config.get("DROP_PATH", 0.0)
    args.use_cls_token = rfdetr_config.get("USE_CLS_TOKEN", False)
    args.position_embedding = rfdetr_config.get("POSITION_EMBEDDING", "sine")
    
    # Pretrain weights (RF-DETR checkpoint)
    args.pretrain_weights = rfdetr_config.get("PRETRAIN_WEIGHTS", "rf-detr-medium.pth")
    
    # Concept settings
    if "MOTIP" in config and "N_CONCEPTS" in config["MOTIP"]:
        args.num_concepts = config["MOTIP"]["N_CONCEPTS"]
        args.concept_loss_coef = config["MOTIP"]["CONCEPT_LOSS_COEF"]
        args.losses = config["MOTIP"]["DETR_LOSSES"]
        if "CONCEPT_CLASSES" in config["MOTIP"]:
            args.concept_classes = [tuple(c) for c in config["MOTIP"]["CONCEPT_CLASSES"]]
        else:
            args.concept_classes = None
    else:
        args.num_concepts = 0
        args.concept_loss_coef = 0
        args.losses = ['labels', 'boxes', 'cardinality']
        args.concept_classes = None
    
    return args


def build(config: dict):
    # Generate DETR args:
    detr_args = Args()
    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    # 2. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]
    
    # Add concept-related parameters for MOTIP
    if "MOTIP" in config and "N_CONCEPTS" in config["MOTIP"]:
        detr_args.num_concepts = config["MOTIP"]["N_CONCEPTS"]
        detr_args.concept_loss_coef = config["MOTIP"]["CONCEPT_LOSS_COEF"]
        detr_args.losses = config["MOTIP"]["DETR_LOSSES"]
        # Multi-concept support: CONCEPT_CLASSES is a list of (name, num_classes, unknown_label)
        # e.g., [["gender", 3, 2], ["upper_body", 13, 12]]
        if "CONCEPT_CLASSES" in config["MOTIP"]:
            # Convert from list of lists to list of tuples
            detr_args.concept_classes = [tuple(c) for c in config["MOTIP"]["CONCEPT_CLASSES"]]
        else:
            # Legacy single concept (gender only)
            detr_args.concept_classes = None
    else:
        # Fall back to top-level config if MOTIP section not available
        detr_args.num_concepts = config.get("NUM_CONCEPTS", 0)
        detr_args.concept_loss_coef = config.get("CONCEPT_LOSS_COEF", 0)
        detr_args.losses = ['labels', 'boxes', 'cardinality']
        detr_args.concept_classes = None

    # Get DETR framework, default to deformable_detr for backward compatibility
    detr_framework = config.get("DETR_FRAMEWORK", "deformable_detr").lower()
    match detr_framework:
        case "deformable_detr":
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case "rf_detr" | "rfdetr" | "rf-detr":
            # Build RF-DETR with MOTIP integration
            from models.rfdetr import build_rfdetr
            rfdetr_args = build_rfdetr_args(config)
            detr, detr_criterion, _ = build_rfdetr(args=rfdetr_args)
        case _:
            raise NotImplementedError(f"DETR framework {config['DETR_FRAMEWORK']} is not supported.")

    # Build each component:
    # 1. trajectory modeling (currently, only FFNs are used):
    # For RF-DETR, hidden_dim may be different
    hidden_dim = config.get("RFDETR", {}).get("HIDDEN_DIM", config["DETR_HIDDEN_DIM"]) if detr_framework in ["rf_detr", "rfdetr", "rf-detr"] else config["DETR_HIDDEN_DIM"]
    _trajectory_modeling = TrajectoryModeling(
        detr_dim=hidden_dim,
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
    ) if config["ONLY_DETR"] is False else None
    # 2. ID decoder:
    _id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
    ) if config["ONLY_DETR"] is False else None

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
    )

    return motip_model, detr_criterion
