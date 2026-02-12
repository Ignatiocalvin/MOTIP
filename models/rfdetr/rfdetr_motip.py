# Copyright (c) Ruopeng Gao. All Rights Reserved.
# RF-DETR integration for MOTIP - Main Model

"""
RF-DETR MOTIP Integration Model

This module wraps RF-DETR's LWDETR model to be compatible with MOTIP's tracking framework.
Key modifications:
1. Add concept prediction heads (gender, clothing, etc.)
2. Output hidden states for trajectory modeling
3. Match MOTIP's DeformableDETR output format
"""

import sys
import os
import math
import copy
from typing import Optional, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add rf-detr to path
RFDETR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'rf-detr'))
print(f"[DEBUG rfdetr_motip.py] RFDETR_PATH: {RFDETR_PATH}")
print(f"[DEBUG rfdetr_motip.py] Path exists: {os.path.exists(RFDETR_PATH)}")
if os.path.exists(RFDETR_PATH):
    print(f"[DEBUG rfdetr_motip.py] Contents of rf-detr/: {os.listdir(RFDETR_PATH)}")
    rfdetr_pkg = os.path.join(RFDETR_PATH, 'rfdetr')
    print(f"[DEBUG rfdetr_motip.py] rfdetr package exists: {os.path.exists(rfdetr_pkg)}")
    # Check if it's at the root level
    if not os.path.exists(rfdetr_pkg):
        # Try checking if models/ exists at root (common RF-DETR structure)
        models_dir = os.path.join(RFDETR_PATH, 'models')
        if os.path.exists(models_dir):
            print(f"[DEBUG rfdetr_motip.py] Found models/ at root, using RFDETR_PATH as is")
            RFDETR_PATH = RFDETR_PATH  # It's at the root
        else:
            print(f"[ERROR rfdetr_motip.py] Cannot find rfdetr package structure")
if RFDETR_PATH not in sys.path:
    sys.path.insert(0, RFDETR_PATH)
    print(f"[DEBUG rfdetr_motip.py] Added to sys.path[0]: {RFDETR_PATH}")

from rfdetr.models.backbone import build_backbone
from rfdetr.models.transformer import build_transformer
from rfdetr.util.misc import NestedTensor, nested_tensor_from_tensor_list


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RFDETR_MOTIP(nn.Module):
    """
    RF-DETR model adapted for MOTIP tracking with concept prediction.
    
    This class wraps RF-DETR's LWDETR architecture and adds:
    1. Concept prediction heads for multi-attribute classification
    2. Hidden state output for trajectory modeling
    3. MOTIP-compatible output format
    
    The architecture uses:
    - DINOv2 backbone (windowed attention variant)
    - Deformable attention transformer
    - Multi-scale feature projection
    """
    
    def __init__(
        self,
        backbone,
        transformer,
        num_classes: int,
        num_queries: int,
        num_concepts: int = 0,
        concept_classes: Optional[List[int]] = None,
        aux_loss: bool = True,
        group_detr: int = 1,
        two_stage: bool = True,
        lite_refpoint_refine: bool = True,
        bbox_reparam: bool = True,
    ):
        """
        Initialize RF-DETR for MOTIP.
        
        Args:
            backbone: DINOv2 backbone with projector
            transformer: Deformable transformer
            num_classes: Number of object classes (typically 1 for person tracking)
            num_queries: Number of object queries
            num_concepts: Number of concept types (deprecated, use concept_classes)
            concept_classes: List of (name, num_classes, unknown_label) tuples for each concept
            aux_loss: Whether to use auxiliary decoding losses
            group_detr: Number of groups for accelerated training
            two_stage: Use two-stage detection
            lite_refpoint_refine: Use lightweight reference point refinement
            bbox_reparam: Use box reparameterization
        """
        super().__init__()
        print(f"[DEBUG] Initializing RFDETR_MOTIP...")
        
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        print(f"[DEBUG] Transformer d_model: {hidden_dim}")
        
        # Detection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Query embeddings
        query_dim = 4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)
        
        # Concept prediction heads
        self.concept_classes = concept_classes
        if concept_classes is not None and len(concept_classes) > 0:
            self.num_concepts = len(concept_classes)
            # Create separate prediction head for each concept
            self.concept_embeds = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, n_classes, 3) 
                for n_classes in concept_classes
            ])
        elif num_concepts > 0:
            # Legacy single concept mode
            self.num_concepts = 1
            self.concept_classes = [num_concepts]
            self.concept_embeds = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, num_concepts, 3)
            ])
        else:
            self.num_concepts = 0
            self.concept_classes = []
            self.concept_embeds = None
        
        # Backbone and settings
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr
        self.two_stage = two_stage
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam
        print(f"[DEBUG] RFDETR_MOTIP initialization complete")
        
        # Store patch size and num_windows for input preprocessing
        # Default values for DINOv2 with windowed attention
        self.patch_size = 14
        self.num_windows = 2
        self.block_size = self.patch_size * self.num_windows  # 28
        
        # Reference point refinement
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None
        
        # Initialize prior probability for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        # Initialize bbox embed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        # Initialize concept heads
        if self.num_concepts > 0 and self.concept_embeds is not None:
            for concept_embed in self.concept_embeds:
                nn.init.constant_(concept_embed.layers[-1].bias.data, bias_value)
        
        # Two-stage encoder heads
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)]
            )
        
        self._export = False

    def export(self):
        """Prepare model for export/inference."""
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if (hasattr(m, "export") and isinstance(m.export, Callable) and 
                hasattr(m, "_export") and not m._export):
                m.export()

    def forward(self, samples: NestedTensor, targets=None):
        """
        Forward pass of RF-DETR for MOTIP.
        
        Args:
            samples: NestedTensor with:
                - tensors: batched images [B, 3, H, W]
                - mask: padding mask [B, H, W]
            targets: Optional target dicts for training
            
        Returns:
            dict with:
                - pred_logits: [B, num_queries, num_classes]
                - pred_boxes: [B, num_queries, 4]
                - pred_concepts: list of [B, num_queries, n_classes] for each concept
                - outputs: hidden states [B, num_queries, hidden_dim] for tracking
                - aux_outputs: optional intermediate layer outputs
                - enc_outputs: optional encoder outputs (two-stage)
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # RF-DETR with DINOv2 windowed attention requires:
        # 1. Square images (H == W) for windowed attention to work correctly
        # 2. Dimensions divisible by block_size (patch_size * num_windows = 28)
        tensors = samples.tensors
        B, C, H, W = tensors.shape
        
        # Calculate target size: make square and divisible by block_size
        max_dim = max(H, W)
        target_size = ((max_dim + self.block_size - 1) // self.block_size) * self.block_size
        
        # Resize to square target size using bilinear interpolation
        if H != target_size or W != target_size:
            tensors = F.interpolate(tensors, size=(target_size, target_size), 
                                   mode='bilinear', align_corners=False)
            # Create new mask for resized tensor
            mask = torch.zeros((B, target_size, target_size), 
                              dtype=torch.bool, device=tensors.device)
            samples = NestedTensor(tensors, mask)
        
        # Backbone feature extraction
        features, poss = self.backbone(samples)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None
        
        # Query embeddings
        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # Only use one group during inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]
        
        # Transformer forward
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight
        )
        
        out = {}
        
        if hs is not None:
            # Box predictions with reparameterization
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + 
                    ref_unsigmoid[..., :2]
                )
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            
            # Class predictions
            outputs_class = self.class_embed(hs)
            
            # Concept predictions
            outputs_concepts = None
            if self.num_concepts > 0 and self.concept_embeds is not None:
                # Apply each concept head to the last layer's hidden states
                outputs_concepts = []
                for concept_idx in range(self.num_concepts):
                    concept_logits = self.concept_embeds[concept_idx](hs[-1])
                    outputs_concepts.append(concept_logits)
            
            # Build output dict
            out = {
                'pred_logits': outputs_class[-1],
                'pred_boxes': outputs_coord[-1],
            }
            
            # Add concept predictions
            if outputs_concepts is not None:
                out['pred_concepts'] = outputs_concepts
            
            # IMPORTANT: Output hidden states for trajectory modeling
            out['outputs'] = hs[-1]
            
            # Auxiliary outputs for intermediate layers
            if self.aux_loss:
                aux_outputs = []
                for lvl in range(hs.shape[0] - 1):
                    aux_out = {
                        'pred_logits': outputs_class[lvl],
                        'pred_boxes': outputs_coord[lvl],
                    }
                    if self.num_concepts > 0 and self.concept_embeds is not None:
                        # Concept predictions for aux layers
                        aux_concepts = []
                        for concept_idx in range(self.num_concepts):
                            aux_concept_logits = self.concept_embeds[concept_idx](hs[lvl])
                            aux_concepts.append(aux_concept_logits)
                        aux_out['pred_concepts'] = aux_concepts
                    aux_outputs.append(aux_out)
                out['aux_outputs'] = aux_outputs
        
        # Two-stage encoder outputs
        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)
            
            if hs is not None:
                out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
            else:
                out = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                out['outputs'] = hs_enc  # For tracking
        
        return out
    
    def forward_export(self, tensors):
        """Export-friendly forward pass."""
        srcs, _, poss = self.backbone(tensors)
        
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight = self.query_feat.weight[:self.num_queries]
        
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight
        )
        
        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + 
                    ref_unsigmoid[..., :2]
                )
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            
            outputs_class = self.class_embed(hs)
            
            # Concept predictions for export
            outputs_concepts = None
            if self.num_concepts > 0 and self.concept_embeds is not None:
                outputs_concepts = []
                for concept_idx in range(self.num_concepts):
                    concept_logits = self.concept_embeds[concept_idx](hs[-1])
                    outputs_concepts.append(concept_logits)
            
            return outputs_coord, outputs_class, outputs_concepts
        else:
            assert self.two_stage
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            return ref_enc, outputs_class, None


def build(args):
    """
    Build RF-DETR model for MOTIP.
    
    Args:
        args: Namespace with model configuration
        
    Returns:
        tuple: (model, criterion, postprocessor)
    """
    from .criterion import SetCriterion, build_matcher
    
    print(f"[DEBUG] Building RF-DETR model...")
    print(f"[DEBUG] Encoder: {args.encoder}, Resolution: {args.resolution}")
    
    num_classes = args.num_classes
    device = torch.device(args.device)
    
    # Build backbone (DINOv2 + projector)
    # Note: force_no_pretrain overrides load_dinov2_weights
    load_weights = getattr(args, 'load_dinov2_weights', True)
    if getattr(args, 'force_no_pretrain', False):
        load_weights = False
    
    print(f"[DEBUG] Load DINOv2 weights: {load_weights}")
    print(f"[DEBUG] Building backbone...")
        
    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=getattr(args, 'pretrained_encoder', None),
        window_block_indexes=getattr(args, 'window_block_indexes', None),
        drop_path=getattr(args, 'drop_path', 0.0),
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=getattr(args, 'use_cls_token', False),
        hidden_dim=args.hidden_dim,
        position_embedding=getattr(args, 'position_embedding', 'sine'),
        freeze_encoder=getattr(args, 'freeze_encoder', False),
        layer_norm=getattr(args, 'layer_norm', True),
        target_shape=(args.resolution, args.resolution),
        rms_norm=getattr(args, 'rms_norm', False),
        backbone_lora=getattr(args, 'backbone_lora', False),
        force_no_pretrain=getattr(args, 'force_no_pretrain', False),
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
        load_dinov2_weights=load_weights,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
    )
    print(f"[DEBUG] Backbone built successfully")
    
    # Set number of feature levels
    args.num_feature_levels = len(args.projector_scale)
    
    # Build transformer
    print(f"[DEBUG] Building transformer...")
    transformer = build_transformer(args)
    print(f"[DEBUG] Transformer built successfully")
    
    # Extract concept classes from args
    concept_classes = None
    num_concepts = 0
    if hasattr(args, 'concept_classes') and args.concept_classes is not None:
        # concept_classes is list of (name, num_classes, unknown_label)
        concept_classes = [c[1] for c in args.concept_classes]  # Extract num_classes
        num_concepts = len(concept_classes)
    elif hasattr(args, 'num_concepts') and args.num_concepts > 0:
        num_concepts = args.num_concepts
    
    # Build model
    model = RFDETR_MOTIP(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_concepts=num_concepts,
        concept_classes=concept_classes,
        aux_loss=getattr(args, 'aux_loss', True),
        group_detr=getattr(args, 'group_detr', 1),
        two_stage=getattr(args, 'two_stage', True),
        lite_refpoint_refine=getattr(args, 'lite_refpoint_refine', True),
        bbox_reparam=getattr(args, 'bbox_reparam', True),
    )
    
    # Build matcher
    matcher = build_matcher(args)
    
    # Weight dict for losses
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }
    
    # Add concept loss weight
    if num_concepts > 0:
        weight_dict['loss_concepts'] = getattr(args, 'concept_loss_coef', 0.5)
    
    # Auxiliary losses
    if getattr(args, 'aux_loss', True):
        aux_weight_dict = {}
        dec_layers = getattr(args, 'dec_layers', 3)
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if getattr(args, 'two_stage', True):
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    # Losses to compute
    losses = getattr(args, 'losses', ['labels', 'boxes', 'cardinality'])
    if num_concepts > 0 and 'concepts' not in losses:
        losses = list(losses) + ['concepts']
    
    # Build criterion
    criterion = SetCriterion(
        num_classes=num_classes,
        num_concepts=num_concepts,
        concept_classes=getattr(args, 'concept_classes', None),
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=getattr(args, 'focal_alpha', 0.25),
        losses=losses,
        group_detr=getattr(args, 'group_detr', 1),
        ia_bce_loss=getattr(args, 'ia_bce_loss', True),
    )
    criterion.to(device)
    
    # Simple postprocessor
    postprocessors = None  # MOTIP handles postprocessing
    
    return model, criterion, postprocessors
