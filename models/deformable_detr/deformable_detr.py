# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from utils import box_ops
from utils.nested_tensor import NestedTensor, nested_tensor_from_tensor_list
from models.misc import inverse_sigmoid, accuracy, interpolate
from utils.misc import is_distributed, distributed_world_size
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized, inverse_sigmoid)

from models.mlp import MLP
from models.deformable_detr.backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_concepts, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_concepts = num_concepts # Number of concepts
        if self.num_concepts > 0:
            self.concept_embed = MLP(hidden_dim, hidden_dim, self.num_concepts, 3)
        self.class_embed = nn.Linear(hidden_dim, num_classes) 
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # 4. Initialize concept_embed head (Classification Prior Principle)
        if self.num_concepts > 0:
            # As a multi-label classification head, we follow the same
            # principle as the class_embed head: set a low prior probability
            # for each concept being "present" to ensure training stability.
            nn.init.constant_(self.concept_embed.layers[-1].bias.data, bias_value)
            # The weights (self.concept_embed.layers[-1].weight.data) are
            # left to their default (Kaiming/Xavier) initialization,
            # which is consistent with the class_embed head.

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            # A critical and non-obvious step is to ensure the new concept_embed head is correctly handled by the logic 
            # for auxiliary decoder losses. The DeformableDETR clones or repeats its prediction heads for each decoder layer. This logic must be extended to include self.concept_embed.
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if self.num_concepts > 0:
                self.concept_embed = _get_clones(self.concept_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if self.num_concepts > 0:
                self.concept_embed = nn.ModuleList([self.concept_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_concepts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.num_concepts > 0:
                outputs_concepts.append(self.concept_embed[lvl](hs[lvl])) # Apply the corresponding concept head to the hidden state of this layer
                                                                        # We output logits, as nn.BCEWithLogitsLoss is preferred for multi-label concepts
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if self.num_concepts > 0:
            outputs_concept = torch.stack(outputs_concepts)

        # The newly generated outputs_concept tensor must be added to the output dictionaries returned by the forward method.
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.num_concepts > 0:
            out['pred_concepts'] = outputs_concept[-1]

        if self.aux_loss:
            if self.num_concepts > 0:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_concept)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        # In the "two-stage" architecture, the encoder generates initial "proposals." These are intermediate predictions.
        # Our objective is to get concept predictions for the final objects that are used for tracking. These final predictions are generated by the decoder.
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # Output the outputs of last decoder layer.
        # We need these outputs to generate the embeddings for objects.
        out["outputs"] = hs[-1]
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_concept=None):
        if outputs_concept is not None:
            # zip all three
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_concepts': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_concept[:-1])]
        else:
            # original behavior
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, num_concepts=0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.num_concepts = num_concepts

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_concepts(self, outputs, targets, indices, num_boxes, **kwargs):
        """Classification loss for concepts (e.g., gender).
        We compute the cross-entropy loss over the matched queries.
        """
        assert 'pred_concepts' in outputs
        src_logits = outputs['pred_concepts'] # (B, N_queries, num_concepts)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["concepts"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # P-DESTRE's gender attribute '2' means 'Unknown'.
        # We must ignore these labels during training.
        # Create a mask for valid (non-unknown) concept labels.
        # We only compute loss for matched queries (idx) that have a valid label.
        
        # Get matched source logits
        src_logits_matched = src_logits[idx]
        
        # Create mask for valid targets (where target!= 2)
        valid_mask = (target_classes_o!= 2)
        
        if valid_mask.sum() == 0:
            # Handle case where all targets in batch are 'Unknown'
            losses = {'loss_concepts': torch.tensor(0.0, device=src_logits.device)}
            return losses

        # Apply mask to both logits and targets
        src_logits_valid = src_logits_matched[valid_mask]
        target_classes_valid = target_classes_o[valid_mask]

        # Get predictions for logging
        pred_concepts = torch.argmax(src_logits_valid, dim=-1)
        
        # Calculate accuracy for logging
        correct_predictions = (pred_concepts == target_classes_valid).float()
        concept_accuracy = correct_predictions.mean() * 100.0

        # Compute cross-entropy loss only on valid, matched predictions
        loss_concepts = F.cross_entropy(src_logits_valid, target_classes_valid, reduction='none')

        losses = {}
        losses['loss_concepts'] = loss_concepts.sum() / num_boxes
        losses['concept_accuracy'] = concept_accuracy
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def debug_concept_targets(self, outputs, targets, indices):
        """Enhanced debugging for concept/ID target analysis"""
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        self._debug_step += 1
        
        # Only debug every 100 steps to avoid spam
        if self._debug_step % 100 != 0:
            return True  # Assume valid to avoid spam
        
        print(f"\n🔍 DEBUG STEP {self._debug_step}")
        print(f"Batch size: {len(targets)}")
        
        total_concepts = 0
        total_ids = 0
        valid_targets = 0
        
        for i, target in enumerate(targets):
            has_concepts = 'concepts' in target and target['concepts'] is not None
            has_ids = 'track_ids' in target and target['track_ids'] is not None
            
            if has_concepts:
                concepts = target['concepts']
                if hasattr(concepts, '__len__'):
                    total_concepts += len(concepts)
                else:
                    total_concepts += 1
            
            if has_ids:
                ids = target['track_ids'] 
                if hasattr(ids, '__len__'):
                    total_ids += len(ids)
                else:
                    total_ids += 1
            
            if has_concepts or has_ids:
                valid_targets += 1
            
            if i < 3:  # Show details for first 3 targets
                print(f"  Target {i}: concepts={has_concepts}, ids={has_ids}, keys={list(target.keys())}")
        
        print(f"Summary: {valid_targets}/{len(targets)} targets have concept/ID data")
        print(f"Total concepts: {total_concepts}, Total IDs: {total_ids}")
        
        # Check if we have matched indices
        if indices:
            matched_targets = len(indices)
            print(f"Matched targets from Hungarian matching: {matched_targets}")
            
            try:
                # Check actual concept targets after matching
                target_concepts = torch.cat([t["concepts"][J] for t, (_, J) in zip(targets, indices)])
                print(f"Matched concept targets: {len(target_concepts)}")
                
                target_ids = torch.cat([t["track_ids"][J] for t, (_, J) in zip(targets, indices)]) 
                print(f"Matched ID targets: {len(target_ids)}")
                
            except Exception as e:
                print(f"Error extracting matched targets: {e}")
        
        return total_concepts > 0 or total_ids > 0

    def log_concept_predictions(self, outputs, targets, indices, num_samples=3):
        """Enhanced logging with both concept and ID predictions vs ground truth."""
        if 'pred_concepts' not in outputs or self.num_concepts == 0:
            return
        
        # Add debug information
        has_valid_targets = self.debug_concept_targets(outputs, targets, indices)
        if not has_valid_targets:
            print("[Concepts] No concept/ID targets found - DEBUG CONFIRMED", flush=True)
            return
            
        concept_labels = ['Male', 'Female', 'Unknown']
        
        # Get predictions and targets
        src_logits = outputs['pred_logits']  # For ID predictions
        src_concept_logits = outputs['pred_concepts']  # For concept predictions
        idx = self._get_src_permutation_idx(indices)
        
        # Check if we have any targets with concepts
        try:
            target_concepts = torch.cat([t["concepts"][J] for t, (_, J) in zip(targets, indices)])
            target_ids = torch.cat([t["track_ids"][J] for t, (_, J) in zip(targets, indices)])
        except:
            print("[Concepts] No concept/ID targets found", flush=True)
            return
        
        # Get matched predictions
        pred_concept_logits = src_concept_logits[idx]
        pred_concepts = torch.argmax(pred_concept_logits, dim=-1)
        
        # Get ID predictions (assuming class predictions represent object instances/IDs)
        pred_id_logits = src_logits[idx]
        pred_ids = torch.argmax(pred_id_logits, dim=-1)
        
        # Count all targets (including Unknown concepts)
        total_targets = len(target_concepts)
        unknown_count = (target_concepts == 2).sum().item()
        
        # Filter out Unknown labels (=2) for concept accuracy calculation
        valid_concept_mask = (target_concepts != 2)
        
        # Concept Statistics
        if valid_concept_mask.sum() == 0:
            print(f"[Concepts] Targets: {total_targets} (all Unknown) - skipping concept accuracy", flush=True)
            concept_accuracy = 0
            pred_male = pred_female = gt_male = gt_female = 0
        else:
            # Apply filter for concepts
            pred_concepts_valid = pred_concepts[valid_concept_mask]
            target_concepts_valid = target_concepts[valid_concept_mask]
            
            # Calculate concept accuracy
            correct_concepts = (pred_concepts_valid == target_concepts_valid).sum().item()
            valid_concept_count = len(pred_concepts_valid)
            concept_accuracy = correct_concepts / valid_concept_count * 100 if valid_concept_count > 0 else 0
            
            # Count concept predictions and ground truth
            pred_male = (pred_concepts_valid == 0).sum().item()
            pred_female = (pred_concepts_valid == 1).sum().item()
            gt_male = (target_concepts_valid == 0).sum().item()
            gt_female = (target_concepts_valid == 1).sum().item()
        
        # ID Statistics
        correct_ids = (pred_ids == target_ids).sum().item()
        total_ids = len(pred_ids)
        id_accuracy = correct_ids / total_ids * 100 if total_ids > 0 else 0
        
        # Count unique IDs
        unique_pred_ids = len(torch.unique(pred_ids))
        unique_gt_ids = len(torch.unique(target_ids))
        
        # Comprehensive output
        print(f"\n[Tracking] Targets: {total_targets} objects", flush=True)
        print(f"[Concepts] Accuracy: {concept_accuracy:.1f}% ({unknown_count} Unknown, {total_targets - unknown_count} valid)", flush=True)
        print(f"[Concepts] Preds: {pred_male}M {pred_female}F | GT: {gt_male}M {gt_female}F", flush=True)
        print(f"[IDs] Accuracy: {id_accuracy:.1f}% ({correct_ids}/{total_ids})", flush=True)
        print(f"[IDs] Unique: Pred={unique_pred_ids}, GT={unique_gt_ids}", flush=True)
        
        # Show detailed examples
        num_show = min(num_samples, total_targets)
        if num_show > 0:
            examples = []
            for i in range(num_show):
                # Concept info
                if i < len(target_concepts):
                    concept_idx = target_concepts[i].item()
                    if concept_idx == 2:  # Unknown
                        concept_match = "?"
                        concept_info = "U"
                    else:
                        pred_concept_idx = pred_concepts[i].item()
                        p_label = concept_labels[pred_concept_idx][:1]  # Just M/F
                        t_label = concept_labels[concept_idx][:1]  # Just M/F
                        concept_match = "✓" if pred_concept_idx == concept_idx else "✗"
                        concept_info = f"{p_label}→{t_label}"
                else:
                    concept_match = "?"
                    concept_info = "?"
                
                # ID info
                pred_id = pred_ids[i].item()
                gt_id = target_ids[i].item()
                id_match = "✓" if pred_id == gt_id else "✗"
                id_info = f"{pred_id}→{gt_id}"
                
                examples.append(f"[{concept_match}{concept_info}|{id_match}{id_info}]")
            
            print(f"[Examples] {' '.join(examples)}", flush=True)
            print(f"[Legend] [Concept_match Pred→GT | ID_match Pred→GT]", flush=True)

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        assert "batch_len" in kwargs, f"batch_len is not in kwargs"
        batch_len = kwargs["batch_len"]
        kwargs = {}     # to default setting

        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'concepts': self.loss_concepts
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        # Organize the batch data:
        loss_dict = {}
        iter_idxs = torch.tensor(list(range(0, len(targets))), dtype=torch.int64, device=outputs['pred_logits'].device)
        from train import batch_iterator, tensor_dict_index_select
        for batch_iter_idxs, batch_targets, batch_indices in batch_iterator(
            batch_len, iter_idxs, targets, indices
        ):
            batch_outputs = tensor_dict_index_select(outputs, batch_iter_idxs, dim=0)
            batch_loss_dict = loss_map[loss](batch_outputs, batch_targets, batch_indices, 1, **kwargs)  # num_boxes=1
            for k, v in batch_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = v
                else:
                    loss_dict[k] += v
        # Average the loss:
        if loss == "labels" or loss == "boxes" or loss == "masks":
            for k in loss_dict.keys():
                loss_dict[k] /= num_boxes
        pass
        return loss_dict
        # return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if "batch_len" not in kwargs:
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = []
            iter_idxs = torch.tensor(
                list(range(0, len(targets))), dtype=torch.int64, device=outputs_without_aux['pred_logits'].device
            )
            from train import batch_iterator, tensor_dict_index_select
            for batch_iter_idxs, batch_targets in batch_iterator(
                    kwargs["batch_len"], iter_idxs, targets
            ):
                batch_outputs_without_aux = tensor_dict_index_select(outputs_without_aux, batch_iter_idxs, dim=0)
                _ = self.matcher(batch_outputs_without_aux, batch_targets)
                indices += _
                pass

        batch_len = kwargs["batch_len"]         # HELLORPG Added
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_distributed():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / distributed_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {"batch_len": kwargs["batch_len"]}         # HELLORPG Added
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # Log concept predictions vs ground truth (for debugging)
        if hasattr(self, 'num_concepts') and self.num_concepts > 0 and 'pred_concepts' in outputs:
            if not hasattr(self, '_log_counter'):
                self._log_counter = 0
            self._log_counter += 1
            
            # Simple logging: every 100 batches, keep it clean
            if self._log_counter % 100 == 0:
                try:
                    self.log_concept_predictions(outputs, targets, indices, num_samples=3)
                except Exception as e:
                    print(f"[Concepts] Logging error: {e}", flush=True)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    kwargs["batch_len"] = batch_len     # HELLORPG Added
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, indices


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     num_classes = 250
    num_classes = args.num_classes
    num_concepts = getattr(args, 'num_concepts', 0)
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_concepts=num_concepts
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    # Read concept_loss_coef from args, default to 0
    concept_loss_coef = getattr(args, 'concept_loss_coef', 0)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if concept_loss_coef > 0:
        weight_dict['loss_concepts'] = concept_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.masks:
        losses += ["masks"]

    # Handle the 'losses' argument from the config
    # If not provided, default to original behavior
    losses = getattr(args, 'losses', ['labels', 'boxes', 'cardinality'])

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25, num_concepts=0
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, num_concepts=num_concepts)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

