# Copyright (c) Ruopeng Gao. All Rights Reserved.
# RF-DETR Criterion for MOTIP - Loss Computation with Concept Support

"""
Loss computation for RF-DETR MOTIP integration.

This module extends RF-DETR's SetCriterion with:
1. Concept prediction losses (cross-entropy for gender, clothing, etc.)
2. Multi-concept support
3. Logging and debugging utilities
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

# Add rf-detr to path
RFDETR_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'rf-detr')
if RFDETR_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(RFDETR_PATH))

from rfdetr.util import box_ops
from rfdetr.util.misc import (
    accuracy, get_world_size, is_dist_avail_and_initialized
)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Focal Loss for classification.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for RF-DETR.
    Matches predictions to ground truth using optimal bipartite matching.
    """
    
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """
        Performs the matching between predictions and ground truth.
        
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        from scipy.optimize import linear_sum_assignment
        
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten to compute cost matrix
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # Concat target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute classification cost
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        # Compute L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute GIoU cost
        cost_giou = -box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(out_bbox),
            box_ops.box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            # Handle group_detr
            if group_detr > 1:
                c_split = c[i].chunk(group_detr, dim=0)
                index_pairs = [linear_sum_assignment(c_part) for c_part in c_split]
                # Combine indices from all groups
                all_i = []
                all_j = []
                for g_idx, (idx_i, idx_j) in enumerate(index_pairs):
                    all_i.extend(idx_i + g_idx * (num_queries // group_detr))
                    all_j.extend(idx_j)
                indices.append((
                    torch.as_tensor(all_i, dtype=torch.int64),
                    torch.as_tensor(all_j, dtype=torch.int64)
                ))
            else:
                idx_i, idx_j = linear_sum_assignment(c[i])
                indices.append((
                    torch.as_tensor(idx_i, dtype=torch.int64),
                    torch.as_tensor(idx_j, dtype=torch.int64)
                ))
        
        return [(i.to(out_prob.device), j.to(out_prob.device)) for i, j in indices]


def build_matcher(args):
    """Build the Hungarian matcher."""
    return HungarianMatcher(
        cost_class=getattr(args, 'set_cost_class', 2.0),
        cost_bbox=getattr(args, 'set_cost_bbox', 5.0),
        cost_giou=getattr(args, 'set_cost_giou', 2.0),
        focal_alpha=getattr(args, 'focal_alpha', 0.25),
    )


class SetCriterion(nn.Module):
    """
    Loss computation for RF-DETR with MOTIP concept prediction support.
    
    Computes:
    1. Classification loss (focal loss)
    2. Box regression loss (L1 + GIoU)
    3. Cardinality error (for logging)
    4. Concept prediction losses (cross-entropy for each concept type)
    """
    
    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        concept_classes: Optional[List] = None,
        matcher: nn.Module = None,
        weight_dict: Dict[str, float] = None,
        focal_alpha: float = 0.25,
        losses: List[str] = None,
        group_detr: int = 1,
        ia_bce_loss: bool = True,
    ):
        """
        Initialize the criterion.
        
        Args:
            num_classes: Number of object classes
            num_concepts: Number of concept types (e.g., gender, clothing)
            concept_classes: List of (name, num_classes, unknown_label) for each concept
            matcher: Hungarian matcher
            weight_dict: Loss weights
            focal_alpha: Alpha for focal loss
            losses: List of loss types to compute
            group_detr: Number of groups for training
            ia_bce_loss: Use IoU-aware BCE loss
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.concept_classes = concept_classes or []
        self.matcher = matcher
        self.weight_dict = weight_dict or {}
        self.focal_alpha = focal_alpha
        self.losses = losses or ['labels', 'boxes', 'cardinality']
        self.group_detr = group_detr
        self.ia_bce_loss = ia_bce_loss
        
        # Logging counter
        self._log_counter = 0
        
        # Concept labels for logging
        self.concept_labels_map = {
            'gender': ['Male', 'Female', 'Unknown'],
            'upper_body': [
                'T-Shirt', 'Blouse', 'Sweater', 'Coat', 'Bikini', 'Naked',
                'Dress', 'Uniform', 'Shirt', 'Suit', 'Hoodie', 'Cardigan', 'Unknown'
            ],
        }

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss using focal loss or IoU-aware BCE.
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        if self.ia_bce_loss:
            # IoU-aware BCE loss
            alpha = self.focal_alpha
            gamma = 2
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            
            iou_targets = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes)
            )[0])
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob ** gamma
            
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            
            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()
            
            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
            
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            loss_ce = loss_ce.sum() / num_boxes
        else:
            # Standard focal loss
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes,
                dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o
            
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            
            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_boxes,
                alpha=self.focal_alpha, gamma=2
            ) * src_logits.shape[1]
        
        losses = {'loss_ce': loss_ce}
        
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute cardinality error (for logging only)."""
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute L1 and GIoU losses for bounding boxes."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses

    def loss_concepts(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        Compute concept prediction losses.
        
        Supports multiple concept types (gender, clothing, etc.)
        Handles unknown labels by excluding them from loss computation.
        """
        if 'pred_concepts' not in outputs or self.num_concepts == 0:
            return {}
        
        losses = {}
        total_loss = torch.tensor(0.0, device=outputs['pred_logits'].device)
        
        # Get predictions for each concept
        pred_concepts = outputs['pred_concepts']
        if not isinstance(pred_concepts, list):
            pred_concepts = [pred_concepts]
        
        idx = self._get_src_permutation_idx(indices)
        
        # Process each concept type
        for concept_idx, concept_logits in enumerate(pred_concepts):
            # Get concept name and unknown label
            if self.concept_classes and concept_idx < len(self.concept_classes):
                if isinstance(self.concept_classes[concept_idx], tuple):
                    concept_name, n_classes, unknown_label = self.concept_classes[concept_idx]
                else:
                    concept_name = f'concept_{concept_idx}'
                    n_classes = self.concept_classes[concept_idx]
                    unknown_label = n_classes - 1
            else:
                concept_name = f'concept_{concept_idx}'
                unknown_label = -1  # No unknown label
            
            # Get predictions for matched objects
            src_concept_logits = concept_logits[idx]
            
            # Get target concepts
            try:
                if 'concepts' in targets[0]:
                    # Concepts might be single column or multi-column
                    target_concepts_all = torch.cat([
                        t["concepts"][J] for t, (_, J) in zip(targets, indices)
                    ])
                    # Handle 2D tensor (multi-concept) - extract the specific concept column
                    if target_concepts_all.dim() == 2:
                        # Multi-concept case: select the column for this concept
                        if concept_idx < target_concepts_all.shape[1]:
                            target_concepts = target_concepts_all[:, concept_idx]
                        else:
                            continue  # This concept index doesn't exist
                    else:
                        # Single concept case (1D tensor)
                        if concept_idx == 0:
                            target_concepts = target_concepts_all
                        else:
                            continue  # Only one concept available
                elif f'{concept_name}_concepts' in targets[0]:
                    # Named concept
                    target_concepts = torch.cat([
                        t[f"{concept_name}_concepts"][J] for t, (_, J) in zip(targets, indices)
                    ])
                else:
                    continue  # Skip if no targets for this concept
            except (KeyError, IndexError):
                continue
            
            # Filter out unknown labels
            if unknown_label >= 0:
                valid_mask = target_concepts != unknown_label
                src_concept_valid = src_concept_logits[valid_mask]
                target_concepts_valid = target_concepts[valid_mask]
            else:
                src_concept_valid = src_concept_logits
                target_concepts_valid = target_concepts
            
            # Compute cross-entropy loss
            if len(src_concept_valid) > 0:
                concept_loss = F.cross_entropy(
                    src_concept_valid, 
                    target_concepts_valid.long(),
                    reduction='mean'
                )
                losses[f'loss_{concept_name}'] = concept_loss
                total_loss = total_loss + concept_loss
                
                # Compute accuracy for logging
                with torch.no_grad():
                    pred_labels = src_concept_valid.argmax(dim=-1)
                    correct = (pred_labels == target_concepts_valid).float().mean()
                    losses[f'{concept_name}_accuracy'] = correct * 100
                    
                    # Store for detailed logging
                    losses[f'{concept_name}_predictions'] = pred_labels
                    losses[f'{concept_name}_targets'] = target_concepts_valid
        
        # Combined concept loss
        losses['loss_concepts'] = total_loss / max(1, len(pred_concepts))
        
        return losses

    def _get_src_permutation_idx(self, indices):
        """Get source permutation indices for matched pairs."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Get target permutation indices for matched pairs."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Get the specified loss."""
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'concepts': self.loss_concepts,
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, batch_len=None, **kwargs):
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs dict
            targets: List of target dicts
            batch_len: Batch length for memory-efficient processing (MOTIP compatibility)
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Tuple of (losses dict, indices list) for MOTIP compatibility
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        
        # Match predictions to targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)
        
        # Compute number of boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes * group_detr, 1)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        # Auxiliary losses
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    loss_kwargs = {}
                    if loss == 'labels':
                        loss_kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes, **loss_kwargs)
                    # Filter out non-tensor values for aux outputs
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items() if isinstance(v, torch.Tensor)}
                    losses.update(l_dict)
        
        # Encoder outputs (two-stage)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            enc_indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                if loss == 'concepts':
                    continue  # Skip concepts for encoder
                loss_kwargs = {'log': False} if loss == 'labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, targets, enc_indices, num_boxes, **loss_kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items() if isinstance(v, torch.Tensor)}
                losses.update(l_dict)
        
        # Return (losses, indices) tuple for MOTIP compatibility
        return losses, indices

    def log_concept_predictions(self, outputs, targets, indices, num_samples=3):
        """Log concept predictions for debugging."""
        if 'pred_concepts' not in outputs or self.num_concepts == 0:
            return
        
        pred_concepts = outputs['pred_concepts']
        if not isinstance(pred_concepts, list):
            pred_concepts = [pred_concepts]
        
        idx = self._get_src_permutation_idx(indices)
        
        for concept_idx, concept_logits in enumerate(pred_concepts):
            if self.concept_classes and concept_idx < len(self.concept_classes):
                if isinstance(self.concept_classes[concept_idx], tuple):
                    concept_name, _, _ = self.concept_classes[concept_idx]
                else:
                    concept_name = f'concept_{concept_idx}'
            else:
                concept_name = f'concept_{concept_idx}'
            
            labels = self.concept_labels_map.get(concept_name, None)
            
            src_logits = concept_logits[idx]
            pred_labels = src_logits.argmax(dim=-1)
            
            try:
                if 'concepts' in targets[0]:
                    target_concepts = torch.cat([
                        t["concepts"][J] for t, (_, J) in zip(targets, indices)
                    ])
                else:
                    continue
            except:
                continue
            
            correct = (pred_labels == target_concepts).sum().item()
            total = len(pred_labels)
            
            print(f"\n[{concept_name.upper()}] Accuracy: {100*correct/max(1,total):.1f}% ({correct}/{total})")
            
            if labels and num_samples > 0:
                for i in range(min(num_samples, len(pred_labels))):
                    pred_label = labels[pred_labels[i].item()] if pred_labels[i] < len(labels) else f"ID{pred_labels[i]}"
                    gt_label = labels[target_concepts[i].item()] if target_concepts[i] < len(labels) else f"ID{target_concepts[i]}"
                    match = "✓" if pred_labels[i] == target_concepts[i] else "✗"
                    print(f"  [{match}] Pred: {pred_label}, GT: {gt_label}")
