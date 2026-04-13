# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import numpy as np
import torch
import torch.nn.functional as F
import einops
from PIL import Image
from scipy.optimize import linear_sum_assignment

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh
from models.misc import get_model


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
            # Concept bottleneck mode: "hard" or "soft"
            concept_bottleneck_mode: str = "hard",
            # SAM concept bottleneck parameters
            use_concept_bottleneck: bool = False,
            object_mask_root: str = None,
            sequence_name: str = None,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype
        self.sequence_hw = sequence_hw
        
        # SAM concept bottleneck settings
        self.use_concept_bottleneck = use_concept_bottleneck
        self.object_mask_root = object_mask_root
        self.sequence_name = sequence_name
        
        # Concept bottleneck mode (for DETR concept heads)
        self.concept_bottleneck_mode = concept_bottleneck_mode.lower()
        if self.concept_bottleneck_mode not in ["hard", "soft"]:
            raise ValueError(f"concept_bottleneck_mode must be 'hard' or 'soft', got '{concept_bottleneck_mode}'")

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.num_id_vocabulary = get_model(model).num_id_vocabulary
        
        # Get feature_dim from trajectory_modeling (handles both deformable-detr and rf-detr)
        actual_model = get_model(model)
        if hasattr(actual_model, 'trajectory_modeling') and actual_model.trajectory_modeling is not None:
            self.feature_dim = actual_model.trajectory_modeling.feature_dim
        else:
            self.feature_dim = 256  # Default fallback

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, self.feature_dim), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )
        # Trajectory concepts: 
        # - "hard" mode: (T, N, num_concepts) integer labels
        # - "soft" mode: (T, N, total_concept_classes) float probabilities
        # - SAM concept bottleneck: (T, N, feature_dim) float features
        if self.use_concept_bottleneck:
            # SAM concept bottleneck uses feature_dim (256) as concept dimension
            self.trajectory_concepts = torch.zeros(
                (0, 0, self.feature_dim), dtype=torch.float32, device=distributed_device(),
            )
        else:
            self.trajectory_concepts = torch.zeros(
                (0, 0, 0), dtype=torch.float32 if self.concept_bottleneck_mode == "soft" else torch.int64, 
                device=distributed_device(),
            )
        # self.trajectory_features = torch.zeros(())

        self.current_track_results = {}
        return

    # ---------- SAM Concept Bottleneck Helper Methods ----------
    
    def _get_mask_frame_dir(self, image_path: str):
        """Get the directory containing SAM masks for a given frame."""
        if self.object_mask_root is None:
            return None
        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.object_mask_root, self.sequence_name, frame_name)

    def _load_frame_masks(self, image_path: str):
        """Load all SAM masks for a given frame."""
        frame_dir = self._get_mask_frame_dir(image_path)
        if frame_dir is None or not os.path.isdir(frame_dir):
            return torch.zeros(
                (0, self.sequence_hw[0], self.sequence_hw[1]),
                dtype=torch.bool, device=distributed_device(),
            )

        mask_files = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir) if f.endswith(".png")
        ])

        masks = []
        for p in mask_files:
            m = Image.open(p).convert("L")
            m = np.array(m)
            m = torch.from_numpy(m > 0)
            masks.append(m)

        if len(masks) == 0:
            return torch.zeros(
                (0, self.sequence_hw[0], self.sequence_hw[1]),
                dtype=torch.bool, device=distributed_device(),
            )
        return torch.stack(masks, dim=0).to(distributed_device())  # [M, H, W]

    def _box_xywh_to_xyxy(self, boxes_xywh: torch.Tensor):
        """Convert boxes from xywh to xyxy format."""
        x, y, w, h = boxes_xywh.unbind(-1)
        return torch.stack([x, y, x + w, y + h], dim=-1)

    def _compute_mask_boxes_xyxy(self, masks: torch.Tensor):
        """Compute bounding boxes from masks. masks: [M, H, W], returns: [M, 4] in xyxy."""
        if masks.shape[0] == 0:
            return torch.zeros((0, 4), dtype=self.dtype, device=distributed_device())

        boxes = []
        for m in masks:
            ys, xs = torch.where(m)
            if len(xs) == 0:
                boxes.append(torch.tensor([0, 0, 0, 0], dtype=self.dtype, device=distributed_device()))
            else:
                x1 = xs.min().to(self.dtype)
                y1 = ys.min().to(self.dtype)
                x2 = xs.max().to(self.dtype)
                y2 = ys.max().to(self.dtype)
                boxes.append(torch.stack([x1, y1, x2, y2]))
        return torch.stack(boxes, dim=0)

    def _pairwise_iou_xyxy(self, boxes1: torch.Tensor, boxes2: torch.Tensor):
        """Compute pairwise IoU between two sets of boxes. boxes1: [N, 4], boxes2: [M, 4]."""
        if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=self.dtype, device=distributed_device())

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-6)

    def _masked_average_pool(self, feature_map: torch.Tensor, masks: torch.Tensor):
        """Pool features from feature map using masks. feature_map: [C, Hf, Wf], masks: [N, H, W], returns: [N, C]."""
        C, Hf, Wf = feature_map.shape
        N = masks.shape[0]

        if N == 0:
            return torch.zeros((0, C), dtype=feature_map.dtype, device=feature_map.device)

        masks_resized = F.interpolate(
            masks[:, None].to(feature_map.dtype), size=(Hf, Wf), mode="nearest",
        )[:, 0]

        feat_flat = feature_map.view(C, -1)
        masks_flat = masks_resized.view(N, -1)
        weighted_sum = masks_flat @ feat_flat.t()
        denom = masks_flat.sum(dim=1, keepdim=True)

        valid = denom.squeeze(1) > 0
        concept_features = torch.zeros((N, C), dtype=feature_map.dtype, device=feature_map.device)
        concept_features[valid] = weighted_sum[valid] / denom[valid]
        return concept_features

    def _get_current_detection_concepts_sam(self, image_path: str, boxes: torch.Tensor, detr_out: dict):
        """Get SAM-based concept features for detected boxes. boxes: [N, 4] in cxcywh normalized, returns: [N, C]."""
        if (not self.use_concept_bottleneck) or (self.object_mask_root is None) or (image_path is None):
            return torch.zeros((boxes.shape[0], self.feature_dim), dtype=self.dtype, device=distributed_device())

        frame_masks = self._load_frame_masks(image_path)  # [M, H, W]
        if frame_masks.shape[0] == 0:
            return torch.zeros((boxes.shape[0], self.feature_dim), dtype=self.dtype, device=distributed_device())

        feature_map = detr_out["feature_map"][0]  # [C, Hf, Wf]
        mask_concepts = self._masked_average_pool(feature_map, frame_masks)  # [M, C]

        # Convert detection boxes to xyxy for IoU matching
        boxes_xywh = box_cxcywh_to_xywh(boxes) * self.bbox_unnorm
        det_boxes_xyxy = self._box_xywh_to_xyxy(boxes_xywh)
        mask_boxes_xyxy = self._compute_mask_boxes_xyxy(frame_masks)
        ious = self._pairwise_iou_xyxy(det_boxes_xyxy, mask_boxes_xyxy)  # [N, M]

        if ious.shape[1] == 0:
            return torch.zeros((boxes.shape[0], self.feature_dim), dtype=self.dtype, device=distributed_device())

        best_mask_idx = ious.argmax(dim=1)
        return mask_concepts[best_mask_idx]

    # ---------- End SAM Helper Methods ----------

    @torch.no_grad()
    def update(self, image, image_path=None):
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds, concepts = self._get_activate_detections(detr_out=detr_out)
        
        # SAM Concept Bottleneck: compute concepts from feature map + masks
        if self.use_concept_bottleneck:
            sam_concepts = self._get_current_detection_concepts_sam(
                image_path=image_path, boxes=boxes, detr_out=detr_out
            )
        else:
            sam_concepts = None
        
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # Pass concepts for ID prediction (SAM concepts or DETR concepts)
            if self.use_concept_bottleneck:
                id_pred_labels = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds, concepts=None, sam_concepts=sam_concepts)
            else:
                id_pred_labels = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds, concepts=concepts)
        
        # Filter out illegal newborn detections:
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]
        # Handle concepts as a list of tensors (one per concept type) or None
        if concepts is not None:
            if isinstance(concepts, list):
                concepts = [c[keep_idxs] for c in concepts]
            else:
                concepts = concepts[keep_idxs]
        # Filter SAM concepts
        if sam_concepts is not None:
            sam_concepts = sam_concepts[keep_idxs]

        # A hack implementation, before assign new id labels, update the id_queue to ensure the uniqueness of id labels:
        n_activate_id_labels = 0
        n_newborn_targets = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn_targets += 1

        # Make sure the length of newborn instances is less than the length of remaining IDs:
        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_keep_idxs = torch.ones(len(newborn_idxs), dtype=torch.bool, device=newborn_idxs.device)
            newborn_keep_idxs[n_remaining_ids:] = False
            keep_idxs[newborn_idxs] = newborn_keep_idxs
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]
            # Also filter concepts
            if isinstance(concepts, list):
                concepts = [c[keep_idxs] for c in concepts]
            elif concepts is not None:
                concepts = concepts[keep_idxs]
            # Also filter SAM concepts
            if sam_concepts is not None:
                sam_concepts = sam_concepts[keep_idxs]
        pass

        # Assign new id labels:
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels)

        if len(torch.unique(id_labels)) != len(id_labels):
            print(id_labels, id_labels.shape)
            exit(-1)

        # Convert concepts to a format suitable for per-object iteration
        # SAM concept bottleneck: use SAM concepts directly (float features)
        if self.use_concept_bottleneck and sam_concepts is not None:
            concepts_for_results = sam_concepts  # Already [N, C] float tensor
        # DETR concept heads: convert list to tensor based on mode
        elif isinstance(concepts, list) and len(concepts) > 0:
            # Check if we have any objects
            if concepts[0].shape[0] > 0:
                if self.concept_bottleneck_mode == "soft":
                    # Soft mode: concatenate softmax probability distributions
                    concept_probs = [torch.softmax(c, dim=-1) for c in concepts]
                    concepts_for_results = torch.cat(concept_probs, dim=-1)  # (N_objects, total_concept_classes)
                else:
                    # Hard mode: get predicted class for each concept (argmax)
                    concept_preds = torch.stack([c.argmax(dim=-1) for c in concepts], dim=-1)  # (N_objects, N_concepts)
                    concepts_for_results = concept_preds
            else:
                # No objects, create empty tensor with correct shape
                if self.concept_bottleneck_mode == "soft":
                    # Soft mode: total_concept_classes dimension
                    total_classes = sum(c.shape[-1] for c in concepts)
                    concepts_for_results = torch.empty((0, total_classes), dtype=torch.float32, device=concepts[0].device)
                else:
                    # Hard mode: N_concepts dimension
                    n_concepts = len(concepts)
                    concepts_for_results = torch.empty((0, n_concepts), dtype=torch.int64, device=concepts[0].device)
        elif concepts is not None:
            concepts_for_results = concepts
        else:
            concepts_for_results = torch.empty((0,), dtype=torch.int64)

        # Update the results:
        self.current_track_results = {
            "score": scores,
            "category": categories,
            # "bbox": boxes * self.bbox_unnorm,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": torch.tensor(
                [self.id_label_to_id[_] for _ in id_labels.tolist()], dtype=torch.int64,
            ),
            "concepts": concepts_for_results
        }

        # Update id_queue:
        for _ in range(len(id_labels)):
            self.id_queue.add(id_labels[_].item())

        # Update trajectory infos (including concepts):
        self._update_trajectory_infos(boxes=boxes, output_embeds=output_embeds, id_labels=id_labels, concepts=concepts_for_results)

        # Filter out inactive tracks:
        self._filter_out_inactive_tracks()
        pass
        return

    def get_track_results(self):
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        # Handle case when model has no concepts (N_CONCEPTS = 0)
        pred_concepts = detr_out.get("pred_concepts", None)
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        # Selecting:
        # logits = logits[activate_indices]
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        # Handle pred_concepts - can be None when N_CONCEPTS = 0
        if pred_concepts is None:
            concepts = None
        elif isinstance(pred_concepts, list):
            # Each tensor in list is (B, N_queries, n_classes) or (N_queries, n_classes)
            concepts = [c[0][activate_indices] if c.dim() == 3 else c[activate_indices] for c in pred_concepts]
        else:
            # Single tensor case (legacy)
            concepts = pred_concepts[0][activate_indices] if pred_concepts.dim() == 3 else pred_concepts[activate_indices]
        return scores, categories, boxes, output_embeds, concepts

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor, concepts=None, sam_concepts=None):
        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            
            # Prepare current concepts based on mode:
            # - SAM concept bottleneck: use sam_concepts directly
            # - DETR concept heads: process concepts list
            if self.use_concept_bottleneck and sam_concepts is not None:
                # SAM concept bottleneck: sam_concepts is [N, C] float tensor
                current_concept_data = sam_concepts[None, ...]  # (1, N, C)
            elif concepts is not None and isinstance(concepts, list) and len(concepts) > 0:
                if self.concept_bottleneck_mode == "soft":
                    # Soft mode: concatenate softmax probability distributions
                    concept_probs = [torch.softmax(c, dim=-1) for c in concepts]  # list of (N, n_classes_i)
                    current_concept_data = torch.cat(concept_probs, dim=-1)  # (N, total_concept_classes)
                    current_concept_data = current_concept_data[None, ...]  # (1, N, total_concept_classes)
                else:
                    # Hard mode: get argmax labels for each concept
                    current_concept_data = torch.stack([c.argmax(dim=-1) for c in concepts], dim=-1)  # (N, num_concepts)
                    current_concept_data = current_concept_data[None, ...]  # (1, N, num_concepts)
            else:
                current_concept_data = None
            
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            
            # Add concept data to seq_info if available
            if self.trajectory_concepts.shape[-1] > 0:
                seq_info["trajectory_concepts"] = self.trajectory_concepts[None, None, ...]
            if current_concept_data is not None:
                seq_info["unknown_concepts"] = current_concept_data[None, None, ...]
            
            # 3. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 5. assign id labels:
            # Different assignment protocols:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                # case "object-priority": id_labels = self._object_priority_assignment(id_scores=id_scores)
                case _: raise NotImplementedError

            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            return id_pred_labels

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            newborn_id_labels = torch.tensor(
                list(self.id_queue)[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
            )
            for _ in range(len(newborn_id_labels)):
                if self.trajectory_id_labels.shape[0] > 0:
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                if newborn_id_labels[_].item() in self.id_label_to_id:
                    self.id_label_to_id.pop(newborn_id_labels[_].item())
            # remove from trajectory infos:
            self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
            self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
            self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
            self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
            self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
            if self.trajectory_concepts.shape[-1] > 0:
                self.trajectory_concepts = self.trajectory_concepts[:, ~trajectory_remove_idxs]
            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.next_id += 1

            return pred_id_labels

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor, concepts=None):
        """
        Update trajectory information with new detections.
        
        Args:
            boxes: Detection boxes (N, 4)
            output_embeds: Detection embeddings (N, feature_dim)
            id_labels: Assigned ID labels (N,)
            concepts: Concept predictions - either (N, num_concepts) int labels for hard mode
                      or (N, total_concept_classes) float probs for soft mode
        """
        # Determine concept dimension and dtype based on mode
        concept_dim = concepts.shape[-1] if concepts is not None and concepts.dim() > 1 else 0
        # SAM concept bottleneck uses float tensors, same as soft mode
        concept_dtype = torch.float32 if (self.concept_bottleneck_mode == "soft" or self.use_concept_bottleneck) else torch.int64
        
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        if self.trajectory_concepts.shape[-1] > 0:
            self.trajectory_concepts = self.trajectory_concepts[-self.miss_tolerance + 2:, ...]
        
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels = _id_labels - already_id_labels
        
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels) > 0:
            newborn_id_labels = torch.tensor(list(newborn_id_labels), dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0]
            _N = len(newborn_id_labels)
            _id_labels = einops.repeat(newborn_id_labels, 'n -> t n', t=_T)
            _boxes = torch.zeros((_T, _N, 4), dtype=self.dtype, device=distributed_device())
            _times = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N,
            )
            _features = torch.zeros(
                (_T, _N, self.feature_dim), dtype=self.dtype, device=distributed_device(),
            )
            _masks = torch.ones((_T, _N), dtype=torch.bool, device=distributed_device())
            # 3.1. padding to trajectory infos:
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _id_labels], dim=1)
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, _boxes], dim=1)
            self.trajectory_times = torch.cat([self.trajectory_times, _times], dim=1)
            self.trajectory_features = torch.cat([self.trajectory_features, _features], dim=1)
            self.trajectory_masks = torch.cat([self.trajectory_masks, _masks], dim=1)
            
            # Also pad concepts if tracking them
            if concept_dim > 0:
                if self.trajectory_concepts.shape[-1] == 0:
                    # Initialize trajectory_concepts with correct shape
                    self.trajectory_concepts = torch.zeros(
                        (_T, self.trajectory_id_labels.shape[1] - _N, concept_dim),
                        dtype=concept_dtype, device=distributed_device()
                    )
                _concepts_pad = torch.zeros((_T, _N, concept_dim), dtype=concept_dtype, device=distributed_device())
                self.trajectory_concepts = torch.cat([self.trajectory_concepts, _concepts_pad], dim=1)
        
        # 4. update trajectory infos:
        _N = self.trajectory_id_labels.shape[1]
        current_id_labels = self.trajectory_id_labels[0] if self.trajectory_id_labels.shape[0] > 0 else id_labels
        current_features = torch.zeros((_N, self.feature_dim), dtype=self.dtype, device=distributed_device())
        current_boxes = torch.zeros((_N, 4), dtype=self.dtype, device=distributed_device())
        current_times = self.trajectory_id_labels.shape[0] * torch.ones((_N,), dtype=torch.int64, device=distributed_device())
        current_masks = torch.ones((_N,), dtype=torch.bool, device=distributed_device())
        if concept_dim > 0:
            current_concepts = torch.zeros((_N, concept_dim), dtype=concept_dtype, device=distributed_device())
        
        # 4.1. find out the same id labels (matching):
        indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(as_tuple=False)
        current_idxs = indices[:, 0]
        idxs = indices[:, 1]
        # 4.2. fill in the infos:
        current_id_labels[current_idxs] = id_labels[idxs]
        current_features[current_idxs] = output_embeds[idxs]
        current_boxes[current_idxs] = boxes[idxs]
        current_masks[current_idxs] = False
        if concept_dim > 0 and concepts is not None:
            current_concepts[current_idxs] = concepts[idxs]
        
        # 4.3. cat to trajectory infos:
        self.trajectory_features = torch.cat([self.trajectory_features, current_features[None, ...]], dim=0).contiguous()
        self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes[None, ...]], dim=0).contiguous()
        self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels[None, ...]], dim=0).contiguous()
        self.trajectory_times = torch.cat([self.trajectory_times, current_times[None, ...]], dim=0).contiguous()
        self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks[None, ...]], dim=0).contiguous()
        if concept_dim > 0:
            if self.trajectory_concepts.shape[-1] == 0:
                # First time adding concepts - initialize with proper history
                _T = self.trajectory_id_labels.shape[0]
                self.trajectory_concepts = torch.zeros((_T - 1, _N, concept_dim), dtype=concept_dtype, device=distributed_device())
                self.trajectory_concepts = torch.cat([self.trajectory_concepts, current_concepts[None, ...]], dim=0).contiguous()
            elif self.trajectory_concepts.shape[1] != _N:
                # Dimension mismatch - need to pad trajectory_concepts to match current trajectory count
                _T_concepts = self.trajectory_concepts.shape[0]
                _N_concepts = self.trajectory_concepts.shape[1]
                if _N > _N_concepts:
                    # Pad with zeros for new trajectories
                    _pad = torch.zeros((_T_concepts, _N - _N_concepts, concept_dim), dtype=concept_dtype, device=distributed_device())
                    self.trajectory_concepts = torch.cat([self.trajectory_concepts, _pad], dim=1).contiguous()
                else:
                    # This shouldn't happen (trajectories removed), but handle it
                    self.trajectory_concepts = self.trajectory_concepts[:, :_N, :].contiguous()
                self.trajectory_concepts = torch.cat([self.trajectory_concepts, current_concepts[None, ...]], dim=0).contiguous()
            else:
                self.trajectory_concepts = torch.cat([self.trajectory_concepts, current_concepts[None, ...]], dim=0).contiguous()
        
        # 4.4. a hack implementation to fix "times":
        self.trajectory_times = einops.repeat(
            torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
            't -> t n', n=self.trajectory_times.shape[1],
        ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        if self.trajectory_concepts.shape[-1] > 0:
            self.trajectory_concepts = self.trajectory_concepts[:, is_active]
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        if len(id_scores) > 1:
            id_scores_newborn_repeat = id_scores[:, -1:].repeat(1, len(id_scores) - 1)
            id_scores = torch.cat((id_scores, id_scores_newborn_repeat), dim=-1)
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())
        match_rows, match_cols = linear_sum_assignment(1 - id_scores.cpu())
        for _ in range(len(match_rows)):
            _id = match_cols[_]
            if _id not in trajectory_id_labels_set:
                id_labels.append(self.num_id_vocabulary)
            elif _id >= self.num_id_vocabulary:
                id_labels.append(self.num_id_vocabulary)
            elif id_scores[match_rows[_], _id] < self.id_thresh:
                id_labels.append(self.num_id_vocabulary)
            else:
                id_labels.append(_id)
        return id_labels

    def _object_max_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        object_max_confs, object_max_id_labels = torch.max(id_scores, dim=-1)   # get the target ID labels and confs
        # Get the max confs of each ID label:
        id_max_confs = dict()
        for conf, id_label in zip(object_max_confs.tolist(), object_max_id_labels.tolist()):
            if id_label not in id_max_confs:
                id_max_confs[id_label] = conf
            else:
                # if conf == id_max_confs[id_label]:  # a very rare case
                #     conf = conf - 0.0001
                id_max_confs[id_label] = max(id_max_confs[id_label], conf)
        if self.num_id_vocabulary in id_max_confs:
            id_max_confs[self.num_id_vocabulary] = 0.0  # special token

        # Assign ID labels:
        for _ in range(len(object_max_id_labels)):
            if object_max_id_labels[_].item() not in trajectory_id_labels_set:         # not in tracked IDs -> newborn
                id_labels.append(self.num_id_vocabulary)
            else:
                _id_label = object_max_id_labels[_].item()
                _conf = object_max_confs[_].item()
                if _conf < self.id_thresh or _conf < id_max_confs[_id_label]:  # low conf or not the max conf -> newborn
                    id_labels.append(self.num_id_vocabulary)
                elif _id_label in id_labels:
                    id_labels.append(self.num_id_vocabulary)
                else:                                                          # normal case
                    id_labels.append(_id_label)

        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = [self.num_id_vocabulary] * len(id_scores)  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        id_max_confs, id_max_obj_idxs = torch.max(id_scores, dim=0)
        # Get the max confs of each object:
        object_max_confs = dict()
        for conf, object_idx in zip(id_max_confs.tolist(), id_max_obj_idxs.tolist()):
            if object_idx not in object_max_confs:
                object_max_confs[object_idx] = conf
            else:
                if conf == object_max_confs[object_idx]:    # a very rare case
                    conf = conf - 0.0001
                object_max_confs[object_idx] = max(object_max_confs[object_idx], conf)

        # Assign ID labels:
        for _ in range(len(id_max_obj_idxs)):
            _obj_idx, _id_label, _conf = id_max_obj_idxs[_].item(), _, id_max_confs[_].item()
            if _conf < self.id_thresh or _conf < object_max_confs[_obj_idx]:
                pass
            elif _id_label not in trajectory_id_labels_set:
                pass
            else:
                id_labels[_obj_idx] = _id_label

        return id_labels
