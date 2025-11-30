# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops

from utils.nested_tensor import nested_tensor_from_tensor_list
import logging
from typing import List

logger = logging.getLogger(__name__)

def read_file_to_list(file_path: str) -> List[str]:
    """
    Reads a file and returns a list of lines, stripping whitespace.
    Handles FileNotFoundError.
    """
    if not file_path:
        logger.error("read_file_to_list: received empty file_path.")
        return []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found at: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

def is_legal(annotation: dict):
    assert "id" in annotation, "Annotation must have 'id' field."
    assert "category" in annotation, "Annotation must have 'category' field."
    assert "bbox" in annotation, "Annotation must have 'bbox' field."
    assert "visibility" in annotation, "Annotation must have 'visibility' field."
    assert "concepts" in annotation, "Annotation must have 'concepts' field."
    
    assert len(annotation["id"]) == len(annotation["category"]) \
           == len(annotation["bbox"]) == len(annotation["visibility"]) \
           == len(annotation["concepts"]), \
           "The length of 'id', 'category', 'bbox', 'visibility', and 'concepts' must be the same."

    # assert torch.unique(annotation["id"]).size(0) == annotation["id"].size(0), f"IDs must be unique."
    _id_unique = torch.unique(annotation["id"]).size(0) == annotation["id"].size(0)     # for PersonPath22

    # A hack implementation for DETR (300 queries):
    # TODO: to make it more general, maybe pass the number of queries as an parameter.
    leq_300 = annotation["id"].shape[0] <= 300

    # return len(annotation["id"]) > 0
    return len(annotation["id"]) > 0 and _id_unique and leq_300


def append_annotation(
        annotation: dict,
        obj_id: int,
        category: int,
        bbox: list,
        visibility: float,
        concepts: int,
):
    annotation["id"] = torch.cat([
        annotation["id"],
        torch.tensor([obj_id], dtype=torch.int64)
    ])
    annotation["category"] = torch.cat([
        annotation["category"],
        torch.tensor([category], dtype=torch.int64)
    ])
    annotation["bbox"] = torch.cat([
        annotation["bbox"],
        torch.tensor([bbox], dtype=torch.float32)
    ])
    annotation["visibility"] = torch.cat([
        annotation["visibility"],
        torch.tensor([visibility], dtype=torch.float32)
    ])
    annotation["concepts"] = torch.cat([
        annotation["concepts"],
        torch.tensor([concepts], dtype=torch.int64) # Store as tensor
    ])
    return annotation


def collate_fn(batch):
    images, annotations, metas = zip(*batch)    # (B, T, ...)
    _B = len(batch)
    _T = len(images[0])
    images_list = [clip[_] for clip in images for _ in range(len(clip))]
    size_divisibility = metas[0][0]["size_divisibility"]
    nested_tensor = nested_tensor_from_tensor_list(images_list, size_divisibility=size_divisibility)
    # Reshape the nested tensor:
    nested_tensor.tensors = einops.rearrange(
        nested_tensor.tensors, "(b t) c h w -> b t c h w", b=_B, t=_T
    )
    nested_tensor.mask = einops.rearrange(
        nested_tensor.mask, "(b t) h w -> b t h w", b=_B, t=_T
    )
    # Above is prepared for DETR.
    # Below is prepared for MOTIP, pre-padding the annotations:
    max_N = max(annotation[0]["trajectory_id_labels"].shape[-1] for annotation in annotations)
    # Padding the ID annotations:
    for b in range(len(annotations)):
        for t in range(len(annotations[b])):
            _G, _, _N = annotations[b][t]["trajectory_id_labels"].shape
            if _N < max_N:
                annotations[b][t]["trajectory_id_labels"] = torch.cat([
                    annotations[b][t]["trajectory_id_labels"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["trajectory_id_masks"] = torch.cat([
                    annotations[b][t]["trajectory_id_masks"],
                    torch.ones((_G, 1, max_N - _N), dtype=torch.bool)
                ], dim=-1)
                annotations[b][t]["trajectory_ann_idxs"] = torch.cat([
                    annotations[b][t]["trajectory_ann_idxs"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["trajectory_times"] = torch.cat([
                    annotations[b][t]["trajectory_times"],
                    t * torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_id_labels"] = torch.cat([
                    annotations[b][t]["unknown_id_labels"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_id_masks"] = torch.cat([
                    annotations[b][t]["unknown_id_masks"],
                    torch.ones((_G, 1, max_N - _N), dtype=torch.bool)
                ], dim=-1)
                annotations[b][t]["unknown_ann_idxs"] = torch.cat([
                    annotations[b][t]["unknown_ann_idxs"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_times"] = torch.cat([
                    annotations[b][t]["unknown_times"],
                    t * torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
            pass
    return {
        "images": nested_tensor,
        "annotations": annotations,
        "metas": metas,
    }
