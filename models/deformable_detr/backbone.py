# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.

Key Features:

    Multi-scale Feature Extraction: Can extract features at different scales (8x, 16x, 32x downsampling)
    Flexible Training: Can freeze/unfreeze different parts of the backbone
    Position Encoding Integration: Seamlessly combines visual features with positional information
    Mask Handling: Properly handles attention masks for variable-sized inputs
    Memory Efficient: Uses frozen BatchNorm to reduce memory usage during training

"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import ResNet50_Weights
from typing import Dict, List

from utils.nested_tensor import NestedTensor
from utils.misc import is_main_process

from models.deformable_detr.position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Batch statistics and affine parameters are frozen (not updated during training.

    Used instead of regular BatchNorm when you want to keep the normalization parameters fixed
    Includes an epsilon value before the square root operation to prevent numerical instability

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Removing the num_batches_tracked parameter from the state dictionary before 
        loading it into the FrozenBatchNorm2d module.

        Regular PyTorch BatchNorm2d layers track a parameter called num_batches_tracked which counts how many batches have been processed during training
        FrozenBatchNorm2d doesn't need this parameter because it's frozen - the statistics don't update
        """
        num_batches_tracked_key = prefix + 'num_batches_tracked' # The code constructs the key name: prefix + 'num_batches_tracked' (where prefix identifies the specific layer)
        if num_batches_tracked_key in state_dict: # Check if the key exists in the state_dict. If it does, it means that the state_dict contains a parameter that is not needed for FrozenBatchNorm2d
            del state_dict[num_batches_tracked_key] # Deletes the num_batches_tracked parameter from the state_dict

        # Call the parent class's _load_from_state_dict method to handle the rest of the loading process
        # This ensures that the rest of the parameters (weight, bias, running_mean, running_var) are loaded correctly
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """
     Selectively freezes parameters - only layers 2, 3, and 4 are trainable if train_backbone is True

    Two modes:
        Intermediate layers: Returns features from layers 2, 3, 4 with strides [8, 16, 32] and channels [512, 1024, 2048]
        Single layer: Returns only layer 4 features with stride [32] and channels [2048]

    Uses IntermediateLayerGetter to extract features from specific layers
    """

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        """
        Args:
            backbone (nn.Module): The backbone model to extract features from.
            train_backbone (bool): Whether to train the backbone or freeze its parameters.
            return_interm_layers (bool): Whether to return intermediate layers or only the final layer.
        
        Two Training Scenarios:
            Scenario 1: train_backbone = False
                All backbone parameters are frozen (no gradients computed)
                The entire backbone acts as a fixed feature extractor
                Only the transformer parts of the model will be trained
        
            Scenario 2: train_backbone = True
                Only layers 2, 3, and 4 are trainable
                Layer 1 and earlier layers (conv1, bn1, etc.) remain frozen
                This is a common fine-tuning strategy

        Why This Design?
            1. Computational Efficiency: Earlier layers learn low-level features (edges, textures) that transfer well across tasks, so freezing them saves computation

            2. Stability: Early layers contain fundamental visual representations that shouldn't change much during fine-tuning

            3. Memory Savings: Frozen parameters don't need gradient storage, reducing memory usage

            4. Transfer Learning: This approach leverages pre-trained ImageNet features while allowing task-specific adaptation in deeper layers

        ResNet Layer Structure:
            Layer 1: Basic feature extraction (frozen)
            Layer 2: Mid-level features (trainable if train_backbone=True)
            Layer 3: Higher-level features (trainable if train_backbone=True)
            Layer 4: Most abstract features (trainable if train_backbone=True)
        """
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False) # Freezes parameters
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        if name == "resnet50":
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=ResNet50_Weights.IMAGENET1K_V1 if is_main_process() else None, norm_layer=norm_layer)
        else:
            raise NotImplementedError(f"Do not support backbone name {name}.")
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    """
    Joins the backbone and position encoding modules.

    Takes a backbone and a position encoding module, applies the backbone to the input,
    and then applies the position encoding to the output features.

    Args:
        backbone (BackboneBase): The backbone module to extract features.
        position_embedding (nn.Module): The position encoding module to apply to the features.

    Returns:
        out (List[NestedTensor]): List of feature maps from the backbone.
        pos (List[Tensor]): List of position encodings corresponding to the feature maps.


    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    Builds the backbone model with position encoding.
    Args:
        args (argparse.Namespace): Arguments containing backbone configuration.

            lr_backbone: Whether to train backbone (if > 0)
            masks or num_feature_levels: Whether to return intermediate layers
            backbone: Backbone architecture name
            dilation: Whether to use dilation

    Returns:
        model (Joiner): A Joiner model that combines the backbone and position encoding.
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_intern_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_intern_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
