# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn

from models.ffn import FFN


class TrajectoryModeling(nn.Module):
    def __init__(
            self,
            detr_dim: int,
            ffn_dim_ratio: int,
            feature_dim: int,
            use_concept_bottleneck: bool = False,  # SAM concept bottleneck
    ):
        super().__init__()

        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim
        self.use_concept_bottleneck = use_concept_bottleneck

        # SAM concept bottleneck fusion
        if self.use_concept_bottleneck:
            # Input: cat(features[dim], concepts[dim]) → output: features[dim]
            self.concept_fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
            self.concept_fusion_norm = nn.LayerNorm(feature_dim)
        else:
            self.concept_fusion = None
            self.concept_fusion_norm = None

        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _fuse_feature_and_concept(self, features, concepts):
        """Fuse DETR features with SAM concept features."""
        if not self.use_concept_bottleneck:
            return features
        x = torch.cat([features, concepts], dim=-1)   # [..., 2*dim]
        x = self.concept_fusion(x)                    # [..., dim]
        x = self.concept_fusion_norm(x)
        return x

    def _model_one_stream(self, features, concepts):
        """Process a single stream (trajectory or unknown) with concept fusion."""
        features = self._fuse_feature_and_concept(features, concepts)
        features = features + self.adapter(features)
        features = self.norm(features)
        features = features + self.ffn(features)
        features = self.ffn_norm(features)
        return features

    def forward(self, seq_info):
        trajectory_features = seq_info["trajectory_features"]
        unknown_features = seq_info["unknown_features"]

        if self.use_concept_bottleneck:
            # SAM mode: fuse concepts with features
            trajectory_concepts = seq_info["trajectory_concepts"]
            unknown_concepts = seq_info["unknown_concepts"]
            trajectory_features = self._model_one_stream(trajectory_features, trajectory_concepts)
            unknown_features = self._model_one_stream(unknown_features, unknown_concepts)
        else:
            # Original path (no SAM concepts)
            trajectory_features = trajectory_features + self.adapter(trajectory_features)
            trajectory_features = self.norm(trajectory_features)
            trajectory_features = trajectory_features + self.ffn(trajectory_features)
            trajectory_features = self.ffn_norm(trajectory_features)

            # Also process unknown features for consistency
            unknown_features = unknown_features + self.adapter(unknown_features)
            unknown_features = self.norm(unknown_features)
            unknown_features = unknown_features + self.ffn(unknown_features)
            unknown_features = self.ffn_norm(unknown_features)

        seq_info["trajectory_features"] = trajectory_features
        seq_info["unknown_features"] = unknown_features
        return seq_info
