# Copyright (c) Ruopeng Gao. All Rights Reserved.
# RF-DETR integration for MOTIP

"""
RF-DETR integration module for MOTIP tracking.

This module provides a wrapper around RF-DETR (Roboflow's Detection Transformer)
that makes it compatible with MOTIP's tracking framework, including:
- Concept prediction heads (gender, clothing attributes, etc.)
- Output format matching MOTIP's DeformableDETR interface
- Loss computation with concept losses
"""

from .rfdetr_motip import RFDETR_MOTIP, build as build_rfdetr
from .criterion import SetCriterion

__all__ = ['RFDETR_MOTIP', 'build_rfdetr', 'SetCriterion']
