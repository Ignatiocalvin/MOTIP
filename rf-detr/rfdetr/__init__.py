# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# NOTE: RFDETRBase/Large/etc. imports (rfdetr.detr → rfdetr.main → peft)
# are omitted here because peft hangs on HPC nodes without internet access.
# MOTIP only uses rfdetr.models.* — this __init__ is intentionally lightweight.

