#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Vendored and trimmed from https://github.com/Megvii-BaseDetection/YOLOX (tag 0.1.1rc0),
# Apache-2.0 -- see ../LICENSE. The training branch (``targets``/loss-dict path) is removed --
# see the note in yolo_head.py; this fork never trains.

import torch.nn as nn

from scarebox.vendor.yolox.models.yolo_head import YOLOXHead
from scarebox.vendor.yolox.models.yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module (inference-only fork). The module list is defined by
    create_yolov3_modules function. The network returns detection results.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        return self.head(fpn_outs)
