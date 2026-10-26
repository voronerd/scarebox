"""Vendored YOLOX model definition (inference-only), Apache-2.0.

Source: https://github.com/Megvii-BaseDetection/YOLOX, tag ``0.1.1rc0``.
Copyright (c) 2014-2021 Megvii Inc. Licensed under the Apache License,
Version 2.0 -- full text in ``LICENSE`` next to this file.

Why vendored instead of ``pip install yolox``: the upstream package tries to build a C++ NMS
extension at install time that does not reliably build on aarch64 (02-01 KNOWN FACTS). The
model definition itself is pure PyTorch and needs no build step, so only ``models/`` and the
``postprocess``/``bboxes_iou`` helpers from ``utils/boxes.py`` are carried over here, with the
training-only code paths (loss computation, SimOTA label assignment) trimmed from
``yolo_head.py`` and ``yolox.py`` -- this fork only ever runs ``model.eval()`` inference with a
pretrained checkpoint, never trains. NMS itself uses ``torchvision.ops.batched_nms`` (confirmed
present in the ``IMAGE_TORCH`` base image, so no custom NMS was needed).
"""
