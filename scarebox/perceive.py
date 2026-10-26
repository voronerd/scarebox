"""perceive: detect, track, trigger, group, best-crop. Spec §4.3.

`Detector` wraps the vendored YOLOX-s (Apache-2.0, Megvii -- `scarebox.vendor.yolox`),
person-class only. `Perceiver` keeps a ByteTrack tracker (supervision's built-in
`sv.ByteTrack`, Apache-2.0 -- a deliberate deviation from the design: the Roboflow
`trackers` package was not installed, since `supervision` is already a required dependency
and ships the same ByteTrack algorithm) and runs the trigger-line / grouping / best-crop
state machine per frame.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

from scarebox.config import Config
from scarebox.events import Event
from scarebox.vendor.yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from scarebox.vendor.yolox.utils import postprocess

# YOLOX-s exp defaults (Megvii's yolox_s.py): depth/width multipliers for the CSPDarknet
# backbone + PAFPN neck + decoupled head. COCO has 80 classes; class 0 is "person".
_YOLOX_S_DEPTH = 0.33
_YOLOX_S_WIDTH = 0.50
_COCO_NUM_CLASSES = 80
_COCO_PERSON_CLASS = 0

# A track that already fired is ignored for this long before it is allowed to fire again
# (spec §4.3: "its ID is remembered for 60s"). In practice a track ID is rarely reused this
# fast, so this is a safety net against a tracker ID wrapping back onto the same person.
_FIRE_MEMORY_S = 60.0


def build_yolox_s() -> YOLOX:
    """The vendored YOLOX model graph, sized for the `yolox_s` checkpoint."""
    backbone = YOLOPAFPN(depth=_YOLOX_S_DEPTH, width=_YOLOX_S_WIDTH)
    head = YOLOXHead(_COCO_NUM_CLASSES, width=_YOLOX_S_WIDTH)
    return YOLOX(backbone=backbone, head=head)


class Detector:
    """YOLOX-s person detector. FP16 on CUDA, 640px letterboxed input (spec §4.3)."""

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        input_size: int = 640,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> None:
        self.device = device
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.half = device.startswith("cuda")

        model = build_yolox_s()
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        if self.half:
            model = model.half()
        self.model = model.to(device)

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, float]:
        """YOLOX's own preprocessing (yolox/data/data_augment.py `preproc`): letterbox to a
        square `input_size`, pad with grey (114), BGR, NO mean/std normalisation -- the
        released YOLOX weights were trained on raw 0-255 BGR."""
        h, w = image.shape[:2]
        r = min(self.input_size / h, self.input_size / w)
        resized = cv2.resize(
            image, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR
        )
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        chw = padded.transpose(2, 0, 1).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        if self.half:
            tensor = tensor.half()
        return tensor.to(self.device), r

    def detect(self, image: np.ndarray) -> sv.Detections:
        tensor, ratio = self._preprocess(image)
        with torch.no_grad():
            raw = self.model(tensor)
            outputs = postprocess(
                raw, _COCO_NUM_CLASSES, self.conf_threshold, self.nms_threshold
            )
        pred = outputs[0]
        if pred is None:
            return sv.Detections.empty()
        pred = pred.detach().to("cpu", dtype=torch.float32).numpy()
        boxes = pred[:, :4] / ratio
        obj_conf = pred[:, 4]
        class_conf = pred[:, 5]
        class_id = pred[:, 6].astype(int)
        scores = obj_conf * class_conf
        person_mask = class_id == _COCO_PERSON_CLASS
        if not person_mask.any():
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=boxes[person_mask].astype(np.float32),
            confidence=scores[person_mask].astype(np.float32),
            class_id=class_id[person_mask],
        )


def _line_side(trigger_line, x: float, y: float) -> float:
    """Sign of the 2-D cross product between the trigger-line segment's direction vector
    and the vector from the line's first point to (x, y). A crossing is a sign change
    between two frames' bottom-centre points (spec §4.3)."""
    (x0, y0), (x1, y1) = trigger_line
    dx, dy = x1 - x0, y1 - y0
    cross = dx * (y - y0) - dy * (x - x0)
    if cross == 0:
        return 0.0
    return math.copysign(1.0, cross)


def _line_distance(trigger_line, x: float, y: float) -> float:
    """Perpendicular distance from (x, y) to the trigger line (infinite line through its
    two endpoints), in the same normalised coordinate units as `trigger_line` itself. Used
    for `perceive.group_band` (hub amendment 2026-10-15): how close another track's
    bottom-centre must be to the line, at the trigger frame, to be swept into the group."""
    (x0, y0), (x1, y1) = trigger_line
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0.0:
        return math.hypot(x - x0, y - y0)
    cross = dx * (y - y0) - dy * (x - x0)
    return abs(cross) / length


def crop_score(crop: np.ndarray) -> float:
    """area x sharpness (variance of the Laplacian) -- spec §4.3's best-crop metric."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area = float(crop.shape[0] * crop.shape[1])
    return area * sharpness


def crop_and_resize(
    image: np.ndarray, bbox_xyxy, pad_frac: float = 0.10, max_px: int = 512
) -> np.ndarray:
    """Pads `bbox_xyxy` by `pad_frac`, clips to the frame, and resizes so the long side is
    at most `max_px` (spec §4.3 best-crop)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - bw * pad_frac, x2 + bw * pad_frac
    y1, y2 = y1 - bh * pad_frac, y2 + bh * pad_frac
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return np.zeros((0, 0, 3), dtype=image.dtype)
    crop = image[iy1:iy2, ix1:ix2]
    ch, cw = crop.shape[:2]
    scale = max_px / max(ch, cw)
    if scale < 1.0:
        crop = cv2.resize(
            crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_AREA
        )
    return crop


class Perceiver:
    """Detect -> track -> trigger-line crossing -> group -> best-crop, per frame.

    `detector`/`tracker` are injectable (real `Detector`/`sv.ByteTrack()` by default) so
    tests can drive the trigger/group/crop state machine with synthetic per-frame boxes and
    track IDs, without needing a GPU or a real tracker's IOU matching.

    GROUPING (hub amendment 2026-10-15, supersedes the original wait-out-the-window
    design, which left this as an open question): an `Event` is created the instant
    the first track crosses, never held open waiting to see who else crosses. Its group is
    that track plus every other qualifying (bbox-height-passing, not-already-fired) track
    whose bottom-centre lies within `perceive.group_band` of the trigger line AT THAT SAME
    FRAME -- they're swept in and marked fired without needing to literally cross
    themselves. A track that crosses while that event is still open (< `best_crop_frames`
    frames since its trigger) joins it too, with no proximity check -- in the rare case an
    unrelated crossing lands inside that tiny window, it merges in; this is the amendment's
    accepted simplification, not a bug. `group_window_s` is no longer read: the only cap on
    how long an event stays open is `best_crop_frames` (shared by the whole event, not
    per-track), so it is never held longer than that many frames.
    """

    def __init__(self, cfg: Config, detector=None, tracker=None) -> None:
        self.cfg = cfg
        p = cfg.perceive
        self.trigger_line = p.trigger_line
        self.min_bbox_height = p.min_bbox_height
        self.group_band = p.group_band
        self.best_crop_frames = p.best_crop_frames
        self.crop_max_px = cfg.judge.crop_max_px

        self.detector = detector if detector is not None else Detector(
            weights_path=Path("models/yolox") / f"{p.detector}.pth",
            conf_threshold=p.min_confidence,
        )
        self.tracker = tracker if tracker is not None else sv.ByteTrack()

        self._fired_at: dict[int, float] = {}
        self._prev_side: dict[int, float] = {}
        self._pending: Event | None = None
        self._pending_age: int = 0
        self._candidates: dict[int, list[tuple[np.ndarray, float]]] = {}
        self._last_tracked: sv.Detections | None = None

    def process_frame(self, image: np.ndarray, t_capture: float) -> list[Event]:
        detections = self.detector.detect(image)
        tracked = self.tracker.update_with_detections(detections)
        self._last_tracked = tracked
        h, w = image.shape[:2]
        events: list[Event] = []

        tracked_boxes: dict[int, np.ndarray] = {}
        newly_fired: list[tuple[int, float]] = []
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            bbox = tracked.xyxy[i]
            tracked_boxes[track_id] = bbox
            x1, y1, x2, y2 = bbox
            bottom_x = (x1 + x2) / 2.0 / w
            bottom_y = y2 / h
            bbox_h_frac = (y2 - y1) / h

            side = _line_side(self.trigger_line, bottom_x, bottom_y)
            prev_side = self._prev_side.get(track_id)
            self._prev_side[track_id] = side

            crossed = prev_side is not None and side != prev_side and side != 0.0 and prev_side != 0.0
            if crossed and bbox_h_frac >= self.min_bbox_height:
                last_fire = self._fired_at.get(track_id)
                if last_fire is None or (t_capture - last_fire) >= _FIRE_MEMORY_S:
                    newly_fired.append((track_id, bbox_h_frac))

        # Grouping (hub amendment 2026-10-15): the first crossing this frame opens (or
        # joins) the pending event; if it opens one, every other qualifying, not-recently-
        # fired track within `group_band` of the line at THIS frame is swept in too, marked
        # fired so it never gets its own event.
        grouped_this_frame: set[int] = set()
        for track_id, bbox_h_frac in newly_fired:
            if track_id in grouped_this_frame:
                continue
            self._fired_at[track_id] = t_capture
            grouped_this_frame.add(track_id)

            if self._pending is not None:
                self._pending.track_ids.append(track_id)
                self._pending.bbox_heights.append(bbox_h_frac)
                self._pending.group_size = len(self._pending.track_ids)
                self._candidates.setdefault(track_id, [])
                continue

            self._pending = Event(
                t_trigger=t_capture,
                track_ids=[track_id],
                group_size=1,
                bbox_heights=[bbox_h_frac],
            )
            self._pending_age = 0
            self._candidates = {track_id: []}

            for other_id, other_bbox in tracked_boxes.items():
                if other_id == track_id or other_id in grouped_this_frame:
                    continue
                last_fire = self._fired_at.get(other_id)
                if last_fire is not None and (t_capture - last_fire) < _FIRE_MEMORY_S:
                    continue
                ox1, oy1, ox2, oy2 = other_bbox
                other_bottom_x = (ox1 + ox2) / 2.0 / w
                other_bottom_y = oy2 / h
                other_h_frac = (oy2 - oy1) / h
                if other_h_frac < self.min_bbox_height:
                    continue
                if _line_distance(self.trigger_line, other_bottom_x, other_bottom_y) <= self.group_band:
                    self._fired_at[other_id] = t_capture
                    grouped_this_frame.add(other_id)
                    self._pending.track_ids.append(other_id)
                    self._pending.bbox_heights.append(other_h_frac)
                    self._pending.group_size = len(self._pending.track_ids)
                    self._candidates[other_id] = []

        if self._pending is not None:
            for track_id in self._pending.track_ids:
                bbox = tracked_boxes.get(track_id)
                if bbox is None:
                    continue
                crop = crop_and_resize(image, bbox, max_px=self.crop_max_px)
                if crop.size:
                    self._candidates.setdefault(track_id, []).append((crop, crop_score(crop)))

            self._pending_age += 1
            if self._pending_age >= self.best_crop_frames:
                events.append(self._finalize_pending())

        return events

    def _finalize_pending(self) -> Event:
        assert self._pending is not None
        ev = self._pending
        best_per_track = []
        for track_id in ev.track_ids:
            cands = self._candidates.get(track_id, [])
            if cands:
                best_crop, best_score = max(cands, key=lambda c: c[1])
                best_per_track.append((best_crop, best_score, track_id))
        best_per_track.sort(key=lambda c: c[1], reverse=True)
        ev.crops = [c[0] for c in best_per_track[:3]]
        ev.crop_track_ids = [c[2] for c in best_per_track[:3]]

        self._pending = None
        self._pending_age = 0
        self._candidates = {}
        return ev

    def current_tracked_boxes(self) -> dict[int, np.ndarray]:
        """track_id -> xyxy bbox from the most recent `process_frame` call (04-04): the aim
        page's colour-coded `/stream.mjpg` overlay needs box geometry each frame without
        reaching into `_last_tracked`/the tracker itself -- same encapsulation idea as
        `debug_draw`, just returning data instead of pixels."""
        if self._last_tracked is None:
            return {}
        return {
            int(self._last_tracked.tracker_id[i]): self._last_tracked.xyxy[i]
            for i in range(len(self._last_tracked))
        }

    def debug_draw(self, image: np.ndarray) -> np.ndarray:
        """Boxes, track IDs and the trigger line, for the aim page (spec §4.3). Draws over
        a copy -- never mutates the frame passed to detection."""
        out = image.copy()
        h, w = out.shape[:2]
        (x0, y0), (x1, y1) = self.trigger_line
        cv2.line(out, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 0, 255), 2)
        tracked = self._last_tracked
        if tracked is not None:
            for i in range(len(tracked)):
                x1b, y1b, x2b, y2b = (int(v) for v in tracked.xyxy[i])
                track_id = int(tracked.tracker_id[i])
                cv2.rectangle(out, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)
                cv2.putText(
                    out, str(track_id), (x1b, max(0, y1b - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
        return out
