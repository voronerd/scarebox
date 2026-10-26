"""Perceiver trigger/group/best-crop tests, driven with a fake detector + an identity
tracker (bypassing real YOLOX + ByteTrack, which need a GPU / real IOU matching) so the
state machine itself -- crossing detection, grouping, crop selection -- is exercised
deterministically. Spec §4.3; plan 02-01 Task 2 (hub amendment 2026-10-15: grouping at
the first crossing via `group_band`, no `group_window_s` hold).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# scarebox.perceive imports OpenCV, PyTorch and supervision at module level; on a CPU-only
# dev machine without them, skip with a reason instead of erroring at collection.
for _mod in ("cv2", "torch", "supervision"):
    pytest.importorskip(_mod, reason=f"scarebox.perceive needs {_mod} (OpenCV/GPU stack), not installed")

from scarebox.perceive import Perceiver  # noqa: E402


class _FakeDetections:
    """Duck-types just enough of `sv.Detections` for `Perceiver` to consume:
    `len()`, `.xyxy` (Nx4), `.tracker_id` (N,)."""

    def __init__(self, xyxy, tracker_id):
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.tracker_id = np.array(tracker_id, dtype=int)

    def __len__(self):
        return len(self.xyxy)


class _FakeDetector:
    """Returns one preset `_FakeDetections` per call, in order."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def detect(self, image):
        frame = self._frames[self._i]
        self._i += 1
        return frame


class _IdentityTracker:
    """The fake detector already attaches track IDs, so tracking is a no-op."""

    def update_with_detections(self, detections):
        return detections


def make_cfg(**perceive_overrides):
    perceive = SimpleNamespace(
        detector="yolox_s",
        min_confidence=0.5,
        trigger_line=((0.5, 0.0), (0.5, 1.0)),
        min_bbox_height=0.2,
        group_band=0.08,
        group_window_s=1.5,  # deprecated, unread by Perceiver; kept for config parity
        best_crop_frames=1,
    )
    for key, value in perceive_overrides.items():
        setattr(perceive, key, value)
    judge = SimpleNamespace(crop_max_px=512)
    return SimpleNamespace(perceive=perceive, judge=judge)


def make_perceiver(detections_sequence, **perceive_overrides):
    cfg = make_cfg(**perceive_overrides)
    detector = _FakeDetector(detections_sequence)
    return Perceiver(cfg, detector=detector, tracker=_IdentityTracker())


def _blank(w=100, h=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_single_crossing_fires_once():
    # track 1: left (x_center=40 -> norm 0.40) then right (x_center=60 -> norm 0.60):
    # crosses the x=0.5 trigger line. bbox height 30/100 = 0.3 clears min_bbox_height=0.2.
    frames = [
        _FakeDetections([[35, 0, 45, 30]], [1]),
        _FakeDetections([[55, 0, 65, 30]], [1]),  # crosses here, t_trigger; best_crop_frames=1
    ]
    p = make_perceiver(frames, best_crop_frames=1)
    events = []
    events += p.process_frame(_blank(), 0.0)
    events += p.process_frame(_blank(), 0.1)
    assert len(events) == 1
    assert events[0].track_ids == [1]
    assert events[0].group_size == 1


def test_non_crossing_track_never_fires():
    # Stays on the left side (x_center=40, norm 0.40) for every frame: never fires.
    frames = [_FakeDetections([[35, 0, 45, 30]], [1]) for _ in range(5)]
    p = make_perceiver(frames, best_crop_frames=1)
    events = []
    for i in range(5):
        events += p.process_frame(_blank(), i * 0.1)
    assert events == []


def test_small_box_does_not_fire():
    # Crosses the line, but bbox height is 20/100=0.2 < min_bbox_height=0.5: must not fire.
    frames = [
        _FakeDetections([[35, 0, 45, 20]], [1]),
        _FakeDetections([[55, 0, 65, 20]], [1]),  # would cross, but box too small
        _FakeDetections([[70, 0, 80, 20]], [1]),
    ]
    p = make_perceiver(frames, best_crop_frames=1, min_bbox_height=0.5)
    events = []
    events += p.process_frame(_blank(), 0.0)
    events += p.process_frame(_blank(), 0.1)
    events += p.process_frame(_blank(), 0.3)
    assert events == []


def test_side_by_side_runners_make_one_group_at_first_crossing():
    # track 1 crosses (0.40 -> 0.60). track 2 sits at 0.46 throughout -- never itself
    # crosses the x=0.5 line -- but at the moment track 1 crosses, track 2's
    # bottom-centre is |0.46 - 0.5| = 0.04 from the line, inside group_band=0.08, so it is
    # swept into the SAME event, marked fired, at the first-crossing frame (no delay).
    frames = [
        _FakeDetections([[35, 0, 45, 30], [41, 0, 51, 30]], [1, 2]),  # t=0.0: 1@.40, 2@.46
        _FakeDetections([[55, 0, 65, 30], [41, 0, 51, 30]], [1, 2]),  # t=0.05: 1 crosses
    ]
    p = make_perceiver(frames, best_crop_frames=1)
    events = []
    events += p.process_frame(_blank(), 0.0)
    events += p.process_frame(_blank(), 0.05)
    assert len(events) == 1
    assert events[0].group_size == 2
    assert sorted(events[0].track_ids) == [1, 2]


def test_crossing_after_event_closes_is_a_separate_event():
    # track 1 crosses and its event closes (best_crop_frames=2 frames later). Only once
    # that event has finalized does track 2 cross -- it must NOT be folded into track 1's
    # already-closed event.
    frames = [
        _FakeDetections([[35, 0, 45, 30]], [1]),  # t=0.0: track1 left
        _FakeDetections([[55, 0, 65, 30]], [1]),  # t=0.05: track1 crosses -> event A opens
        _FakeDetections([[70, 0, 80, 30]], [1]),  # t=0.1: collect frame 2 -> event A closes
        _FakeDetections([[15, 0, 25, 30]], [2]),  # t=2.0: track2 appears, left
        _FakeDetections([[55, 0, 65, 30]], [2]),  # t=2.05: track2 crosses -> event B opens
        _FakeDetections([[70, 0, 80, 30]], [2]),  # t=2.1: collect frame 2 -> event B closes
    ]
    p = make_perceiver(frames, best_crop_frames=2)
    events = []
    for t in (0.0, 0.05, 0.1, 2.0, 2.05, 2.1):
        events += p.process_frame(_blank(), t)
    assert len(events) == 2
    assert events[0].track_ids == [1]
    assert events[0].group_size == 1
    assert events[1].track_ids == [2]
    assert events[1].group_size == 1


def test_solo_runner_emitted_within_best_crop_frames_no_hold():
    # Frame timestamps are 10s apart -- far more than the old 1.5s group_window_s hold --
    # to prove emission is gated on FRAME COUNT (best_crop_frames), not elapsed time. If
    # the old time-based deadline were still in effect, the event would have closed after
    # the very next frame (t=10.0 > 1.5s); instead it must wait for exactly
    # best_crop_frames=3 frames of collection before emitting.
    frames = [
        _FakeDetections([[35, 0, 45, 30]], [1]),  # t=0.0: left
        _FakeDetections([[55, 0, 65, 30]], [1]),  # t=10.0: crosses -> opens, age=1
        _FakeDetections([[60, 0, 70, 30]], [1]),  # t=20.0: collect -> age=2
        _FakeDetections([[65, 0, 75, 30]], [1]),  # t=30.0: collect -> age=3 == best_crop_frames
    ]
    p = make_perceiver(frames, best_crop_frames=3)
    events = []
    events += p.process_frame(_blank(), 0.0)
    events += p.process_frame(_blank(), 10.0)
    events += p.process_frame(_blank(), 20.0)
    assert events == []  # not yet -- only 2 of 3 collection frames have passed
    events += p.process_frame(_blank(), 30.0)
    assert len(events) == 1
    assert events[0].track_ids == [1]


def test_best_crop_picks_the_sharper_of_two():
    frames = [
        _FakeDetections([[35, 0, 45, 30]], [1]),  # left, pre-crossing (not collected)
        _FakeDetections([[55, 0, 65, 40]], [1]),  # crosses -> collect #1 (sharp image)
        _FakeDetections([[55, 0, 65, 40]], [1]),  # collect #2 (flat image) -> closes
    ]
    p = make_perceiver(frames, best_crop_frames=2, min_bbox_height=0.0)
    rng = np.random.default_rng(0)
    sharp_img = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    blurry_img = np.full((100, 100, 3), 128, dtype=np.uint8)

    events = []
    events += p.process_frame(blurry_img, 0.0)  # pre-crossing, nothing collected
    events += p.process_frame(sharp_img, 0.05)  # crossing + collect #1 (sharp)
    events += p.process_frame(blurry_img, 0.06)  # collect #2 (flat) -> finalize

    assert len(events) == 1
    crops = events[0].crops
    assert len(crops) == 1
    # The flat crop has std == 0; the noisy crop's std is far above that -- the sharper one won.
    assert crops[0].std() > 10
