"""capture: a per-run log for tuning over the video's timeline (2026-10-19, Martin: test the
box on a busy stream of kids, "temporal optimisation needed").

OFF in the race config (`capture.enabled: false`): race day keeps its no-recording promise.
On for replays of a test video, where the video itself is the dataset, so frames are
referenced by number, not saved. Writes to `<dir>/<run-id>/`:

  meta.json      config summary + source, written at start
  tracks.jsonl   one line per frame: {"f": frame seq, "t": capture time, "b": [[track_id,
                 x1, y1, x2, y2], ...]} -- every tracked person, every frame
  events.jsonl   one line per trigger ("kind": "trigger": frame, tracks, boxes), per judged
                 result ("kind": "result": the orchestrator's record: verdict, line, score,
                 timings), and per speaker outcome ("kind": "played" / "dropped" + reason)
  crops/<event id>.jpg   the exact crop the judge was given, for offline re-judging

The offline tools (silver labels, the policy simulator) read only these files.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class Capture:
    def __init__(self, root: str | Path, meta: dict | None = None) -> None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / run_id
        (self.dir / "crops").mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps({"run_id": run_id, **(meta or {})}, indent=1))
        self._tracks = (self.dir / "tracks.jsonl").open("a")
        self._events = (self.dir / "events.jsonl").open("a")
        self._n_frames = 0
        logger.info("capture.start dir=%s", self.dir)

    def frame(self, seq: int, t: float, boxes: dict) -> None:
        rows = [[int(tid), *(round(float(v), 1) for v in box[:4])] for tid, box in boxes.items()]
        self._tracks.write(json.dumps({"f": int(seq), "t": round(float(t), 4), "b": rows}) + "\n")
        self._n_frames += 1
        if self._n_frames % 300 == 0:
            self._tracks.flush()

    def trigger(self, event, seq: int, boxes: dict, crop_jpeg: bytes | None) -> None:
        if crop_jpeg:
            (self.dir / "crops" / f"{event.id}.jpg").write_bytes(crop_jpeg)
        own = {tid: boxes[tid] for tid in event.track_ids if tid in boxes}
        self.event(
            "trigger",
            id=event.id,
            f=int(seq),
            t=round(float(event.t_trigger), 4),
            tracks=list(event.track_ids),
            boxes={str(k): [round(float(v), 1) for v in b[:4]] for k, b in own.items()},
        )

    def event(self, kind: str, **fields) -> None:
        self._events.write(json.dumps({"kind": kind, "wall": round(time.time(), 3), **fields}, default=str) + "\n")
        self._events.flush()

    def close(self) -> None:
        for fh in (self._tracks, self._events):
            try:
                fh.close()
            except OSError:
                pass
