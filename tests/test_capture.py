"""capture (2026-10-19): the per-run tuning log the timeline simulator reads."""
import json
import types

import numpy as np

from scarebox.capture import Capture
from scarebox.stage import PlayItem, Stage


def test_capture_writes_tracks_events_and_crops(tmp_path):
    cap = Capture(tmp_path, meta={"source": "file"})
    boxes = {7: np.array([10.0, 20.0, 110.0, 320.0]), 9: np.array([200.0, 25.0, 280.0, 300.0])}
    cap.frame(41, 1.5, boxes)
    ev = types.SimpleNamespace(id="e1", t_trigger=1.5, track_ids=[7])
    cap.trigger(ev, 41, boxes, b"\xff\xd8crop")
    cap.event("result", id="e1", category="pirate", line="Arrr!", score=2)
    cap.close()

    run = next(tmp_path.iterdir())
    assert json.loads((run / "meta.json").read_text())["source"] == "file"
    frame = json.loads((run / "tracks.jsonl").read_text().splitlines()[0])
    assert frame["f"] == 41 and sorted(r[0] for r in frame["b"]) == [7, 9]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["trigger", "result"]
    assert events[0]["boxes"] == {"7": [10.0, 20.0, 110.0, 320.0]}  # only the event's own track
    assert (run / "crops" / "e1.jpg").read_bytes() == b"\xff\xd8crop"


def test_stage_reports_every_line_outcome():
    cfg = types.SimpleNamespace(
        stage=types.SimpleNamespace(cooldown_s=0.0, queue_max=1, stings=False, alsa_device="x"),
        orchestrator=types.SimpleNamespace(max_line_age_s=30.0),
    )

    class Sink:
        def play(self, pcm, sr):
            pass

    seen = []
    stage = Stage(cfg, Sink(), clock=lambda: 10.0, sfx_dir="no/such/dir")
    stage.on_outcome = lambda kind, item, **kw: seen.append((kind, item.event_id, kw.get("reason")))
    tone = np.zeros(10, dtype=np.int16)
    stage.enqueue(PlayItem(event_id="a", t_trigger=9.0, audio=tone))
    stage.enqueue(PlayItem(event_id="b", t_trigger=9.5, audio=tone))  # queue_max 1 -> a dropped
    import asyncio

    asyncio.run(stage.try_play_next())
    assert seen == [("dropped", "a", "queue_full"), ("played", "b", None)]
