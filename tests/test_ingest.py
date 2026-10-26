import asyncio
import time
from pathlib import Path

import av
import numpy as np
import pytest

from scarebox.ingest import FileSource, LatestFrameBuffer, Frame


def _write_tiny_mp4(path: Path, n_frames: int = 10, fps: int = 10) -> None:
    """A tiny synthetic MP4: solid-colour frames, each a different shade so decode order
    is easy to sanity-check, at a known fps so FileSource's real-time pacing is testable."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    for i in range(n_frames):
        array = np.full((48, 64, 3), fill_value=(i * 20) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        frame = frame.reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def tiny_mp4(tmp_path) -> Path:
    path = tmp_path / "tiny.mp4"
    _write_tiny_mp4(path, n_frames=10, fps=10)
    return path


def test_file_source_yields_monotonic_seq(tiny_mp4):
    async def collect():
        source = FileSource(tiny_mp4, realtime=False)
        seqs = []
        async for frame in source.frames():
            seqs.append(frame.seq)
        return seqs

    seqs = asyncio.run(collect())
    assert seqs == list(range(1, 11))


def test_file_source_realtime_pacing(tiny_mp4):
    """10 frames at 10fps is ~1s of footage; realtime=True should take roughly that long,
    not return instantly (decode-as-fast-as-possible would finish in a few ms)."""

    async def collect():
        source = FileSource(tiny_mp4, realtime=True)
        frames = []
        async for frame in source.frames():
            frames.append(frame)
        return frames

    start = time.monotonic()
    frames = asyncio.run(collect())
    elapsed = time.monotonic() - start
    assert len(frames) == 10
    # Real-time for ~0.9s of footage (10 frames after the first, at 10fps): generous bounds
    # since CI/bench hosts vary, but it must not look like "as fast as possible" (a few ms).
    assert elapsed > 0.5


def test_file_source_loop_restarts(tiny_mp4):
    async def collect(limit):
        source = FileSource(tiny_mp4, realtime=False, loop=True)
        seqs = []
        async for frame in source.frames():
            seqs.append(frame.seq)
            if len(seqs) >= limit:
                break
        return seqs

    # Looping means seq keeps counting up across the loop boundary (never resets to 0),
    # so 15 frames out of a 10-frame clip proves the loop fired without duplicating seq.
    seqs = asyncio.run(collect(15))
    assert seqs == list(range(1, 16))


def test_latest_frame_buffer_skips_under_slow_consumer():
    """A producer pushing frames much faster than the consumer polls must never hand the
    consumer a backlog -- the consumer sees strictly increasing seq with gaps, not every
    seq in order, proving the buffer overwrites rather than queues."""

    async def run():
        buf = LatestFrameBuffer()
        seen = []

        async def producer():
            # 400 frames at 1ms each (~400ms) comfortably outlasts the consumer's 5-poll,
            # 20ms-apart window (~100ms) -- otherwise the producer can exhaust its frames
            # before the consumer's last poll, which then waits forever for a frame newer
            # than the last one that will ever exist (found via local repro, 2026-10-15).
            for i in range(1, 401):
                buf.put(Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), t_capture=0.0, seq=i))
                await asyncio.sleep(0.001)

        async def consumer():
            last = 0
            # A deliberately slow consumer: it would see at most ~5 of the 50 pushed
            # frames in the time the producer takes to push them all (~50ms).
            for _ in range(5):
                frame = await buf.get_new(last)
                last = frame.seq
                seen.append(last)
                await asyncio.sleep(0.02)

        await asyncio.gather(producer(), consumer())
        return seen

    seen = asyncio.run(run())
    assert len(seen) == 5
    # Strictly increasing...
    assert seen == sorted(set(seen))
    # ...and with gaps: the consumer did not see every seq from 1 to its final value.
    assert seen[-1] - seen[0] + 1 > len(seen)


# --- UsbSource (2026-10-19: the Dell webcam on gbox1 is the race camera) ------------------

from scarebox.ingest import UsbSource, find_usb_camera  # noqa: E402


def _fake_sysfs(tmp_path, nodes):
    sys_dir = tmp_path / "video4linux"
    for node, name, index in nodes:
        d = sys_dir / node
        d.mkdir(parents=True)
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(f"{index}\n")
    return str(sys_dir)


def test_find_usb_camera_picks_lowest_index_node_by_name(tmp_path):
    sys_dir = _fake_sysfs(tmp_path, [
        ("video0", "Full HD webcam", 0), ("video1", "Full HD webcam", 1),
        ("video2", "Dell Webcam WB7022", 0), ("video3", "Dell Webcam WB7022", 1),
        ("video4", "Dell Webcam WB7022", 2),
    ])
    assert find_usb_camera("WB7022", "/host-dev", sys_dir) == "/host-dev/video2"
    assert find_usb_camera("wb7022", "/dev", sys_dir) == "/dev/video2"  # case-insensitive
    assert find_usb_camera("Logitech", "/dev", sys_dir) is None


class _FakeCap:
    def __init__(self, frames):
        self._frames = list(frames)

    def isOpened(self):  # noqa: N802 -- cv2 API
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        pass


def test_usb_source_reconnects_after_camera_vanishes():
    # The first open delivers 3 frames, then the camera "unplugs" (reads fail). The source
    # must re-run discovery and reopen (as a replug would) and keep seq increasing.
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    caps = [_FakeCap([img] * 3), _FakeCap([img] * 50)]
    opened, found = [], []

    def opener(path, w, h, fps):
        opened.append((path, w, h, fps))
        return caps.pop(0) if caps else _FakeCap([])

    def finder(name, dev_dir):
        found.append(name)
        return f"{dev_dir}/video{len(found) - 1}"

    src = UsbSource("WB7022", dev_dir="/host-dev", reconnect_backoff=(0.01,), opener=opener, finder=finder)

    async def take_until(seq):
        async for f in src.frames():
            if f.seq >= seq:
                return f.seq

    last = asyncio.run(asyncio.wait_for(take_until(5), timeout=5))
    src.stop()
    assert last >= 5  # frames kept coming after the drop
    assert opened[0] == ("/host-dev/video0", 1920, 1080, 30)
    assert opened[1][0] == "/host-dev/video1"  # re-discovered after the drop


def test_make_source_usb(monkeypatch):
    import dataclasses

    from scarebox.config import load_config
    from scarebox.ingest import make_source

    for k in ("CAMERA_HOST", "CAMERA_USER", "CAMERA_PASSWORD", "AIM_TOKEN"):
        monkeypatch.setenv(k, "x")
    cfg = load_config("config/bench.yaml")
    cfg = dataclasses.replace(cfg, source=dataclasses.replace(cfg.source, kind="usb"))
    src = make_source(cfg)
    assert isinstance(src, UsbSource)
    assert (src.name, src.dev_dir, src.width, src.height, src.fps) == ("WB7022", "/host-dev", 1920, 1080, 30)
