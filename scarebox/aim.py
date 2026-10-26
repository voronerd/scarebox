"""aim: the race-morning phone "control room" (04-02 base, 04-04 upgrade). Shows a live,
colour-coded MJPEG stream of what the camera sees (trigger line + per-track state), recent
event cards (what led to each line played), a live settings panel, PTZ nudge, and the
trigger-line drag. Runs inside `app.py` on `aim.listen` (0.0.0.0:8090); every route needs
`AIM_TOKEN`.

Design: every camera/vision operation (cv2 JPEG encode, `perceive.debug_draw`,
`Camera.ptz_nudge`) is injected into this module as a plain callable in `AimDeps`, supplied
by `app.py` -- the only module that needs cv2/torch/`Camera` at import time. `aim.py` itself
imports nothing heavier than the stdlib, so `tests/test_aim.py` runs on a bare venv, same bar
as the rest of the pure-logic test files (no cv2 in a CPU-only dev venv -- see `perceive.py`'s own
test exclusion).

Snapshots and stream frames are generated on demand, in memory, and never written to disk
(spec §2; 04-04 extends the same rule to the event-card crop JPEGs and the whole event ring
buffer -- see `app.py`'s `EventRing`, RAM-only by construction). This module has no
file-writing path for a rendered frame or a crop at all -- the bytes come back from an
injected callable and go straight to the HTTP response.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

AIM_HTML_PATH = Path(__file__).parent / "aim.html"
DEFAULT_TRIGGER_LINE_PATH = Path("runs/trigger_line.json")
VALID_PTZ_DIRS = ("Left", "Right", "Up", "Down")
# 04-04: `/stream.mjpg` -- a 3rd phone/laptop watching the live view at once gets a 503
# instead of silently starving the first two (and the frame-render CPU budget) of bandwidth.
_MAX_STREAM_CLIENTS = 2
# ~10 fps -- smooth enough for aiming/judging by eye, far short of the 1 fps `/snapshot.jpg`
# this supplements (kept working, unchanged, for anything still polling it).
_STREAM_FRAME_INTERVAL_S = 0.1
_STREAM_BOUNDARY = "scareboxframe"


class AimError(Exception):
    """Raised for an empty `AIM_TOKEN` -- refuse to serve an unprotected aim page."""


def _client_trusted(client_ip: str, networks) -> bool:
    """True when the request comes from one of `networks` (CIDR strings). A malformed entry or
    address never grants access."""
    try:
        ip = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    for net in networks or ():
        try:
            if ip in ipaddress.ip_network(net, strict=False):
                return True
        except ValueError:
            continue
    return False


def check_token(provided: str | None, expected: str) -> bool:
    if not expected:
        raise AimError("AIM_TOKEN is empty -- refusing to serve an unprotected aim page")
    return provided is not None and provided == expected


def save_trigger_line(line, path: str | Path = DEFAULT_TRIGGER_LINE_PATH) -> None:
    """Write-tmp-then-`os.replace` (same pattern as `bake.write_manifest_atomic`) so a
    reader never sees a half-written file. `line` is `((x1, y1), (x2, y2))`, normalised
    0..1 frame coordinates -- the same shape `perceive.Perceiver.trigger_line` expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps([list(pt) for pt in line]))
    os.replace(tmp, path)


def load_trigger_line(path: str | Path = DEFAULT_TRIGGER_LINE_PATH):
    """`None` if nothing has been saved yet -- `app.py` keeps the config value in that case.
    Otherwise the saved line, as `((x1, y1), (x2, y2))` of floats."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return tuple(tuple(float(v) for v in pt) for pt in data)


@dataclasses.dataclass
class AimDeps:
    """Everything the aim HTTP handler needs, as plain callables/paths. `app.py` wires the
    real cv2 snapshot/stream encoders, `Camera.ptz_nudge`, the live `Perceiver.trigger_line`,
    `Library`/`Stage` for the test button, the RAM-only event ring buffer, and the `Tuning`
    settings object; tests inject fakes for all of it."""

    token: str
    snapshot_jpeg: Callable[[], bytes | None]
    # 04-04: the colour-coded, ~960px-wide overlay frame for `/stream.mjpg` -- a SEPARATE
    # callable from `snapshot_jpeg` (different overlay, different cadence), so `/snapshot.jpg`
    # keeps behaving exactly as it did in 04-02.
    stream_frame_jpeg: Callable[[], bytes | None]
    ptz_nudge: Callable[[str, int, float], None]
    get_trigger_line: Callable[[], tuple]
    set_trigger_line: Callable[[tuple], None]
    # 04-04: the RAM-only event ring buffer (`app.EventRing`), replacing the 04-02
    # `events.jsonl`-tailing `/events` route -- `events()` returns JSON-safe card dicts (no
    # image bytes ever included), `event_crop(id)` returns that one event's crop JPEG bytes
    # (or `None` for an unknown/evicted id -> 404).
    events: Callable[[], list[dict]]
    event_crop: Callable[[str], bytes | None]
    test_line: Callable[[], str | None]
    status: Callable[[], dict]
    # 04-04: the live Settings panel -- `get_settings()` returns `{values, defaults, ranges}`;
    # `set_settings(patch)` validates+clamps+applies a POST body, returning `(ok, error)`;
    # `reset_settings()` restores config defaults. All three read/write through `app.py`'s
    # `Tuning` instance, never a second copy of the same number.
    get_settings: Callable[[], dict]
    set_settings: Callable[[dict], tuple[bool, str | None]]
    reset_settings: Callable[[], dict]
    trigger_line_path: Path = DEFAULT_TRIGGER_LINE_PATH
    # 04-04: `/stream.mjpg?view=full` -- the UNCROPPED frame with the crop rectangle drawn on
    # it, for the Crop-mode editor (dragging the rectangle needs to see what's being cut
    # off). Falls back to `stream_frame_jpeg` when unset (e.g. a test double that doesn't
    # care about Crop mode).
    stream_frame_full_jpeg: Callable[[], bytes | None] | None = None
    # Detection report (2026-10-19, a PDF for HCA): `event_image(id, kind)` returns that
    # event's "wide" shot or whole "frame" JPEG (or the judge's "crop"); None when unknown.
    event_image: Callable[[str, str], bytes | None] | None = None
    # `report_pdf(max_events)` -> PDF bytes (app.py: `report_pdf.build_report_pdf`). Built on
    # the box with compressed pages, because a phone's print-to-PDF of /report.html came out
    # at 85 MB. None means the route 404s (tests, or a build without Pillow).
    report_pdf: Callable[[int], bytes] | None = None
    # Networks that may use the page WITHOUT the token (Martin, 2026-10-23: phones on the race
    # travel router's password-protected Wi-Fi). Everything else, e.g. the home LAN, still needs it.
    trusted_networks: tuple = ()


_SCORE_MEANING = {
    3: "names this exact costume",
    2: "costume-category line",
    1: "no-costume goading",
    0: "generic (couldn't tell)",
}
_PATH_MEANING = {
    "live": "written live for this runner",
    "library": "pre-recorded",
    "dropped": "nothing played",
    "silent": "no line (non-costumes switched off)",
}


def render_report(cards: list[dict], image: Callable[[str, str], bytes | None]) -> str:
    """A self-contained, print-ready HTML detection report (images inlined as data URIs, so
    "Print -> Save as PDF" on a phone, or Chrome headless, gives a standalone PDF). Newest
    first; only events the judge actually resolved -- skipped crossings never get a card."""
    import base64
    import html
    import time as _time

    def esc(s) -> str:
        return html.escape("" if s is None else str(s))

    def img(eid: str, kind: str, cls: str) -> str:
        data = image(eid, kind)
        if not data:
            return ""
        b64 = base64.b64encode(data).decode()
        return f'<img class="{cls}" src="data:image/jpeg;base64,{b64}" alt="{kind}">'

    rows = [c for c in reversed(cards) if c.get("resolved") and c.get("path") != "skipped"]
    costumed = sum(1 for c in rows if c.get("is_costume"))
    packs = sum(1 for c in rows if c.get("pack"))
    blocks = []
    for c in rows:
        eid = c["id"]
        when = _time.strftime("%H:%M:%S", _time.localtime(c["wall_time"])) if c.get("wall_time") else ""
        if not c.get("is_costume"):
            verdict = "No costume"
        elif c.get("pack_match_display"):
            verdict = f'{esc(c["pack_match_display"])} <span class="cat">({esc((c.get("category") or "").replace("_", " "))})</span>'
        else:
            verdict = f'{esc(c.get("costume"))} <span class="cat">({esc((c.get("category") or "").replace("_", " "))})</span>'
        path = c.get("path") or ""
        how = "costume pack: " + esc(c.get("pack_match_display") or c.get("pack")) if c.get("pack") else esc(
            _PATH_MEANING.get(path.split(":")[0], path)
        )
        score = c.get("score")
        score_txt = f"{score} / 3, {_SCORE_MEANING.get(score, '')}" if score is not None else "n/a"
        ms = c.get("ms") or {}
        timing = " &middot; ".join(
            f"{name} {ms[k] / 1000:.1f} s"
            for k, name in (("judge", "judge"), ("quip", "line writing"), ("voice_first_audio", "voice"), ("total", "total"))
            if ms.get(k) is not None
        )
        details = []
        if c.get("is_costume") and c.get("pack_match_display") and c.get("costume"):
            details.append(f"<dt>Judge&rsquo;s words</dt><dd>&ldquo;{esc(c['costume'])}&rdquo;</dd>")
        if (c.get("group_size") or 1) > 1:
            details.append(f"<dt>Group</dt><dd>{int(c['group_size'])} people</dd>")
        blocked = (
            f'<dt>Blocked</dt><dd>live line &ldquo;{esc(c["blocked_text"])}&rdquo; stopped by the safety filter '
            f'({esc(c.get("blocked_reason"))})</dd>'
            if c.get("blocked_text") else ""
        )
        blocks.append(f"""
<section class="ev">
  <div class="pics">{img(eid, "frame", "main") or img(eid, "wide", "main") or img(eid, "crop", "main")}</div>
  <div class="txt">
    <div class="when">{esc(when)}</div>
    <h2>{verdict}</h2>
    <p class="line">{"&ldquo;" + esc(c.get("line")) + "&rdquo;" if c.get("line") else "<em>No line played</em>"}</p>
    <dl>
      <dt>Line</dt><dd>{how}</dd>
      <dt>Priority</dt><dd>{esc(score_txt)}</dd>
      {"".join(details)}
      {blocked}
      <dt>Timing</dt><dd>{timing or "n/a"}</dd>
    </dl>
    {img(eid, "wide", "close") if image(eid, "frame") else ""}
  </div>
</section>""")
    generated = _time.strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scarebox detection report</title>
<style>
  @page {{ size: A4; margin: 14mm; }}
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1d1a26; margin: 0 auto; max-width: 900px; padding: 16px; }}
  header h1 {{ font-size: 26px; margin: 0 0 4px; }}
  header p {{ margin: 4px 0; color: #444; line-height: 1.45; }}
  .stats {{ font-weight: 600; color: #1d1a26; }}
  .ev {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 16px; padding: 14px 0; border-top: 1px solid #ccc; break-inside: avoid; page-break-inside: avoid; }}
  .pics {{ display: grid; gap: 8px; align-content: start; }}
  .main {{ width: 100%; max-height: 300px; object-fit: contain; border-radius: 4px; }}
  .close {{ display: block; height: 150px; margin-top: 8px; border-radius: 4px; }}
  .when {{ color: #777; font-size: 12px; }}
  h2 {{ font-size: 19px; margin: 2px 0 6px; }}
  .cat {{ font-weight: 400; color: #666; font-size: 15px; }}
  .line {{ font-size: 16px; font-style: italic; margin: 6px 0 10px; }}
  dl {{ display: grid; grid-template-columns: auto 1fr; gap: 3px 10px; margin: 0; font-size: 13px; }}
  dt {{ color: #777; }} dd {{ margin: 0; }}
  @media (max-width: 600px) {{ .ev {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<header>
  <h1>Scarebox: detection report</h1>
  <p>A camera watches runners near the finish. A person detector spots each runner as they cross a line on the
  camera image, a vision model judges their costume from that picture, and &ldquo;Sir Reginald&rdquo;, a posh
  English heckler voice, calls out a G-rated, costume-specific line urging them on. Each entry below is one
  detection: the picture the system judged (the yellow box marks the person), what it decided, and the line it chose.</p>
  <p><b>Priority</b> ranks lines when it&rsquo;s crowded (3 = names that exact costume, 2 = costume-category line,
  1 = plain encouragement for someone without a costume). In a crowd only 2&ndash;3 are spoken. <b>Timing</b> is from
  the runner crossing the line.</p>
  <p class="stats">{len(rows)} detections &middot; {costumed} in costume &middot; {packs} matched a specific costume pack &middot; generated {esc(generated)}</p>
</header>
{"".join(blocks) or "<p>No detections yet.</p>"}
</body></html>"""


def _make_handler(deps: AimDeps) -> type[BaseHTTPRequestHandler]:
    # Shared across every connection this ONE aim server handles (closed over, not a class
    # attribute, so two independently-constructed servers in the same test process never
    # share a counter). `ThreadingHTTPServer` runs each connection on its own thread, so this
    # needs a lock, not just an int.
    stream_lock = threading.Lock()
    stream_clients = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        server_version = "scarebox-aim/1.0"

        def _token_ok(self, parsed) -> bool:
            if _client_trusted(self.client_address[0], deps.trusted_networks):
                return True
            qs = parse_qs(parsed.query)
            provided = qs.get("token", [None])[0] or self.headers.get("X-Token")
            try:
                return check_token(provided, deps.token)
            except AimError:
                return False

        def _send_bytes(
            self, code: int, body: bytes, content_type: str, cache_s: int = 0
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            if cache_s:
                # An event's crop never changes once taken, so the phone keeps it rather than
                # refetching every card image on every refresh. `private`: never a shared cache.
                self.send_header("Cache-Control", f"private, max-age={cache_s}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, obj) -> None:
            self._send_bytes(code, json.dumps(obj).encode(), "application/json")

        def _forbidden(self) -> None:
            self._send_bytes(403, b"forbidden", "text/plain")

        def _not_found(self) -> None:
            self.send_response(404)
            self.end_headers()

        def _stream_mjpeg(self, full: bool) -> None:
            """Blocks for the lifetime of the connection, pushing one JPEG frame every
            `_STREAM_FRAME_INTERVAL_S` until the client goes away. The write (not the sleep)
            is what notices a dead client -- a closed socket raises on `self.wfile.write`,
            caught below to end the loop cleanly instead of logging a scary traceback for
            every ordinary "phone locked/tab closed" disconnect.

            `full` (04-04, `?view=full`): the UNCROPPED frame with the crop rectangle drawn
            on it, for the Crop-mode editor -- falls back to the normal cropped+overlay
            frame if `app.py` never wired a `stream_frame_full_jpeg`."""
            get_frame = deps.stream_frame_jpeg
            if full and deps.stream_frame_full_jpeg is not None:
                get_frame = deps.stream_frame_full_jpeg
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={_STREAM_BOUNDARY}"
            )
            self.end_headers()
            try:
                while True:
                    frame = get_frame()
                    if frame is not None:
                        self.wfile.write(f"--{_STREAM_BOUNDARY}\r\n".encode())
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(_STREAM_FRAME_INTERVAL_S)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def do_GET(self) -> None:  # noqa: N802 (stdlib override)
            parsed = urlparse(self.path)
            if not self._token_ok(parsed):
                self._forbidden()
                return
            if parsed.path in ("/", "/index.html"):
                self._send_bytes(200, AIM_HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/snapshot.jpg":
                jpeg = deps.snapshot_jpeg()
                if jpeg is None:
                    self._send_json(503, {"error": "no frame yet"})
                    return
                self._send_bytes(200, jpeg, "image/jpeg")
                return
            if parsed.path == "/stream.mjpg":
                qs = parse_qs(parsed.query)
                full = qs.get("view", [""])[0] == "full"
                with stream_lock:
                    if stream_clients["n"] >= _MAX_STREAM_CLIENTS:
                        self._send_json(503, {"error": "stream busy"})
                        return
                    stream_clients["n"] += 1
                try:
                    self._stream_mjpeg(full)
                finally:
                    with stream_lock:
                        stream_clients["n"] -= 1
                return
            if parsed.path == "/events":
                self._send_json(200, {"events": deps.events()})
                return
            if parsed.path == "/report.pdf":
                if deps.report_pdf is None:
                    self._not_found()
                    return
                try:
                    n = max(1, min(60, int((parse_qs(parsed.query).get("n") or ["30"])[0])))
                except ValueError:
                    n = 30
                body = deps.report_pdf(n)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", 'attachment; filename="scarebox-detection-report.pdf"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/report.html":
                cards = deps.events()
                image = deps.event_image or (lambda eid, kind: deps.event_crop(eid) if kind == "crop" else None)
                body = render_report(cards, image).encode()
                self._send_bytes(200, body, "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/event/") and parsed.path.endswith(".jpg"):
                event_id = parsed.path[len("/event/") : -len(".jpg")]
                kind = (parse_qs(parsed.query).get("kind") or ["crop"])[0]
                if kind != "crop" and deps.event_image is not None:
                    crop = deps.event_image(event_id, kind)
                else:
                    crop = deps.event_crop(event_id)
                if crop is None:
                    self._not_found()
                    return
                self._send_bytes(200, crop, "image/jpeg", cache_s=3600)
                return
            if parsed.path == "/status":
                self._send_json(200, deps.status())
                return
            if parsed.path == "/settings":
                self._send_json(200, deps.get_settings())
                return
            self._not_found()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not self._token_ok(parsed):
                self._forbidden()
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"})
                return

            if parsed.path == "/ptz":
                try:
                    direction = payload["dir"]
                    if direction not in VALID_PTZ_DIRS:
                        raise ValueError(f"invalid dir: {direction!r}")
                    speed = int(payload.get("speed", 1))
                    ms = float(payload.get("ms", 300))
                    deps.ptz_nudge(direction, speed, ms / 1000.0)
                except Exception as exc:  # noqa: BLE001 -- bad payload or a dead camera
                    self._send_json(400, {"error": str(exc)})
                    return
                self._send_json(200, {"ok": True})
                return

            if parsed.path == "/trigger":
                try:
                    line = (
                        (float(payload["x1"]), float(payload["y1"])),
                        (float(payload["x2"]), float(payload["y2"])),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self._send_json(400, {"error": f"bad trigger payload: {exc}"})
                    return
                save_trigger_line(line, deps.trigger_line_path)
                deps.set_trigger_line(line)
                self._send_json(200, {"ok": True, "line": [list(pt) for pt in line]})
                return

            if parsed.path == "/test":
                try:
                    result = deps.test_line()
                except Exception as exc:  # noqa: BLE001
                    self._send_json(500, {"error": str(exc)})
                    return
                self._send_json(200, {"queued": result is not None, "line": result})
                return

            if parsed.path == "/settings":
                if not isinstance(payload, dict):
                    self._send_json(400, {"error": "bad settings payload: expected an object"})
                    return
                try:
                    ok, error = deps.set_settings(payload)
                except Exception as exc:  # noqa: BLE001 -- a bad apply should 400, never 500
                    self._send_json(400, {"error": str(exc)})
                    return
                if not ok:
                    self._send_json(400, {"error": error})
                    return
                self._send_json(200, deps.get_settings())
                return

            if parsed.path == "/settings/reset":
                self._send_json(200, deps.reset_settings())
                return

            self._not_found()

        def log_message(self, fmt, *args) -> None:  # quieter default logging
            logger.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def run_aim_server(deps: AimDeps, listen: str) -> ThreadingHTTPServer:
    """Builds (does not `serve_forever` -- caller's choice) the aim HTTP server. `listen` is
    `"host:port"` (`cfg.aim.listen`, e.g. `"0.0.0.0:8090"`); port `0` binds an ephemeral port
    (tests). Split out, same shape as `bake.run_vet_server`, so tests run it in a background
    thread without touching the real port."""
    host, _, port_str = listen.rpartition(":")
    host = host or "0.0.0.0"
    handler = _make_handler(deps)
    server = ThreadingHTTPServer((host, int(port_str)), handler)
    # 04-04: `/stream.mjpg` blocks its worker thread for the connection's whole lifetime --
    # `daemon_threads` makes sure a lingering stream connection never blocks process exit.
    server.daemon_threads = True
    return server
