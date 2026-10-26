"""tests/test_aim.py (04-02 base, 04-04 upgrade): pure-logic unit tests + a real-socket
integration fixture, same shape as tests/test_bake.py's vet-server tests. Every camera/cv2
operation is a fake callable injected via `AimDeps` -- aim.py itself never imports cv2, so
this file runs on a bare CPU-only venv (no GPU deps) same as the rest of the pure-logic suite.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from scarebox.aim import (
    AimDeps,
    AimError,
    _client_trusted,
    check_token,
    load_trigger_line,
    run_aim_server,
)


def _get(url, token=None, path="/"):
    full = f"{url}{path}"
    if token is not None:
        sep = "&" if "?" in full else "?"
        full = f"{full}{sep}token={token}"
    return urllib.request.urlopen(full, timeout=5)


def _post(url, path, token, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}{path}?token={token}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=5)


@pytest.fixture
def aim_server(tmp_path):
    trigger_path = tmp_path / "trigger_line.json"
    calls = {"ptz": [], "test": 0}
    state = {"trigger_line": ((0.5, 0.0), (0.5, 1.0))}

    # 04-04: a tiny in-memory stand-in for `app.EventRing` -- real crop bytes, real cap,
    # NEVER touches disk, same contract the real one has to meet.
    events_store: dict[str, dict] = {}
    order: list[str] = []
    crops: dict[str, bytes] = {}

    def events_fn():
        return [events_store[eid] for eid in order]

    def event_crop_fn(event_id):
        return crops.get(event_id)

    # 04-04: a tiny in-memory settings stand-in -- real validate/clamp semantics, exercised
    # through the HTTP layer the same way `app.py`'s real `Tuning`-backed closures would be.
    settings = {
        "values": {"stage_cooldown_s": 1.0, "min_bbox_height": 0.25},
        "defaults": {"stage_cooldown_s": 1.0, "min_bbox_height": 0.25},
        "ranges": {"stage_cooldown_s": [0.0, 10.0], "min_bbox_height": [0.05, 0.9]},
    }

    def get_settings_fn():
        return settings

    def set_settings_fn(patch):
        for key, value in patch.items():
            if key not in settings["ranges"]:
                return False, f"unknown setting: {key}"
            lo, hi = settings["ranges"][key]
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False, f"{key}: expected a number"
            settings["values"][key] = min(hi, max(lo, value))
        return True, None

    def reset_settings_fn():
        settings["values"] = dict(settings["defaults"])
        return settings

    def ptz_nudge(direction, speed, duration_s):
        calls["ptz"].append((direction, speed, duration_s))

    def set_trigger_line(line):
        state["trigger_line"] = line

    def test_line():
        calls["test"] += 1
        return "boo!"

    deps = AimDeps(
        token="secret-token",
        snapshot_jpeg=lambda: b"\xff\xd8fakejpeg",
        stream_frame_jpeg=lambda: b"\xff\xd8streamjpeg",
        ptz_nudge=ptz_nudge,
        get_trigger_line=lambda: state["trigger_line"],
        set_trigger_line=set_trigger_line,
        events=events_fn,
        event_crop=event_crop_fn,
        test_line=test_line,
        status=lambda: {"ok": True},
        get_settings=get_settings_fn,
        set_settings=set_settings_fn,
        reset_settings=reset_settings_fn,
        trigger_line_path=trigger_path,
    )
    server = run_aim_server(deps, "127.0.0.1:0")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    harness = {
        "events_store": events_store,
        "order": order,
        "crops": crops,
        "calls": calls,
        "state": state,
        "settings": settings,
        "deps": deps,
    }
    yield f"http://127.0.0.1:{port}", trigger_path, harness
    server.shutdown()
    thread.join(timeout=5)


def _add_event(harness, event_id, crop=None, **fields):
    card = {"id": event_id, **fields}
    harness["events_store"][event_id] = card
    harness["order"].append(event_id)
    if crop is not None:
        harness["crops"][event_id] = crop


# -- check_token (pure logic) ----------------------------------------------------------------


def test_check_token_rejects_empty_expected():
    with pytest.raises(AimError):
        check_token("x", "")


def test_check_token_rejects_none_provided():
    assert check_token(None, "secret") is False


def test_check_token_rejects_mismatch():
    assert check_token("wrong", "secret") is False


def test_check_token_accepts_match():
    assert check_token("secret", "secret") is True


# -- token required on every route (plan's drift check 1) ------------------------------------


def test_root_requires_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(base, timeout=5)
    assert exc_info.value.code == 403


def test_trusted_network_needs_no_token_but_others_still_do(aim_server):
    # Martin, 2026-10-23: phones on the race travel router's Wi-Fi open the page with no token.
    base, _, harness = aim_server
    harness["deps"].trusted_networks = ("127.0.0.0/8",)
    assert _get(base).status == 200
    harness["deps"].trusted_networks = ("198.51.100.0/24",)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(base, timeout=5)
    assert exc_info.value.code == 403


@pytest.mark.parametrize("ip, nets, ok", [
    ("198.51.100.40", ("198.51.100.0/24",), True),
    ("203.0.113.20", ("198.51.100.0/24",), False),
    ("198.51.100.40", (), False),
    ("not-an-ip", ("0.0.0.0/0",), False),
    ("198.51.100.40", ("garbage", "198.51.100.0/24"), True),
])
def test_client_trusted(ip, nets, ok):
    assert _client_trusted(ip, nets) is ok


def test_root_rejects_wrong_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(base, token="nope")
    assert exc_info.value.code == 403


def test_root_accepts_correct_token_serves_html(aim_server):
    base, *_ = aim_server
    resp = _get(base, token="secret-token")
    assert resp.status == 200
    assert b"scarebox" in resp.read().lower()


def test_snapshot_requires_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base}/snapshot.jpg", timeout=5)
    assert exc_info.value.code == 403


def test_snapshot_returns_injected_jpeg_bytes(aim_server):
    base, *_ = aim_server
    resp = _get(base, token="secret-token", path="/snapshot.jpg")
    assert resp.status == 200
    assert resp.headers.get("Content-Type") == "image/jpeg"
    assert resp.read().startswith(b"\xff\xd8")


def test_ptz_requires_token(aim_server):
    base, *_ = aim_server
    data = json.dumps({"dir": "Left"}).encode()
    req = urllib.request.Request(f"{base}/ptz", data=data, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 403


def test_status_requires_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base}/status", timeout=5)
    assert exc_info.value.code == 403


# -- trigger POST persists AND reloads (plan's required Task 1 test) -------------------------


def test_trigger_post_persists_to_disk(aim_server):
    base, trigger_path, _harness = aim_server
    resp = _post(base, "/trigger", "secret-token", {"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.8})
    assert resp.status == 200
    on_disk = load_trigger_line(trigger_path)
    assert on_disk == ((0.1, 0.2), (0.9, 0.8))


def test_trigger_post_updates_in_memory_line(aim_server):
    base, _trigger_path, harness = aim_server
    _post(base, "/trigger", "secret-token", {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})
    assert harness["state"]["trigger_line"] == ((0.0, 0.0), (1.0, 1.0))


def test_trigger_line_reloads_after_restart(tmp_path):
    """The persistence half of "persists and reloads", independent of the server: a fresh
    `load_trigger_line` call (as `app.py`'s next startup would make) sees exactly what the
    previous session's `/trigger` POST wrote."""
    from scarebox.aim import save_trigger_line

    path = tmp_path / "trigger_line.json"
    assert load_trigger_line(path) is None  # nothing saved yet
    save_trigger_line(((0.2, 0.3), (0.7, 0.6)), path)
    assert load_trigger_line(path) == ((0.2, 0.3), (0.7, 0.6))


def test_trigger_post_rejects_bad_payload(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post(base, "/trigger", "secret-token", {"x1": 0.1})
    assert exc_info.value.code == 400


def test_trigger_requires_token(aim_server):
    base, *_ = aim_server
    data = json.dumps({"x1": 0, "y1": 0, "x2": 1, "y2": 1}).encode()
    req = urllib.request.Request(f"{base}/trigger", data=data, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 403


# -- events: RAM-only ring buffer, no image data in the JSON (04-04) --------------------------


def test_events_returns_cards_no_image_data(aim_server):
    base, _trigger_path, harness = aim_server
    _add_event(
        harness,
        "a1",
        crop=b"\xff\xd8realcropbytes",
        category="ghost",
        costume="ghost",
        path="library",
        line="boo",
    )
    _add_event(harness, "a2", category="group", costume="", path="dropped", line=None)
    resp = _get(base, token="secret-token", path="/events")
    body = json.loads(resp.read())
    assert len(body["events"]) == 2
    assert body["events"][0]["id"] == "a1"
    blob = json.dumps(body)
    for forbidden in ("realcropbytes", "\\xff\\xd8"):
        assert forbidden not in blob


def test_events_requires_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base}/events", timeout=5)
    assert exc_info.value.code == 403


def test_events_empty_when_nothing_happened_yet(aim_server):
    base, *_ = aim_server
    resp = _get(base, token="secret-token", path="/events")
    assert json.loads(resp.read()) == {"events": []}


# -- event crop endpoint (04-04) ---------------------------------------------------------------


def test_event_crop_returns_injected_jpeg(aim_server):
    base, _trigger_path, harness = aim_server
    _add_event(harness, "e1", crop=b"\xff\xd8thecropbytes")
    resp = _get(base, token="secret-token", path="/event/e1.jpg")
    assert resp.status == 200
    assert resp.headers.get("Content-Type") == "image/jpeg"
    assert resp.read() == b"\xff\xd8thecropbytes"


def test_report_renders_resolved_events_with_inline_images():
    # 2026-10-19: the HCA detection report. Self-contained (data: URI images), newest first,
    # skipped/unresolved events left out, model text escaped.
    from scarebox.aim import render_report

    cards = [
        {"id": "a", "resolved": True, "path": "library", "is_costume": True, "costume": "Spider-Man costume",
         "category": "superhero", "pack": "spider-man", "pack_match_display": "Spider-Man",
         "line": "Sprint, Spider-Man <b>now</b>!", "score": 3, "ms": {"judge": 800, "total": 3000}},
        {"id": "b", "resolved": True, "path": "skipped"},
        {"id": "c", "resolved": False, "path": None},
        {"id": "d", "resolved": True, "path": "library", "is_costume": False, "line": "Nearly there!", "score": 1},
    ]
    html_out = render_report(cards, lambda eid, kind: b"\xff\xd8jpg" if kind == "wide" else None)
    assert "data:image/jpeg;base64," in html_out
    assert "Spider-Man" in html_out and "Nearly there!" in html_out
    assert "&lt;b&gt;now&lt;/b&gt;" in html_out  # escaped, not live HTML
    assert html_out.index("Nearly there!") < html_out.index("Sprint, Spider-Man")  # newest first
    assert "2 detections" in html_out  # skipped + unresolved left out


def test_report_route_requires_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base}/report.html", timeout=5)
    assert exc_info.value.code == 403
    resp = _get(base, token="secret-token", path="/report.html")
    assert resp.status == 200 and b"detection report" in resp.read()


def test_event_crop_404s_for_unknown_id(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(base, token="secret-token", path="/event/nonexistent.jpg")
    assert exc_info.value.code == 404


def test_event_crop_requires_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base}/event/e1.jpg", timeout=5)
    assert exc_info.value.code == 403


# -- /stream.mjpg: boundary, content type, client cap (04-04) ----------------------------------


def test_stream_requires_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base}/stream.mjpg", timeout=5)
    assert exc_info.value.code == 403


def test_stream_returns_multipart_boundary_and_frame(aim_server):
    base, *_ = aim_server
    resp = _get(base, token="secret-token", path="/stream.mjpg")
    assert resp.status == 200
    content_type = resp.headers.get("Content-Type")
    assert "multipart/x-mixed-replace" in content_type
    assert "boundary=" in content_type
    boundary = content_type.split("boundary=")[1]
    chunk = resp.read(200)
    assert boundary.encode() in chunk
    assert b"image/jpeg" in chunk
    resp.close()


def test_stream_rejects_a_third_concurrent_client(aim_server):
    base, *_ = aim_server
    resp1 = _get(base, token="secret-token", path="/stream.mjpg")
    resp1.read(50)
    resp2 = _get(base, token="secret-token", path="/stream.mjpg")
    resp2.read(50)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(base, token="secret-token", path="/stream.mjpg")
        assert exc_info.value.code == 503
    finally:
        resp1.close()
        resp2.close()


# -- ptz nudge ---------------------------------------------------------------------------------


def test_ptz_post_calls_nudge_with_seconds(aim_server):
    base, _trigger_path, harness = aim_server
    resp = _post(base, "/ptz", "secret-token", {"dir": "Left", "speed": 2, "ms": 200})
    assert resp.status == 200
    assert harness["calls"]["ptz"] == [("Left", 2, 0.2)]


def test_ptz_rejects_invalid_direction(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post(base, "/ptz", "secret-token", {"dir": "Sideways"})
    assert exc_info.value.code == 400


# -- test button --------------------------------------------------------------------------------


def test_test_button_enqueues_a_line(aim_server):
    base, _trigger_path, harness = aim_server
    resp = _post(base, "/test", "secret-token", {})
    body = json.loads(resp.read())
    assert body["queued"] is True
    assert body["line"] == "boo!"
    assert harness["calls"]["test"] == 1


# -- status --------------------------------------------------------------------------------------


def test_status_returns_deps_status(aim_server):
    base, *_ = aim_server
    resp = _get(base, token="secret-token", path="/status")
    assert json.loads(resp.read()) == {"ok": True}


def test_unknown_path_is_404(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(base, token="secret-token", path="/nonsense")
    assert exc_info.value.code == 404


# -- settings: GET/POST/reset (04-04) -----------------------------------------------------------


def test_settings_get_requires_token(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base}/settings", timeout=5)
    assert exc_info.value.code == 403


def test_settings_get_returns_values_defaults_ranges(aim_server):
    base, *_ = aim_server
    resp = _get(base, token="secret-token", path="/settings")
    body = json.loads(resp.read())
    assert "values" in body
    assert "defaults" in body
    assert "ranges" in body
    assert body["values"]["stage_cooldown_s"] == 1.0


def test_settings_post_applies_and_clamps(aim_server):
    base, _trigger_path, harness = aim_server
    resp = _post(base, "/settings", "secret-token", {"stage_cooldown_s": 99.0})
    assert resp.status == 200
    body = json.loads(resp.read())
    assert body["values"]["stage_cooldown_s"] == 10.0  # clamped to the range's hi
    assert harness["settings"]["values"]["stage_cooldown_s"] == 10.0


def test_settings_post_rejects_unknown_key(aim_server):
    base, *_ = aim_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post(base, "/settings", "secret-token", {"not_a_real_setting": 1})
    assert exc_info.value.code == 400


def test_settings_post_requires_token(aim_server):
    base, *_ = aim_server
    data = json.dumps({"stage_cooldown_s": 2.0}).encode()
    req = urllib.request.Request(f"{base}/settings", data=data, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 403


def test_settings_reset_restores_defaults(aim_server):
    base, _trigger_path, harness = aim_server
    _post(base, "/settings", "secret-token", {"stage_cooldown_s": 5.0})
    resp = urllib.request.urlopen(
        urllib.request.Request(
            f"{base}/settings/reset?token=secret-token", data=b"{}", method="POST"
        ),
        timeout=5,
    )
    body = json.loads(resp.read())
    assert body["values"]["stage_cooldown_s"] == 1.0


def test_settings_reset_requires_token(aim_server):
    base, *_ = aim_server
    req = urllib.request.Request(f"{base}/settings/reset", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 403
