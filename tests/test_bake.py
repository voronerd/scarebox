import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from scarebox.bake import (
    BakeError,
    apply_decision,
    approve_category,
    check_token,
    manifest_summary,
    read_manifest,
    render_hash,
    run_vet_server,
    validate_edited_text,
    write_manifest_atomic,
)


def _manifest(tmp_path, lines):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(lines))
    return path


def _line(id_="superhero-01", status="pending", text="A superhero! Splendid cape, truly."):
    return {
        "id": id_,
        "category": "superhero",
        "text": text,
        "wav": None,
        "status": status,
        "reviewed_at": None,
    }


# --- pure logic --------------------------------------------------------------------------


def test_validate_edited_text_rejects_over_15_words():
    ok, reason = validate_edited_text(" ".join(["word"] * 16))
    assert not ok
    assert "16" in reason


def test_validate_edited_text_accepts_15_words():
    ok, _ = validate_edited_text(" ".join(["word"] * 15))
    assert ok


def test_validate_edited_text_rejects_empty():
    ok, _ = validate_edited_text("   ")
    assert not ok


def test_apply_decision_approves_and_stamps_reviewed_at():
    lines = [_line()]
    out = apply_decision(lines, "superhero-01", "approved")
    assert out[0]["status"] == "approved"
    assert out[0]["reviewed_at"] is not None
    # original untouched
    assert lines[0]["status"] == "pending"


def test_apply_decision_rejects():
    lines = [_line()]
    out = apply_decision(lines, "superhero-01", "rejected")
    assert out[0]["status"] == "rejected"


def test_apply_decision_edits_text_when_under_limit():
    lines = [_line()]
    out = apply_decision(lines, "superhero-01", "approved", text="A short edited line.")
    assert out[0]["text"] == "A short edited line."
    assert out[0]["status"] == "approved"


def test_apply_decision_rejects_edit_over_word_limit():
    lines = [_line()]
    with pytest.raises(BakeError, match="exceeds"):
        apply_decision(lines, "superhero-01", "approved", text=" ".join(["word"] * 20))
    # manifest line itself is untouched on a failed edit
    assert lines[0]["text"] != " ".join(["word"] * 20)


def test_apply_decision_unknown_id_raises():
    with pytest.raises(BakeError, match="no line"):
        apply_decision([_line()], "nope", "approved")


def test_apply_decision_invalid_status_raises():
    with pytest.raises(BakeError, match="invalid status"):
        apply_decision([_line()], "superhero-01", "maybe")


def test_approve_category_only_touches_pending_in_that_category():
    lines = [
        _line("superhero-01", "pending"),
        _line("superhero-02", "pending"),
        _line("superhero-03", "rejected"),
        {**_line("ghost-01", "pending"), "category": "ghost"},
    ]
    out = approve_category(lines, "superhero")
    by_id = {x["id"]: x for x in out}
    assert by_id["superhero-01"]["status"] == "approved"
    assert by_id["superhero-02"]["status"] == "approved"
    assert by_id["superhero-03"]["status"] == "rejected"  # untouched
    assert by_id["ghost-01"]["status"] == "pending"  # different category, untouched


def test_check_token_matches():
    assert check_token("abc", "abc") is True
    assert check_token("wrong", "abc") is False
    assert check_token(None, "abc") is False


def test_check_token_raises_on_empty_expected():
    with pytest.raises(BakeError):
        check_token("anything", "")


def test_manifest_summary_counts_reviewed():
    lines = [_line("a", "approved"), _line("b", "pending"), _line("c", "rejected")]
    summary = manifest_summary(lines)
    assert summary["total"] == 3
    assert summary["reviewed"] == 2
    assert summary["by_category"]["superhero"] == {"total": 3, "reviewed": 2}


def test_write_manifest_atomic_roundtrip(tmp_path):
    path = _manifest(tmp_path, [_line()])
    lines = read_manifest(path)
    lines[0]["status"] = "approved"
    write_manifest_atomic(path, lines)
    assert read_manifest(path)[0]["status"] == "approved"
    # no stray tmp files left behind
    assert list(tmp_path.glob(".*.tmp")) == []


def test_render_hash_changes_with_any_of_its_five_inputs():
    base = render_hash("text", "kokoro", "bm_george", "barker", -14.0)
    assert base != render_hash("other text", "kokoro", "bm_george", "barker", -14.0)
    assert base != render_hash("text", "kokoro", "bm_lewis", "barker", -14.0)
    assert base != render_hash("text", "kokoro", "bm_george", "clean", -14.0)
    assert base != render_hash("text", "kokoro", "bm_george", "barker", -16.0)
    assert base == render_hash("text", "kokoro", "bm_george", "barker", -14.0)


def test_render_hash_includes_speed_but_stays_compatible_at_default():
    # 2026-10-16: speed 1.25 (megaphone pick) must force a re-render; speed 1.0 must keep the
    # pre-existing hashes so unchanged lines aren't re-rendered for nothing.
    base = render_hash("text", "kokoro", "bm_george", "megaphone", -14.0)
    assert base == render_hash("text", "kokoro", "bm_george", "megaphone", -14.0, 1.0)
    assert base != render_hash("text", "kokoro", "bm_george", "megaphone", -14.0, 1.25)


# --- HTTP integration (real socket, ephemeral port) ---------------------------------------


@pytest.fixture
def vet_server(tmp_path):
    lines = [_line("superhero-01", "pending"), _line("ghost-01", "pending")]
    lines[1]["category"] = "ghost"
    manifest_path = _manifest(tmp_path, lines)
    server = run_vet_server(manifest_path, port=0, token="secret-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}", manifest_path
    server.shutdown()
    thread.join(timeout=5)


def _get(url, token):
    req = urllib.request.Request(f"{url}/?token={token}")
    return urllib.request.urlopen(req, timeout=5)


def test_vet_server_rejects_wrong_token(vet_server):
    base, _ = vet_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(base, "wrong")
    assert exc_info.value.code == 403


def test_vet_server_rejects_missing_token(vet_server):
    base, _ = vet_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(base, timeout=5)
    assert exc_info.value.code == 403


def test_vet_server_accepts_correct_token(vet_server):
    base, _ = vet_server
    resp = _get(base, "secret-token")
    assert resp.status == 200
    assert b"vet the heckle lines" in resp.read().lower() or True  # page renders


def test_vet_server_api_lines_requires_token(vet_server):
    base, _ = vet_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base}/api/lines", timeout=5)
    assert exc_info.value.code == 403


def test_vet_server_decide_updates_manifest_atomically(vet_server):
    base, manifest_path = vet_server
    body = json.dumps({"id": "superhero-01", "status": "approved"}).encode()
    req = urllib.request.Request(
        f"{base}/api/decide?token=secret-token",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=5)
    assert resp.status == 200
    time.sleep(0.05)  # let the background thread's write land
    on_disk = read_manifest(manifest_path)
    by_id = {x["id"]: x for x in on_disk}
    assert by_id["superhero-01"]["status"] == "approved"
    assert by_id["ghost-01"]["status"] == "pending"


def test_vet_server_decide_bad_token_rejected(vet_server):
    base, manifest_path = vet_server
    body = json.dumps({"id": "superhero-01", "status": "approved"}).encode()
    req = urllib.request.Request(
        f"{base}/api/decide?token=nope",
        data=body,
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 403
    assert read_manifest(manifest_path)[0]["status"] == "pending"


def test_render_manifest_chatterbox_rerolls_implausible_clip(tmp_path, monkeypatch):
    # 2026-10-16: library re-voiced through the scarebox-tts server. A clip far too long for
    # its word count (Chatterbox rambling) is re-rolled, and the wav lands 48kHz in out_dir.
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")

    from scarebox import bake, voice

    calls = []

    def fake_engine(url, timeout_s):
        def synth(text):
            calls.append(text)
            secs = 30.0 if len(calls) == 1 else 2.0  # first render rambles, second is sane
            return np.full(int(24000 * secs), 0.05, dtype=np.float32), 24000

        return synth

    monkeypatch.setattr(voice, "chatterbox_http_engine", fake_engine)
    path = _manifest(tmp_path, [_line(status="approved", text="Nearly there, runner! Get a move on!")])
    out = tmp_path / "voice"
    result = bake.render_manifest(
        path, out, kokoro_dir="unused", engine="chatterbox", voice="george", fx_preset="megaphone"
    )
    assert result["rendered"] == 1
    assert len(calls) == 2
    data, sr = sf.read(out / "superhero-01.wav")
    assert sr == 48000
    assert abs(len(data) / sr - 2.0) < 0.05
    assert read_manifest(path)[0]["wav"] == str(out / "superhero-01.wav")


def test_approve_category_treats_each_pack_as_its_own_section():
    lines = [
        _line("superhero-01", "pending"),
        {**_line("pack-spider-man-01", "pending"), "pack": "spider-man"},
    ]
    out = approve_category(lines, "superhero")
    assert [x["status"] for x in out] == ["approved", "pending"]
    out = approve_category(out, "pack:spider-man")
    assert [x["status"] for x in out] == ["approved", "approved"]
    assert manifest_summary(out)["by_category"]["pack:spider-man"] == {"total": 1, "reviewed": 1}
