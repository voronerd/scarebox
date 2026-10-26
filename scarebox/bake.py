"""`scarebox bake`: the line-baking toolchain (spec §6.3).

Three subcommands, none of which needs a running pipeline:

- `vet`: a tiny token-protected HTTP page (stdlib `http.server` only, no new deps) so Martin
  can approve/reject/edit candidate lines from his phone. Decisions are written back to
  `library/manifest.json` atomically (write-tmp-then-`os.replace`) so a crash or a concurrent
  read never sees a half-written file.
- `pull`: copies the gbox1 manifest back to this host via `scp` (the lines are ours to
  publish; no API, no secrets in the payload).
- `render`: voices every eligible line (approved, plus pending if `--include-pending`) with
  Kokoro, applies `BarkerFX` + `loudnorm_offline`, and writes
  `library/voice/<engine>_<voice>/<id>.wav`, skipping a line whose render hash (text + engine
  + voice + FX preset + loudness target) is unchanged from what's already on disk (hub
  amendment, WWMD W5 -- a text-only hash would silently keep a stale voice after gate G1
  picks a different one).

The HTTP pieces are deliberately split from the pure logic (`apply_decision`,
`approve_category`, `validate_edited_text`, `check_token`, `read_manifest`,
`write_manifest_atomic`) so the vetting behaviour is unit-testable without a live socket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scarebox.prompts import MAX_WORDS

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
VET_PAGE = REPO_ROOT / "scarebox" / "vet_page.html"

_VALID_STATUSES = frozenset({"approved", "rejected", "pending"})


class BakeError(Exception):
    """Raised for any bake.py precondition failure (bad manifest, bad token, ...)."""


# --------------------------------------------------------------------------------------------
# Pure logic -- unit-testable without a socket.
# --------------------------------------------------------------------------------------------


def read_manifest(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def write_manifest_atomic(path: str | Path, lines: list[dict]) -> None:
    """Write-tmp-then-`os.replace` in the same directory, so a reader never sees a partial
    file and a crash mid-write never corrupts the on-disk manifest."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(lines, indent=2) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def validate_edited_text(text: str) -> tuple[bool, str]:
    """Spec §6.3 Task 2: "an edited line goes back through the length check"."""
    if not isinstance(text, str) or not text.strip():
        return False, "text is empty"
    words = text.split()
    if len(words) > MAX_WORDS:
        return False, f"{len(words)} words exceeds the {MAX_WORDS}-word limit"
    return True, ""


def apply_decision(
    lines: list[dict], line_id: str, status: str, text: str | None = None
) -> list[dict]:
    """Returns a NEW list with `line_id`'s status (and optionally text) updated. Raises
    `BakeError` if the id is unknown, the status is invalid, or edited text fails the length
    check -- callers must not partially apply a bad decision."""
    if status not in _VALID_STATUSES:
        raise BakeError(f"invalid status {status!r}")
    found = False
    out = []
    for line in lines:
        if line["id"] != line_id:
            out.append(line)
            continue
        found = True
        new_line = dict(line)
        if text is not None and text != line["text"]:
            ok, reason = validate_edited_text(text)
            if not ok:
                raise BakeError(f"edited text for {line_id} rejected: {reason}")
            new_line["text"] = text
        new_line["status"] = status
        new_line["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out.append(new_line)
    if not found:
        raise BakeError(f"no line with id {line_id!r}")
    return out


def vet_group(line: dict) -> str:
    """The vet page's section for a line: its costume pack ("pack:spider-man") if it has one,
    else its category. Pack lines get their own section so they're approved on their own."""
    return f"pack:{line['pack']}" if line.get("pack") else line["category"]


def approve_category(lines: list[dict], category: str) -> list[dict]:
    """"Approve all remaining in this section" -- every `pending` line whose `vet_group` is
    `category` (a category name, or "pack:<key>") becomes `approved`; lines already
    `approved`/`rejected` are untouched."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = []
    for line in lines:
        if vet_group(line) == category and line["status"] == "pending":
            new_line = dict(line)
            new_line["status"] = "approved"
            new_line["reviewed_at"] = now
            out.append(new_line)
        else:
            out.append(line)
    return out


def check_token(provided: str | None, expected: str) -> bool:
    if not expected:
        raise BakeError("AIM_TOKEN is empty -- refusing to serve an unprotected vet page")
    return provided is not None and provided == expected


def load_env_token(env_path: str | Path = REPO_ROOT / ".env") -> str:
    """`AIM_TOKEN` from the process environment if set, else parsed out of `.env`
    (`KEY=value` lines, `#` comments ignored) -- convenient for running `bake vet` directly
    without sourcing the file first."""
    if os.environ.get("AIM_TOKEN"):
        return os.environ["AIM_TOKEN"]
    env_path = Path(env_path)
    if not env_path.exists():
        raise BakeError(f"AIM_TOKEN not in env and {env_path} does not exist")
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == "AIM_TOKEN":
            return val.strip()
    raise BakeError(f"AIM_TOKEN not found in env or {env_path}")


def manifest_summary(lines: list[dict]) -> dict:
    total = len(lines)
    reviewed = sum(1 for x in lines if x["status"] != "pending")
    by_category: dict[str, dict[str, int]] = {}
    for line in lines:
        cat = by_category.setdefault(vet_group(line), {"total": 0, "reviewed": 0})
        cat["total"] += 1
        if line["status"] != "pending":
            cat["reviewed"] += 1
    return {"total": total, "reviewed": reviewed, "by_category": by_category}


# --------------------------------------------------------------------------------------------
# HTTP wiring
# --------------------------------------------------------------------------------------------


def _make_handler(manifest_path: Path, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "scarebox-bake/1.0"

        def _token_ok(self, parsed) -> bool:
            qs = parse_qs(parsed.query)
            provided = qs.get("token", [None])[0] or self.headers.get("X-Token")
            try:
                return check_token(provided, token)
            except BakeError:
                return False

        def _send_json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _forbidden(self) -> None:
            body = b"forbidden"
            self.send_response(403)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib override)
            parsed = urlparse(self.path)
            if not self._token_ok(parsed):
                self._forbidden()
                return
            if parsed.path in ("/", "/index.html"):
                body = VET_PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/lines":
                lines = read_manifest(manifest_path)
                self._send_json(200, {"lines": lines, "summary": manifest_summary(lines)})
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not self._token_ok(parsed):
                self._forbidden()
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "bad json"})
                return
            try:
                lines = read_manifest(manifest_path)
                if parsed.path == "/api/decide":
                    lines = apply_decision(
                        lines,
                        payload["id"],
                        payload["status"],
                        payload.get("text"),
                    )
                elif parsed.path == "/api/approve_category":
                    lines = approve_category(lines, payload["category"])
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                write_manifest_atomic(manifest_path, lines)
            except (BakeError, KeyError) as e:
                self._send_json(400, {"error": str(e)})
                return
            self._send_json(200, {"summary": manifest_summary(lines)})

        def log_message(self, fmt, *args) -> None:  # quieter default logging
            logger.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def run_vet_server(manifest_path: str | Path, port: int, token: str) -> ThreadingHTTPServer:
    """Builds and starts (but does not `serve_forever` -- caller's choice) the vet HTTP
    server. Split out so tests can run it in a background thread on an ephemeral port."""
    handler = _make_handler(Path(manifest_path), token)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    return server


def cmd_vet(args: argparse.Namespace) -> None:
    token = load_env_token()
    server = run_vet_server(args.manifest, args.port, token)
    logger.info("scarebox bake vet listening on :%d (manifest=%s)", args.port, args.manifest)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def cmd_pull(args: argparse.Namespace) -> None:
    """scp the gbox1 manifest back to this host's `library/manifest.json`."""
    host = args.host
    remote_path = f"{host}:{args.remote_dir}/library/manifest.json"
    local_path = REPO_ROOT / "library" / "manifest.json"
    subprocess.run(["scp", "-q", remote_path, str(local_path)], check=True)
    lines = read_manifest(local_path)
    summary = manifest_summary(lines)
    print(f"pulled {summary['total']} lines, {summary['reviewed']} reviewed, from {host}")


# --------------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------------


def render_hash(
    text: str, engine: str, voice: str, fx_preset: str, loudness: float, speed: float = 1.0
) -> str:
    """Hash of text + engine + voice + FX preset + loudness target (hub amendment, WWMD W5):
    re-rendering skips a line only when ALL five are unchanged, so a voice/FX change after
    gate G1 forces a re-render instead of silently keeping stale audio."""
    # speed joins the key only when it differs from 1.0, so existing hashes stay valid.
    key = f"{text}|{engine}|{voice}|{fx_preset}|{loudness}" + (f"|{speed}" if speed != 1.0 else "")
    key = key.encode()
    return hashlib.sha256(key).hexdigest()[:16]


def render_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    kokoro_dir: str | Path,
    engine: str = "kokoro",
    voice: str = "bm_george",
    fx_preset: str = "barker",
    target_lufs: float = -14.0,
    out_sr: int = 48000,
    speed: float = 1.0,
    include_pending: bool = False,
    progress_cb=None,
    tts_url: str = "http://127.0.0.1:8102",
) -> dict:
    """Renders every eligible line to `out_dir/<id>.wav`, updating `wav` + `render_hash` in
    the manifest. Imports torch/kokoro/scipy lazily so this module stays importable (and
    testable) on a host with no GPU/Kokoro installed -- only `cmd_render` and the actual bake
    render run need them."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    from scarebox.fx import BarkerFX, loudnorm_offline

    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if engine == "chatterbox":
        from scarebox.voice import chatterbox_http_engine

        http_synth = chatterbox_http_engine(tts_url, timeout_s=60.0)

        def synth(text: str):
            # Chatterbox samples, so a render can occasionally ramble or clip short. Re-roll
            # when the clip length is implausible for the word count (about 0.15-0.9 s/word
            # at Sir Reginald's shouting pace) -- cheap insurance across a big batch.
            words = max(len(text.split()), 1)
            for _attempt in range(3):
                audio, sr = http_synth(text)
                if 0.15 <= (len(audio) / sr) / words <= 0.9:
                    break
                logger.warning("chatterbox re-roll (%.2fs for %d words): %s", len(audio) / sr, words, text)
            return audio, sr

    else:
        import torch
        from kokoro import KModel, KPipeline

        kokoro_dir = Path(kokoro_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = (
            KModel(config=str(kokoro_dir / "config.json"), model=str(kokoro_dir / "kokoro-v1_0.pth"))
            .to(device)
            .eval()
        )
        pipeline = KPipeline(lang_code="b", model=model)
        voice_path = str(kokoro_dir / "voices" / f"{voice}.pt")

        def synth(text: str):
            chunks = [audio for _gs, _ps, audio in pipeline(text, voice=voice_path, speed=speed)]
            audio = (np.concatenate([np.asarray(c) for c in chunks]) if chunks else np.zeros(1)).astype(np.float32)
            return audio, 24000  # Kokoro's native rate

    lines = read_manifest(manifest_path)
    eligible_statuses = {"approved"} | ({"pending"} if include_pending else set())

    rendered, skipped, failed = 0, 0, 0
    for line in lines:
        if line["status"] not in eligible_statuses:
            continue
        rhash = render_hash(line["text"], engine, voice, fx_preset, target_lufs, speed)
        wav_path = out_dir / f"{line['id']}.wav"
        if line.get("render_hash") == rhash and line.get("wav") and wav_path.exists():
            skipped += 1
            continue
        try:
            audio, sr = synth(line["text"])
            # Spec §6.1/§5.3 wants 48kHz mono 16-bit output.
            g = np.gcd(out_sr, sr)
            audio_48k = resample_poly(audio, out_sr // g, sr // g).astype(np.float32)
            fx = BarkerFX(sr=out_sr, preset=fx_preset)
            audio_fx = fx.process(audio_48k)
            audio_norm = loudnorm_offline(audio_fx, out_sr, target_lufs=target_lufs)
            sf.write(wav_path, audio_norm, out_sr, subtype="PCM_16")
            # Stored as given (not relative_to(manifest_path.parent)): callers pass
            # --manifest and --out-dir both relative to the repo root (the convention every
            # config file + Library.pick()'s Path(wav).exists() check assumes), so the wav
            # path needs to stay repo-root-relative, not library/-relative.
            line["wav"] = str(wav_path)
            line["render_hash"] = rhash
            rendered += 1
            if progress_cb:
                progress_cb(line["id"])
        except Exception as e:  # noqa: BLE001 -- one bad line must not kill the whole batch
            logger.error("render failed for %s: %s", line["id"], e)
            failed += 1
        write_manifest_atomic(manifest_path, lines)

    return {"rendered": rendered, "skipped": skipped, "failed": failed, "total": len(lines)}


def cmd_render(args: argparse.Namespace) -> None:
    result = render_manifest(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        kokoro_dir=args.kokoro_dir,
        engine=args.engine,
        voice=args.voice,
        fx_preset=args.fx_preset,
        target_lufs=args.target_lufs,
        include_pending=args.include_pending,
        speed=args.speed,
        tts_url=args.tts_url,
    )
    print(json.dumps(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scarebox bake")
    sub = parser.add_subparsers(dest="command", required=True)

    vet = sub.add_parser("vet", help="serve the token-protected line vetting page")
    vet.add_argument("--manifest", default="library/manifest.json")
    vet.add_argument("--port", type=int, default=8091)
    vet.set_defaults(func=cmd_vet)

    pull = sub.add_parser("pull", help="scp the gbox1 manifest back to this host")
    pull.add_argument("--host", default="gbox1-scarebox")
    pull.add_argument("--remote-dir", default="scarebox")
    pull.set_defaults(func=cmd_pull)

    render = sub.add_parser("render", help="voice every eligible line (Kokoro, or Chatterbox via scarebox-tts) + BarkerFX")
    render.add_argument("--manifest", default="library/manifest.json")
    render.add_argument("--out-dir", default="library/voice/kokoro_bm_george")
    render.add_argument("--kokoro-dir", default=str(Path.home() / "models" / "Kokoro-82M"))
    render.add_argument("--engine", default="kokoro")
    render.add_argument("--voice", default="bm_george")
    render.add_argument("--fx-preset", default="barker")
    render.add_argument("--target-lufs", type=float, default=-14.0)
    render.add_argument("--speed", type=float, default=1.0, help="Kokoro delivery speed (1.25 = race-day megaphone pick)")
    render.add_argument("--tts-url", default="http://127.0.0.1:8102", help="engine chatterbox: the scarebox-tts server")
    render.add_argument("--include-pending", action="store_true")
    render.set_defaults(func=cmd_render)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
