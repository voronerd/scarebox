"""Config loader for scarebox.

Reads a YAML config, expands ``${VAR}`` references (process env first, then an
env file), and returns nested frozen dataclasses mirroring every section of
``config/scarebox.example.yaml``. An unknown key, a missing key, or a missing
env var all raise :class:`ConfigError` naming the offending key/var.

Installation overlay: if a ``local.yaml`` sits in the same directory as the loaded
config, it is deep-merged over it before validation. It is git-ignored and holds the
values only one installation needs (the camera's MAC, the networks the aim page trusts);
``config/local.example.yaml`` documents its keys. A missing or empty overlay changes
nothing; an overlay that is not a mapping, or names a key the schema doesn't know, raises
:class:`ConfigError`.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import Any

import yaml

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(Exception):
    """Raised for any malformed or incomplete scarebox config."""


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    host: str
    mac: str
    user: str
    password: str
    rtsp_path: str
    set_clock_on_start: bool


@dataclasses.dataclass(frozen=True)
class SourceConfig:
    kind: str  # usb | rtsp | file
    file: str
    loop: bool
    # kind usb (2026-10-19): a webcam plugged into gbox1, found by (part of) its V4L2 name.
    # usb_dev_dir is where its device nodes are visible: /dev on the host, /host-dev inside the
    # app container (compose bind-mounts /dev there so a replugged camera is seen live).
    usb_name: str
    usb_dev_dir: str
    width: int
    height: int
    fps: int


@dataclasses.dataclass(frozen=True)
class PerceiveConfig:
    detector: str
    min_confidence: float
    trigger_line: tuple
    min_bbox_height: float
    group_band: float
    group_window_s: float
    best_crop_frames: int


@dataclasses.dataclass(frozen=True)
class JudgeConfig:
    endpoint: str
    model: str
    crop_max_px: int
    timeout_s: float
    min_confidence: float
    # 03-03 lever 2: "full" is the 03-02 six-key c/k/f/i/n/d schema; "lean" (default) drops
    # f/i/n -- see prompts.py's LEAN_JUDGE_* comment for why each is safe to drop.
    schema: str = "lean"


@dataclasses.dataclass(frozen=True)
class QuipConfig:
    endpoint: str
    model: str
    max_words: int
    roast_non_costumes: bool
    timeout_s: float
    single_call: bool
    # 03-03: `judge_and_quip()`'s own timeout -- 03-02's live-components run found the combined
    # verdict+lines call needs materially more decode budget than the shared `timeout_s`
    # (1.5s) gives it (0/20 survived, all timeout). Only used by `judge_and_quip`, never by
    # the plain two-call `quip()`.
    single_call_timeout_s: float = 2.5


@dataclasses.dataclass(frozen=True)
class CensorConfig:
    blocklist: str
    guard_endpoint: str
    guard_model: str
    fail_closed: bool
    guard_timeout_s: float = 0.6


@dataclasses.dataclass(frozen=True)
class VoiceConfig:
    engine: str
    voice: str
    reference_clip: str
    exaggeration: float
    fx_chain: str
    kokoro_dir: str = "models/Kokoro-82M"
    # Kokoro delivery speed. Martin picked variant E (2026-10-16): 1.25x + megaphone FX.
    speed: float = 1.0
    # 03-03 lever 3: emit each Kokoro pipeline segment's audio as soon as it's synthesised
    # instead of waiting for the whole line -- see voice.py's Voice.speak(). Off by default
    # (whole-clip render, unchanged 03-02 behaviour) so the hub can step the ladder without
    # code edits.
    stream_segments: bool = False
    # engine "chatterbox": synthesis runs in the separate scarebox-tts container (its own torch
    # build -- Chatterbox can't share the NGC app image), reached over loopback HTTP.
    # deploy/chatterbox/server.py owns the cloning settings (reference clip, exaggeration).
    tts_url: str = "http://127.0.0.1:8102"
    tts_timeout_s: float = 3.0


@dataclasses.dataclass(frozen=True)
class StageConfig:
    alsa_device: str
    cooldown_s: float
    queue_max: int
    stings: bool
    # 04-01: explicit sink choice. "alsa" (race default) -> `app._make_sink` must get a real
    # device or REPORT the failure (never silently substitute WavSink, 02-02's bug); "wav" ->
    # bench/CI, no device expected or probed at all.
    sink: str = "alsa"


@dataclasses.dataclass(frozen=True)
class OrchestratorConfig:
    budget_s: float
    max_line_age_s: float
    max_live_inflight: int = 1


@dataclasses.dataclass(frozen=True)
class LibraryConfig:
    manifest: str
    require_approved: bool
    # Costume packs (2026-10-16): alias table mapping judge costume strings to pack keys.
    packs: str


@dataclasses.dataclass(frozen=True)
class AimConfig:
    listen: str
    # Where the phone page's settings persist. The replay config points this elsewhere so a
    # tuning run never inherits (or overwrites) the live box's crop and gaps.
    tuning_file: str = "runs/tuning.json"
    # CIDRs allowed in without AIM_TOKEN (the race travel router's LAN). Empty = token always.
    trusted_networks: tuple = ()


@dataclasses.dataclass(frozen=True)
class WatchdogConfig:
    # 04-01: `app._watchdog` self-exits (so `restart: unless-stopped` brings a fresh
    # container up) if no camera frame has landed in `frame_timeout_s`, or the
    # frame-consuming loop hasn't ticked in `tick_timeout_s` -- Docker's healthcheck only
    # marks a container unhealthy, it never restarts one on its own.
    frame_timeout_s: float
    tick_timeout_s: float
    # 04-03: `docker pause` freezes the watchdog task too (it can't trip), so proving the
    # self-exit watchdog works under a real hang needs a software-injected one instead.
    # `false` everywhere except the soak overlay (`config/soak.yaml`) -- when `true`,
    # touching `runs/debug_block_mainloop` makes `app.main`'s per-frame loop sleep 35s once
    # (the file is removed before sleeping, so a restart never re-triggers it).
    debug_hooks: bool = False


@dataclasses.dataclass(frozen=True)
class CaptureConfig:
    # 2026-10-19: per-run tuning log (scarebox/capture.py). false in the race config: race day
    # keeps its no-recording promise. Meant for replays of a test video.
    enabled: bool
    dir: str


@dataclasses.dataclass(frozen=True)
class Config:
    camera: CameraConfig
    source: SourceConfig
    perceive: PerceiveConfig
    judge: JudgeConfig
    quip: QuipConfig
    censor: CensorConfig
    voice: VoiceConfig
    stage: StageConfig
    orchestrator: OrchestratorConfig
    library: LibraryConfig
    aim: AimConfig
    watchdog: WatchdogConfig
    capture: CaptureConfig


_SECTIONS: dict[str, type] = {
    "camera": CameraConfig,
    "source": SourceConfig,
    "perceive": PerceiveConfig,
    "judge": JudgeConfig,
    "quip": QuipConfig,
    "censor": CensorConfig,
    "voice": VoiceConfig,
    "stage": StageConfig,
    "orchestrator": OrchestratorConfig,
    "library": LibraryConfig,
    "aim": AimConfig,
    "watchdog": WatchdogConfig,
    "capture": CaptureConfig,
}


def _default_env_path() -> Path:
    """``.env`` (or ``.env.<profile>``) next to the repo root, i.e. cwd."""
    profile = os.environ.get("SCAREBOX_PROFILE", "").strip()
    name = ".env" if not profile else f".env.{profile}"
    return Path.cwd() / name


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[key] = value
    return env


def _expand(value: Any, file_env: dict[str, str]) -> Any:
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            name = match.group(1)
            if name in os.environ:
                return os.environ[name]
            if name in file_env:
                return file_env[name]
            raise ConfigError(f"missing env var: {name}")

        return _VAR_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand(v, file_env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, file_env) for v in value]
    return value


def _freeze(value: Any) -> Any:
    """Recursively turn lists into tuples so nested values stay immutable."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _build_section(cls: type, name: str, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{name}: expected a mapping")
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ConfigError(f"unknown key: {name}.{sorted(unknown)[0]}")
    optional = {f.name for f in dataclasses.fields(cls)
                if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING}
    missing = fields - set(data) - optional
    if missing:
        raise ConfigError(f"missing key: {name}.{sorted(missing)[0]}")
    values = {key: _freeze(val) for key, val in data.items()}
    return cls(**values)


def _build_config(data: Any) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("config root: expected a mapping")
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"unknown key: {sorted(unknown)[0]}")
    missing = set(_SECTIONS) - set(data)
    if missing:
        raise ConfigError(f"missing section: {sorted(missing)[0]}")
    sections = {
        name: _build_section(cls, name, data[name]) for name, cls in _SECTIONS.items()
    }
    return Config(**sections)


OVERLAY_NAME = "local.yaml"


def _read_overlay(path: Path) -> dict:
    """The ``local.yaml`` overlay as a mapping: ``{}`` when absent or empty."""
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: not valid YAML ({exc.__class__.__name__})") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path.name}: unreadable ({exc.__class__.__name__})") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: expected a mapping")
    for name, section in data.items():
        if name not in _SECTIONS:
            raise ConfigError(f"{path.name}: unknown key: {name}")
        if not isinstance(section, dict):
            raise ConfigError(f"{path.name}: {name}: expected a mapping")
        fields = {f.name for f in dataclasses.fields(_SECTIONS[name])}
        unknown = set(section) - fields
        if unknown:
            raise ConfigError(f"{path.name}: unknown key: {name}.{sorted(unknown, key=str)[0]}")
    return data


def _deep_merge(base: Any, overlay: Any) -> Any:
    """``overlay`` over ``base``: mappings merge key by key, anything else replaces."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = _deep_merge(base[key], value) if key in base else value
        return merged
    return overlay


def load_config(path: str | os.PathLike, env_file: str | os.PathLike | None = None) -> Config:
    """Load, overlay, expand and validate a scarebox YAML config into a :class:`Config`.

    ``local.yaml`` beside ``path`` (if present) is deep-merged over it first; the merged
    result is validated exactly like a base config.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    overlay_path = path.parent / OVERLAY_NAME
    if overlay_path.resolve() != path.resolve() and overlay_path.is_file():
        overlay = _read_overlay(overlay_path)
        if overlay:
            raw = _deep_merge(raw, overlay)
    env_path = Path(env_file) if env_file is not None else _default_env_path()
    file_env = _parse_env_file(env_path) if env_path.exists() else {}
    expanded = _expand(raw, file_env)
    return _build_config(expanded)
