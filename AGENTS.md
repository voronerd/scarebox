# AGENTS.md: adapting scarebox

For a coding agent (or a person) moving scarebox onto hardware other than the author's. Read
`README.md` first for what the thing is. This file says where every assumption about the
author's box lives, how to test a change, and which rules are not yours to change.

## Terms

- **gbox1**: the author's box, an ASUS Ascent GX10: NVIDIA GB10 chip, aarch64 (64-bit ARM)
  CPU, 121 GiB of memory shared between CPU and GPU ("unified memory"). Every box-specific
  value in this repo was measured there.
- **Event**: one runner (or a group) crossing the trigger line; the unit of work for the whole
  pipeline (`scarebox/events.py`).
- **Verdict**: the vision model's answer for an event: costume category, costume words, a short
  description (`scarebox/events.py`, schema in `scarebox/prompts.py`).
- **Category**: one of the 25 fixed costume categories in `scarebox/prompts.py` `CATEGORIES`
  (plus `meta` for the self-test's status lines). It is the join key between the judge and the
  line library. Never rename one without renaming it in `library/manifest.json` too, and that
  file is not yours to edit (see the rules below).
- **Library**: `library/manifest.json`, 1,809 human-approved lines, each with a category, text,
  review status and the path its rendered WAV is written to. `library/costume_packs.json` maps
  specific costumes (Spider-Man, Elsa, ...) to aliases so the picker can prefer lines that name
  them.
- **Live line**: a line written at race time by `quip.py`, which must pass `censor.py`.
- **Overlay**: `config/local.yaml`, git-ignored, deep-merged over whichever config is loaded.
  `config/local.example.yaml` documents every key. It holds installation values (camera MAC,
  trusted networks); secrets go in `.env` (template: `.env.example`).
- **Sink**: where audio goes: `alsa` (a real device, race), `wav` (files, bench), `timed`
  (silent but busy for each line's real length, replay).

## Module map

All under `scarebox/`. The pipeline order is ingest, perceive, judge, quip (with censor),
voice, stage; `orchestrator.py` runs it per event and `app.py` wires it up.

| Module | What it does | Heavy imports? |
|---|---|---|
| `__main__.py` | CLI: `python -m scarebox run --config <yaml> [--mode live\|library]` | no |
| `config.py` | Loads a YAML config, expands `${VAR}` from the environment and `.env`, merges the overlay, validates into frozen dataclasses. Unknown or missing keys are errors. | no |
| `events.py` | `Event`, `Verdict`, `LineResult` dataclasses shared by every stage | no |
| `ingest.py` | Frame sources: `UsbSource` (webcam found by V4L2 name), `RtspSource` (IP camera), `FileSource` (MP4 replay). Keeps only the newest frame so nothing lags. | PyAV, OpenCV for USB |
| `camera.py` | Dahua/Amcrest CGI client (clock sync, encoder config, pan/tilt) for the RTSP camera only | no |
| `discover.py` | Finds an RTSP camera by MAC on whatever /24 the box is on (port 554 probe, then the ARP table) | no |
| `perceive.py` | YOLOX-s person detector (vendored in `scarebox/vendor/yolox/`), ByteTrack tracker (from `supervision`), trigger line, grouping, best crop | torch, cv2, supervision |
| `judge.py` | One vision-model call per event: crop in, JSON-schema-constrained `Verdict` out. Any error returns `None` (library fallback). | no (HTTP) |
| `quip.py` | Writes the live line with the same vision model (text only), few-shot from the library, filters candidates through the censor | no (HTTP) |
| `prompts.py` | Persona prompt, categories, JSON schemas for judge and quip | no |
| `censor.py` | Two-layer fail-closed content filter: blocklist (`config/blocklist.txt`) then Granite Guardian over HTTP | no (HTTP) |
| `voice.py` | Streaming TTS: Chatterbox over HTTP (`deploy/chatterbox/server.py`) or Kokoro in-process, then the FX chain | Kokoro only if used |
| `fx.py` | The loud-hailer / barker voice effect and offline loudness normalisation, numpy and scipy only | no |
| `library.py` | Picks an approved, rendered line by category or costume pack, avoiding recent repeats | no |
| `stage.py` | Audio-out queue: one line at a time, cooldown, drops stale lines, plays via `aplay` | no |
| `orchestrator.py` | Per-event state machine racing the live path against `orchestrator.budget_s`, falling back to the library | no |
| `selftest.py` | Start-up checks (camera, judge, censor, voice, audio device); speaks one status line | no |
| `app.py` | `scarebox run`: wires everything, the watchdog, the heartbeat file, the aim page | cv2, torch |
| `aim.py` + `aim.html` | The phone "control room" on port 8090: live MJPEG, trigger-line drag, settings, event cards | no (cv2 injected by `app.py`) |
| `tuning.py` | The mutable settings the aim page can change mid-run, saved to `runs/tuning.json` | no |
| `capture.py` | Per-run timeline log for replays of test footage. Off in the race config. | no |
| `report_pdf.py` | Detection report as a compact PDF | matplotlib |
| `bake.py` + `vet_page.html` | Line tooling: `vet` (phone page to approve lines), `pull`, `render` (voice the library) | soundfile, TTS |

The pure-logic modules stay importable without OpenCV, torch or a GPU, and that is what lets the
test suite run on a CPU machine. Keep it that way: new hardware code takes its dependency as an
injected callable (the pattern `judge.py`, `quip.py`, `voice.py`, `aim.py` and `discover.py`
already use), and heavy imports happen inside the function that needs them.

Other directories: `deploy/` (Dockerfiles, `compose.yaml`, Chatterbox server, `deploy/host/`
files for the host OS), `config/` (the configs), `scripts/` (staging, benchmarks, tuning and
replay tools, the leak gate and the export), `homelab/` (the author's own box-switching scripts,
an example only), `docs/` (runbook, hardware, race report, design, models, decision records).

## Configs

| File | Use |
|---|---|
| `config/field.yaml` | The race config: USB webcam, real ALSA dongle, approved lines only, capture off |
| `config/scarebox.example.yaml` | The annotated reference for every key; copy it to start your own |
| `config/replay.yaml` | Your own MP4 through the real pipeline, silent `timed` sink, capture on |
| `config/bench.yaml`, `config/soak.yaml` | Bench and long-run soak on a file source, `wav` sink |
| `config/local.example.yaml` | The overlay template: copy to `config/local.yaml` and fill in |
| `config/blocklist.txt` | The censor's deterministic blocklist. Not yours to edit. |

`deploy/compose.yaml` picks the config through `SCAREBOX_CONFIG` (default
`config/scarebox.example.yaml`).

## Every hardware and box assumption, and where it lives

### The chip: GB10, aarch64, CUDA, unified memory

- **Container images are arm64.** `deploy/images.env` pins `IMAGE_VLLM` (NVIDIA NGC vLLM) and
  `IMAGE_TORCH` (NVIDIA NGC PyTorch) by digest, arm64 builds. On x86_64, replace both digests
  with the amd64 builds of the same tags (or newer ones) and re-run the benchmarks.
- **Never reinstall torch from PyPI in the app image.** `deploy/Dockerfile.app` builds on
  `IMAGE_TORCH` because its CUDA-matched torch is the only one that worked on aarch64. That is
  also why the pip `yolox` package isn't used (its C++ NMS extension doesn't build on aarch64);
  the model definition is vendored in `scarebox/vendor/yolox/` and NMS comes from torchvision.
- **The Chatterbox image is pinned to CUDA 13.** `deploy/Dockerfile.chatterbox` starts from
  `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04`, installs a matched `torch==2.11.0` and
  `torchaudio==2.11.0` from the `cu130` wheel index, then `chatterbox-tts==0.1.7 --no-deps`
  (its declared torch pin has no aarch64/cu130 wheel). On another GPU, pick the wheel index
  that matches your driver; keep torch and torchaudio a matched pair.
- **Memory pins assume unified memory.** In `deploy/compose.yaml`, `vlm` runs with
  `--gpu-memory-utilization 0.3 --kv-cache-memory-bytes 4294967296 --max-model-len 4096
  --max-num-seqs 4`, and `guard` with `0.15`, `1073741824` and `--max-model-len 2048`. On GB10
  vLLM's start-up check measures against the whole shared pool, so the fractions are small and
  the KV cache is set in bytes. On a discrete card these fractions are of VRAM: both models plus
  the Chatterbox and app processes must fit on the card(s). `deploy/bench/compose.vlm.yaml`
  explains the failure the byte setting avoids.
- **GPU access.** Every service reserves NVIDIA GPUs with `driver: nvidia, count: all`
  (needs the NVIDIA container toolkit). The app also runs YOLOX on the GPU (`perceive.py`
  `Detector`, `device="cuda"`, half precision).
- **Ports and host networking.** All services use `network_mode: host`: vLLM judge/quip on
  127.0.0.1:8100, guard on 127.0.0.1:8101, Chatterbox on 127.0.0.1:8102, the aim page on
  0.0.0.0:8090 (`aim.listen`), the vet page on 8091. The endpoints are in each config's
  `judge`, `quip`, `censor` and `voice` sections; move the ports in both places together.
- **Offline at run time.** `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1` for TTS) are set in
  compose. The app image bakes in spaCy's `en_core_web_sm` at build time because Kokoro's English
  G2P would otherwise download it at first use. Anything you add must work with no network.

### Models and their paths

- vLLM mounts `${HOME}/models/Qwen3-VL-4B-Instruct-FP8` and
  `${HOME}/models/granite-guardian-3.2-3b-a800m` at `/model`; served names `qwen3-vl-4b-fp8`
  and `granite-guardian` must match `judge.model`, `quip.model` and `censor.guard_model`.
- The app mounts `${HOME}/models` at `/app/models`: YOLOX-s weights at
  `models/yolox/yolox_s.pth` (`perceive.py`; `perceive.detector` names the file, and only
  `yolox_s` is implemented whatever the example config's comment suggests), Kokoro at
  `voice.kokoro_dir` (`models/Kokoro-82M`, only if `voice.engine: kokoro`).
- Chatterbox loads from `CB_MODEL` (default `/models/chatterbox`, i.e. `~/models/chatterbox`)
  and clones the voice in `CB_REF`, default `library/voice/reference_excited.wav` (ships; the
  race voice). `reference_calm.wav` and `reference.wav` are calmer alternatives; provenance in
  `library/voice/SOURCES.md`. Any replacement must be synthetic, never a real person's voice.
- `scripts/stage_models.sh` downloads the candidates into `~/models` (`SCAREBOX_MODELS`
  overrides). It assumes an `hf` CLI at `~/.venv/bin/hf` (`HF_BIN` overrides).
- Rendered library audio is not shipped. `python -m scarebox.bake render --engine chatterbox
  --out-dir library/voice/chatterbox_george --fx-preset megaphone`, run where the TTS server is
  reachable, writes it to the paths the manifest already names. Library lines without a WAV on
  disk are never picked, and the self-test's status lines are library lines too.

### The camera

- **USB webcam (race default).** `source.kind: usb` in `config/field.yaml`. `ingest.py`
  `find_usb_camera` picks the `/sys/class/video4linux` node whose name contains
  `source.usb_name` (`WB7022`, the Dell; `""` matches any) and has the lowest index, every time
  it opens, so a replug that changes `/dev/videoN` doesn't matter. It requests
  `source.width`/`height`/`fps` (1920x1080 at 30, MJPG). Ask only for a mode your camera
  supports: an unsupported mode request is the likeliest thing that knocked the author's
  webcam off the bus. Check modes with `v4l2-ctl --list-formats-ext`.
- **Container access to the webcam.** `deploy/compose.yaml` bind-mounts the host's `/dev`
  read-only at `/host-dev` (`source.usb_dev_dir`), grants video nodes with
  `device_cgroup_rules: "c 81:* rmw"`, and adds the host `video` group by number
  (`group_add: "44"`). Check your GID with `getent group video`.
- **USB power saving.** `deploy/host/99-scarebox-camera.rules` (udev) sets `power/control` to
  `on` for the webcam (`413c:c015`), its presence sensor (`413c:d001`) and the hubs between it
  and the box, by USB ID. A different camera or hub needs its IDs (from `lsusb`) here. The
  author's USB controller died once mid-stream; keep the camera and the audio dongle on
  different USB controllers.
- **RTSP camera (supported fallback).** `source.kind: rtsp`, Dahua/Amcrest-style:
  `camera.rtsp_path` (`/cam/realmonitor?channel=1&subtype=1`), digest-auth CGI in `camera.py`,
  credentials from `.env` (`CAMERA_USER`, `CAMERA_PASSWORD`, `CAMERA_HOST`). With `camera.host`
  empty, `discover.py` finds the camera by `camera.mac` from the overlay. Pan/tilt on the aim
  page only works with this source.
- **Framing.** `perceive.trigger_line`, `min_bbox_height` (a person must be a quarter of the
  frame height) and `group_band` were tuned for that webcam's view; the aim page changes the
  trigger line and several others live.

### Audio

- **ALSA card name.** `stage.alsa_device: scarebox_out` in every config. The name comes from
  `deploy/host/99-scarebox-audio.rules` (udev), which matches the GeneralPlus dongle by USB ID
  `1b3f:2008`, sets the card's `id` to `scarebox_out` and turns its autosuspend off. A different
  dongle: change both ID pairs in that rule. `stage.AlsaSink` turns the card name into
  `plughw:CARD=scarebox_out,DEV=0` and plays through `aplay` (alsa-utils, installed in
  `deploy/Dockerfile.app`).
- **Container access to sound.** `/dev/snd` is bind-mounted (not passed as `devices:`, which
  copied the nodes once and lost a replugged dongle), with `device_cgroup_rules: "c 116:* rmw"`
  and the host `audio` group by number (`group_add: "29"`; check `getent group audio`).
- **Keep the desktop off the dongle.** `deploy/host/51-scarebox-ignore-dongle.lua` (WirePlumber
  0.4.x, `main.lua.d/`) or `51-scarebox-ignore-dongle.conf` (WirePlumber 0.5) disables the
  device for PipeWire, matched by its `device.name`. A different dongle needs that name changed.
  A box with no desktop session doesn't need either.
- **Volume** is stored on the host with `sudo alsactl store scarebox_out` (80% on the author's
  dongle, so the PA's line-in doesn't distort); set the loudness on the PA.
- **Fail loudly.** The race config uses `stage.sink: alsa`, which needs a real device. Do not
  make it fall back to a silent sink: a box that runs perfectly and never says a word is the
  worst failure there is.

### The host

- **Users and paths.** Containers run as uid/gid `1001:1001` (the author's `scarebox` user).
  `deploy/host/scarebox.service` (systemd, a boot-time `compose up -d`) assumes
  `User=scarebox` and the repo at `/home/scarebox/scarebox`. Every compose command runs from the
  repo root, because the mounts use `${PWD}` and `${HOME}`.
- **Restart and watchdog.** Compose sets `restart: unless-stopped` and healthchecks; the app's
  own watchdog (`watchdog.frame_timeout_s`, `tick_timeout_s` in the config, `_watchdog` in
  `app.py`) exits the process on a hang, and its healthcheck reads `runs/heartbeat`. The
  systemd unit covers a cold power-on, which `unless-stopped` alone doesn't.
- **Host files need root.** Everything in `deploy/host/` is installed by hand with `sudo`; its
  `README.md` has the steps and the verification commands.

### The network

- **The aim page's address.** The author's box keeps one fixed address on its built-in Ethernet
  port alongside its DHCP lease, so the phone page is always at the same URL on the travel
  router. `deploy/host/90-NM-builtin-port.yaml` is a placeholder example of that
  NetworkManager profile (interface name, UUID and address are yours to fill in); the README in
  that directory gives the `nmcli` commands.
- **Token-free networks.** `aim.trusted_networks` (CIDR list) in `config/local.yaml`: clients
  there reach the aim page without `AIM_TOKEN`. Empty means the token is always required. The
  examples use the documentation range 198.51.100.0/24; use your own.
- **The author's homelab.** `homelab/gbox1-mode.sh` and `homelab/router-link-test.sh` are
  specific to a box that also serves a large model to the author's home network. Don't wire
  them into anything; see `homelab/README.md`. `scripts/sync-gbox1.sh` (git bundle over SSH to
  the box) and `python -m scarebox.bake pull` default to the author's SSH alias
  `gbox1-scarebox`; override `SCAREBOX_HOST` / `--host`.

## Testing

Run from the repo root (tests use repo-relative paths):

```bash
pip install -e '.[dev]' av
python -m pytest -q tests
```

If another copy of scarebox is installed in the same environment, put this tree first:
`PYTHONPATH=$PWD python -m pytest -q tests`.

On a CPU-only machine: 457 passed, 6 skipped. The skips are by design, each with a reason:
`tests/test_app.py` and `tests/test_perceive.py` (need cv2 and the GPU stack),
`tests/test_report_pdf.py` (matplotlib), one test in `tests/test_bake.py` (soundfile), and two
live tests in `tests/test_censor.py` (need the guard server on 127.0.0.1:8101). The full set,
skips included, runs inside the app image on the GPU box with the stack up.

What to keep true when you change things:

- The suite passes on a CPU machine with no deselects. A test that needs hardware skips with a
  reason (`pytest.importorskip`, or a reachability check like `tests/test_censor.py`), it never
  errors at collection.
- `tests/test_censor.py` and the red-team files in `tests/redteam/` stay green; they are the
  evidence that the censor still catches what it must.
- Test the real thing after a hardware change: `config/replay.yaml` with your own footage for the
  pipeline, then the self-test line on the box for the devices.

## Rules that must not change

These are the product, not implementation details. Adapt the hardware all you like; do not
weaken any of these, and if a task seems to need it, stop and ask a human.

1. **The censor is fail-closed.** `censor.Censor.check` returns `Pass` only when the blocklist
   layer finds nothing AND the guard answers, in time, that none of its six risks apply. A
   guard timeout, a guard error, an exception anywhere: `Reject`. Never add a path that returns
   `Pass` on an error, never skip the guard because it is slow or down (the orchestrator plays a
   library line instead), and leave `censor.fail_closed: true`. Never remove entries from
   `config/blocklist.txt`. Never change `scarebox/censor.py` to be more permissive.
2. **Lines are G-rated and about the costume only.** Nothing about anyone's body, size, age,
   skin, gender, disability or face. Don't edit the text of `library/manifest.json` or
   `library/costume_packs.json`. New lines go in as `pending`, are approved by a human on the
   vet page (`python -m scarebox.bake vet`), and `library.require_approved: true` in the race
   config makes the app refuse to start with any unreviewed line. Keep it `true` anywhere real
   people are in front of the camera. Don't loosen the persona prompt's safety rules in
   `scarebox/prompts.py`.
3. **No recording.** Frames and crops live in memory only; the aim page's stream, snapshots and
   event crops are never written to disk; logs hold text. `capture.enabled` stays `false` in
   any config pointed at real people (it exists for replaying your own test footage). Don't add
   frame dumps, debug image writes or audio recording to the live path.
4. **No real person's voice.** The voice is cloned from a synthetic clip
   (`library/voice/reference.wav`, synthesised with Kokoro). Don't clone a real person.
5. **Permissive licences only.** Apache-2.0, MIT, BSD or CC0 for code, models and voices.
   `docs/MODELS.md` lists what was excluded and why (for example Ultralytics YOLO is AGPL, and
   the FX chain is hand-written because the obvious audio-FX package is GPL).
6. **Offline at run time.** Nothing in the run path may need the internet.
7. **Silence is not a status.** The speaker is the only display at the race: keep the self-test
   speaking one status line, and keep the race sink failing loudly when the device is missing.

## If you publish your fork

`scripts/export_public.sh DEST` copies an allowlist of tracked files into `DEST` and runs
`scripts/leak_gate.py` over it, which fails on private-network and tailnet addresses, MAC
addresses, chat-session links, unexpected email addresses, the literal values in your `.env`
and any word in a git-ignored deny file (`scripts/leak_gate.deny`, one per line: your
hostnames, people's names). Examples in this repo use RFC 5737 documentation addresses
(192.0.2.x, 198.51.100.x, 203.0.113.x) and the locally administered MAC `02:00:00:00:00:01`;
keep yours that way.
