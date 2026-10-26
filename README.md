# scarebox

An offline AI heckler for a Halloween fun run. A webcam watches the course, a vision model works
out what costume each runner is wearing, and a pompous English gentleman ("Sir Reginald") yells
a family-friendly one-liner about that costume through a PA speaker further down the track.

> "Frida Kahlo's artistry demands a final, glorious dash, darling!"
>
> (written on the spot for a runner dressed as Frida Kahlo, race day 2026-10-25)

Martin Sawtell built it as a volunteer for the [Hopkinton Center for the Arts](https://www.hopartscenter.org)
Halloween 5k in Hopkinton, Massachusetts: a family fun run where most of the field, kids
especially, runs in costume. It ran there for real on 2026-10-25, in a field by the course, on a
battery, with no internet, on one small GPU box.

This is a showcase that is close to runnable, not a product. It was built for one box, one
webcam and one audio dongle, and it says so wherever that matters. If you have similar hardware,
or a coding agent willing to adapt it, this README and [`AGENTS.md`](AGENTS.md) are meant to get
you there.

## Words used here

- **gbox1**: the author's box, an ASUS Ascent GX10 (NVIDIA GB10 chip, aarch64/ARM, 121 GiB of
  memory shared by CPU and GPU). The name appears throughout the repo as the example machine.
- **Library line**: one of 1,809 jokes written ahead of time, each read and approved by a human,
  and voiced ahead of time. Instant to play.
- **Live line**: a joke written on the spot by the vision model for the costume it just saw,
  checked by the safety filter, then voiced.
- **Trigger line**: a line drawn across the camera picture. A runner crossing it is an "event"
  and gets a joke.
- **Aim page**: a web page the box serves on port 8090 for a phone: live picture, a draggable
  trigger line, a TEST button and live settings.

## Race result (2026-10-25)

About two hours at the HCA Halloween 5k (10:08 to 12:09 US Eastern), unattended apart from one
fix at the start:

- **104 lines played**: 75 library lines and 29 live-written ones.
- **74 dropped**: lines that went stale before the speaker was free (the runner had gone), or
  were outranked in a crowd. Dropping is deliberate: a joke about a runner who has already
  passed is worse than silence.
- Two phones used the aim page over the travel router's Wi-Fi.
- **What broke:** at power-on the webcam's USB 3 link didn't come up. Reseating the plug fixed
  it, and the camera then held for the whole run. While the camera was missing the box stayed
  silent instead of announcing the fault; that is an open bug.

Tuning on 19 minutes of test footage beforehand raised the share of eye-catching costumes
actually named aloud from 12.4% to 31.3%. Details: [`docs/RACE-REPORT.md`](docs/RACE-REPORT.md).

## In the field

On race morning the box sat beside the course with a webcam looking up the path and a PA speaker
further down it, so the joke lands as the runner reaches the speaker rather than the camera. A
portable power station ran everything. A pocket travel router gave a phone a Wi-Fi link to the
control page (live picture, settings, a test button); it had no internet and didn't need any.
Setup was: place the camera and speaker, plug in, power on, wait about two minutes for Sir
Reginald to announce he is seated.

People loved it. The generic lines got smiles, but when it named something specific about a
costume, runners reacted with surprise and delight: it had noticed *them*. The best moment was a
runner dressed as Frida Kahlo. No pre-written line covers Frida Kahlo, so the vision model
recognised the costume and wrote the line at the top of this page on the spot, the safety filter
passed it, and Sir Reginald delivered it as she ran past.

Crowds are the hard part. A single runner is easy; a pack of kids arriving at once is not, and
one voice can only say one line every few seconds. In a crowd the box looks at each person in a
group, gives the line to the most eye-catching costume, and lets lines about runners who have
already passed the speaker go unsaid.

## Hardware it assumes

| Part | What ran on race day | How fixed is it? |
|---|---|---|
| Compute | ASUS Ascent GX10 (NVIDIA GB10, aarch64, 121 GiB unified memory) on its stock 240 W USB-C brick | Hard: the images and memory settings are for this chip. See `AGENTS.md`. |
| Camera | Dell UltraSharp webcam (WB7022), USB-C straight into the box, 1080p MJPG at 30 fps | Any UVC webcam: change `source.usb_name` and the mode. A Dahua/Amcrest IP camera over RTSP is also supported. |
| Audio | GeneralPlus USB audio dongle (USB ID `1b3f:2008`) into the PA's line-in over a 10 to 15 m cable | Any USB dongle: change two IDs in a udev rule. |
| Speaker | A PA speaker with a line input | Anything with a line input. |
| Network | A small travel router, no internet, only so a phone can reach the aim page | Optional. |
| Power | A 1000 Wh portable power station, box on its AC outlet | Anything that runs a 240 W brick for a few hours. |

Software on the box: Linux with Docker, Docker Compose and the NVIDIA container toolkit. The
models are downloaded once while online; at the race nothing touches the internet. Full parts
list, wiring and cold-boot behaviour: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## How it works

```
 webcam ─▶ ingest ─▶ perceive ──────▶ judge ────────▶ quip ────────▶ censor ───▶ voice ────▶ stage ─▶ PA
           (newest   (YOLOX person    (vision model   (same model    (blocklist  (Chatterbox  (one line
            frame     detector,        names the       writes a       + Granite   TTS + a      at a time;
            only)     ByteTrack        costume)        one-liner)     Guardian)   loud-hailer  stale lines
                      tracker,            │                              │        effect)      dropped)
                      trigger line)       │                              │
                                          └─── too slow, or rejected ────┴──▶ line picker: an approved
                                                                              library line for that
                                                                              costume, already voiced
```

- **Camera upstream of the speaker.** The webcam sits with the box; the speaker sits roughly 8 to
  20 m further down the course at the end of a long audio cable, so the runner is still arriving
  while the line is written and voiced. Every line names the costume aloud, so it lands on the
  right person even when runners move at different speeds.
- **Deadline rule.** Each runner gets a time budget (`orchestrator.budget_s`, 3.5 s). If the
  live line isn't ready in time, a library line for that costume category plays instead. Most
  lines on race day were library lines; that is the design working, not failing.
- **Two models, one GPU.** A 4-billion-parameter vision model (Qwen3-VL-4B, FP8) both names the
  costume and writes the joke. A small safety classifier (IBM Granite Guardian 3.2) checks every
  live line. Both are served by vLLM on the box. Speech is Chatterbox, cloned from a synthetic
  reference voice, then run through a loud-hailer effect written in numpy and scipy.
- **The speaker is the status display.** There's no screen at the race. On start-up a self-test
  checks the camera, the vision model, the safety filter, the voice and the speaker, then says
  "The adjudicator is seated. Let the parade commence!" or names the first thing that's broken.
- **It looks after itself.** Containers restart on failure, an internal watchdog exits the app
  if frames or the main loop stop (so the restart has something to act on), and a boot unit
  starts the whole stack from a cold power-on.

The original design is in [`docs/DESIGN.md`](docs/DESIGN.md); the race-morning procedure is
[`docs/RUNBOOK.md`](docs/RUNBOOK.md); model choices and their measurements are in
[`docs/MODELS.md`](docs/MODELS.md).

## Ground rules (non-negotiable)

- **Costume only.** Lines are about the costume, never the person: nothing about body, size,
  age, skin, gender, disability or face. Every live line passes a blocklist and a safety
  classifier, and is dropped if either objects, errors or doesn't answer in time (fail-closed).
- **G-rated.** The library is reviewed line by line, and the race config refuses to start if any
  line is unreviewed.
- **No recording.** Frames live in memory only. Logs hold text: costume label, line, timings.
- **No real person's voice.** The character voice is cloned from a clip synthesised by Kokoro.
- **Permissive licences only** (Apache-2.0, MIT, BSD, CC0) for code, models and voices.

## Quick start

### 1. Run the tests on any machine (no GPU needed)

```bash
git clone https://github.com/voronerd/scarebox && cd scarebox
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]' av        # av (PyAV) is needed by the frame-source tests
python -m pytest -q tests         # from the repo root: the tests use repo-relative paths
```

On a CPU-only machine this gives 457 passed and 6 skipped. The skips are expected:
`tests/test_app.py` and `tests/test_perceive.py` need OpenCV and the GPU stack,
`tests/test_report_pdf.py` needs matplotlib, one test in `tests/test_bake.py` needs soundfile,
and two live tests in `tests/test_censor.py` need the safety-filter server running.

### 2. Run it on a GB10-class box

Commands run from the repo root on the box. The author's layout is a `scarebox` user with the
repo at `/home/scarebox/scarebox` and model weights under `~/models`.

1. **Download the models** while online. `scripts/stage_models.sh` fetches the candidates into
   `~/models` (including some that were tried and rejected; trim its list if you like). The ones
   in use: `Qwen3-VL-4B-Instruct-FP8`, `granite-guardian-3.2-3b-a800m` and the YOLOX-s weights
   (`models/yolox/yolox_s.pth`). Chatterbox's weights go in `~/models/chatterbox`; the script
   doesn't fetch those.
2. **Get the images.** The vLLM image is pinned by digest in `deploy/images.env`. Build the
   other two:
   ```bash
   docker build -f deploy/Dockerfile.app \
     --build-arg IMAGE_TORCH=$(grep ^IMAGE_TORCH deploy/images.env | cut -d= -f2) -t scarebox/app:dev .
   docker build -t scarebox/tts-chatterbox:dev -f deploy/Dockerfile.chatterbox deploy
   ```
3. **The voice.** Chatterbox clones Sir Reginald from `library/voice/reference_excited.wav`, the
   clip the race used (it ships). Calmer alternatives are `reference_calm.wav` and
   `reference.wav`; set `CB_REF` on the `tts` service to switch. All three are synthetic
   (Kokoro `bm_george`), never a real person: `library/voice/SOURCES.md`.
4. **Secrets.** `cp .env.example .env && chmod 600 .env`, then set `AIM_TOKEN` to a long random
   string. The camera values only matter for an RTSP camera.
5. **Installation values.** `cp config/local.example.yaml config/local.yaml` and put in your own
   networks (and the camera MAC if you use an IP camera). This file is git-ignored.
6. **Host files** (root, once): the udev rules that name the dongle and keep the webcam awake,
   the WirePlumber rule that keeps the desktop's audio off the dongle, and the boot unit. All in
   `deploy/host/`, with install steps in its `README.md`.
7. **Voice the library.** The 1,809 lines ship as text; their audio is rendered on your box.
   Start the voice server, then render (this takes a while):
   ```bash
   docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env up -d tts
   docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env run --rm --no-deps app \
     python -m scarebox.bake render --engine chatterbox --out-dir library/voice/chatterbox_george --fx-preset megaphone
   ```
8. **Start it** with the race config:
   ```bash
   SCAREBOX_CONFIG=config/field.yaml docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env up -d
   ```
   Within about five minutes the speaker should say "The adjudicator is seated. Let the parade commence!" Open
   `http://<box address>:8090/?token=<AIM_TOKEN>` on a phone to aim the camera and drag the
   trigger line.

No webcam yet? `config/replay.yaml` plays your own MP4 through the real pipeline with a silent
speaker, which is how the crowd tuning was done.

## What is box-specific

These are the things that won't carry over unchanged. Each one, and the file it lives in, is in
[`AGENTS.md`](AGENTS.md):

- **The chip.** The container images are arm64 NVIDIA NGC builds, the Chatterbox image uses
  CUDA 13 PyTorch wheels, and the vLLM memory settings in `deploy/compose.yaml` assume GB10's
  unified memory. On an x86 machine with a discrete NVIDIA card, the images and those memory
  settings change.
- **The USB devices.** The webcam is found by name (`WB7022`); the dongle is matched by USB ID
  (`1b3f:2008`) in a udev rule that names it `scarebox_out`; a second udev rule keeps that webcam
  and its hubs out of USB power saving.
- **The host.** The container user is uid 1001 with the host's `audio` (29) and `video` (44)
  groups, and the boot unit assumes the `/home/scarebox/scarebox` layout.
- **The network.** A fixed address on the box's Ethernet port for the aim page, and the networks
  that page trusts without a token, live in your `config/local.yaml`.
- **The author's homelab.** `homelab/` holds scripts that switch the author's box between its
  everyday job and race duty. They are an example only; a box that runs nothing but scarebox
  doesn't need them.

## How it was built

Most of it was built in about a day of "vibe coding": Martin designed it and made the calls, and
AI coding agents (Claude Code) wrote most of the code, tests and docs. Another day went on tuning.
The trick that worked for tuning indoors: point the webcam at a tablet playing YouTube videos of
Halloween parades, where a steady stream of kids in costume walks past all day, then score
whether the box picked the right costumes to name (see [`docs/RACE-REPORT.md`](docs/RACE-REPORT.md)).
The pre-written lines were generated by several language models and every one was read and
approved by a human before it could be spoken.

## Layout

- `scarebox/`: the Python package (pipeline, aim page, self-test, line-baking tools)
- `tests/`: the test suite (pytest), including red-team lines for the safety filter
- `config/`: the race config (`field.yaml`), replay, bench and soak configs, the example config,
  the overlay example and the safety blocklist
- `library/`: the line library (`manifest.json`), costume packs, the voice reference clip and
  synthetic sound effects
- `deploy/`: Dockerfiles, `compose.yaml`, the Chatterbox server, and host files (udev, systemd,
  WirePlumber, a NetworkManager example)
- `scripts/`: model staging, benchmarks, tuning and replay tools, the leak gate and the export
- `docs/`: runbook, hardware, race report, design, models, decision records
- `homelab/`: the author's own box-switching scripts, kept as an example

## Licence

Apache-2.0. See `LICENSE`. Models, voices, libraries and container images keep their own
licences; [`THIRD_PARTY.md`](THIRD_PARTY.md) lists each with its source.
