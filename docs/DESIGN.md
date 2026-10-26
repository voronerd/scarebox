# Scarebox: original design

- **Date:** 2026-10-15, the day the design was approved; measured numbers from the first
  benchmarks were folded in the same day.
- **Event:** Hopkinton Center for the Arts (HCA) Halloween 5k, 2026-10-25, morning (daylight).
- **Owner:** Martin. Built by Claude sessions, mostly unattended.
- **Decision record:** `docs/decisions/0001-offline-live-pipeline-with-vetted-library-fallback.md`.

This is the plan as it stood before the build, lightly edited to remove details of the author's
home network. It is kept because it explains *why* the pieces are shaped the way they are. Several
things changed in the ten calendar days between design and race; where the design and the code disagree, the code wins.

**What changed after this was written** (details in `docs/RACE-REPORT.md` and `docs/HARDWARE.md`):

- **Camera:** the Amcrest network camera (§3.1) was replaced on 2026-10-19 by a Dell USB webcam
  plugged straight into the box. RTSP is still supported. No camera network, no clock to set, no
  pan/tilt to re-aim, and the router no longer carries video.
- **Voice:** Chatterbox won by ear on 2026-10-16 (Kokoro was the provisional pick). It is slower,
  about 4 s from crossing to voice, which was accepted for the delivery. It runs in its own
  container.
- **Router:** a TP-Link travel router, plugged into the box's built-in Ethernet port, with a fixed
  address for the aim page, instead of the GL.iNet model planned in §3.1.
- **Speaker distance:** further downstream than §3.3 suggests (15 to 20 m for joggers), and a
  line may start up to 9 s after the crossing instead of the 8 s line age here.
- **Crowds:** "spotlight" (in a group, the line goes to the most eye-catching costume), an
  eye-catching rating from the judge, and a busy mode that switches itself on in crowds.
- **Line library:** 1,809 approved lines including 108 costume-specific packs, against the
  roughly 300 planned in §6.2.
- **Single call:** the combined judge-and-joke call (§4.1) was built behind `quip.single_call`,
  but the race ran with two calls.

## 0. Glossary

| Term | Meaning |
|---|---|
| **gbox1** | Martin's ASUS Ascent GX10: an NVIDIA GB10 "Grace-Blackwell" box with an ARM (aarch64) CPU, a Blackwell GPU, and 121 GiB of memory shared by both. Normally it serves a large language model to the rest of Martin's home network; that model is unloaded so the scarebox can have the memory. |
| **RTSP** | The standard protocol IP cameras use to stream video. The camera offers a **main stream** (full resolution) and a **sub-stream** (low resolution, cheaper to decode). We use the sub-stream. |
| **CGI** | The camera's own HTTP API (`/cgi-bin/...`, digest auth). It sets the clock, encoder settings and pan/tilt. |
| **ONVIF** | A camera-control standard. We avoid it because its login checks the camera clock, and this camera's clock resets every power cycle. |
| **Detector** | A small vision model that draws boxes around people in each frame (YOLOX or RF-DETR). |
| **Tracker** | ByteTrack: links the detector's boxes across frames so each runner keeps one ID while in view. |
| **Track** | One runner (one tracker ID) followed across frames. |
| **Trigger line** | A line drawn across the camera image. A track crossing it creates an **event**. |
| **Event** | "Heckle this runner (or group)": the unit of work flowing through the pipeline. |
| **VLM** | Vision-language model: takes an image plus text, answers in text. Here it names the costume. |
| **TTFT** | Time to first token: how long a model takes before it starts answering. With an image this is mostly "prefill", the time spent reading the image. |
| **TTS** | Text-to-speech. Turns the line into audio. |
| **Guard model** | A small classifier that flags unsafe text (Granite Guardian). |
| **Library** | Pre-written, pre-voiced, human-approved lines grouped by costume category, used whenever the live path is late or fails. |
| **Budget / deadline** | The maximum time from trigger to audio start (default 3.5 s). Miss it and a library line plays instead. |
| **Replay** | Feeding a recorded MP4 through the pipeline instead of the live camera, for testing. |
| **ALSA** | Linux's low-level sound layer. We play audio straight to the USB dongle through it, with no desktop sound server. |
| **p50 / p95** | The median, and the value 95% of measurements come in under. |

## 1. Purpose and success criteria

A box at the side of the course that watches runners near the finish, works out each costume,
and has a posh, pretentious English gentleman heckle them through a PA. It must be funny,
family-friendly (G-rated: mostly kids), fast enough that the line reaches the right runner, and
must run unattended for 2 hours off a battery with no internet.

**Success at the race:**

1. It runs 2 hours with no crash. If something does crash, it restarts itself within 30 seconds.
2. Most runners hear a line while they are near the speaker (target: at least 80% of solo runners).
3. When the line names a costume, the costume is right at least 70% of the time.
4. Nothing inappropriate is ever said. **Every played line is either human-approved (library) or has passed both safety filters.**
5. Nothing is recorded.

**Non-goals:** recognising individuals, storing images, internet features, a polished UI,
multi-camera, a web kill switch. The kill switch is unplugging the speaker.

## 2. Constraints

- **Ten days.** The library path must work end to end by about day 4, so there's always something to run.
- **Offline at the race.** All models, containers and audio assets get downloaded while the box is on the home network.
- **The box is firmware-fragile and shared:**
  - No system upgrades or firmware updates during the project, and no restarting or pruning other people's containers.
  - A unified-memory out-of-memory can freeze the box until someone pulls the power, and an out-of-memory killer acts at 8 GiB or less of available memory. Keep at least 16 GiB available at all times. Pin vLLM's KV-cache size in bytes; never rely on a bare `--gpu-memory-utilization`.
  - The boot path is **unproven** (the box hadn't been rebooted in weeks), so the first reboot is an attended test before race day.
  - Several ports are already taken. Scarebox uses **8100** (vision model), **8101** (guard model) and **8090** (aim page).
- **Publishable:** Apache-2.0. Every dependency, model and voice must allow redistribution and use: Apache/MIT/BSD, no AGPL, no non-commercial licences. No private addresses, hostnames or secrets in the tree; everything goes through `config/*.yaml` plus `.env`.
- **No recording:** frames stay in RAM. Logs are text only. Test footage lives in `testdata/` (git-ignored) and is never committed.
- **No real person's voice:** the character voice is cloned from a synthetic reference clip.

## 3. Hardware and network

### 3.1 Parts (as planned)

| Part | Facts that matter |
|---|---|
| gbox1 (GX10) | aarch64 with CUDA (compute capability 12.1). Idle about 40 W, full load about 200 W. Powered by its stock 240 W USB-C brick from the power station's **AC outlet** (most power stations' USB-C ports cap at 100 to 140 W, and nobody has tested the GX10 on less). |
| Camera: Amcrest IP4M-1041B | **Indoor** pan/tilt camera: 5 V DC power, no PoE, not weatherproof (needs a rain cover if wet). 90° field of view. Main stream 2688×1520 at 30 fps; sub-stream about 704×480. **No RTC battery: the clock resets to 2000-02-01 on every power loss** (confirmed 2026-10-15). **Pan zero recalibrates at power-up**, so saved positions don't survive a reboot. |
| Router: GL.iNet Slate AX (GL-AXT1800) | 2 LAN ports (camera + gbox1, no switch), USB-C 5 V power, works fully offline as a plain DHCP LAN. |
| Audio | USB audio dongle into the PA's line-in. No Bluetooth: it adds 150 to 300 ms and drops pairing. PA line-in type (3.5 mm or RCA) checked at rehearsal. |
| Power | 1000 Wh power station. Expected draw about 150 W average, so 2.5 h uses about 430 Wh including inverter loss: more than 2× headroom. |

### 3.2 Topology

```
HOME BENCH                                     RACE DAY (rehearsal + race)

 home LAN (e.g. 192.0.2.0/24)                  travel router LAN (e.g. 198.51.100.0/24), no WAN
   ├─ gbox1 (wired)                              ├─ gbox1 (wired, DHCP reservation)
   ├─ camera (Wi-Fi, static)                     ├─ camera (WIRED, DHCP reservation)
   └─ dev workstation (ssh in)                   └─ phone (Wi-Fi) → aim page :8090
 gbox1 USB → dongle → bench speaker            gbox1 USB → dongle → PA line-in
```

**Finding the camera:** config takes `camera.host`. If `camera.mac` is set instead, startup
resolves the IP by probing the local /24 for RTSP port 554 and matching the MAC in `ip neigh`.
That makes the home-to-race subnet change zero-config.

### 3.3 Physical layout at the course

```
   runners →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→ finish
        [camera]-------- D metres --------[PA speaker]
           ^ trigger line in the camera image
```

D is tuned at the rehearsal. Rule of thumb: D ≈ (expected trigger-to-audio latency) × (typical
speed) + half a line's duration × speed. Adult runner about 2.7 m/s, kid walking about 1.2 m/s.
For a 3 s latency that's about 8 m for runners and 4 m for walkers, so start at **D ≈ 8 to 10 m**
and rely on the line naming the costume to cover the spread. Point the camera across the course,
not into the low morning sun in the east.

## 4. Architecture

### 4.1 Approach (chosen)

**Live generation with a pre-baked library as the deadline fallback, built library-first.**
The live path makes the magic; the library guarantees a vetted line always plays.
Build order: library path end to end first, live path on top.

**Rejected:**

- **Library only:** safe and fast but can't react to specifics ("a Spider-Man with a juice box"). It survives as the fallback.
- **Fully live, no library:** one slow model call means silence. Unacceptable with kids waiting for their line.
- **One VLM call writing costume plus joke:** measurably less funny than label-then-joke, and it gives the filter nowhere to sit. Kept as a speed fallback in case two calls blow the budget. The first benchmarks showed they do (3083 ms p95 against a 2000 ms allocation, §4.5), so a combined `judge_and_quip` call was added behind `quip.single_call`.
- **Cloud models:** no internet at the course.
- **Ultralytics YOLO:** AGPL would force AGPL on the repo.
- **ONVIF control:** clock-skew auth failures (see glossary).
- **Bluetooth audio:** latency and pairing drops.

### 4.2 Pipeline

```
ingest ─frames─▶ perceive ─Event─▶ orchestrator ─┬─▶ judge ─▶ quip ─▶ censor ─▶ voice ─┐
                                                 │                                    ├─▶ stage ─▶ ALSA
                                                 └── deadline ─▶ library ─────────────┘
```

One Python 3.12 package, `scarebox/`, run as a single asyncio process (`scarebox run`) with
model servers as separate containers. Stages talk over in-process `asyncio.Queue`s. Model calls
go over HTTP to OpenAI-compatible servers on localhost, so engines are swappable.

### 4.3 Component contracts

**`ingest`**: `FrameSource` interface: `async def frames() -> AsyncIterator[Frame]`.
`Frame = {image: np.ndarray (BGR), t_capture: float (monotonic), seq: int}`.

- `RtspSource`: decodes the sub-stream. It keeps **only the newest frame** (a single-slot overwrite buffer, reader thread), so a slow consumer never builds lag.
- `FileSource`: plays an MP4 at real-time pace (or as fast as possible with `--fast`). Same interface.
- On start: if `camera.set_clock_on_start`, push the box's time to the camera via CGI `global.cgi?action=setCurrentTime&time=YYYY-MM-DD%20HH:MM:SS`.
- On stream loss: reconnect with backoff (0.5 s, 1 s, 2 s, capped at 5 s), and log `ingest.reconnect`.

**`perceive`**: `async def run(frames) -> AsyncIterator[Event]`.

- Detector (YOLOX-s first; RF-DETR-base if it's more accurate at equal speed) runs the person class only, at `min_confidence`.
- ByteTrack assigns track IDs.
- **Trigger:** a track's box bottom-centre crosses `trigger_line` while the box height is at least `min_bbox_height` of the frame. Each track fires **once** and its ID is remembered for 60 s.
- **Groups:** tracks near the line when the first one crosses join the same event, with `group_size = n`.
- **Best crop:** after triggering, collect up to `best_crop_frames` crops (about 330 ms at 15 fps) and pick the one with the largest area × sharpness (variance of the Laplacian). Crops are padded 10% and resized so the long side is at most `crop_max_px` (512). If the budget is tight, `best_crop_frames: 1` disables the wait.
- `Event = {id, t_trigger, track_ids, group_size, crops: [ndarray], bbox_heights}`.

**`judge`**: `async def judge(event) -> Verdict`.

- One VLM call with the best crop (for a group, the 2 or 3 largest crops in one request), JSON-schema-constrained output: `Verdict = {costume: str, category: Category, confidence: float, is_costume: bool, group_size: int, details: str}`.
- `costume` is a short noun phrase ("inflatable T. rex"). `details` holds one or two concrete visual specifics the joke can use ("holding a juice box").
- `category` is from the fixed list in §6.2. That list is the join key to the library.
- Times out at `judge.timeout_s` and returns `None`, and the orchestrator falls back.

**`quip`**: `async def quip(verdict) -> str`.

- A text-only call to the same model server (no image, so it's fast). The persona prompt is in §5.3.
- Output: one line, at most `max_words` (15), must contain the costume noun.
- It asks for 3 candidates in one call. Censor filters them, and the shortest passing one wins.
- If `is_costume` is false and `roast_non_costumes` is off, skip straight to the library's `no_costume` category, or to silence if that category is disabled.

**`censor`**: `def check(line) -> Pass | Reject(reason)`. Two layers, both must pass:

1. **Deterministic:** a lowercase, de-leeted match against `config/blocklist.txt` (profanity, slurs, body, size, age, appearance, disability, violence, death-threat, romance words). Plus rules: no second-person physical descriptors ("your face/legs/hair/teeth..."), at most 15 words, no digits beyond a small allow-list, ASCII-ish text only. Runs in under 1 ms.
2. **Guard model:** Granite Guardian (Apache-2.0) served on :8101, asked for harm / social bias / profanity / unethical behaviour. Any positive verdict, or a timeout, means **reject** (fail closed).

**`voice`**: `async def speak(text) -> PCM chunks` (an async iterator, streaming where the engine supports it).

- The engine comes from the bake-off: Chatterbox with `exaggeration` and a `reference_clip` synthesised from Kokoro `bm_george` (no real voice), or Kokoro `bm_george` directly.
- The **barker FX chain** is applied per chunk: band-pass 300 Hz to 5 kHz, mild drive, +4 dB peak at 2 kHz, compression, short room reverb, then a fixed gain stage. Loudness is calibrated once to about −14 LUFS, because per-line loudness normalisation needs the whole clip.
- Output: 48 kHz, 16-bit mono.

**`library`**: `def pick(category, exclude_recent) -> LibraryLine`. Picks at random from approved lines in that category, never repeating any of the last 10 played. Falls back to `mystery` if the category is empty. See §6.

**`stage`**: owns the ALSA device.

- One line plays at a time. The queue holds at most 2 pending items; if it's full when a new one arrives, the **oldest pending is dropped** (that runner has gone).
- Before playing, it drops anything whose `t_trigger` is older than `max_line_age_s` (8 s).
- `cooldown_s` gap between lines, plus an optional short SFX sting before a line (`stings`).
- The device is pinned by a udev rule to the name `scarebox_out` (vendor:product match), with USB autosuspend disabled for it.
- Without a device (bench, CI) it writes to a `NullSink` or a WAV sink in `runs/` (bench only, not on race day).

**`orchestrator`**: per-event state machine (§4.4) plus the deadline timer.

**`aim`**: a tiny HTTP page on `:8090`, protected by a token from `.env` (`AIM_TOKEN`). It shows:

- the current frame (1 fps JPEG, generated on demand and never stored) with the trigger line and boxes drawn on it;
- PTZ nudge buttons (CGI `ptz.cgi?action=start&code=Left|Right|Up|Down&arg2=<speed>` then `stop`);
- trigger-line drag-to-set, saved back to config;
- the last 20 events as text (category, line, latencies);
- a "test line" button that plays a library line.

It exists because pan zero resets on power-up, so the camera has to be aimed by eye on race morning.

### 4.4 Event state machine

```
TRIGGERED ──▶ JUDGING ──▶ QUIPPING ──▶ CENSORING ──▶ VOICING ──▶ QUEUED ──▶ PLAYED
    │            │            │             │            │
    └────────────┴────────────┴─────────────┴────────────┴──▶ (deadline fires before QUEUED)
                                                                     │
                                         FALLBACK: library.pick(category or 'mystery') ──▶ QUEUED
any stage error ─▶ FALLBACK          censor rejects all candidates ─▶ FALLBACK (same category)
late live line arriving after FALLBACK ─▶ discarded (logged as late)
```

The deadline is `t_trigger + budget_s`. The category from `judge` is kept even if `quip` or `voice`
is late, so the fallback line still matches the costume. Each event logs one JSON line to
`runs/events.jsonl` (text only):
`{id, t_trigger, group_size, category, costume, confidence, path: live|library|dropped, line, ms: {detect, crop_wait, judge, quip, censor, voice_first_audio, queue_wait, total}}`.

### 4.5 Latency budget (measured in the first benchmarks)

| Stage | Measured | Notes |
|---|---|---|
| Camera glass to decoded frame | median 1339 ms / **p95 1452 ms** | Measured by timing a pan/tilt move, so it includes motor start (~50–150 ms) and is an **upper bound**; the camera was on 2.4 GHz Wi-Fi, and race day was planned wired. This lag happens *before* `t_trigger`: it shifts *where the runner physically is* when the line plays (the distance D), not this budget. |
| Detect + track | ~10–20 ms/frame | Estimate, measured later on replay footage |
| Best-crop wait | 0–330 ms | Computed from `best_crop_frames` (5 frames at 15 fps), not separately measured |
| judge (VLM, 1 crop at 512 px, ~55 output tokens) | p50 1292 ms / **p95 1513 ms** | Qwen3-VL-4B-Instruct-FP8; decode 44.8 tok/s, time to first token ~71 ms p50 |
| quip (text only, 3 × 15 words) | p50 ~1361 ms / **p95 ~1570 ms** | Derived: judge+quip combined p95 3083 ms minus judge p95 1513 ms |
| censor | 50–200 ms | Estimate; blocklist is sub-ms, guard call unmeasured |
| TTS to first audio | p50 60 ms / **p95 83 ms** | Kokoro-82M, ~89× real-time |
| ALSA | ~20–50 ms | Estimate |
| **Total, trigger to audio (what `budget_s` measures)** | **~3.7 s p95** | Crop wait 0.33 + judge 1.51 + quip ~1.57 + censor ~0.2 + TTS 0.08 + ALSA. **About 0.2–0.3 s over the 3.5 s budget**; `budget_s` stays 3.5 and the token cuts below aim well under it. |
| **Camera lag (before trigger, not in the budget)** | ~1.45 s p95, upper bound on Wi-Fi | Sets camera-to-speaker distance D (§3.3). Tune at rehearsal. |

The lever that matters is generated tokens, not model size: judge+quip latency is decode-bound
(TTFT ~71 ms). The plan applies terse JSON schemas first (judge ≤ 30 completion tokens, quip ≤ 50
via guided JSON) to kill both the latency and the ~15–17% quip schema-miss rate, then the remaining
levers in order of expected payoff: **(1)** `best_crop_frames: 1` (free, up to 330 ms), **(2)** a
single combined `judge_and_quip` call (one round-trip instead of two), **(3)** a smaller crop
(384 px; prefill is a small share of the latency at 512 px, so this barely moves the number; last
resort).

## 5. Models and serving

### 5.1 Candidates (the bake-off picks the winners)

| Role | First choice | Alternatives | Licence |
|---|---|---|---|
| Detector | YOLOX-s | RF-DETR-base | Apache-2.0 |
| Tracker | ByteTrack | norfair | Apache-2.0 / BSD |
| VLM | Qwen3-VL-4B-Instruct | Qwen3-VL-8B; Gemma-4-26B-A4B (licence to confirm); SmolVLM2-2.2B as speed floor | Apache-2.0 (Qwen3-VL, SmolVLM2) |
| Guard | Granite Guardian (smallest variant) | Llama Guard 3-1B (Llama licence, avoid if possible) | Apache-2.0 |
| TTS | Chatterbox Turbo + synthetic reference | Kokoro-82M `bm_george`; Piper en_GB (check accent by ear) | MIT / Apache-2.0 / MIT |

Every pick gets its licence verified from the model card at download time and recorded in
`docs/MODELS.md` (name, revision, licence, source URL). What was finally used, and every
library and container image under it, is in `THIRD_PARTY.md`.

### 5.2 Serving on the box

- **Containers only.** Use NVIDIA's DGX Spark vLLM image for the VLM and guard: the only maintained aarch64 + sm_121 build. Detector, tracker, TTS and the app go in an `nvcr.io/nvidia/pytorch` based image. ONNX Runtime GPU has no aarch64 wheels, so it's out.
- **Memory pins:** each vLLM server pins `--kv-cache-memory-bytes` and `--max-model-len`, and is started with a high `oom_score_adj`. Target total scarebox footprint is **under 40 GiB**.
- Models are pre-downloaded into `models/` (a local path on the box, git-ignored) and mounted read-only. Containers run with `HF_HUB_OFFLINE=1`.
- One `compose.yaml`: `vlm`, `guard`, `app`, all `restart: unless-stopped`. The app's healthcheck waits for both model servers. The startup self-test line plays only when camera, models and audio are all live.
- The app container gets the sound devices and the GPU. Nothing runs privileged and nothing touches other containers.

### 5.3 Persona prompt (draft, tuned later)

> You are Sir Reginald Pumpernickel-Fortescue, a gloriously pompous English gentleman
> adjudicating a children's Halloween fun run from a deckchair. You deliver ONE short line
> (max 15 words) to a runner, about their COSTUME ONLY. Tone: theatrical, haughty, affectionate,
> silly, like a pantomime dame crossed with a snooty art critic. Always name the costume.
> Never mention the person's body, size, face, age, skin, hair, gender, speed, or ability.
> No insults that would upset a seven-year-old. No scary threats, no romance, no brand slogans.
> Return JSON: {"lines": ["...", "...", "..."]}.

Plus 6 to 8 few-shot examples per category family from the approved library, refreshed after vetting.
The prompt in use is in `scarebox/prompts.py`.

## 6. Line library

### 6.1 Format

```
library/
  manifest.json             # [{id, category, text, wav, status: approved|rejected|pending, reviewed_at}]
  voice/<engine>/<id>.wav   # rendered after approval (regenerated by `bake render`, not shipped)
  sfx/*.wav                 # CC0 only; each listed in library/sfx/SOURCES.md
  voice/reference.wav       # synthetic reference clip for cloning (Kokoro-generated)
```

### 6.2 Categories (the join key between judge and library)

`superhero, princess, fairy_tutu, witch_wizard, ghost, skeleton, zombie, vampire, mummy,
pumpkin, monster, dinosaur, animal, inflatable, food, pirate, robot_space, movie_character,
sports_uniform, homemade_box, group, no_costume, dog, stroller, mystery`.
Target about 12 to 15 lines per category, about 300 total. Plus about 10 **meta lines** for
startup, the test button and long idle ("The adjudicator is seated. Proceed, ye costumed hordes.").

### 6.3 Bake flow

1. `scarebox bake generate`: an LLM (any OpenAI-compatible endpoint; cloud is fine, since nothing here is sensitive) writes candidates per category with the persona prompt. Every candidate passes `censor` first.
2. **Human vetting (async gate):** candidates go on a phone page with approve/reject per line (`python -m scarebox.bake vet`). Martin taps through on his phone whenever.
3. `scarebox bake pull` reads the decisions back into `manifest.json`.
4. `scarebox bake render` voices approved lines with the chosen TTS and FX chain, then loudness-normalises them **properly** (offline, so full loudness normalisation is fine).

Until Martin vets, a "pending" library is usable for **bench testing only**. The race config refuses
to start unless every library line is `approved` (`library.require_approved: true`).

## 7. Content-safety policy

- **Allowed subject:** the costume (what it is, how it's made, the character's lore, props). Group jokes about the group's costumes. **Finish-line goading** (Martin, 2026-10-16: the box sits on the last stretch): pantomime-style pushes to hurry, such as "Not long to go!", "Get a move on!", "Run, don't crawl!", "Chop chop, the finish awaits!".
- **Banned:** the person's body, size, weight, height, face, hair, skin, age, gender, disability, voice, fitness, clothing that isn't costume; **calling anyone slow, lazy, unfit, tired or behind** (goading pushes the pace up; it never remarks on how they're doing); profanity of any degree (including "hell", "damn", "crap"); violence or threats beyond cartoon-spooky; romance; religion; politics; brands as insults; real public figures as insults.
- **Uncertain costume:** the judge's `confidence` under `min_confidence` means the line uses the category only, or the `mystery` library.
- **Non-costume runners:** gentle generic lines, behind the `roast_non_costumes` toggle (default on, per Martin).
- **Fail closed** everywhere: a filter error, timeout or unknown means no live line.
- **Red-team set:** `tests/redteam/lines.txt`, at least 50 lines that must be rejected (body comments, mild profanity, leetspeak, sneaky phrasings), and `tests/redteam/ok.txt`, at least 50 that must pass. The tests require 100% and 100%.

## 8. Development and unattended operation

### 8.1 Where code lives and runs

- **Repo:** a git repository on Martin's workstation, local only at first. Pure logic (orchestrator, censor, library, config) is unit-tested there, on a CPU.
- **Execution target:** gbox1, as an **unprivileged** user `scarebox` (docker group, no sudo), at `~scarebox/scarebox`. GPU work, replay runs and latency tests run there.
- **Sync:** `scripts/sync-gbox1.sh` makes a `git bundle` of HEAD, copies it to the box, and fast-forwards the box's checkout from it. Untracked files on the box (`models/`, `.env`, `runs/`) are untouched.
- **Remote commands:** over SSH as the `scarebox` user only. Never `sudo`, never other users' containers, never restart or prune Docker, and check `free -g` shows at least 16 GiB available before starting any model server.

### 8.2 Human gates (asynchronous: the build never blocks on them)

| Gate | Needed by | Default if Martin hasn't answered |
|---|---|---|
| G1 Voice pick (listen to bake-off samples) | Start of the live path | Fastest engine meeting the budget; Chatterbox if both meet it |
| G2 Library vetting (approve/reject lines) | Race config only | Bench uses pending lines; race refuses to start until vetted |
| G3 Audio hardware present (dongle + PA line-in) | Soak test | Soak runs on a WAV sink; real-audio test deferred to the rehearsal |
| G4 Home rehearsal | Rehearsal | No default. Martin must be there |

Each gate sends a push notification and gets a line on a status page.

### 8.3 Phases

| # | Phase | Mode | Deliverable / exit gate |
|---|---|---|---|
| 0 | Box + credentials setup | Attended | `scarebox` user; SSH access; camera credentials in `.env` both ends; the box's usual model unloaded; base images and candidate weights pulled; sync round-trip works; ≥ 90 GiB available; hardware checklist answered |
| 1 | Spikes | Unattended | Camera tuned and frame age measured; ~80 labelled costume crops; VLM bake-off table; TTS samples; picks recorded; `docs/MODELS.md` |
| 2 | Library path end-to-end | Unattended | ingest → perceive → trigger → judge(category) → library → stage on replay MP4s; `bake generate` + vetting page; events log; unit tests |
| 3 | Live path | Unattended | quip + censor + voice + deadline orchestrator; red-team 100%; p95 ≤ budget on replay; G1 resolved or defaulted |
| 4 | Hardening + ops | Unattended | compose autostart; self-test line; aim page; camera auto-discovery by MAC; field profile; 2 h replay soak with power and memory logged; runbooks in `docs/RUNBOOK.md` |
| 5 | Home rehearsal | Attended (Martin) | §10.2 checklist passes; D and trigger line tuned; box back in its everyday role |

## 9. Testing

### 9.1 Measurements

- **Camera frame age:** point the camera at a screen showing a millisecond clock driven by the box's time. Grab a decoded frame, read the clock (OCR, or ask the VLM), and compare with now. Report median and p95 over 100 frames, for each encoder setting tried.
- **Per-stage latency:** from the `ms` block in `events.jsonl`. `scripts/latency_report.py` prints p50/p95/max per stage and per path.
- **Power:** read nvidia-smi power and `/proc/meminfo` every few seconds during the soak. The box's total wall draw isn't visible in software, so the rehearsal reads the power station's display.

### 9.2 Test data

- `scripts/fetch_testdata.sh`: `yt-dlp` pulls a handful of public clips (costume fun runs, Halloween parades, marathon finish lines) into `testdata/`. Git-ignored and never redistributed. (The published script takes your own list of clips.)
- Labelled crop set: about 80 person crops cut from those clips, labelled with costume + category by a strong cloud VLM (public footage only) as **silver** labels, with spot checks recorded. It scores judge accuracy (exact category match plus a costume-name similarity judgement).
- Martin's own run-pasts at the rehearsal are the real test. Nothing is recorded; results are tallied from the events log plus notes.

### 9.3 Automated tests (pytest)

- `orchestrator`: fake clock; the deadline fires, a late live line is discarded, category is carried into the fallback, queue overflow drops the oldest, max line age.
- `perceive`: synthetic tracks crossing the line fire once; groups merge; min height respected.
- `censor`: red-team and OK sets (100% / 100%), plus guard-timeout means reject.
- `library`: no repeats in the last 10; empty category falls to mystery; the race config refuses pending lines.
- `ingest`: FileSource pacing; RtspSource reconnect.
- **Replay end-to-end:** a 5-minute clip through the full stack on the box with a WAV sink. Assert no exceptions, every event reaches PLAYED or DROPPED, and p95 ≤ budget.

## 10. Runbooks (expanded into `docs/RUNBOOK.md`)

### 10.1 Bench session (home)

Precondition: the box's usual model unloaded, and `free -g` available ≥ 56 GiB (40 scarebox + 16
reserve). Run `docker compose up -d` in the repo on the box and watch `runs/events.jsonl`.
Unattended sessions never load or unload the usual model (that needs sudo); if memory is short
they stop and flag it.

### 10.2 Home rehearsal (attended)

1. Check that no firmware capsule is staged: `ls /boot/efi/EFI/UpdateCapsule/` must be empty. Then cleanly power the box off.
2. Move it to the power station's AC outlet, the travel router and the wired camera. Power on in this order: router, camera, box.
3. Wait for the self-test line (target: under 5 min from power-on). If it never comes, connect a laptop to the router and SSH in.
4. On the phone (joined to the router's Wi-Fi), open the aim page. Aim the camera and set the trigger line.
5. Do run-pasts in different costumes, at walking and running pace, solo and in pairs. Tune D and `budget_s`.
6. Read the power draw off the power station. Check that the memory floor held.
7. Power down with a short press of the power button (verify it does a clean shutdown), back onto the home network, boot, then restore the box's usual model (`homelab/gbox1-mode.sh normal` is the author's version) and confirm its health check is clear.

### 10.3 Race morning

Same as rehearsal steps 2 to 6, at the course. Bring a rain cover for the camera, gaffer tape,
spare USB-C and Ethernet cables, and the laptop. Kill switch: unplug the speaker.

### 10.4 Return home

Rehearsal step 7. Then write up the results (hit rate, crashes, power): see `docs/RACE-REPORT.md`.

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| The box fails to boot after power-off (unproven boot path; forum reports of GB10 boxes not powering back on) | Low–Med | Rehearsal does the first reboot at home with days to spare. If it fails: RMA, and the race needs a fallback box (out of scope) |
| VLM too slow with an image | Med | Bake-off first; levers in §4.5; library covers misses |
| Unfunny lines | Med | Two-stage generation, 3 candidates, curated few-shots from the approved library |
| Something inappropriate said | Low | Two-layer fail-closed filter, red-team tests, vetted library, physical kill switch |
| Camera clock / PTZ reset at power-up | Certain | Clock set by the app; aim page |
| Audio dongle missing or flaky | Med | G3 gate; udev pin; autosuspend off; spare dongle |
| Rain on the indoor camera | Med | Rain cover; camera under the box's lid |
| Crowd bunching at the finish | Med | Group events; queue drop-oldest; cooldown |
| Time (10 days) | Med | Library path first: something runnable by about day 4 |

## 12. Open items, and how they were answered

1. USB audio dongle on hand, and a PA line-in? Yes, a dongle (2026-10-15). The line-in type was left to the rehearsal.
2. Exact travel router? First the GL.iNet Slate AX; on 2026-10-23 a TP-Link travel router on the box's Ethernet instead.
3. Gemma-4-26B-A4B licence? Never confirmed as permissive, so it was never a candidate.
4. Does a short power-button press shut the box down cleanly? Yes (2026-10-23), and it boots back into scarebox.
5. When does the box's usual model come off? Only when Martin said go (2026-10-15). Switching both ways is scripted (`homelab/gbox1-mode.sh`, specific to the author's box; see `homelab/README.md`).
