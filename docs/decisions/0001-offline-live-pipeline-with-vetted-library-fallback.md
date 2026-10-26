# 0001 — Offline Live Pipeline With Vetted Library Fallback

**Status:** Accepted
**Date:** 2026-10-15
**Deciders:** Martin

> **Authoring note.** Decided in a brainstorm session on 2026-10-15, informed by three web-research
> passes (vision stack, offline TTS, camera/router/power). Full design: `docs/DESIGN.md` (its
> architecture section holds the chosen approach and the rejected ones).

## Trigger and evidence

Martin volunteered a costume-heckling "scarebox" for the HCA Halloween 5k (about 2026-10-25). The
architecture had to be fixed before any code, because the 10-day window leaves no room to
rebuild. Evidence that shaped the decision:

- No published DGX Spark / GB10 benchmark measures vision-model latency with an image input, so live generation speed is an unknown until Phase 1 measures it.
- Ultralytics YOLO is AGPL-3.0, which would force AGPL on a repo intended for public release.
- Coqui XTTS-v2 (CPML) and F5-TTS (CC-BY-NC) are non-commercial.
- The Amcrest IP4M-1041B has no RTC battery: its clock read 2000-02-01 on 2026-10-15, so ONVIF's time-checked login is unreliable offline.
- Humour research (HumorChain, arXiv 2511.21732) found that structured multi-step generation beat a single pass (67% vs 45% humorous) more than model size did.

## Context

The box must say something funny, G-rated, and correct about each runner's costume within a few
seconds, unattended for 2 hours, with no internet, on gbox1 (an ASUS GX10 with 121 GiB of unified memory). The
audience is mostly kids, so a bad line is worse than no line, and silence is worse than a
canned line. Martin wants to publish the code afterwards.

## Decision

We will build scarebox as a **fully offline, live-generation pipeline bounded by a per-event
deadline, with a pre-baked, human-vetted line library as the fallback**, built library-first.

### Sub-decisions

#### 1a — Live path

Detector + tracker → vision-model costume **label** → separate text-only **joke** call →
two-layer **fail-closed censor** (deterministic blocklist/rules, then a guard model) → TTS with a
character FX chain → one-at-a-time audio stage.

#### 1b — Deadline and fallback

Every event gets a budget (default 3.5 s from trigger to audio start). If the live line isn't
queued by then, or any stage errors, or the censor rejects every candidate, a library line for the
same costume category plays instead. A live line that arrives late is discarded.

#### 1c — Library first

The library path (detect → category → library → audio) is built and proven before the live path.
The race configuration refuses to start unless every library line has been approved by a human.

#### 1d — Permissive licences only

Code, models, voices and sound effects must be Apache-2.0, MIT, BSD or CC0. No AGPL (no
Ultralytics), no non-commercial models, and no cloning of a real person's voice: the character
voice is cloned from a synthetic reference clip.

#### 1e — Camera control via RTSP + CGI, not ONVIF

Video comes over RTSP. Clock and pan/tilt control use the camera's digest-auth CGI API. The app
sets the camera clock at startup.

#### 1f — Offline on gbox1

All models are served locally in containers on gbox1, as an unprivileged user, within a memory
budget that respects gbox1's out-of-memory rules. Nothing at the race depends on the internet.

## Scope

**In scope:** the runtime architecture, its fallback semantics, the licence policy, and the camera control path.

**Out of scope (explicit):** which specific models win (Phase 1 bake-off, recorded in
`docs/MODELS.md`); persona wording; UI beyond the aim
page; any cloud or connected mode.

## Consequences

- **Good:** there's always a vetted line to play, so latency misses degrade to "less specific", not "silent" or "unsafe". The library path gives a runnable system early. The licence policy keeps publication clean.
- **Bad:** two pipelines to build and test. Lines have to be vetted by hand (about 300). Two model calls cost more latency than one.
- **Accepted costs:** the best-known detector family (Ultralytics) is off the table. Some TTS engines with richer voices are excluded by licence.

### Falsifiability anchor / When to revisit

- If Phase 1 measures p95 trigger-to-audio above 3.5 s with the smallest viable vision model, revisit 1a: switch to a single combined label-and-joke call, or to library-only for the race.
- If more than 30% of live lines are rejected by the censor during replay, revisit the prompt or the guard model before blaming the architecture.
- If the project is never published, 1d's exclusions can be relaxed.

## Cross-references

- `docs/DESIGN.md`: architecture, line library, content safety
- `docs/MODELS.md`: which models won the Phase 1 bake-off
- `THIRD_PARTY.md`: the licence and upstream source of every model, voice, library and container image finally used (§1d)
- `docs/HARDWARE.md`: the hardware as built. Note on §1e: on 2026-10-19 the race camera became a USB webcam plugged into the box, so the race used neither RTSP nor CGI; the RTSP + CGI path stays supported for IP cameras
- `docs/RACE-REPORT.md`: how the decision held up on 2026-10-25 (104 lines played, 75 from the library, 29 live)

## Sources

- Brainstorm session transcript, 2026-10-15 (Martin + Claude).
- Research reports (2026-10-15): vision stack (Ultralytics AGPL; Qwen3-VL Apache-2.0; RF-DETR and YOLOX Apache-2.0; https://docs.ultralytics.com/guides/nvidia-dgx-spark; https://build.nvidia.com/spark/vllm/overview), offline TTS (https://www.resemble.ai/chatterbox-turbo; https://forums.developer.nvidia.com/t/running-kokoro-tts-on-nvidia-dgx-spark-arm64-gb10/368846), camera/power (https://support.amcrest.com/hc/en-us/articles/4412609373709-Accessing-Amcrest-Products-Using-RTSP; https://www.servethehome.com/asus-ascent-gx10-review-a-new-nvidia-gb10-solution/4/).
- HumorChain: https://arxiv.org/pdf/2511.21732
- Camera clock reading: ONVIF `GetSystemDateAndTime` against the camera on the home LAN, 2026-10-15 14:11 UTC → 2000-02-01.
