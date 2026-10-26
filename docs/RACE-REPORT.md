# Race report: HCA Halloween 5k, 2026-10-25

How scarebox did on its one real outing, how it was tuned beforehand, and what broke on the way
there and on the day, with the fix for each.

## Terms

- **Library line**: a joke written ahead of time, approved by a human and voiced ahead of time.
  Instant to play.
- **Live line**: a joke written on the spot by the vision model for the costume it just saw, passed
  by the two-layer safety filter (the "censor"), then voiced.
- **Trigger line**: a line drawn across the camera picture; a runner crossing it is an **event**.
- **Dropped**: a line that was never played because it went stale (the runner had gone by the time
  the speaker was free) or was outranked by a better line in a crowd. Deliberate: a joke about a
  runner who has already passed is worse than silence.
- **Stand-out**: a costume a crowd would point at. The judge rates every costume 1 to 3 for this,
  and in a group it gives the line to the most eye-catching one ("spotlight").
- **Busy mode**: switches itself on when lots of people cross at once; it judges fewer runners,
  shortens the gap between lines and voices only stand-outs.
- **Aim page**: the phone page the box serves on port 8090.
- **gbox1**: the author's box, an ASUS Ascent GX10 (NVIDIA GB10). Hardware: `docs/HARDWARE.md`.

## Results

About two hours at the Hopkinton Center for the Arts (HCA) Halloween 5k, 10:08 to 12:09 US Eastern
on 2026-10-25, unattended apart from one fix at the start.

| Measure | Result |
|---|---|
| Lines played | **104** |
| of which library lines | 75 |
| of which live lines | 29 (28% of what played) |
| Lines dropped | 74 (stale, or outranked in a crowd) |
| Phones on the aim page | 2, over the travel router's Wi-Fi |
| Safety | every line played was either human-approved (library) or passed both safety layers (live); that is how the pipeline is built, not a count from the day |

These numbers come from the app's text log for the run. Nothing was recorded (no video, no
audio), by design, so there is no ground truth for how often the costume was named correctly at
the race itself. On test footage the judge named the wrong costume roughly 1 time in 15.

Most lines being library lines is the design working, not failing. The voice is Chatterbox, which
sounds far better than the faster Kokoro but takes about 4 seconds from crossing to voice; with
runners arriving close together, many live lines missed their deadline and a pre-voiced line for
the same costume category played instead.

## Tuning before the race

Done on 19 minutes of the author's own video of a costumed crowd, replayed through the real
pipeline with a silent speaker (`config/replay.yaml`) and scored with `scripts/timeline_sim.py`.
The question asked was: of the must-announce costumes (the stand-outs), what share actually hears
a line naming them?

| Change | Stand-outs named |
|---|---|
| Starting point | 12.4% |
| Spotlight: in a group, the line goes to the most eye-catching costume | 20.9% |
| The judge rates each costume 1 to 3, and the queue ranks on it | 23.3% |
| A stand-out's line may start up to 9 s after the crossing (was 5 s) | **31.3%** |

The last change mattered because 5 s was shorter than judging plus one line already playing, so
stand-outs kept expiring in the queue: 71 of 123 did at 5 s. The same work cut group lines that
named nobody from 37% to 0.5%.

From the same replays: a line started a median 5 s after the crossing, 9 in 10 within 7.7 s. That
set the speaker about 15 to 20 m past the camera for joggers and 8 to 12 m for walking children.
The phone page's "Latest a line may start after the crossing" slider should be roughly speaker
distance divided by runner speed, plus 2 s.

Settings in force on the day (`config/field.yaml`):

| Setting | Value | Why |
|---|---|---|
| `orchestrator.budget_s` | 3.5 s | Deadline for a live line before the library line plays |
| `orchestrator.max_line_age_s` | 9 s | Speaker downstream; 5 s expired stand-outs (above) |
| `stage.cooldown_s` | 2.5 s after a line ends | Lines ran together without it |
| `stage.queue_max` | 6 | Busy mode picks the best of these |
| `perceive.best_crop_frames` | 1 | No waiting for a sharper crop: up to 330 ms saved |
| `perceive.min_bbox_height` | 0.25 | A runner must fill a quarter of the frame height |
| `voice.engine` | Chatterbox, exaggeration 0.7, speed 1.25, loud-hailer effect | Chosen by ear; about 4 s to voice accepted for it |
| `voice.stream_segments` | on | One voice request per sentence: 9.5% live share against 3% for whole lines in an A/B test |
| `stage.stings` | off | No chime before each line |
| Dongle output level | 80% | Full level distorted the PA's line-in |

Earlier, on 2026-10-15, a 2-hour soak on replay footage (with the then-voice, Kokoro) passed:
4,013 events, 29% live, trigger-to-audio p95 2.93 s, no exceptions, at least 68.9 GiB of memory
free throughout. Killing the app container brought it back in 6 s; killing the vision model gave
67 s of library-only lines and then live lines again; hanging the main loop for 35 s tripped the
watchdog and the app was back in 5 s.

## What broke, and the fixes

### On race day

- **The webcam's USB 3 link didn't come up at power-on.** Only the USB 2 half connected, which
  can't carry the video. Reseating the USB-C plug fixed it, with a helper at home guiding over SSH
  through a laptop at the course. The camera then held for the whole run. Lesson in
  `docs/HARDWARE.md`: seat the plug fully and check the link speed.
- **Open bug: silent while the camera was missing.** The self-test is meant to say "I say, I can't
  see a blessed thing" when there is no picture, and repeat it every 2 minutes. While the camera
  was missing on race morning the box said nothing at all, so the only sign was the absence of
  jokes. Not yet fixed; it will be tracked on this repository's issue tracker.

### Before race day

| What broke | Found | Fix |
|---|---|---|
| A USB controller died mid-stream with the webcam attached (`HC died`); only a reboot revived it | 2026-10-19 | `deploy/host/99-scarebox-camera.rules` turns USB power saving off for the camera chain; camera on the box's own port; camera and dongle on separate controllers. Likely fix, not proven. |
| The webcam dropped off the box on a different cable and port | 2026-10-18 | One labelled, proven cable into the box's own USB-C port; the app asks only for a supported mode (1080p MJPG at 30 fps) |
| After a hub swap every line failed with "No such device": the container's copy of the sound devices had lost the re-enumerated dongle | 2026-10-19 | Compose bind-mounts `/dev/snd` with a device cgroup rule, the same live-device approach as the webcam |
| Playing a line froze the whole app for the line's length | 2026-10-19 | Playback moved off the main loop |
| At the first real cold boot the app started about 25 s before the model servers (Docker restarts everything at once and ignores `depends_on`); the self-test failed and the app stayed library-only for the whole run | 2026-10-23 | The app waits up to 5 minutes for the vision model, guard and voice server before the self-test |
| The voice warm-up crashed the app twice on every boot, because it ran a real synthesis before the voice server was up | 2026-10-23 | The same wait now runs before the voice is built |
| Re-entering scarebox mode started only the app: `systemctl enable --now` on a boot unit already marked active from an earlier boot runs nothing | 2026-10-24 | `homelab/gbox1-mode.sh scarebox` enables and then restarts the unit |
| The hub with the router's Ethernet port was replaced by one without | 2026-10-24 | Router moved to the box's built-in Ethernet port, with the fixed aim-page address on that port |
| Stand-outs expired in the queue at a 5 s line age | 2026-10-19 | 9 s, and a phone-page slider (see Tuning) |
| Lines ran together | 2026-10-19 | 2.5 s of silence after each line ends |

The field cold boot test on 2026-10-23 (no home network, no internet) passed once the first fix
was in: the stack came up by itself, the vision model was ready after about 70 s, the self-test
passed, a phone used the aim page with no token over the router's Wi-Fi, and 29 lines played. A
single tap of the power button shut the box down cleanly.

## What wasn't measured

- Whole-box power draw at the race (software can only see the GPU; read the power station's
  display next time).
- Costume accuracy at the race, since nothing was recorded.
- How long the webcam holds over many hours: it held the two race hours after the reseat, and the
  USB power-saving fix hasn't otherwise been soak-tested.
