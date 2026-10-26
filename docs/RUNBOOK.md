# Scarebox runbook

Step-by-step for the rehearsal and race morning. Written for your phone, possibly
stressed. Every term is explained on first use.

**What scarebox is:** a camera watches runners, a computer judges the costume, and a PA
(public address) speaker further down the track says a one-line joke about it. "gbox1" is
the one small computer doing the thinking. The camera is a USB webcam (Dell UltraSharp,
model WB7022) plugged straight into gbox1, so it needs no network and no power of its own.
Only gbox1, the router (there so your phone can reach gbox1) and the speaker need power.

---

## 1. Packing list

- gbox1 + its own stock USB-C power brick (240 W). No substitute chargers — nothing
  weaker has been tested.
- Travel router (TP-Link) + its power cable + an Ethernet cable from it into **gbox1's
  built-in Ethernet port** (the one the home network uses at home; the USB hub has no network
  port). gbox1 keeps a fixed address on that port (198.51.100.10 in this runbook; yours is
  whatever you set in `deploy/host/90-NM-builtin-port.yaml`). The router is only there so
  a phone can open the aim page (§4) and a laptop can get in if something's wrong; the camera
  doesn't use it. It needs no internet.
- **The webcam (Dell WB7022) and the USB-C cable that's proven with it.** On 2026-10-18 it
  dropped off gbox1 on a different cable/port and stayed gone; on this cable and port it
  then ran 20 minutes non-stop at 30 frames a second with zero dropped frames. Plug it into
  **gbox1's own USB-C port**, not a hub or dock, and label that cable. It's an **indoor**
  camera: pack something to shade it from direct sun and keep rain off.
- A mount or clamp to hold the webcam steady, pointed across the course.
- USB audio dongle (plugs into gbox1, feeds the PA) + **a long audio cable** to the PA's
  line-in. gbox1 and the camera sit together (the camera's USB cable is short), so the
  audio cable is what spans the gap to the speaker: plan on 10–15 m. **Confirm whether the
  PA wants 3.5 mm or RCA** — pack an adapter for the one you don't have; unconfirmed as of
  writing.
- PA speaker + its power.
- Portable power station, charged the night before.
- Gaffer tape, a spare USB-C cable, a spare Ethernet cable.
- A laptop — backup way into gbox1 if the audio self-test never plays (§6, §10).
- Your phone, charged — used to aim the camera (§4).

---

## 2. Before leaving home (the day before)

Then **leave the stack running** — the last thing you do is cut the power, not shut
anything down in software.

1. **Check free memory:** `free -g` on gbox1. Plenty of headroom expected; if it's tight,
   sort that out before you leave.
2. **Check no firmware update is staged:** `ls /boot/efi/EFI/UpdateCapsule/` must be
   empty. If not, gbox1 will try to apply it unattended at the course — don't let that
   happen.
3. **Check every heckle line is approved.** The race config refuses to start if even one
   pre-written line hasn't been reviewed. Confirm zero "pending" lines before you leave.
4. **Put gbox1 in scarebox mode:** `homelab/gbox1-mode.sh scarebox` from your workstation. That unloads
   the fleet model, starts the stack with the race settings (`config/field.yaml`), and turns
   on the boot unit that starts scarebox on every power-on while in this mode. With the
   webcam and dongle plugged in, listen for "The adjudicator is seated" (§3).
   `homelab/gbox1-mode.sh status` should say `scarebox boot unit: enabled`.
5. **Leave it running, then just turn gbox1 off** (power button, or unplug). At the course
   it comes back by itself, with no laptop and no typing. Don't run `gbox1-mode.sh normal`
   until you're home: that switches the boot unit off.

**Remote help at the course (optional):** gbox1 has no VPN of its own, but a helper at home
can reach it by SSH-jumping through a laptop at the course (`ssh -J <laptop> gbox1`). The
laptop must be on the TP-Link's Wi-Fi AND have internet: either the TP-Link's uplink is on
(your phone hotspot), or the laptop is USB-tethered to your phone. With the hotspot on, your
phone can't join the TP-Link, so open the page in the laptop's browser instead
(`http://198.51.100.10:8090/`).

---

## 3. Setup at the course

Set up gbox1 and the webcam **together, upstream** (the camera end of the stretch), and run
the long audio cable down to the speaker (§5).

1. **Webcam into gbox1** — its labelled USB-C cable, into gbox1's own USB-C port. Mount it
   pointing across the course, out of direct sun. It can go in before or after power-on:
   gbox1 finds it by name, and keeps looking if it isn't there yet.
2. **Router** — power on, wait for its lights to settle, then the Ethernet cable from the
   router to gbox1.
3. **gbox1** — its stock power brick into the power station's **AC outlet** (looks like a
   wall socket), not a USB port — USB ports on most power stations can't supply enough.

Then wait — you left the stack running (§2), so **you don't type anything**.

**What you'll hear:** once camera, costume judgement, the joke-safety filter, the voice
and the speaker have all checked out, the PA says, in character: *"The adjudicator is
seated. Let the parade commence!"* That's your all-clear — go aim the camera (§4).

**Timing:** budget up to **5 minutes** from power-on to that line. Nothing after 5
minutes (not even an error line) — connect the laptop to the router's Wi-Fi and SSH into
gbox1.

Hear something else instead? See §6.

---

## 4. Aiming the camera

1. **Join the router's Wi-Fi** on your phone. The phone must be ON the router's Wi-Fi: a
   phone that is the router's internet hotspot sits outside it and can't open the page.
2. **gbox1's address is fixed: 198.51.100.10** in this runbook (set on gbox1 itself,
   2026-10-23, so it doesn't depend on the router; use your own). If that ever fails, the
   router's settings page (here http://198.51.100.1) lists it among connected devices, under
   whatever lease the router handed out.
3. **Open the aim page:** `http://198.51.100.10:8090/` (or scan the QR code taped to the
   box). No token needed from the router's Wi-Fi (its password is the lock). From the home
   network the page still wants `?token=<AIM_TOKEN>` from gbox1's `.env`.
4. **On the page:** a live photo with a line drawn across it (the "trigger line" — where a
   runner has to cross for the joke to fire), a TEST button (fires a line right now, good
   for checking the speaker), and recent events. The pan/tilt arrows don't apply to the
   webcam: they'll say it's aimed by hand.
5. **Aim:** move the webcam itself until the course fills the picture side to side, and
   the runners you want to catch are big in the frame (at least a quarter of its height)
   when they reach the line. Not into low morning sun. Drag the trigger line to where you
   want runners caught. Hit TEST a couple of times to confirm the speaker.

**No phone?** Plug a screen into gbox1's HDMI port instead: the same page opens in a
browser there at `http://localhost:8090/?token=<the token>`.

Once aimed, it stays aimed: the webcam has no motor, so unlike the old camera it doesn't
forget its position when the power drops. Just don't knock it.

---

## 5. Placing the speaker

The camera and speaker are deliberately apart: the camera spots the runner, the computer
thinks, and the joke needs to come out while that runner is still close. The speaker
sits distance **D** down the course from the camera to cover that thinking time. gbox1
stays with the camera; the long audio cable is what reaches the speaker.

**Rule of thumb:** D ≈ (time from "runner crosses the line" to "joke plays") × (runner's
speed), plus a bit for the line's own length.

- **Camera lag** (light hitting the lens to a usable frame): a USB webcam is typically well
  under a quarter of a second. Not measured for this one: Martin's call (2026-10-19) is to
  tune D by moving the speaker rather than measure it.
- **Trigger-to-audio time** (line crossed to joke playing): about **3–4 seconds** with Sir
  Reginald's current voice (Chatterbox) in the slowest cases. Most lines are pre-recorded
  and come out faster.

Together, roughly **4 seconds** at worst. At running pace (~2.7 m/s) that's ~11 m; at
walking pace with a kid (~1.2 m/s) that's ~5 m. **Start the speaker about 10 metres down
from the camera** — don't over-tune this, every joke names the costume aloud, which covers
the speed spread either way. If jokes land after people have passed, move the speaker
further down. Confirm at the rehearsal with real walk/run-pasts.

---

## 6. What you'll hear if something's wrong

No screen at the course — the speaker is the status display. One line, not a list.

| You hear | What's broken | What to do |
|---|---|---|
| "I say, I can't see a blessed thing." | Camera — no picture reaching gbox1 (USB cable loose, wrong port, or the webcam needs a replug). | Unplug the webcam, count to ten, plug it back into gbox1's own USB-C port with its labelled cable. gbox1 keeps looking for it and re-announces every 2 min until it's back; nothing to type. |
| "The judging mind has gone quite blank. One moment, please." | Costume-judging AI not answering. | Falls back to pre-written lines only (still funny, not freshly written). Give it a minute; if it persists past the rehearsal, get Claude's help. |
| "My better judgement has abandoned me. Standing by, regardless." | The joke safety filter not answering. | Same fallback as above — nothing unvetted plays either way. |
| "My voice has utterly failed me. How frightfully embarrassing." | Text-to-speech engine failed. | Pre-written lines are pre-recorded and still play; anything needing a freshly spoken line won't. |
| "The speaker itself has gone silent. Do fetch an engineer." | Audio hardware — dongle or its cable to the PA. | Check the dongle's seated and the PA cable's in. Nothing plays until fixed, including this warning (it'd be logged, not spoken, if the speaker itself is dead). |

Nothing at all after 5 minutes, not even an error line? See §10.

---

## 7. Kill switch

**Unplug the speaker.** Instant silence, nothing else to touch, and nothing else stops
working for anyone still enjoying it once you plug back in.

---

## 8. Shutdown

Same deliberate way as §2 step 5: power button or unplug directly. No software
stop/shutdown command needed first.

---

## 9. Back home

1. Reconnect gbox1 to your home network, power on.
2. Run the switch-back-to-normal step (exact command in the appendix) — stops the
   scarebox software (it auto-starts and holds memory otherwise) and reloads your
   everyday AI model.
3. Confirm that model's automatic health-check comes back clear before calling it done.
   If it doesn't, get help rather than walking away.

---

## 10. Troubleshooting

| Problem | Likely cause | What to do |
|---|---|---|
| No sound, no error line either | Dead PA power, bad line-in cable, or gbox1 never finished booting | Check PA power/cable. If gbox1 seems unresponsive, laptop + router Wi-Fi + SSH in. |
| Wrong costume called out | The judging AI isn't perfect — wrong roughly 1 in 15 on test footage, and never comments on the person, only the guessed costume | An accepted error rate, not a live fix. Flag it afterward if it's constant, not occasional. |
| Too many canned-sounding lines | The live joke-writing missed its time budget and fell back to a pre-written line — designed behaviour | Normal, especially with runners close together. If it's nearly every runner, check §6 for a matching error line. |
| Camera aimed wrong | The webcam got knocked | Re-open the aim page (§4), move it back. Under a minute. |
| Jokes land after runners have passed | Speaker too close to the camera | Move the speaker further down the course (§5). |
| Power station reading low | Running longer than planned, or not on the AC outlet | Confirm you're on the AC outlet, not a USB port. Carry a spare charged station for an all-day event. |

---

## Author's homelab appendix

*(Specific to Martin's home network — reference-only for whoever's helping set this up.
Still no secrets. Public body above uses placeholders, not real addresses.)*

- gbox1 runs the everyday big AI model when it isn't running scarebox — the two can't run
  at once; scarebox needs the memory that model would be using. Since 2026-10-22 that model
  is Qwen3.8-Flash-Next (container `gbox1-q38-vllm`); the older "B1" stays on disk as the
  rollback.
- Switch-over script, run from Martin's workstation: `homelab/gbox1-mode.sh`
  - `scarebox` — frees memory by unloading whichever big model is loaded (do this before
    testing scarebox at home).
  - `normal` — stops the scarebox containers, reloads Qwen3.8 via the sanctioned
    `b1-cold-reload.sh` script (this is §9 step 2). `ENGINE=b1` in front reloads B1 instead.
  - `status` — read-only: what's loaded, memory free.
- "Confirm the canary is clear" (§9 step 3): check the automated watchdog that pages if
  the big model comes back unhealthy (`gbox1-canary`, runs every 10 min from Martin's workstation) hasn't fired —
  ask Claude, or check for a Telegram alert.
- **The camera is the USB webcam** (`source.kind: usb` in `config/field.yaml`, Martin's call
  2026-10-19). The app finds it by name (`usb_name: WB7022`), not by `/dev/videoN` (the
  number changes on replug), and reads 1080p MJPG at 30 fps, the only mode it asks for: a
  request for an unsupported mode is the likeliest thing that knocked it off the bus on
  2026-10-18. The app container sees the host's `/dev` read-only at `/host-dev`, so a
  replugged camera is found without restarting anything. The Amcrest is still supported
  (`source.kind: rtsp`) as a fallback. gbox1's router-side address is fixed (198.51.100.10
  here) on the built-in Ethernet port (`deploy/host/90-NM-builtin-port.yaml`), as an
  extra address beside whatever DHCP hands out, so §4 needs no router lookup.
- Exact command for §2 step 4, from the repo root on gbox1 as the `scarebox` user:
  `SCAREBOX_CONFIG=config/field.yaml docker compose -p scarebox -f deploy/compose.yaml --env-file deploy/images.env up -d`
- **The voice is its own container** (2026-10-16). Sir Reginald is Chatterbox, which runs as
  `scarebox-tts` from its own image, separate from the app. The `up -d` above starts it, and the
  app waits until it reports healthy, about 30 s after the camera model. If the image is ever
  missing (say after a re-image), rebuild it while online, from the repo root:
  `docker build -t scarebox/tts-chatterbox:dev -f deploy/Dockerfile.chatterbox deploy`.
  Its weights live in `~/models/chatterbox`. It starts with no network (verified).
- **One `.env` file, home and race.** The ordinary `.env` file is all the race settings
  need (checked 2026-10-16: `field.yaml` loads cleanly from `.env`). There's no separate
  race `.env.field` to maintain. The aim-page token in §4 is the `AIM_TOKEN` line in `.env`.

**Still unproven, honestly, until the rehearsal:** real speaker audio through the USB
dongle (tested so far only with a fake audio file or simulated hardware); an actual power-
off/power-on cycle at the course (§2–§3's mechanism is reasoned through, not yet run for
real); the travel router's own networking (phone to aim page); the webcam outdoors in
daylight, and where the trigger line and minimum runner size need to sit for its wider
view; the boot-time backstop unit (needs a sudo password gbox1's automated builds don't
have — a rehearsal-day install, not already running).
