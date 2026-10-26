# Hardware

What scarebox ran on at the 2026-10-25 race, how it was wired, what a cold boot does, and what the
hardware taught us along the way. For the software side of each assumption (which config key or
file holds it) see `AGENTS.md`; for the race-morning procedure see `docs/RUNBOOK.md`.

## Terms

- **gbox1**: the author's box, an ASUS Ascent GX10. It is built on NVIDIA's **GB10** chip: a
  64-bit ARM (aarch64) CPU and a Blackwell GPU sharing 121 GiB of memory ("unified memory").
- **Dongle**: a small USB sound card. Its headphone-style output feeds the PA speaker's line input.
- **PA**: a public-address speaker, the loud kind with its own amplifier and power.
- **Line-in**: a speaker's input for an unamplified audio signal (3.5 mm jack or RCA plugs).
- **UVC**: "USB Video Class", the standard most USB webcams follow, so Linux drives them with no
  extra software.
- **Cold boot**: starting from the power being off, with nobody at a keyboard.
- **Aim page**: the web page the box serves on port 8090 so a phone can see the camera picture and
  set the trigger line.
- **USB controller**: the chip inside the computer that a group of USB ports hangs off. If one
  stops responding, every port on it goes dead.

## Parts

| Part | What ran on race day | Facts that matter |
|---|---|---|
| Compute | ASUS Ascent GX10 (NVIDIA GB10, aarch64, 121 GiB unified memory) | CUDA compute capability 12.1. Its I/O is USB-C ports, HDMI and a built-in Ethernet port. Powered by its own stock **240 W USB-C** brick; nothing weaker was tried. The design estimate was about 40 W idle and 200 W flat out; in the 2-hour replay soak the GPU alone averaged 22 W and peaked at 91 W (nvidia-smi; whole-box draw isn't visible in software). |
| Camera | Dell UltraSharp **webcam** WB7022 (USB ID `413c:c015`) | A UVC webcam on a USB-C cable straight into one of gbox1's own USB-C ports, no hub. Read as 1920x1080 MJPG at 30 fps, the one mode the app asks for; that held 30 fps with zero failed reads over a 20-minute test. It also shows up as a presence sensor (`413c:d001`) and two internal hubs (`0bda:0412`, `0bda:5412`). No motor, so it stays where you point it. Indoor camera: shade it and keep rain off. |
| Audio | GeneralPlus USB audio **dongle** (USB ID `1b3f:2008`) | Into the PA's line-in over a long cable (10 to 15 m). Output level stored at 80% (`sudo alsactl store scarebox_out`) because full level distorted the PA's line-in; loudness is set on the PA. No Bluetooth: it adds 150 to 300 ms and drops pairing. |
| Speaker | A PA speaker with a line input and its own power | Check whether it wants 3.5 mm or RCA, and pack an adapter. |
| Network | A small TP-Link travel router, no internet uplink | Only there so a phone (and, if needed, a laptop) can reach the box. Ethernet from the router into gbox1's **built-in** Ethernet port. gbox1 keeps one fixed extra address on that port (198.51.100.10 in these docs) so the aim page is always at the same URL. |
| USB hub | A USB-C hub for anything that isn't the webcam | The webcam does not go through it. Keep the dongle on a different USB controller from the webcam (see the lessons below). |
| Power | 1000 Wh portable power station | gbox1's brick goes into the station's **AC outlet** (looks like a wall socket), not its USB-C port: most power stations' USB-C ports stop at 100 to 140 W. Design estimate: about 150 W average, so a 2.5-hour outing uses about 430 Wh including inverter loss. |
| Spares | Gaffer tape, a spare USB-C cable, a spare Ethernet cable, a laptop | The laptop is the way in if the speaker never says its start-up line. |

The camera the project started with, an Amcrest IP4M-1041B pan/tilt network camera, is still
supported over RTSP (`source.kind: rtsp`) but was dropped on 2026-10-19. It has no clock battery
(its clock reset to the year 2000 on every power cut, which breaks ONVIF's time-checked login) and
it recalibrates its pan position at power-up, so it had to be re-aimed after every boot. It also
needed its own 5 V power and a network port. The webcam needs none of that.

## Wiring

```
                        travel router (Wi-Fi for phones, no internet)
                              │ Ethernet
                              ▼
 webcam ──USB-C──▶  gbox1 (built-in Ethernet port)  ──USB──▶ audio dongle ──10–15 m cable──▶ PA line-in
  (on gbox1's own    │                                                                       (speaker
   USB-C port)       │ stock 240 W USB-C brick                                                down the
                     ▼                                                                         course)
             power station AC outlet
```

Physically: gbox1 and the webcam sit together at the upstream end, the webcam pointing across the
course. The speaker goes downstream, because a line is written and voiced in the seconds after a
runner crosses the trigger line, and it should come out as that runner arrives. On the author's
crowd tuning, a line started a median 5 s after the crossing and 9 in 10 within 7.7 s, which put
the speaker about 15 to 20 m past the camera for joggers and 8 to 12 m for walking children. Every
line names the costume aloud, so it still lands when runners move at different speeds.
`docs/RUNBOOK.md` §5 has the rule of thumb.

## Host setup the hardware needs

All in `deploy/host/` with install steps in its `README.md` (they need root):

- `99-scarebox-audio.rules` (udev): names the dongle's sound card `scarebox_out`, the name every
  config's `stage.alsa_device` uses, and turns its USB power saving off.
- `99-scarebox-camera.rules` (udev): turns USB power saving off for the webcam, its presence sensor
  and every hub between it and the box.
- `51-scarebox-ignore-dongle.lua` / `.conf` (WirePlumber): stops a desktop audio session from
  grabbing the dongle. Installed system-wide, because on a cold boot the login screen runs its own
  audio session too.
- `scarebox.service` (systemd): starts the whole stack once at every boot.
- `90-NM-builtin-port.yaml` (NetworkManager): the example profile for the fixed extra address.

## Cold boot

Proven on 2026-10-23 (home cable out, no internet, power off then on) and again on race morning.
What happens after the power button:

1. Linux boots. Docker brings back the containers that were running when the power went
   (`restart: unless-stopped`), and `scarebox.service` runs `docker compose up -d` so the stack
   comes up even if it had been stopped. **Leave the stack running when you cut the power**; don't
   stop it first.
2. Docker restarts every container at once and ignores compose's `depends_on`, so the app starts
   before the model servers are ready. The app therefore waits up to 5 minutes for the vision
   model, the guard model and the voice server to answer before it builds anything, and keeps its
   own frame watchdog fed while it waits.
3. The self-test checks the camera, the vision model, the safety filter, the voice and the speaker,
   then says "The adjudicator is seated. Let the parade commence!" or names the first broken part
   (the table is in `docs/RUNBOOK.md` §6). On the 2026-10-23 test the vision model was ready about
   70 s after power-on. Budget up to 5 minutes before you go looking for a laptop.
4. The webcam can be plugged in before or after power-on: the app finds it by name and keeps
   looking, re-announcing every 2 minutes until it appears.

Shutting down: a single short tap of the GX10's power button does a clean shutdown, and the box
boots straight back into scarebox next time. Pulling the power also works; the stack is built for it.

Two checks before you rely on a cold boot:

- `ls /boot/efi/EFI/UpdateCapsule/` must be empty. A staged firmware update would try to apply
  itself at the course.
- After a hard power cut, `docker logs` stops at the run of zero bytes the cut leaves in the log
  file. Read the container's JSON log file directly with `sudo grep -a` instead.

## Field lessons

**Seat the USB-C plug firmly, then check the link speed.** On race morning the webcam's USB 3
connection didn't come up at power-on; only its USB 2 half did, which isn't enough for the video.
Pushing the plug fully home fixed it, and the camera then held for the whole two hours. A USB-C
plug can sit far enough in to power a device and still miss the high-speed contacts. After plugging
in, `lsusb -t` should show the webcam at 5000M or faster, not 480M.

**Use the box's own port and one proven cable for the webcam.** On 2026-10-18 the webcam dropped
off the box on a different cable and port and stayed gone. On a labelled cable straight into
gbox1's own USB-C port it then ran 20 minutes at 30 fps with no dropped frames. Ask the webcam only
for a mode it lists (`v4l2-ctl --list-formats-ext`): an unsupported mode request is the likeliest
thing that knocked it off the bus.

**A USB controller can die, and only a reboot brings it back.** On 2026-10-19 one of gbox1's USB
controllers stopped responding mid-stream with the webcam attached (`journalctl -k` showed
`HC died`). No software reset revived it: unbinding and rebinding the driver failed. Only a reboot,
or moving the device to a port on another controller, works. The camera and its hubs were all in
USB power saving at the time, the commonest trigger for that error with USB 3 webcams, so
`99-scarebox-camera.rules` now turns power saving off for the whole camera chain. That is a likely
fix, not a proven one. Keep the camera and the dongle on different controllers so one failure can't
take out both.

**Swapping a hub has knock-on effects.** The hub was swapped twice in the week before the race.
The first swap (2026-10-19) gave the dongle a new USB address, and the app container, which had been
handed a one-off copy of the sound devices at start-up, lost it: every line failed with "No such
device" while everything else looked healthy. Compose now bind-mounts `/dev/snd` (and the host's
`/dev` for the webcam), so a replugged device is visible without a restart. The old hub also had an
Ethernet port, which was the router's way in; its replacement has none, so on 2026-10-24 the router
moved to gbox1's built-in Ethernet port and the fixed address moved with it. After any hub change,
re-run the self-test and re-check the aim page's address.

**The battery: AC outlet, charged the night before, and read the display.** The box's brick goes on
the power station's AC outlet. The draw of the whole box can't be read from software (nvidia-smi
reports the GPU only), and it wasn't logged at the race, so if you need numbers, read them off the
power station's display. At the design estimate of about 150 W, a 1000 Wh station covers the two
hours with more than twice the energy needed. For an all-day event, carry a second charged station.

**Desktop audio will grab the dongle if you let it.** A desktop session's PipeWire treats the
dongle as just another speaker. The WirePlumber rule keeps it off, and it has to be installed
system-wide: per-user wasn't enough, because the login screen on a cold boot runs its own audio
session. Also, `aplay -D scarebox_out` fails because that's a card name, not a playback device; the
app turns it into `plughw:CARD=scarebox_out,DEV=0`.

**Silence is the worst failure.** There is no screen at the course; the speaker is the status
display. The race config uses a real audio device or fails loudly, and the self-test names the
first broken part. One gap remains: on race morning, while the camera was missing, the box stayed
silent instead of saying its camera line. That is an open bug (`docs/RACE-REPORT.md`).
