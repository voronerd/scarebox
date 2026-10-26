# deploy/host/: files for gbox1's host OS, not the containers

Everything here needs root (`sudo`) to install, and the unattended sessions that build
scarebox have NO sudo on gbox1 (project rule). These files are written and reviewed in the
repo, then applied by hand at the Phase 5 rehearsal (or by the hub, hub-direct, if asked).

## `scarebox.service` -- boot backstop, tied to gbox1's mode (2026-10-23)

**What it's for:** a field cold start. `restart: unless-stopped` in `deploy/compose.yaml` only
brings back containers that were running when the power went. A stack that was stopped (which
`homelab/gbox1-mode.sh normal` does) stays down. This oneshot runs `compose up -d` with
`SCAREBOX_CONFIG=config/field.yaml` once at every boot, so the stack comes up from cold either way.

**Only enabled in scarebox mode.** `homelab/gbox1-mode.sh scarebox` installs it, enables it and
starts it; `homelab/gbox1-mode.sh normal` disables it before stopping the stack. A normal-mode
boot therefore never starts scarebox on top of the fleet's big model. Don't enable it by hand.

**Verify:** `homelab/gbox1-mode.sh status` shows `scarebox boot unit: enabled` in scarebox mode,
`disabled` in normal mode.

## Fixed router-side address: `90-NM-builtin-port.yaml` (2026-10-24)

A reference copy, with placeholders, of the netplan file NetworkManager keeps on gbox1 for its
built-in Ethernet port (`enP7s7`, connection "Wired connection 3"), with one addition: an extra
fixed address on the travel router's subnet. These docs use the documentation address
198.51.100.10/24; pick a free address on your own router's subnet, outside its DHCP pool. At home
the port takes its usual DHCP lease and the extra address does nothing; plugged into the travel
router it takes a router lease as well, and the phone page is always at
`http://198.51.100.10:8090/`. Add it in place, with no link bounce, with
`sudo nmcli con modify "Wired connection 3" +ipv4.addresses 198.51.100.10/24 && sudo nmcli dev
reapply enP7s7` (substitute your connection name, address and interface); remove it with
`-ipv4.addresses` the same way. Put the same subnet in `config/local.yaml`'s
`aim.trusted_networks` so phones on the router need no token. Router-link test 2026-10-24
(`homelab/router-link-test.sh`): router lease and the fixed address within 15 s, phone page
loaded, 6 lines played, home lease back on replug.

This replaced an earlier (2026-10-23) profile for the Ethernet port of the old USB hub; that hub
was swapped for an Anker with no network port, and the profile was deleted.

## WirePlumber rule, system-wide (2026-10-23)

`51-scarebox-ignore-dongle.lua` keeps PipeWire off the race audio dongle so only the app container
opens it. gbox1 runs WirePlumber 0.4.17, which reads Lua rules from `main.lua.d/`. It was first
installed per-user for `martin` only; on a cold boot the GDM login screen runs its own PipeWire with
no such rule. `gbox1-mode.sh scarebox` now also installs it to `/etc/wireplumber/main.lua.d/`, which
every user's WirePlumber reads (login screen, `martin`, `scarebox`).

## Audio files (04-01 Task 3): `99-scarebox-audio.rules` + `51-scarebox-ignore-dongle.conf`

**What they're for:** two different programs on gbox1 can both try to grab the USB audio
dongle -- the kernel/ALSA layer (which device name it gets) and Martin's own desktop
PipeWire/WirePlumber session (which would otherwise treat it as just another sink and
start playing desktop audio through it, or worse, open it before the `app` container can).
`99-scarebox-audio.rules` is a **udev** rule (kernel-level, root-installed) that pins the
dongle to a stable ALSA name, `scarebox_out` -- the exact name `stage.alsa_device` in every
`config/*.yaml` already expects, so the app never has to guess which `/dev/snd/pcmC*` node
is the right one. `51-scarebox-ignore-dongle.conf` is a **WirePlumber** rule (desktop-level,
no root, installed as `martin`) that tells the desktop session's own PipeWire to ignore
that device completely, so it never competes with the container for it.

**Installed on gbox1 2026-10-19** for the GeneralPlus dongle (USB `1b3f:2008`). Card 1 is
now named `scarebox_out`. The desktop session no longer opens it. A loopback test (dongle
output cabled into its own mic input) recorded a real library line played by the app
container through the race config. Its output volume is stored at 80% (-9 dB, Martin
2026-10-19, to avoid distorting the PA's line-in) with `sudo alsactl store scarebox_out`;
the PA's own knob sets the loudness.

**Gotchas found while installing:**
- `aplay -D scarebox_out` fails, because that's a card name, not a PCM. `stage.AlsaSink`
  turns it into `plughw:CARD=scarebox_out,DEV=0`.
- gbox1 runs **WirePlumber 0.4.17**, which ignores the 0.5-style
  `51-scarebox-ignore-dongle.conf`. Use `51-scarebox-ignore-dongle.lua` in
  `~/.config/wireplumber/main.lua.d/` instead. Step 6 below describes the 0.5 path. For a
  **different dongle**, change the IDs in the udev rule and the `device.name` match in
  the Lua file.

**Install, in order, once the dongle is plugged into gbox1:**

1. **Find the real IDs:**
   ```bash
   lsusb                                            # note the dongle's "idVendor:idProduct", e.g. 0d8c:0014
   # or, once you know which sound card number it got:
   udevadm info -a -n /dev/snd/controlC<N> | grep -E 'idVendor|idProduct'
   ```
2. **Fill in the udev rule** -- replace both `XXXX`/`YYYY` pairs in
   `99-scarebox-audio.rules` with the real hex IDs from step 1 (same pair, used twice: once
   for the ALSA name pin, once for the autosuspend-off rule).
3. **Install the udev rule** (root; NOT the `scarebox` user, which has none):
   ```bash
   sudo cp /home/scarebox/scarebox/deploy/host/99-scarebox-audio.rules /etc/udev/rules.d/99-scarebox-audio.rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```
4. **Confirm the ALSA name stuck:**
   ```bash
   aplay -l           # should list "scarebox_out" as one of the cards
   cat /proc/asound/cards   # same name, from the kernel's own card table
   ```
5. **Find the WirePlumber device name** (now that the card has a stable name, the desktop
   session has already seen it as a sink -- find its real `device.name` so step 6 can
   target it precisely):
   ```bash
   wpctl status                              # find the dongle's id under Audio > Sinks
   wpctl inspect <id>                         # or: pw-dump | grep -B5 '"device.name"'
   ```
   `device.name` looks like `alsa_card.usb-XXXX_YYYY-00` -- the SAME vendor/product hex
   pair from step 1.
6. **Fill in the WirePlumber rule** -- replace `XXXX`/`YYYY` in
   `51-scarebox-ignore-dongle.conf` with the same IDs, then install it (as `martin`, the
   desktop user -- **not** `scarebox`, and no root needed):
   ```bash
   mkdir -p ~martin/.config/wireplumber/wireplumber.conf.d
   cp /home/scarebox/scarebox/deploy/host/51-scarebox-ignore-dongle.conf \
     ~martin/.config/wireplumber/wireplumber.conf.d/51-scarebox-ignore-dongle.conf
   systemctl --user restart wireplumber     # as martin, in martin's own session
   ```
7. **Confirm the desktop session is no longer touching it:**
   ```bash
   wpctl status       # the dongle must NOT appear under Audio > Sinks any more
   ```

**Full verification of all three host files together, once installed:**

```bash
aplay -l                           # "scarebox_out" present (udev rule)
wpctl status                       # scarebox_out NOT listed (WirePlumber rule)
systemctl is-enabled scarebox      # "enabled" (boot backstop unit)
```

## Camera power rule: `99-scarebox-camera.rules` (2026-10-19)

**What it's for:** it keeps the Dell WB7022 webcam, its presence sensor, the hubs inside it,
and the swapped-in hubs between it and gbox1 out of USB power saving (`power/control` = `on`;
the kernel default `auto` suspends after 2 s idle). Background: gbox1's USB controller
`NVDA8000:02` died mid-stream with the camera attached, and no software reset revived it.
Autosuspend is the commonest cause of that kernel error
with USB 3 webcams, so this is the cheap, reversible step. It's a likely fix, not a proven
one: soak-test it.

**Installed on gbox1 2026-10-19:**
```bash
sudo install -m 644 /home/scarebox/scarebox/deploy/host/99-scarebox-camera.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=usb --action=change
```
**Verify:** every device it lists should report `on`:
```bash
for d in /sys/bus/usb/devices/*; do echo "$(cat $d/idVendor 2>/dev/null):$(cat $d/idProduct 2>/dev/null) $(cat $d/power/control 2>/dev/null)"; done | grep -E '413c:c015|413c:d001|0bda:04|0bda:54|2109:0822|2109:2822'
```
**If a controller dies anyway** (`journalctl -k` shows `HC died`): only a reboot, or moving the
camera to a port on another controller, brings it back. `unbind`/`bind` of the `xhci-hcd`
platform device fails with `-110`. Keep the camera and the audio dongle on different
controllers, so one failure can't take out both.
