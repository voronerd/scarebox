-- WirePlumber 0.4.x version of 51-scarebox-ignore-dongle.conf (gbox1 runs 0.4.17, which
-- reads Lua from main.lua.d/ and ignores the 0.5-style wireplumber.conf.d/ file; found
-- 2026-10-19 when the .conf had no effect). Same intent: Martin's desktop PipeWire must
-- leave the race audio dongle alone so only the app container ever opens it.
--
-- Installs as martin (no root): ~/.config/wireplumber/main.lua.d/51-scarebox-ignore-dongle.lua
-- then `systemctl --user restart wireplumber`. See deploy/host/README.md.

table.insert(alsa_monitor.rules, {
  matches = {
    {
      { "device.name", "matches", "alsa_card.usb-GeneralPlus_USB_Audio_Device-*" },
    },
  },
  apply_properties = {
    ["device.disabled"] = true,
  },
})
