# homelab/: one person's setup, kept as an example

Nothing in this directory is part of scarebox itself. These are the author's own scripts for
his own machine, published because they show how a shared GPU box can be switched between its
everyday job and race duty. They will not run as-is anywhere else: they assume his box's name
(`gbox1`, reached over SSH from his workstation), his user accounts, his container names and his
host-side scripts under `/srv/gbox1/`. Read them, borrow the ideas, and write your own.

If your box runs only scarebox, you need none of this: follow `docs/RUNBOOK.md` and the host
files in `deploy/host/`.

## What is here

- **`gbox1-mode.sh`**: switches the box between "normal" mode (a large language model that
  serves the rest of the author's home network) and "scarebox" mode. The two can't run at once,
  because scarebox needs the memory the big model holds.
  - `status` is read-only: running containers, free memory, whether the scarebox boot unit is
    enabled.
  - `scarebox` stops the big model the same way the box's own reload script does, installs and
    enables `deploy/host/scarebox.service` (so the stack comes back after a power cut at the
    course) and the system-wide WirePlumber rule that keeps desktop audio off the race dongle.
  - `normal` disables that boot unit, stops the scarebox containers, checks there is enough free
    memory, and reloads the big model through the box's own reload script.
  - `--dry-run` prints the remote commands instead of running them.
- **`router-link-test.sh`**: a 40-minute watcher, run as root on the box, that logs what the
  built-in Ethernet port does when it is moved from the home network to the travel router (link
  state, addresses, default route, whether each gateway answers) and serves a test page on
  port 8090 so a phone's visit shows up in the log. The interface and both gateway addresses
  are set by environment variables; the defaults are documentation addresses.

## What to change for your own box

- The SSH host name (`GBOX1_HOST`, default `gbox1`) and the user that has `sudo` there.
- Whatever your box runs when it isn't running scarebox. The container names, compose files
  and `/srv/gbox1/compose/b1-cold-reload.sh` in `gbox1-mode.sh` are specific to the author's box;
  replace the stop and reload steps with your own, or drop them.
- The repo path on the box (`/home/scarebox/scarebox` here).
- `NETTEST_IF`, `NETTEST_ROUTER_GW` and `NETTEST_HOME_GW` for `router-link-test.sh`.
