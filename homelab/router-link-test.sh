#!/usr/bin/env bash
# Router-link test for gbox1 (2026-10-24): what happens when gbox1's built-in Ethernet (enP7s7) is
# moved from the home LAN to the scarebox TP-Link travel router. Run as root on gbox1 via
# systemd-run so it survives the SSH session dropping:
#
#   sudo systemd-run --unit=scarebox-nettest --collect /var/tmp/router-link-test.sh [minutes]
#
# Every 5 s it logs to /var/tmp/scarebox-nettest.log: link state, addresses, default route, and
# whether the travel router and the home gateway answer. It also serves a tiny test page on
# :8090 (only if nothing else holds the port, i.e. scarebox is stopped) that logs every client,
# so a phone opening http://<the box's fixed address>:8090/ shows up in the log.
# Read-only apart from the log and the test page. Stops by itself after [minutes] (default 40).
#
# Set these for your own network (the defaults are RFC 5737 documentation addresses):
#   NETTEST_IF         the port under test             (default enP7s7, gbox1's built-in port)
#   NETTEST_ROUTER_GW  the travel router's address     (default 198.51.100.1)
#   NETTEST_HOME_GW    the home LAN gateway's address  (default 203.0.113.1)
# systemd-run does not pass your shell's environment through; add -E VAR=value for each.
set -u
LOG=/var/tmp/scarebox-nettest.log
IF="${NETTEST_IF:-enP7s7}"
ROUTER_GW="${NETTEST_ROUTER_GW:-198.51.100.1}"
HOME_GW="${NETTEST_HOME_GW:-203.0.113.1}"
MINUTES="${1:-40}"
END=$(( $(date +%s) + MINUTES * 60 ))
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "== start: watching $IF for $MINUTES min (router $ROUTER_GW, home $HOME_GW)"
SERVER_PID=""
if ! ss -ltn '( sport = :8090 )' | grep -q 8090; then
    python3 - >> "$LOG" 2>&1 <<'EOF' &
import http.server, socketserver, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<h1>gbox1 reachable</h1><p>scarebox router-link test: you are " + self.client_address[0].encode() + b"</p>"
        self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(body)
    def log_message(self, fmt, *args):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "PAGE", self.client_address[0], fmt % args, flush=True)
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("0.0.0.0", 8090), H).serve_forever()
EOF
    SERVER_PID=$!
    log "test page serving on :8090 (pid $SERVER_PID)"
else
    log "port 8090 already in use (scarebox running?): not starting the test page"
fi

prev=""
while [ "$(date +%s)" -lt "$END" ]; do
    carrier=$(cat /sys/class/net/$IF/carrier 2>/dev/null || echo "?")
    addrs=$(ip -4 -br addr show $IF | awk '{$1=$2=""; print $0}' | xargs)
    route=$(ip route show default | awk '{print $3"@"$5}' | xargs)
    router=$(ping -c1 -W1 "$ROUTER_GW" >/dev/null 2>&1 && echo up || echo down)
    home=$(ping -c1 -W1 "$HOME_GW" >/dev/null 2>&1 && echo up || echo down)
    state="carrier=$carrier addrs=[$addrs] default=[$route] router=$router home=$home"
    # Log every change, and a heartbeat once a minute.
    if [ "$state" != "$prev" ] || [ $(( $(date +%s) % 60 )) -lt 5 ]; then log "$state"; prev="$state"; fi
    sleep 5
done
[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
log "== end"
