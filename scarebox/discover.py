"""discover: find the camera by MAC on whatever subnet it lands on. Spec §4.3's "home-to-race
subnet change is zero-config": the race router hands out DHCP on its own /24 (say
198.51.100.0/24), home is a different /24 entirely, and the race-day config carries the
camera's MAC, not a fixed IP.

`find_camera(mac, subnet=None)` concurrently TCP-probes port 554 (RTSP) across the /24, then
reads `/proc/net/arp` and returns the IP whose MAC matches (case-insensitive). Every I/O step
(the probe, the ARP read, the default-subnet lookup) is an injectable keyword so tests never
touch a real socket, a real `/proc/net/arp`, or a real network interface.
"""

from __future__ import annotations

import ipaddress
import socket
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

_RTSP_PORT = 554
_PROBE_TIMEOUT_S = 0.2
_PROBE_CONCURRENCY = 64
_ARP_PATH = "/proc/net/arp"
_ROUTE_PATH = "/proc/net/route"
_CACHE_PATH = "runs/camera_ip"


class CameraNotFoundError(Exception):
    """Raised when no host on the probed subnet carries the target MAC."""


# -- low-level, individually-fakeable steps ----------------------------------------------


def _tcp_probe(ip: str, port: int = _RTSP_PORT, timeout_s: float = _PROBE_TIMEOUT_S) -> bool:
    """True if `ip:port` accepts a TCP connection within `timeout_s`. The real network path
    -- tests never call this, they inject a fake `probe_fn` instead."""
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _hosts_in_subnet(subnet: str) -> list[str]:
    """Every usable host address in a /24 (or whatever prefix `subnet` carries), as strings."""
    network = ipaddress.ip_network(subnet, strict=False)
    return [str(host) for host in network.hosts()]


def _probe_hosts(
    hosts: Iterable[str],
    *,
    port: int = _RTSP_PORT,
    timeout_s: float = _PROBE_TIMEOUT_S,
    concurrency: int = _PROBE_CONCURRENCY,
    probe_fn: Callable[[str, int, float], bool] = _tcp_probe,
) -> list[str]:
    """Probes every host in `hosts`, `concurrency` at a time (the pool's own size caps how
    many are ever in flight together), and returns the ones that answered."""
    hosts = list(hosts)
    open_ips: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(probe_fn, ip, port, timeout_s): ip for ip in hosts}
        for future in as_completed(futures):
            ip = futures[future]
            if future.result():
                open_ips.append(ip)
    return open_ips


def _read_arp_table(path: str | Path = _ARP_PATH) -> dict[str, str]:
    """Parses `/proc/net/arp`'s fixed-column format into `{ip: mac_lowercase}`. Header line
    is `IP address       HW type     Flags       HW address            Mask     Device`."""
    text = Path(path).read_text()
    table: dict[str, str] = {}
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 4:
            continue
        ip, mac = fields[0], fields[3]
        table[ip] = mac.lower()
    return table


def _default_route_iface(path: str | Path = _ROUTE_PATH) -> str:
    """Parses `/proc/net/route` for the interface carrying the default route (Destination
    `00000000`). Raises :class:`CameraNotFoundError` if there isn't one."""
    text = Path(path).read_text()
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 2:
            continue
        iface, dest = fields[0], fields[1]
        if dest == "00000000":
            return iface
    raise CameraNotFoundError(f"no default route in {path}")


def _iface_ipv4_and_mask(iface: str) -> tuple[str, str]:
    """`SIOCGIFADDR`/`SIOCGIFNETMASK` for `iface` (Linux-only, the real path -- tests inject
    a fake `iface_fn` instead of exercising this)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        ifreq = struct.pack("256s", iface.encode()[:15])
        addr = socket.inet_ntoa(_ioctl(sock, 0x8915, ifreq)[20:24])  # SIOCGIFADDR
        mask = socket.inet_ntoa(_ioctl(sock, 0x891B, ifreq)[20:24])  # SIOCGIFNETMASK
        return addr, mask
    finally:
        sock.close()


def _ioctl(sock: socket.socket, request: int, arg: bytes) -> bytes:
    import fcntl  # Linux-only; imported lazily so this module still imports on non-Linux CI

    return fcntl.ioctl(sock.fileno(), request, arg)


def _default_subnet(
    *,
    route_path: str | Path = _ROUTE_PATH,
    iface_fn: Callable[[str], tuple[str, str]] = _iface_ipv4_and_mask,
) -> str:
    """The /24 (or whatever the interface's real mask is) of the default-route interface,
    as a CIDR string, e.g. `"198.51.100.0/24"`."""
    iface = _default_route_iface(route_path)
    ip, mask = iface_fn(iface)
    network = ipaddress.IPv4Network(f"{ip}/{mask}", strict=False)
    return str(network)


# -- the public entry point ----------------------------------------------------------------


def find_camera(
    mac: str,
    subnet: str | None = None,
    *,
    port: int = _RTSP_PORT,
    timeout_s: float = _PROBE_TIMEOUT_S,
    concurrency: int = _PROBE_CONCURRENCY,
    arp_path: str | Path = _ARP_PATH,
    cache_path: str | Path = _CACHE_PATH,
    probe_fn: Callable[[str, int, float], bool] = _tcp_probe,
    subnet_fn: Callable[[], str] = _default_subnet,
) -> str:
    """Finds the camera carrying `mac` (any case, any separator-free hex) on `subnet` (a
    CIDR string; derived from the default-route interface when omitted).

    Fast path: if `cache_path` holds a previous result, and that IP both answers the probe
    and still maps to `mac` in the ARP table, it's returned without scanning the whole
    subnet. Otherwise this probes every host in the subnet concurrently, reads the ARP
    table once, and returns the first open-port IP whose MAC matches -- writing it to
    `cache_path` for next time. Raises :class:`CameraNotFoundError` if nothing matches.
    """
    target_mac = mac.strip().lower()
    cache = Path(cache_path)
    if cache.exists():
        cached_ip = cache.read_text().strip()
        if cached_ip and probe_fn(cached_ip, port, timeout_s):
            arp = _read_arp_table(arp_path)
            if arp.get(cached_ip, "").lower() == target_mac:
                return cached_ip

    if subnet is None:
        subnet = subnet_fn()

    open_ips = _probe_hosts(
        _hosts_in_subnet(subnet),
        port=port,
        timeout_s=timeout_s,
        concurrency=concurrency,
        probe_fn=probe_fn,
    )
    arp = _read_arp_table(arp_path)
    for ip in open_ips:
        if arp.get(ip, "").lower() == target_mac:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(ip)
            return ip

    raise CameraNotFoundError(f"no host on {subnet} with MAC {mac} answered port {port}")
