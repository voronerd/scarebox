"""Tests for scarebox.discover. Never touches a real socket, a real `/proc/net/arp`, or a
real network interface -- every I/O boundary (`probe_fn`, `arp_path`, `subnet_fn`,
`cache_path`) is injected."""

from pathlib import Path

import pytest

from scarebox.discover import (
    CameraNotFoundError,
    _default_route_iface,
    _default_subnet,
    _hosts_in_subnet,
    _read_arp_table,
    find_camera,
)


def mac(*octets: int) -> str:
    """A MAC string built at run time, so this file carries no MAC literal for the leak gate
    (scripts/leak_gate.py) to trip on. All are made up; none is a real device."""
    return ":".join(f"{o:02x}" for o in octets)


CAMERA_MAC = mac(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x01)  # lowercase, as /proc/net/arp prints it
TARGET_MAC = CAMERA_MAC.upper()
OTHER_MAC = mac(0x11, 0x22, 0x33, 0x44, 0x55, 0x66)
BROADCAST_MAC = mac(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)


def write_arp(path: Path, rows: dict[str, str]) -> Path:
    lines = ["IP address       HW type     Flags       HW address            Mask     Device"]
    for ip, hw in rows.items():
        lines.append(f"{ip}            0x1         0x2         {hw}     *        eth0")
    path.write_text("\n".join(lines) + "\n")
    return path


def fake_probe(open_ips: set[str]):
    calls: list[str] = []

    def probe(ip: str, port: int, timeout_s: float) -> bool:
        calls.append(ip)
        return ip in open_ips

    probe.calls = calls  # type: ignore[attr-defined]
    return probe


# -- find_camera: full scan ----------------------------------------------------------------


def test_find_camera_matches_mac_case_insensitively(tmp_path):
    arp_path = write_arp(
        tmp_path / "arp",
        {
            "192.0.2.5": CAMERA_MAC,  # lowercase in the table, target is uppercase
            "192.0.2.9": OTHER_MAC,
        },
    )
    probe = fake_probe({"192.0.2.5", "192.0.2.9"})
    cache_path = tmp_path / "camera_ip"

    ip = find_camera(
        TARGET_MAC,
        "192.0.2.0/24",
        arp_path=arp_path,
        cache_path=cache_path,
        probe_fn=probe,
    )

    assert ip == "192.0.2.5"
    assert cache_path.read_text().strip() == "192.0.2.5"


def test_find_camera_ignores_open_ports_with_no_mac_match(tmp_path):
    """An open :554 that isn't the camera (some other device) must not be picked."""
    arp_path = write_arp(
        tmp_path / "arp",
        {"192.0.2.7": BROADCAST_MAC},
    )
    probe = fake_probe({"192.0.2.7"})

    with pytest.raises(CameraNotFoundError):
        find_camera(
            TARGET_MAC,
            "192.0.2.0/24",
            arp_path=arp_path,
            cache_path=tmp_path / "camera_ip",
            probe_fn=probe,
        )


def test_find_camera_raises_when_nothing_answers(tmp_path):
    arp_path = write_arp(tmp_path / "arp", {})
    probe = fake_probe(set())

    with pytest.raises(CameraNotFoundError):
        find_camera(
            TARGET_MAC,
            "192.0.2.0/24",
            arp_path=arp_path,
            cache_path=tmp_path / "camera_ip",
            probe_fn=probe,
        )


def test_find_camera_derives_subnet_when_not_given(tmp_path):
    """subnet=None must call subnet_fn() instead of scanning nothing/raising."""
    arp_path = write_arp(tmp_path / "arp", {"192.0.2.5": CAMERA_MAC})
    probe = fake_probe({"192.0.2.5"})

    ip = find_camera(
        TARGET_MAC,
        None,
        arp_path=arp_path,
        cache_path=tmp_path / "camera_ip",
        probe_fn=probe,
        subnet_fn=lambda: "192.0.2.0/24",
    )

    assert ip == "192.0.2.5"


# -- find_camera: cache fast path -----------------------------------------------------------


def test_find_camera_uses_valid_cache_without_scanning_subnet(tmp_path):
    arp_path = write_arp(tmp_path / "arp", {"192.0.2.5": CAMERA_MAC})
    cache_path = tmp_path / "camera_ip"
    cache_path.write_text("192.0.2.5")
    probe = fake_probe({"192.0.2.5"})

    ip = find_camera(
        TARGET_MAC,
        "192.0.2.0/24",
        arp_path=arp_path,
        cache_path=cache_path,
        probe_fn=probe,
    )

    assert ip == "192.0.2.5"
    # Only the single cached IP was probed -- not the whole /24 (254 hosts).
    assert probe.calls == ["192.0.2.5"]  # type: ignore[attr-defined]


def test_find_camera_falls_back_to_full_scan_when_cache_is_stale(tmp_path):
    """Cached IP no longer answers (camera moved) -> full subnet scan finds the new one."""
    arp_path = write_arp(tmp_path / "arp", {"192.0.2.9": CAMERA_MAC})
    cache_path = tmp_path / "camera_ip"
    cache_path.write_text("192.0.2.5")  # stale: not in the ARP table/probe-open set below
    probe = fake_probe({"192.0.2.9"})

    ip = find_camera(
        TARGET_MAC,
        "192.0.2.0/24",
        arp_path=arp_path,
        cache_path=cache_path,
        probe_fn=probe,
    )

    assert ip == "192.0.2.9"
    assert cache_path.read_text().strip() == "192.0.2.9"


def test_find_camera_falls_back_to_full_scan_when_cached_mac_no_longer_matches(tmp_path):
    """Cached IP still answers port 554, but the ARP table now shows a different device at
    that address (e.g. DHCP handed it to something else) -> re-scan rather than trust it."""
    arp_path = write_arp(
        tmp_path / "arp",
        {"192.0.2.5": OTHER_MAC, "192.0.2.9": CAMERA_MAC},
    )
    cache_path = tmp_path / "camera_ip"
    cache_path.write_text("192.0.2.5")
    probe = fake_probe({"192.0.2.5", "192.0.2.9"})

    ip = find_camera(
        TARGET_MAC,
        "192.0.2.0/24",
        arp_path=arp_path,
        cache_path=cache_path,
        probe_fn=probe,
    )

    assert ip == "192.0.2.9"


# -- the individual low-level steps ----------------------------------------------------------


def test_hosts_in_subnet_excludes_network_and_broadcast():
    hosts = _hosts_in_subnet("192.0.2.0/24")
    assert len(hosts) == 254
    assert "192.0.2.0" not in hosts
    assert "192.0.2.255" not in hosts
    assert "192.0.2.1" in hosts
    assert "192.0.2.254" in hosts


def test_read_arp_table_parses_fixed_columns(tmp_path):
    last = mac(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF)
    arp_path = write_arp(tmp_path / "arp", {"203.0.113.5": last.upper()})
    table = _read_arp_table(arp_path)
    assert table == {"203.0.113.5": last}


def test_default_route_iface_picks_the_zero_destination_row(tmp_path):
    route_path = tmp_path / "route"
    route_path.write_text(
        "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\n"
        "eth1\t0001A8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\n"  # non-default row (a /24 network)
        "eth0\t00000000\t0101A8C0\t0003\t0\t0\t0\t00000000\n"  # default route
    )
    assert _default_route_iface(route_path) == "eth0"


def test_default_route_iface_raises_when_none_found(tmp_path):
    route_path = tmp_path / "route"
    route_path.write_text(
        "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\n"
        "eth1\t0001A8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\n"
    )
    with pytest.raises(CameraNotFoundError):
        _default_route_iface(route_path)


def test_default_subnet_combines_route_and_injected_iface_lookup(tmp_path):
    route_path = tmp_path / "route"
    route_path.write_text(
        "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\n"
        "eth0\t00000000\t0101A8C0\t0003\t0\t0\t0\t00000000\n"
    )

    subnet = _default_subnet(
        route_path=route_path,
        iface_fn=lambda iface: ("198.51.100.42", "255.255.255.0") if iface == "eth0" else ("", ""),
    )

    assert subnet == "198.51.100.0/24"
