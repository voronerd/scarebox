"""Dahua/Amcrest CGI client: digest-auth HTTP control for encoder config,
clock sync, and PTZ. RTSP streaming itself is handled elsewhere (ingest).
"""

from __future__ import annotations

import time
from datetime import datetime
from urllib.parse import quote

import httpx


class CameraError(Exception):
    """Raised when the camera's CGI responds with anything but OK."""


class Camera:
    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        timeout: float = 3.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host = host
        self.user = user
        self._password = password
        self.timeout = timeout
        self._auth = httpx.DigestAuth(user, password)
        self._client = httpx.Client(auth=self._auth, timeout=self.timeout, transport=transport)

    def __repr__(self) -> str:
        # repr masks password: never leak it into logs or tracebacks.
        return f"Camera(host={self.host!r}, user={self.user!r}, password=***)"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _base_url(self, cgi: str) -> str:
        return f"http://{self.host}/cgi-bin/{cgi}"

    def _get(self, cgi: str, params: dict) -> str:
        resp = self._client.get(self._base_url(cgi), params=params)
        resp.raise_for_status()
        return resp.text

    # -- encoder config -----------------------------------------------------

    def get_config(self, name: str) -> dict[str, str]:
        """``configManager.cgi?action=getConfig&name=<name>``, parsed into a dict."""
        text = self._get("configManager.cgi", {"action": "getConfig", "name": name})
        result: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
        return result

    def set_config_key(self, key: str, value: str) -> None:
        """Set exactly one config key. Raises :class:`CameraError` unless the body is OK."""
        text = self._get(
            "configManager.cgi",
            {"action": "setConfig", f"{key}": value},
        )
        if text.strip() != "OK":
            raise CameraError(f"setConfig {key}={value} failed: {text.strip()!r}")

    # -- clock ---------------------------------------------------------------

    def get_time(self) -> str:
        """``global.cgi?action=getCurrentTime``."""
        return self._get("global.cgi", {"action": "getCurrentTime"}).strip()

    def set_time(self, dt: datetime) -> None:
        """``global.cgi?action=setCurrentTime&time=YYYY-MM-DD%20HH:MM:SS`` (local time)."""
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        text = self._get("global.cgi", {"action": "setCurrentTime", "time": time_str})
        if text.strip() != "OK":
            raise CameraError(f"setCurrentTime failed: {text.strip()!r}")

    # -- PTZ -------------------------------------------------------------

    def ptz_status(self) -> dict:
        """``ptz.cgi?action=getStatus``, parsed the same way as :meth:`get_config`."""
        text = self._get("ptz.cgi", {"action": "getStatus"})
        result: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
        return result

    def ptz_position(self) -> tuple[float, float, float]:
        """Current (pan, tilt, zoom). The IP4M-1041B firmware spells the keys
        ``status.Postion[0..2]`` (sic); the documented Dahua names are
        ``status.PanPosition`` etc. Accepts either and raises instead of defaulting to 0,
        because a silent 0 "restores" the camera to the wrong place (it did, in 01-01)."""
        s = self.ptz_status()
        for keys in (
            ("status.Postion[0]", "status.Postion[1]", "status.Postion[2]"),
            ("status.PanPosition", "status.TiltPosition", "status.ZoomPosition"),
        ):
            if all(k in s for k in keys):
                return tuple(float(s[k]) for k in keys)  # type: ignore[return-value]
        raise CameraError(f"no PTZ position in status keys: {sorted(s)}")

    def _ptz_action(self, action: str, code: str, arg1=0, arg2=0, arg3=0) -> None:
        text = self._get(
            "ptz.cgi",
            {
                "action": action,
                "channel": 1,
                "code": code,
                "arg1": arg1,
                "arg2": arg2,
                "arg3": arg3,
            },
        )
        if text.strip() != "OK":
            raise CameraError(f"ptz {action} {code} failed: {text.strip()!r}")

    def ptz_nudge(self, direction: str, speed: int = 1, duration_s: float = 0.3) -> None:
        """Move for ``duration_s`` seconds then stop. ``direction`` is one of
        Left/Right/Up/Down. Blocks for the duration (caller times the send)."""
        if direction not in ("Left", "Right", "Up", "Down"):
            raise ValueError(f"invalid direction: {direction!r}")

        self._ptz_action("start", direction, arg1=0, arg2=speed, arg3=0)
        time.sleep(duration_s)
        self._ptz_action("stop", direction, arg1=0, arg2=speed, arg3=0)

    def ptz_goto_abs(self, pan: float, tilt: float, zoom: float) -> None:
        """``code=PositionABS``."""
        self._ptz_action("start", "PositionABS", arg1=pan, arg2=tilt, arg3=zoom)

    # -- RTSP ------------------------------------------------------------

    def rtsp_url(self, subtype: int = 1) -> str:
        """``rtsp://user:pass@host:554/cam/realmonitor?channel=1&subtype=<n>``."""
        user = quote(self.user, safe="")
        password = quote(self._password, safe="")
        return (
            f"rtsp://{user}:{password}@{self.host}:554"
            f"/cam/realmonitor?channel=1&subtype={subtype}"
        )
