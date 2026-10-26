from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from scarebox.camera import Camera, CameraError


def make_camera(handler) -> Camera:
    transport = httpx.MockTransport(handler)
    return Camera("192.0.2.10", "camuser", "s3cret", transport=transport)


def test_get_config_parses_key_value_lines():
    body = (
        "table.Encode[0].ExtraFormat[0].Video.Compression=H.264\n"
        "table.Encode[0].ExtraFormat[0].Video.FPS=25\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/cgi-bin/configManager.cgi"
        qs = parse_qs(urlparse(str(request.url)).query)
        assert qs["action"] == ["getConfig"]
        assert qs["name"] == ["Encode"]
        return httpx.Response(200, text=body)

    cam = make_camera(handler)
    result = cam.get_config("Encode")
    assert result["table.Encode[0].ExtraFormat[0].Video.Compression"] == "H.264"
    assert result["table.Encode[0].ExtraFormat[0].Video.FPS"] == "25"


def test_set_config_key_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        qs = parse_qs(urlparse(str(request.url)).query)
        assert qs["action"] == ["setConfig"]
        assert qs["Encode[0].ExtraFormat[0].Video.FPS"] == ["15"]
        return httpx.Response(200, text="OK")

    cam = make_camera(handler)
    cam.set_config_key("Encode[0].ExtraFormat[0].Video.FPS", "15")


def test_set_config_key_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Error: bad key")

    cam = make_camera(handler)
    with pytest.raises(CameraError):
        cam.set_config_key("Some.Bogus.Key", "1")


def test_repr_masks_password():
    cam = Camera("192.0.2.10", "camuser", "s3cret")
    text = repr(cam)
    assert "s3cret" not in text
    assert "***" in text


def test_rtsp_url_quotes_password():
    cam = Camera("192.0.2.10", "camuser", "p@ss/word:1")
    url = cam.rtsp_url(subtype=1)
    assert "p@ss/word:1" not in url
    assert "p%40ss%2Fword%3A1" in url
    assert url.startswith("rtsp://camuser:")
    assert url.endswith("/cam/realmonitor?channel=1&subtype=1")


def test_ptz_status_parses_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        qs = parse_qs(urlparse(str(request.url)).query)
        assert qs["action"] == ["getStatus"]
        return httpx.Response(200, text="status.PanPosition=120.0\nstatus.TiltPosition=5.0\n")

    cam = make_camera(handler)
    status = cam.ptz_status()
    assert status["status.PanPosition"] == "120.0"


def test_ptz_position_reads_firmware_misspelling():
    # The real IP4M-1041B returns "Postion" (sic), captured live 2026-10-15.
    body = "status.Action=GeneralMove\nstatus.Postion[0]=156.700000\nstatus.Postion[1]=0.000000\nstatus.Postion[2]=1.000000\n"
    cam = make_camera(lambda request: httpx.Response(200, text=body))
    assert cam.ptz_position() == (156.7, 0.0, 1.0)


def test_ptz_position_raises_instead_of_defaulting_to_zero():
    cam = make_camera(lambda request: httpx.Response(200, text="status.Action=GeneralMove\n"))
    import pytest
    with pytest.raises(Exception):
        cam.ptz_position()


def test_get_time():
    def handler(request: httpx.Request) -> httpx.Response:
        qs = parse_qs(urlparse(str(request.url)).query)
        assert qs["action"] == ["getCurrentTime"]
        return httpx.Response(200, text="2026-10-15 09:00:00")

    cam = make_camera(handler)
    assert cam.get_time() == "2026-10-15 09:00:00"
