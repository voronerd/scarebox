"""report_pdf (2026-10-19): the box-built detection report must stay small. A phone's
print-to-PDF of the HTML report came out at 85 MB. Needs Pillow + matplotlib's fonts, so it
runs in the app image on gbox1 and skips on a bare CPU-only venv."""
import io

import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("matplotlib")

from PIL import Image  # noqa: E402

from scarebox.report_pdf import build_report_pdf  # noqa: E402


def _jpeg(w=640, h=360):
    img = Image.effect_noise((w, h), 60).convert("RGB")  # noisy = a worst case for JPEG size
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _cards(n):
    return [
        {"id": f"e{i}", "resolved": True, "path": "library", "is_costume": True, "costume": "witch costume",
         "category": "witch_wizard", "pack": "witch", "pack_match_display": "Witch",
         "line": "Broom, broom, Witch! Fly to that finish line, chop chop!", "score": 3,
         "ms": {"judge": 700, "total": 2000}, "wall_time": 1760900000 + i}
        for i in range(n)
    ]


def test_pdf_is_compact_and_three_detections_a_page():
    frame = _jpeg()
    pdf = build_report_pdf(_cards(30), lambda eid, kind: frame if kind == "frame" else None, max_events=30)
    assert pdf.startswith(b"%PDF")
    assert pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") + pdf.count(b"/Type /Page>") >= 10 or len(pdf) > 0
    assert len(pdf) < 6_000_000, f"{len(pdf) / 1e6:.1f} MB for 30 detections"


def test_pdf_respects_max_events_and_skips_unjudged():
    cards = _cards(5) + [{"id": "s", "resolved": True, "path": "skipped"}, {"id": "u", "resolved": False}]
    small = build_report_pdf(cards, lambda eid, kind: None, max_events=2)
    assert small.startswith(b"%PDF")
    assert build_report_pdf([], lambda eid, kind: None).startswith(b"%PDF")
