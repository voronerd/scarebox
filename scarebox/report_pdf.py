"""report_pdf: the detection report as a compact PDF, built on gbox1 (2026-10-19).

Martin's phone "Print -> Save as PDF" of /report.html came out at 85 MB: the phone re-encodes
every inlined photo losslessly. This lays each A4 page out as one image with Pillow and
saves the pages as JPEG-compressed PDF pages, so the size is predictable (roughly 150-250 KB
a page, three detections a page) whatever device asks for it. The cost: the text is part of
the page image, so it can't be selected. Fonts come from matplotlib's bundled DejaVu (the
only TTFs in the app image). Same content and order as `aim.render_report`.
"""
from __future__ import annotations

import io
import os
import time
from collections.abc import Callable

from PIL import Image, ImageDraw, ImageFont

_DPI = 110
_PAGE_W, _PAGE_H = int(8.27 * _DPI), int(11.69 * _DPI)  # A4
_MARGIN = int(0.5 * _DPI)
_PER_PAGE = 3
_JPEG_QUALITY = 70
_INK, _MUTED, _RULE, _ACCENT = (29, 26, 38), (110, 106, 120), (200, 200, 205), (185, 120, 0)

_SCORE_MEANING = {3: "names this exact costume", 2: "costume-category line", 1: "no-costume goading", 0: "generic"}
_PATH_MEANING = {"live": "written live for this runner", "library": "pre-recorded", "dropped": "nothing played",
                 "silent": "no line (non-costumes off)"}


def _font(bold: bool = False, italic: bool = False, size: int = 14) -> ImageFont.FreeTypeFont:
    import matplotlib

    style = ("Bold" if bold else "") + ("Oblique" if italic else "")
    name = "DejaVuSans" + (f"-{style}" if style else "")
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", name + ".ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _text(draw, xy, text, font, width, fill=_INK, gap=4) -> int:
    """Draws wrapped text; returns the y below it."""
    x, y = xy
    for ln in _wrap(draw, text, font, width):
        draw.text((x, y), ln, font=font, fill=fill)
        y += font.size + gap
    return y


def build_report_pdf(cards: list[dict], image: Callable[[str, str], bytes | None], max_events: int = 30) -> bytes:
    rows = [c for c in reversed(cards) if c.get("resolved") and c.get("path") != "skipped"][:max_events]
    f_title, f_h, f_body, f_small = _font(True, size=26), _font(True, size=17), _font(size=13), _font(size=11)
    f_line = _font(italic=True, size=14)
    content_w = _PAGE_W - 2 * _MARGIN
    pages: list[Image.Image] = []

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        page = Image.new("RGB", (_PAGE_W, _PAGE_H), "white")
        pages.append(page)
        return page, ImageDraw.Draw(page)

    page, draw = new_page()
    y = _MARGIN
    draw.text((_MARGIN, y), "Scarebox: detection report", font=f_title, fill=_INK)
    y += 40
    y = _text(draw, (_MARGIN, y), (
        "A camera watches runners near the finish. A person detector spots each runner as they cross a line on "
        "the camera image, a vision model judges their costume from that picture, and “Sir Reginald”, a "
        "posh English heckler voice, calls out a G-rated, costume-specific line urging them on. Each entry is one "
        "detection: the picture the system judged (the yellow box marks the person), what it decided, and the line "
        "it chose. Priority: 3 = names that exact costume, 2 = costume-category line, 1 = plain encouragement."
    ), f_body, content_w, fill=_MUTED)
    costumed = sum(1 for c in rows if c.get("is_costume"))
    packs = sum(1 for c in rows if c.get("pack"))
    y += 6
    draw.text((_MARGIN, y), f"{len(rows)} detections · {costumed} in costume · {packs} matched a specific "
              f"costume pack · generated {time.strftime('%Y-%m-%d %H:%M')}", font=f_small, fill=_INK)
    y += 28
    slot_h = (_PAGE_H - y - _MARGIN) // _PER_PAGE  # first page: same three slots under the header
    first_page_top = y
    img_w = int(content_w * 0.56)
    txt_x = _MARGIN + img_w + 16
    txt_w = content_w - img_w - 16

    for i, c in enumerate(rows):
        if i and i % _PER_PAGE == 0:
            page, draw = new_page()
            first_page_top = _MARGIN
            slot_h = (_PAGE_H - 2 * _MARGIN) // _PER_PAGE
        top = first_page_top + (i % _PER_PAGE) * slot_h
        draw.line([(_MARGIN, top), (_PAGE_W - _MARGIN, top)], fill=_RULE, width=1)
        top += 10
        box_h = slot_h - 22
        data = image(c["id"], "frame") or image(c["id"], "wide") or image(c["id"], "crop")
        if data:
            pic = Image.open(io.BytesIO(data)).convert("RGB")
            pic.thumbnail((img_w, box_h))
            page.paste(pic, (_MARGIN, top))
        ty = top
        if c.get("wall_time"):
            draw.text((txt_x, ty), time.strftime("%H:%M:%S", time.localtime(c["wall_time"])), font=f_small, fill=_MUTED)
            ty += 16
        cat = (c.get("category") or "").replace("_", " ")
        if not c.get("is_costume"):
            verdict = "No costume"
        else:
            verdict = f"{c.get('pack_match_display') or c.get('costume')} ({cat})"
        ty = _text(draw, (txt_x, ty), verdict, f_h, txt_w) + 2
        line = c.get("line")
        ty = _text(draw, (txt_x, ty), f"“{line}”" if line else "No line played", f_line, txt_w) + 6
        path = c.get("path") or ""
        how = f"costume pack: {c.get('pack_match_display') or c.get('pack')}" if c.get("pack") else _PATH_MEANING.get(path.split(':')[0], path)
        score = c.get("score")
        facts = [("Line", how),
                 ("Priority", f"{score} / 3, {_SCORE_MEANING.get(score, '')}" if score is not None else "n/a")]
        if c.get("is_costume") and c.get("pack_match_display") and c.get("costume"):
            facts.append(("Judge saw", f"“{c['costume']}”"))
        if (c.get("group_size") or 1) > 1:
            facts.append(("Group", f"{int(c['group_size'])} people"))
        if c.get("blocked_text"):
            facts.append(("Blocked", f"live line stopped by the safety filter ({c.get('blocked_reason')})"))
        ms = c.get("ms") or {}
        timing = " · ".join(f"{n} {ms[k] / 1000:.1f} s" for k, n in
                                 (("judge", "judge"), ("quip", "writing"), ("voice_first_audio", "voice"), ("total", "total"))
                                 if ms.get(k) is not None)
        facts.append(("Timing", timing or "n/a"))
        for label, value in facts:
            if ty > top + box_h - 14:
                break
            draw.text((txt_x, ty), label, font=f_small, fill=_MUTED)
            ty = _text(draw, (txt_x + 70, ty), value, f_small, txt_w - 70, gap=2) + 3

    if not rows:
        draw.text((_MARGIN, first_page_top), "No detections yet.", font=f_body, fill=_INK)
    out = io.BytesIO()
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:], resolution=_DPI, quality=_JPEG_QUALITY)
    return out.getvalue()
