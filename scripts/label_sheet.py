#!/usr/bin/env python3
"""Tile testdata/crops/*.jpg into 3x3 contact sheets for silver labelling (gbox1 only).

Each tile is 512px (long side, letterboxed to a 512x512 cell) with its crop filename burned
in. Writes testdata/crops/sheets/sheet_NNN.jpg plus testdata/crops/sheets/index.json, an
ordered [[sheet_file, [tile_filenames...]], ...] list so the labeller can map sheet position
back to the source crop filename without depending on the (small) burned-in text alone.

Needs only Pillow -- run with the host venv, no container required.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = ROOT / "testdata" / "crops"
SHEETS_DIR = CROPS_DIR / "sheets"
TILE = 512
GRID = 3


def load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_tile(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((TILE, TILE))
    cell = Image.new("RGB", (TILE, TILE), (20, 20, 20))
    x = (TILE - im.width) // 2
    y = (TILE - im.height) // 2
    cell.paste(im, (x, y))
    draw = ImageDraw.Draw(cell)
    font = load_font(18)
    label = path.name
    draw.rectangle([0, 0, TILE, 24], fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 0), font=font)
    return cell


def main():
    crops = sorted(p for p in CROPS_DIR.glob("*.jpg"))
    if not crops:
        print("no crops found in testdata/crops", file=sys.stderr)
        sys.exit(1)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    per_sheet = GRID * GRID
    for s in range(0, len(crops), per_sheet):
        batch = crops[s : s + per_sheet]
        sheet = Image.new("RGB", (TILE * GRID, TILE * GRID), (0, 0, 0))
        for i, p in enumerate(batch):
            tile = make_tile(p)
            tx, ty = (i % GRID) * TILE, (i // GRID) * TILE
            sheet.paste(tile, (tx, ty))
        sheet_name = f"sheet_{s // per_sheet:03d}.jpg"
        sheet.save(SHEETS_DIR / sheet_name, quality=82)
        index.append([sheet_name, [p.name for p in batch]])
        print(f"{sheet_name}: {len(batch)} tiles", file=sys.stderr)
    with open(SHEETS_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=1)
    print(f"{len(crops)} crops -> {len(index)} sheets", file=sys.stderr)


if __name__ == "__main__":
    main()
