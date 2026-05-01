#!/usr/bin/env python3
"""Generate site/assets/og.png — industrial / toxic-green / monospace.

1200×630, gunmetal base, hazard yellow edge, JetBrains Mono slogan.
Run from anywhere — paths resolve relative to the repo root.

Usage:
    pip install pillow
    python scripts/build_og.py
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".cache" / "og-fonts"
OUTPUT = REPO / "site" / "assets" / "og.png"
PYPROJECT = REPO / "pyproject.toml"

FONTS = {
    "JetBrainsMono-700.ttf": "https://github.com/google/fonts/raw/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
    "Inter-500.ttf": "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
}

W, H = 1200, 630

BG = (6, 8, 10)
BG_2 = (13, 16, 20)
FG = (236, 241, 247)
FG_DIM = (164, 173, 186)
GREEN = (56, 255, 79)
GREEN_2 = (130, 255, 149)
YELLOW = (255, 176, 0)


def ensure_fonts() -> dict[str, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in FONTS.items():
        p = CACHE / name
        if not p.exists():
            print(f"  downloading {name}")
            urllib.request.urlretrieve(url, p)
        paths[name] = p
    return paths


def current_version() -> str:
    """Pick up `version = "X.Y.Z"` from pyproject.toml so the chip stays in sync."""
    text = PYPROJECT.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return m.group(1) if m else "0.0.0"


def vertical_gradient(w, h, top, bot):
    img = Image.new("RGB", (w, h), top)
    px = img.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3))
        for x in range(w):
            px[x, y] = c
    return img


def hazard_stripes(w, h, stripe=24):
    img = Image.new("RGB", (w, h), YELLOW)
    d = ImageDraw.Draw(img)
    pad = stripe * 2 + max(w, h)
    for i in range(-pad, w + pad, stripe * 2):
        d.polygon(
            [(i, 0), (i + stripe, 0), (i + stripe + h, h), (i + h, h)],
            fill=(17, 17, 17),
        )
    return img


def main() -> None:
    fonts = ensure_fonts()
    version = current_version()

    base = vertical_gradient(W, H, BG, BG_2).convert("RGBA")

    scan = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scan)
    for y in range(0, H, 3):
        sd.line([(0, y), (W, y)], fill=(255, 255, 255, 6))
    base = Image.alpha_composite(base, scan)

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).rectangle((0, 0, W, 4), fill=(*GREEN, 230))
    base = Image.alpha_composite(base, glow.filter(ImageFilter.GaussianBlur(radius=8)))
    ImageDraw.Draw(base).rectangle((0, 0, W, 3), fill=GREEN)

    hazard = hazard_stripes(56, H).convert("RGBA")
    blend = Image.new("RGBA", hazard.size, (0, 0, 0, 80))
    hazard = Image.alpha_composite(hazard, blend)
    base.paste(hazard, (W - 56, 0), hazard)

    draw = ImageDraw.Draw(base)
    h_font = ImageFont.truetype(str(fonts["JetBrainsMono-700.ttf"]), 96)
    chip_font = ImageFont.truetype(str(fonts["JetBrainsMono-700.ttf"]), 24)
    rule_font = ImageFont.truetype(str(fonts["JetBrainsMono-700.ttf"]), 22)
    url_font = ImageFont.truetype(str(fonts["JetBrainsMono-700.ttf"]), 28)
    body_font = ImageFont.truetype(str(fonts["Inter-500.ttf"]), 30)

    chip_text = f"[ KERNEL · ALPHA · v{version} ]"
    bbox = draw.textbbox((0, 0), chip_text, font=chip_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = 80, 90
    pad_x, pad_y = 18, 10
    chip = Image.new("RGBA", (tw + pad_x * 2, th + pad_y * 2 + 6), (*GREEN, 255))
    base.paste(chip, (cx, cy), chip)
    draw.text((cx + pad_x, cy + pad_y - 2), chip_text, font=chip_font, fill=BG)

    draw.text((78, 168), "O(n) attention is", font=h_font, fill=FG)
    draw.text((78, 290), "deception.", font=h_font, fill=GREEN)

    draw.text(
        (80, 420),
        "A backend-neutral kernel of predictive primitives — substrates,",
        font=body_font,
        fill=FG_DIM,
    )
    draw.text(
        (80, 458),
        "memory, gating, routing, readouts. Reusable across descendants.",
        font=body_font,
        fill=FG_DIM,
    )

    draw.text((80, 540), "decepticons → numpy", font=rule_font, fill=GREEN_2)
    draw.text((360, 540), "//  never imports its descendants", font=rule_font, fill=FG_DIM)

    url = "decepticons.win"
    bbox = draw.textbbox((0, 0), url, font=url_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ux = W - 56 - tw - 28
    uy = 540
    draw.text((ux, uy), url, font=url_font, fill=YELLOW)
    draw.line([(ux, uy + th + 8), (ux + tw, uy + th + 8)], fill=YELLOW, width=2)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(OUTPUT, "PNG", optimize=True)
    print(f"wrote {OUTPUT.relative_to(REPO)} (v{version})")


if __name__ == "__main__":
    main()
