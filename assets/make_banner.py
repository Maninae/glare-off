"""Render the README / social banner (1280x640) in the app's own palette, light and dark.

Run from the repo root: `python assets/make_banner.py`. Writes assets/banner-light.png and
assets/banner-dark.png. Colors mirror app/styles/tokens.css; the display font is Charter (ships
with macOS) with a path override via BANNER_SERIF_FONT for other systems.
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BANNER_WIDTH, BANNER_HEIGHT = 1280, 640
ASSETS_DIRECTORY = Path(__file__).resolve().parent
SERIF_FONT_CANDIDATES = [
    os.environ.get("BANNER_SERIF_FONT", ""),
    "/System/Library/Fonts/Supplemental/Charter.ttc",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
]
SANS_FONT_CANDIDATES = ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"]

THEMES = {
    "light": {"bg": "#f5f4ef", "ink": "#191c1b", "ink_soft": "#565c59", "accent": "#17654f", "lens": "#ffffff", "glare": "#f6ead6"},
    "dark": {"bg": "#121514", "ink": "#e9ece9", "ink_soft": "#a6ada9", "accent": "#6cc9a9", "lens": "#1b1f1d", "glare": "#3a3522"},
}


def load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    """Return the first font that loads, falling back to Pillow's default."""
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size)


def draw_glasses_mark(draw: ImageDraw.ImageDraw, left: int, top: int, theme: dict[str, str]) -> None:
    """Draw the spectacles mark: left lens with a glare streak, right lens clean."""
    lens_w, lens_h, gap, stroke = 190, 130, 44, 12
    bridge_y = top + lens_h // 2
    for index in range(2):
        x0 = left + index * (lens_w + gap)
        box = (x0, top, x0 + lens_w, top + lens_h)
        draw.rounded_rectangle(box, radius=40, fill=theme["lens"], outline=theme["ink"], width=stroke)
    # glare streak inside the left lens only: the before/after in one glyph
    streak = [(left + 48, top + 96), (left + 112, top + 32)]
    draw.line(streak, fill=theme["glare"], width=26)
    draw.line([(left + 128, top + 100), (left + 150, top + 78)], fill=theme["glare"], width=14)
    draw.line([(left + lens_w, bridge_y), (left + lens_w + gap, bridge_y)], fill=theme["ink"], width=stroke)
    draw.line([(left - 34, bridge_y - 12), (left + 4, bridge_y)], fill=theme["ink"], width=stroke)
    right_edge = left + 2 * lens_w + gap
    draw.line([(right_edge - 4, bridge_y), (right_edge + 34, bridge_y - 12)], fill=theme["ink"], width=stroke)


def render_banner(theme_name: str) -> Path:
    """Compose one banner and return its path."""
    theme = THEMES[theme_name]
    image = Image.new("RGB", (BANNER_WIDTH, BANNER_HEIGHT), theme["bg"])
    draw = ImageDraw.Draw(image)
    draw_glasses_mark(draw, left=430, top=96, theme=theme)
    title_font = load_font(SERIF_FONT_CANDIDATES, 118)
    tagline_font = load_font(SANS_FONT_CANDIDATES, 34)
    title = "Glare Off"
    title_w = draw.textlength(title, font=title_font)
    draw.text(((BANNER_WIDTH - title_w) / 2, 282), title, font=title_font, fill=theme["ink"])
    tagline = "Take the glare off glasses in your photos."
    tagline_w = draw.textlength(tagline, font=tagline_font)
    draw.text(((BANNER_WIDTH - tagline_w) / 2, 438), tagline, font=tagline_font, fill=theme["ink_soft"])
    promise = "Runs in your browser. Nothing is uploaded."
    promise_w = draw.textlength(promise, font=tagline_font)
    draw.text(((BANNER_WIDTH - promise_w) / 2, 486), promise, font=tagline_font, fill=theme["accent"])
    output_path = ASSETS_DIRECTORY / f"banner-{theme_name}.png"
    image.save(output_path, optimize=True)
    return output_path


if __name__ == "__main__":
    for name in THEMES:
        print(render_banner(name))
