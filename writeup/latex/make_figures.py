"""Build the raster figures for viz_section.tex from writeup/figures/viz/.

Run from the repository root: python3 writeup/latex/make_figures.py
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "writeup/figures/viz"
OUT = Path(__file__).resolve().parent / "figures"

# Screenshots are 1600 x 1000 CSS px at DPR 2.
DPR = 2
# Clearance card of the left column, CSS px (left, top, right, bottom).
CARD = (14, 64, 330, 256)
# Nearest tip, depth bars, colour key and approach trend, CSS px (square).
TREND = (14, 256, 330, 572)
# Tip region of the overview (shadow, drop line, reticle, label), CSS px.
TIP = (1104, 696, 1272, 808)

STATES = [
    ("after/17_live_critical.png", "viz_card_critical"),
    ("after/19_live_held.png", "viz_card_held"),
    ("after/20_live_ghost_last.png", "viz_card_lost"),
]


def save_jpg(img, name, width=None):
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    img.convert("RGB").save(OUT / f"{name}.jpg", quality=92, optimize=True)


def main():
    OUT.mkdir(exist_ok=True)
    overview = Image.open(SRC / "after/15_asymmetric_near.png")
    save_jpg(overview, "viz_overview", width=2400)
    save_jpg(overview.crop(tuple(v * DPR for v in TIP)), "viz_overview_tip")
    for src, name in STATES:
        img = Image.open(SRC / src)
        save_jpg(img.crop(tuple(v * DPR for v in CARD)), name)
    trend = Image.open(SRC / "after/08_approach_trend.png")
    save_jpg(trend.crop(tuple(v * DPR for v in TREND)), "viz_card_trend")
    for src, name in [("design/alert_trace.png", "viz_alert_trace")]:
        Image.open(SRC / src).save(OUT / f"{name}.png", optimize=True)


if __name__ == "__main__":
    main()
