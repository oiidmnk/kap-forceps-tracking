#!/usr/bin/env python3
"""Render screenshot annotations with Pillow (no browser or plotting backend).

Usage: python annotate.py SPEC.json [--out OUT.png] [--pdf OUT.pdf]
Paths in JSON are relative to the spec; CLI output paths are relative to cwd.
Schema (all geometry and font sizes in CSS pixels, BEFORE crop):
  screenshot: PNG path, or "synthetic:dark-ui" for an in-memory 3200x2000 fixture
  css_width: original screenshot CSS width (default 1600); source scale is
      image.width / css_width, or explicitly set device_scale_factor.
  crop: optional [x,y,width,height] in original CSS coordinates
  output_scale: pixels per CSS pixel in output (default 2); dpi: default 300
  out: output PNG path (default spec filename with .png)
  accent: default "#ffd400"; dim_outside: boolean (default false)
  dim_opacity: 0..1 (default .55)
  badge_radius: default 17; font_size: legend font size (default 19)
  callouts: [{number: 1, point: [x,y], target: [x,y,w,h],
              leader: true, highlight: true, rounded: true, text: "Caption"}]
      point is badge center; omit it to place the badge to the right of target.
      At least point or target is required. Numbers default to array index + 1.
      Leaders terminate at the nearest target boundary; highlight defaults true.
  legend: false, or {position: "right"|"below", width: 350, padding: 24,
                     title: "Figure key"}; default false. Text wraps to width.
      Below legends use the image width; right legends expand canvas height
      if necessary. White background, dark text. Crop does not move legend.

Geometry outside the crop is clipped. For publication, place badges inside
the crop and avoid overlapping targets. Output PNG has DPI metadata; optional
PDF embeds the same raster at the requested DPI (not a vector screenshot).
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, max(1, round(size)))
        except OSError:
            pass
    raise RuntimeError("Install Arial or DejaVu Sans for publication typography")


def synthetic():
    im = Image.new("RGB", (3200, 2000), "#15191f")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 3200, 140), fill="#242b35")
    d.text((80, 40), "FORCEPS / TRACKING", font=font(46), fill="#edf0f5")
    for box in ((80, 220, 650, 1810), (730, 220, 2340, 1420),
                (2420, 220, 3120, 1810), (730, 1500, 2340, 1810)):
        d.rounded_rectangle(box, radius=20, fill="#303945", outline="#546172", width=3)
    for y, label in ((300, "Session 04"), (440, "Camera"), (580, "Tracking")):
        d.text((120, y), label, font=font(36), fill="#d1d8e0")
    d.text((820, 280), "LIVE VIEW", font=font(36), fill="#d1d8e0")
    d.line((1100, 1100, 1800, 650, 2000, 820), fill="#a7b9c9", width=35)
    d.text((2480, 300), "Metrics", font=font(40), fill="#d1d8e0")
    d.text((2480, 440), "Confidence\n98.4%", font=font(40), fill="#9cd7bc", spacing=25)
    d.text((820, 1570), "00:24   /   01:30", font=font(40), fill="#d1d8e0")
    return im


def render(spec_path, out=None, pdf=None):
    spec_path = Path(spec_path).resolve()
    s = json.loads(spec_path.read_text())
    source = s["screenshot"]
    im = synthetic() if source == "synthetic:dark-ui" else Image.open(spec_path.parent / source).convert("RGB")
    factor = float(s.get("device_scale_factor", im.width / float(s.get("css_width", 1600))))
    scale = float(s.get("output_scale", 2))
    dpi = float(s.get("dpi", 300))
    if min(factor, scale, dpi) <= 0 or not all(map(math.isfinite, (factor, scale, dpi))):
        raise ValueError("Scale factors and DPI must be finite and positive")
    x, y, w, h = map(float, s.get("crop", [0, 0, im.width / factor, im.height / factor]))
    if x < 0 or y < 0 or min(w, h) <= 0 or (x+w)*factor > im.width or (y+h)*factor > im.height:
        raise ValueError("Crop must be inside screenshot with positive dimensions")
    im = im.crop(tuple(round(v * factor) for v in (x, y, x+w, y+h)))
    im = im.resize((round(w*scale), round(h*scale)), Image.Resampling.LANCZOS)
    def point(p):
        return ((p[0]-x)*scale, (p[1]-y)*scale)
    def box(b):
        return (*point(b[:2]), *point((b[0]+b[2], b[1]+b[3])))
    calls = s.get("callouts", [])
    if s.get("dim_outside"):
        alpha = float(s.get("dim_opacity", .55))
        if not 0 <= alpha <= 1:
            raise ValueError("dim_opacity must be between 0 and 1")
        mask = Image.new("L", im.size, round(255*alpha))
        md = ImageDraw.Draw(mask)
        for c in calls:
            if "target" in c and c.get("highlight", True):
                md.rounded_rectangle(box(c["target"]), radius=8*scale if c.get("rounded", True) else 0, fill=0)
        im = Image.composite(Image.new("RGB", im.size, "black"), im, mask)
    d = ImageDraw.Draw(im)
    accent = s.get("accent", "#ffd400")
    radius = float(s.get("badge_radius", 17))*scale
    for i, c in enumerate(calls, 1):
        target = c.get("target")
        if "point" not in c and target is None:
            raise ValueError(f"Callout {i} needs point or target")
        p = point(c["point"] if "point" in c else (target[0]+target[2]+30, target[1]+target[3]/2))
        if target is not None:
            b = box(target)
            if c.get("highlight", True):
                d.rounded_rectangle(b, radius=8*scale if c.get("rounded", True) else 0,
                                    outline=accent, width=max(1, round(1.5*scale)))
            if c.get("leader", False):
                q = (min(max(p[0], b[0]), b[2]), min(max(p[1], b[1]), b[3]))
                if b[0] < p[0] < b[2] and b[1] < p[1] < b[3]:
                    q = min(((b[0],p[1]), (b[2],p[1]), (p[0],b[1]), (p[0],b[3])),
                            key=lambda q: math.dist(p,q))
                d.line((p, q), fill=accent, width=max(1, round(1.5*scale)))
        d.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius), fill=accent,
                  outline="#171b21", width=max(1, round(1.5*scale)))
        label = str(c.get("number", i))
        bf = font(min(20*scale, radius*2/max(1,len(label))))
        d.text(p, label, font=bf, fill="#171b21", anchor="mm")
    legend = s.get("legend", False)
    if legend:
        if not isinstance(legend, dict) or legend.get("position", "right") not in ("right", "below"):
            raise ValueError("legend needs position right or below")
        right = legend.get("position", "right") == "right"
        lw = round(float(legend.get("width", 350))*scale) if right else im.width
        pad = round(float(legend.get("padding", 24))*scale)
        lf = font(float(s.get("font_size", 19))*scale)
        available = lw-2*pad
        if available < lf.size*3:
            raise ValueError("Legend width is too small for font and padding")
        rows = []
        if legend.get("title"):
            rows.append(str(legend["title"]))
            rows.append("")
        for i, c in enumerate(calls, 1):
            text = f'{c.get("number", i)} — {c.get("text", "")}'
            line = ""
            for word in text.split():
                # Character fallback also wraps long unbroken identifiers.
                if d.textlength((line+" "+word).strip(), font=lf) > available and line:
                    rows.append(line)
                    line = ""
                for char in ((" " if line else "")+word):
                    if d.textlength(line+char, font=lf) > available:
                        rows.append(line)
                        line = ""
                    line += char
            rows.extend((line, ""))
        step = round(lf.size*1.4)
        lh = 2*pad+step*len(rows)
        canvas = Image.new("RGB", (im.width+lw, max(im.height,lh)) if right else (im.width,im.height+lh), "white")
        canvas.paste(im, (0,0))
        cd = ImageDraw.Draw(canvas)
        ox, oy = (im.width+pad,pad) if right else (pad,im.height+pad)
        for row in rows:
            cd.text((ox,oy), row, font=lf, fill="#202630")
            oy += step
        im = canvas
    dest = Path(out) if out else spec_path.parent / s.get("out", spec_path.with_suffix(".png").name)
    im.save(dest, format="PNG", dpi=(dpi,dpi))
    if pdf:
        im.save(pdf, format="PDF", resolution=dpi)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec")
    parser.add_argument("--out")
    parser.add_argument("--pdf")
    args = parser.parse_args()
    print(render(args.spec, args.out, args.pdf))
