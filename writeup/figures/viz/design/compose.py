"""Place rendered annotation PNGs side by side (or stacked with -v) with a heading each.
Usage: compose.py [-v] OUT.png "Title A" a.png "Title B" b.png ..."""
import sys
from PIL import Image, ImageDraw, ImageFont

args = sys.argv[1:]
vertical = args[0] == '-v'
if vertical: args = args[1:]
out, pairs = args[0], args[1:]
items = [(pairs[i], Image.open(pairs[i + 1]).convert('RGB')) for i in range(0, len(pairs), 2)]
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 34)
gap = 28
head_of = lambda t: 64 if t else 0
if vertical:
    W = max(im.width for _, im in items)
    H = sum(head_of(t) + im.height for t, im in items) + gap * (len(items) - 1)
else:
    W = sum(im.width for _, im in items) + gap * (len(items) - 1)
    H = max(head_of(t) for t, _ in items) + max(im.height for _, im in items)
canvas = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(canvas)
x = y = 0
for title, im in items:
    head = head_of(title) if vertical else max(head_of(t) for t, _ in items)
    d.text((x + 4, y + 14), title, font=font, fill='#202630')
    canvas.paste(im, (x, y + head))
    if vertical: y += head + im.height + gap
    else: x += im.width + gap
canvas.save(out, dpi=(200, 200))
print(out)
