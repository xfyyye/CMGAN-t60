#!/usr/bin/env python3
"""Figure 3 v4: compact TSCB flow and Conformer expansion.

The two external and four internal residuals follow models/generator.py:TSCB
and models/conformer.py:ConformerBlock. FFN subscripts and half-scale labels
are omitted from this structural view, as requested. The two FFNs remain
separate modules; the half scaling remains defined in the paper and code.
Creates a new PNG and an editable SVG.
"""
from pathlib import Path
from html import escape
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path,
                    default=Path(__file__).with_name('ts_conformer_v4.png'))
args = parser.parse_args()
PNG = args.output.resolve()
PNG.parent.mkdir(parents=True, exist_ok=True)
W, H, SCALE = 2400, 640, 3
COL = {'ink': '#263442', 'blue': '#ddebfa', 'edge': '#3978b8',
       'gray': '#eceff3', 'gray_edge': '#657386', 'panel': '#f8fbff'}
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
UNICODE = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
img = Image.new('RGB', (W*SCALE, H*SCALE), 'white')
draw = ImageDraw.Draw(img)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>']


def pt(p):
    return tuple(round(t*SCALE) for t in p)


def rect(x, y, w, h, fill, edge, r=18, stroke=4):
    draw.rounded_rectangle(pt((x, y, x+w, y+h)), radius=r*SCALE,
                           fill=fill, outline=edge, width=stroke*SCALE)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
               f'fill="{fill}" stroke="{edge}" stroke-width="{stroke}"/>')


def line(points, color=None, width=4, dash=False):
    color = color or COL['ink']
    if dash:
        for a, b in zip(points, points[1:]):
            length = math.dist(a, b)
            for k in range(0, math.ceil(length), 19):
                t0, t1 = k/length, min(k+10, length)/length
                aa = (a[0]+(b[0]-a[0])*t0, a[1]+(b[1]-a[1])*t0)
                bb = (a[0]+(b[0]-a[0])*t1, a[1]+(b[1]-a[1])*t1)
                draw.line([pt(aa), pt(bb)], fill=color, width=width*SCALE)
    else:
        draw.line([pt(p) for p in points], fill=color, width=width*SCALE,
                  joint='curve')
    coords = ' '.join(f'{x},{y}' for x, y in points)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
               f'stroke-width="{width}" stroke-linejoin="round"' +
               (' stroke-dasharray="10 9"' if dash else '') + '/>')


def arrow(points, color=None, width=4):
    color = color or COL['ink']
    line(points, color, width)
    a, b = points[-2:]
    angle = math.atan2(b[1]-a[1], b[0]-a[0])
    size = 16
    p1 = (b[0]-size*math.cos(angle)+size*.52*math.sin(angle),
          b[1]-size*math.sin(angle)-size*.52*math.cos(angle))
    p2 = (b[0]-size*math.cos(angle)-size*.52*math.sin(angle),
          b[1]-size*math.sin(angle)+size*.52*math.cos(angle))
    draw.polygon([pt(b), pt(p1), pt(p2)], fill=color)
    svg.append(f'<polygon points="{b[0]},{b[1]} {p1[0]},{p1[1]} '
               f'{p2[0]},{p2[1]}" fill="{color}"/>')


def text(x, y, s, size=40, spacing=8):
    needs_unicode = any(0x2070 <= ord(ch) <= 0x209F for ch in s)
    font = ImageFont.truetype(UNICODE if needs_unicode else FONT, size*SCALE)
    family = 'Arial Unicode MS, Arial, sans-serif' if needs_unicode else 'Arial, sans-serif'
    lines = s.split('\n')
    step = size+spacing
    for i, t in enumerate(lines):
        yy = y-(len(lines)-1)*step/2+i*step
        draw.text(pt((x, yy)), t, font=font, fill=COL['ink'], anchor='mm')
        svg.append(f'<text x="{x}" y="{yy}" text-anchor="middle" '
                   f'dominant-baseline="central" font-family="{family}" '
                   f'font-size="{size}" fill="{COL["ink"]}">{escape(t)}</text>')


def dot(x, y):
    r = 5
    draw.ellipse(pt((x-r, y-r, x+r, y+r)), fill=COL['edge'])
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{COL["edge"]}"/>')


def plus(x, y, fill='white'):
    r = 24
    draw.ellipse(pt((x-r, y-r, x+r, y+r)), fill=fill,
                 outline=COL['edge'], width=4*SCALE)
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
               f'stroke="{COL["edge"]}" stroke-width="4"/>')
    line([(x-11, y), (x+11, y)], width=3)
    line([(x, y-11), (x, y+11)], width=3)


def block(x, y, w, h, label, gray=False, size=40):
    rect(x, y-h/2, w, h, COL['gray'] if gray else COL['blue'],
         COL['gray_edge'] if gray else COL['edge'])
    text(x+w/2, y, label, size)


# (a) Time and frequency sequences each have an external residual.
y = 150
nodes = [(80, 160, 'Input', True), (310, 190, 'Reshape', True),
         (580, 300, 'Time\nConformer', False),
         (1100, 190, 'Reshape', True),
         (1370, 300, 'Frequency\nConformer', False),
         (1890, 190, 'Reshape', True), (2150, 170, 'Output', False)]
for x, w, label, gray in nodes:
    block(x, y, w, 110, label, gray)
for a, b in [(240, 310), (500, 580), (880, 966), (1014, 1100),
             (1290, 1370), (1670, 1756), (1804, 1890), (2080, 2150)]:
    arrow([(a, y), (b, y)])
for fork, add in [(540, 990), (1330, 1780)]:
    arrow([(fork, y), (fork, 50), (add, 50), (add, y-24)], COL['edge'])
    dot(fork, y)
    plus(add, y)
text(W/2, 265, '(a) Time–Frequency Conformer block', 36)

# Expansion guide is descriptive: no arrowhead, no shared-parameter symbol.
line([(730, 205), (730, 310)], COL['edge'], width=3, dash=True)

# (b) Internal Conformer. Each residual bypass is an unscaled identity.
rect(40, 310, 2320, 230, COL['panel'], COL['edge'], r=24, stroke=3)
y = 450
ops = [(200, 220, 'FFN', 165, 560),
       (690, 220, 'MHSA', 655, 1010),
       (1140, 260, 'Conv Module', 1105, 1500),
       (1630, 220, 'FFN', 1595, 1990)]
text(96, y, 'Input', 34)
arrow([(145, y), (200, y)])
for i, (x, w, label, fork, add) in enumerate(ops):
    block(x, y, w, 100, label, size=38)
    arrow([(x+w, y), (add-24, y)])
    arrow([(fork, y), (fork, 355), (add, 355), (add, y-24)], COL['edge'])
    dot(fork, y)
    plus(add, y, COL['panel'])
    if i < len(ops)-1:
        arrow([(add+24, y), (ops[i+1][0], y)])
block(2100, y, 220, 100, 'Layer Norm', size=36)
arrow([(2014, y), (2100, y)])
arrow([(2320, y), (2340, y)])
text(W/2, 595, '(b) Conformer block', 36)

svg.append('</svg>')
PNG.with_suffix('.svg').write_text('\n'.join(svg)+'\n', encoding='utf-8')
img.resize((W, H), Image.Resampling.LANCZOS).save(PNG, dpi=(300, 300))
print(PNG)
