#!/usr/bin/env python3
"""Figure 5 v4: compact parallel denoising decoders and complex reconstruction.

Preserves MaskDecoder/ComplexDecoder ordering from models/generator.py and
the compressed-spectrum reconstruction in models/generator_t60.py.
The two dense blocks have independent parameters. The final cyan output
contains real/imaginary residual channels and has no output activation.
"""
from pathlib import Path
from html import escape
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path,
                    default=Path(__file__).with_name('denoise_decoder_v4.png'))
PNG = parser.parse_args().output.resolve()
PNG.parent.mkdir(parents=True, exist_ok=True)
W, H, SCALE = 3000, 650, 3
COL = {'ink': '#263442', 'ge': '#3d9360', 'ce': '#2e7f92',
       'gr': '#657386', 'green': '#dff2e3', 'cyan': '#e2f2f7',
       'gray': '#eceff3', 'panel': '#fbfefc'}
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
img = Image.new('RGB', (W*SCALE, H*SCALE), 'white')
draw = ImageDraw.Draw(img)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
fonts = {}


def font(size, kind='text'):
    key = size, kind
    if key not in fonts:
        name = {'text': 'Arial.ttf', 'roman': 'Times New Roman.ttf',
                'bold': 'Times New Roman Bold.ttf',
                'italic': 'Times New Roman Italic.ttf',
                'unicode': 'Arial Unicode.ttf'}[kind]
        fonts[key] = ImageFont.truetype(str(FONT_DIR/name), round(size*SCALE))
    return fonts[key]


def pt(p):
    return tuple(round(v*SCALE) for v in p)


def line(points, color=None, width=4):
    color = color or COL['ink']
    draw.line([pt(p) for p in points], fill=color, width=round(width*SCALE),
              joint='curve')
    coords = ' '.join(f'{x},{y}' for x, y in points)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
               f'stroke-width="{width}" stroke-linejoin="round"/>')


def arrow(points, color=None, width=4):
    color = color or COL['ink']
    line(points, color, width)
    a, b = points[-2:]
    theta = math.atan2(b[1]-a[1], b[0]-a[0])
    size = 16
    p1 = (b[0]-size*math.cos(theta-.5), b[1]-size*math.sin(theta-.5))
    p2 = (b[0]-size*math.cos(theta+.5), b[1]-size*math.sin(theta+.5))
    draw.polygon([pt(b), pt(p1), pt(p2)], fill=color)
    coords = ' '.join(f'{x},{y}' for x, y in [b, p1, p2])
    svg.append(f'<polygon points="{coords}" fill="{color}"/>')


def text(x, y, s, size=40, color=None, spacing=8):
    color = color or COL['ink']
    lines = s.split('\n')
    for i, t in enumerate(lines):
        yy = y+(i-(len(lines)-1)/2)*(size+spacing)
        draw.text(pt((x, yy)), t, font=font(size), fill=color, anchor='mm')
        svg.append(f'<text x="{x}" y="{yy}" text-anchor="middle" '
                   f'dominant-baseline="central" font-family="Arial, sans-serif" '
                   f'font-size="{size}" fill="{color}">{escape(t)}</text>')


def rect(x, y, w, h, fill, edge, radius=18, stroke=4):
    draw.rounded_rectangle(pt((x, y, x+w, y+h)), radius=round(radius*SCALE),
                           fill=fill, outline=edge, width=round(stroke*SCALE))
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
               f'rx="{radius}" fill="{fill}" stroke="{edge}" stroke-width="{stroke}"/>')


def box(x, cy, w, h, s='', fill=None, edge=None, size=40):
    rect(x, cy-h/2, w, h, fill or COL['green'], edge or COL['ge'])
    if s:
        text(x+w/2, cy, s, size)


def dot(x, y, r=6):
    draw.ellipse(pt((x-r, y-r, x+r, y+r)), fill=COL['ink'])
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{COL["ink"]}"/>')


def operation(x, y, kind):
    r = 28
    draw.ellipse(pt((x-r, y-r, x+r, y+r)), fill='white',
                 outline=COL['ink'], width=3*SCALE)
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" '
               f'stroke="{COL["ink"]}" stroke-width="3"/>')
    if kind == 'multiply':
        line([(x-11, y-11), (x+11, y+11)], width=3.5)
        line([(x-11, y+11), (x+11, y-11)], width=3.5)
    else:
        line([(x-12, y), (x+12, y)], width=3.5)
        line([(x, y-12), (x, y+12)], width=3.5)


def math_label(x, y, runs, size=50, color=None, hat=False):
    """Short mathematical labels; bold Y matches the manuscript's spectrum."""
    color = color or COL['ink']
    widths = [font(size, kind).getlength(s)/SCALE for s, kind in runs]
    cursor = x-sum(widths)/2
    baseline = y+size*.29
    for (s, kind), width in zip(runs, widths):
        draw.text(pt((cursor, baseline)), s, font=font(size, kind),
                  fill=color, anchor='ls')
        family = 'Arial Unicode MS, Arial, sans-serif' if kind == 'unicode' else 'Times New Roman, serif'
        svg.append(f'<text x="{cursor}" y="{baseline}" font-family="{family}" '
                   f'font-size="{size}" font-weight="{700 if kind=="bold" else 400}" '
                   f'font-style="{"italic" if kind=="italic" else "normal"}" '
                   f'fill="{color}">{escape(s)}</text>')
        cursor += width
    if hat:
        top = baseline-size*.86
        line([(x-size*.25, top+size*.09), (x, top),
              (x+size*.25, top+size*.09)], color, width=2)


# One shared input fans out to two independently parameterized decoders.
rect(310, 165, 1635, 445, COL['panel'], COL['ge'], radius=24, stroke=3)
box(40, 400, 190, 120, 'Shared\nfeatures', COL['gray'], COL['gr'], 38)
arrow([(230, 400), (270, 400)])
line([(270, 300), (270, 500)])
dot(270, 400)

text(350+font(36).getlength('Mask Decoder')/SCALE/2,
     205, 'Mask Decoder', 36, COL['ge'])
mask = [(350, 240, 128, 'Dilated\nDense Block', 38),
        (635, 250, 128, 'Sub-pixel\nupsampling', 40),
        (930, 190, 110, 'Conv2d', 40),
        (1165, 280, 128, 'Instance Norm\n+ PReLU', 38),
        (1490, 190, 110, 'Conv2d', 40),
        (1725, 180, 110, 'PReLU', 40)]
for x, w, h, label, size in mask:
    box(x, 300, w, h, label, size=size)
arrow([(270, 300), (350, 300)], COL['ge'])
for a, b in zip(mask, mask[1:]):
    arrow([(a[0]+a[1], 300), (b[0], 300)], COL['ge'])

text(350+font(36).getlength('Complex Decoder')/SCALE/2,
     400, 'Complex Decoder', 36, COL['ce'])
comp = [(350, 240, 128, 'Dilated\nDense Block', 38),
        (635, 250, 128, 'Sub-pixel\nupsampling', 40),
        (930, 280, 128, 'Instance Norm\n+ PReLU', 38),
        (1265, 190, 110, 'Conv2d', 40),
        (1520, 385, 110, 'Complex residual', 40)]
for x, w, h, label, size in comp:
    box(x, 500, w, h, label, COL['cyan'], COL['ce'], size)
arrow([(270, 500), (350, 500)], COL['ce'])
for a, b in zip(comp, comp[1:]):
    arrow([(a[0]+a[1], 500), (b[0], 500)], COL['ce'])

# Multiplication uses the input compressed magnitude, then phase is applied
# before the two-channel complex residual is added.
box(1965, 100, 150, 90, fill=COL['gray'], edge=COL['gr'])
math_label(2040, 100, [('|', 'roman'), ('Y', 'bold'), ('|', 'roman')])
operation(2040, 300, 'multiply')
arrow([(1905, 300), (2012, 300)], COL['ge'])
math_label(1980, 267, [('M', 'italic')], 42, COL['ge'])
arrow([(2040, 145), (2040, 272)], COL['gr'])
box(2220, 300, 330, 130, 'Spectrum\nreconstruction', COL['gray'], COL['gr'], 40)
arrow([(2068, 300), (2220, 300)], COL['ge'])
box(2300, 100, 170, 90, fill=COL['gray'], edge=COL['gr'])
math_label(2385, 100, [('∠', 'unicode'), ('Y', 'bold')])
arrow([(2385, 145), (2385, 235)], COL['gr'])
operation(2710, 300, 'add')
arrow([(2550, 300), (2682, 300)], COL['ge'])
arrow([(1905, 500), (2710, 500), (2710, 328)], COL['ce'])
box(2810, 300, 150, 110)
math_label(2885, 300, [('Y', 'bold')], 50, hat=True)
arrow([(2738, 300), (2810, 300)])

svg.append('</svg>')
PNG.with_suffix('.svg').write_text('\n'.join(svg)+'\n', encoding='utf-8')
img.resize((W, H), Image.Resampling.LANCZOS).save(PNG, dpi=(300, 300))
print(PNG)
