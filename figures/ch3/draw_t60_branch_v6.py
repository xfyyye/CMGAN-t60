#!/usr/bin/env python3
"""Figure 4 v6: compact regression flow, KAN mapping and Fourier expansion.

Retains the five exact waveform functions, 161 samples on [-pi, pi], and
0.41 amplitude factor from draw_t60_branch_v3.py. Layout alone rescales them.
No expansion guides, explanatory sentences or duplicated full equation.
Panel names and local mathematical labels are retained.
"""
from pathlib import Path
from html import escape
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path,
                    default=Path(__file__).with_name('t60_branch_v6.png'))
PNG = parser.parse_args().output.resolve()
PNG.parent.mkdir(parents=True, exist_ok=True)
W, H, SCALE = 2800, 1045, 3
COL = {'ink': '#263442', 'orange': '#fce7cf', 'oe': '#c77825',
       'gray': '#eceff3', 'gr': '#657386', 'blue': '#ddebfa',
       'be': '#3978b8', 'axis': '#c5cdd6', 'edge': '#cbd2da'}
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
img = Image.new('RGB', (W*SCALE, H*SCALE), 'white')
draw = ImageDraw.Draw(img)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
font_cache = {}
Y_OFFSET = 0


def font(size, kind='text'):
    key = size, kind
    if key not in font_cache:
        name = {'text': 'Arial.ttf', 'roman': 'Times New Roman.ttf',
                'italic': 'Times New Roman Italic.ttf'}[kind]
        font_cache[key] = ImageFont.truetype(str(FONT_DIR/name), round(size*SCALE))
    return font_cache[key]


def pt(p):
    return tuple(round((v+(Y_OFFSET if i % 2 else 0))*SCALE) for i, v in enumerate(p))


def line(points, color=None, width=4):
    color = color or COL['ink']
    draw.line([pt(p) for p in points], fill=color,
              width=round(width*SCALE), joint='curve')
    xy = ' '.join(f'{x},{y}' for x, y in points)
    svg.append(f'<polyline points="{xy}" fill="none" stroke="{color}" '
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
    xy = ' '.join(f'{x},{y}' for x, y in [b, p1, p2])
    svg.append(f'<polygon points="{xy}" fill="{color}"/>')


def text(x, y, s, size=38, color=None, spacing=8):
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


def box(x, cy, w, h, label='', blue=False, size=40):
    rect(x, cy-h/2, w, h, COL['blue'] if blue else COL['orange'],
         COL['be'] if blue else COL['oe'])
    if label:
        text(x+w/2, cy, label, size)


def circle(x, y, r, fill, edge, label='', size=36):
    draw.ellipse(pt((x-r, y-r, x+r, y+r)), fill=fill,
                 outline=edge, width=3*SCALE)
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
               f'stroke="{edge}" stroke-width="3"/>')
    if label:
        text(x, y, label, size)


def atom(s, sub='', sup='', italic=False, hat=False, summation=False):
    return dict(s=s, sub=sub, sup=sup, italic=italic, hat=hat, summation=summation)


def math_label(x, y, atoms, size=40, color=None):
    """Baseline-aligned mathematical atoms with real subscripts/superscripts.

    Both outputs keep each mathematical glyph as text, rather than writing
    underscore placeholders or embedding a raster equation inside the SVG.
    """
    color = color or COL['ink']
    metrics = []
    small = size*.64
    for a in atoms:
        fs = size*(1.16 if a['summation'] else 1)
        kind = 'italic' if a['italic'] else 'roman'
        bw = font(fs, kind).getlength(a['s'])/SCALE
        script_w = max(font(small, 'italic').getlength(a['sub'])/SCALE,
                       font(small, 'roman').getlength(a['sup'])/SCALE)
        aw = max(bw, script_w)+12 if a['summation'] else bw+(script_w+2 if script_w else 0)
        metrics.append((a, fs, kind, bw, aw))
    cursor = x-sum(m[4] for m in metrics)/2
    baseline = y+size*.27

    def glyph(xx, yy, s, fs, kind):
        draw.text(pt((xx, yy)), s, font=font(fs, kind), fill=color, anchor='ls')
        svg.append(f'<text x="{xx}" y="{yy}" font-family="Times New Roman, serif" '
                   f'font-size="{fs}" font-style="{"italic" if kind=="italic" else "normal"}" '
                   f'fill="{color}">{escape(s)}</text>')

    for a, fs, kind, bw, aw in metrics:
        bx = cursor+(aw-bw)/2 if a['summation'] else cursor
        glyph(bx, baseline, a['s'], fs, kind)
        if a['summation']:
            if a['sub']:
                sw = font(small, 'italic').getlength(a['sub'])/SCALE
                glyph(cursor+(aw-sw)/2, baseline+size*.95, a['sub'], small, 'italic')
            if a['sup']:
                sw = font(small, 'italic').getlength(a['sup'])/SCALE
                glyph(cursor+(aw-sw)/2, baseline-size*.94, a['sup'], small, 'italic')
        else:
            if a['sub']:
                glyph(cursor+bw+2, baseline+size*.22, a['sub'], small, 'italic')
            if a['sup']:
                glyph(cursor+bw+2, baseline-size*.56, a['sup'], small, 'roman')
        if a['hat']:
            top = baseline-size*.86
            line([(bx+bw*.12, top+size*.09), (bx+bw*.53, top),
                  (bx+bw*.94, top+size*.09)], color, width=2)
        cursor += aw


def curve(x, y, w, h, fn, color, key):
    # Exact v3 formulas, domain, sample count and vertical scaling.
    values = [fn(-math.pi+2*math.pi*i/160) for i in range(161)]
    points = [(x+w*i/160, y+h/2-h*.41*v) for i, v in enumerate(values)]
    line([(x, y+h/2), (x+w, y+h/2)], COL['axis'], width=2)
    line([(x, y), (x, y+h)], COL['axis'], width=2)
    svg.append(f'<g id="{key}">')
    line(points, color, width=4)
    svg.append('</g>')


# (a) Full regression flow. Numerical K and repeated post-layer details
# remain in the paper; they are omitted from this structural overview.
cy = 115
nodes = [(100, 220, 'Shared\nfeatures', True, 40),
         (380, 210, 'Time\naveraging', False, 40),
         (650, 330, 'Frequency pooling\nMean / Max / Std', False, 36),
         (1040, 230, 'Linear\nLayer Norm', False, 38),
         (1330, 280, 'Fourier-KAN', False, 40),
         (1670, 280, 'Fourier-KAN', False, 40),
         (2010, 170, 'Linear', False, 40),
         (2240, 180, 'Sigmoid', False, 40),
         (2480, 220, '', False, 40)]
for x, w, label, blue, size in nodes:
    box(x, cy, w, 120, label, blue, size)
for a, b in zip(nodes, nodes[1:]):
    arrow([(a[0]+a[1], cy), (b[0], cy)])
math_label(2590, cy, [atom('T', sub='60', sup='norm', italic=True, hat=True)], 44)
text(W/2, 228, '(a) T60 regression branch', 38)

# Compact the lower panels after removing explanatory sentences.
Y_OFFSET = -45
svg.append('<g transform="translate(0 -45)">')

# (b) Two illustrative Fourier-KAN layers. Parameters are edge-specific.
xs = [180, 480, 800]
ys = [[480, 585, 690, 795, 900], [480, 795], [635]]
r = 32
for layer in range(2):
    for yy in ys[layer]:
        for zz in ys[layer+1]:
            if layer == 0 and yy == 480 and zz == 480:
                continue
            dx, dy = xs[layer+1]-xs[layer], zz-yy
            length = math.hypot(dx, dy)
            a = (xs[layer]+r*dx/length, yy+r*dy/length)
            b = (xs[layer+1]-r*dx/length, zz-r*dy/length)
            arrow([a, b], COL['edge'], width=3)
arrow([(xs[0]+r, 480), (xs[1]-r, 480)], COL['oe'])
math_label(330, 433, [atom('φ', sub='ji', italic=True), atom('(x)', italic=True)], 38, COL['oe'])
for j, x in enumerate(xs):
    for yy in ys[j]:
        circle(x, yy, r, COL['gray'] if j == 0 else COL['orange'],
               COL['gr'] if j == 0 else COL['oe'], 'Σ' if j else '')
        if j == 0:
            math_label(x, yy, [atom('x', sub='i' if yy == 480 else '', italic=True)], 34)
for x, label in zip(xs, ['Inputs', 'Hidden features', 'KAN features']):
    text(x, 375, label, 32)
text(480, 1040, '(b) Fourier-KAN mapping', 38)

# (c) Single-edge Fourier expansion. Four representative basis functions
# and the composed curve are mathematically identical to v3.
rect(1010, 340, 1720, 645, 'white', COL['oe'], radius=22, stroke=3)
text(1210, 390, 'Fourier basis', 34)
text(1530, 390, 'Learnable coefficients', 32)
text(1795, 390, 'Sum', 34)
text(2260, 390, 'Edge function', 34)
rows = [480, 590, 790, 900]
labels = ['cos(x)', 'sin(x)', 'cos(Kx)', 'sin(Kx)']
coefs = [('a', 'ji1'), ('c', 'ji1'), ('a', 'jiK'), ('c', 'jiK')]
for i, y in enumerate(rows):
    k = 1 if i < 2 else 4
    trig = math.cos if i % 2 == 0 else math.sin
    curve(1110, y-28.8, 200, 57.6,
          lambda z, k=k, trig=trig: trig(k*z),
          COL['be'] if i % 2 == 0 else COL['oe'], f'basis-{i}')
    text(1210, y+47, labels[i], 28, COL['gr'])
    box(1430, y, 200, 64)
    letter, sub = coefs[i]
    math_label(1530, y, [atom('× '), atom(letter, sub=sub, italic=True)], 38)
    arrow([(1330, y), (1430, y)], COL['oe'])
    # All four visible branches converge on a sum over every k, with the
    # omitted middle orders denoted by vertical ellipses.
    bend = (1680, y)
    dx, dy = 1795-bend[0], 690-bend[1]
    length = math.hypot(dx, dy)
    tip = (1795-32*dx/length, 690-32*dy/length)
    arrow([(1630, y), bend, tip], COL['oe'])
for x in [1210, 1530]:
    # Geometry avoids a missing vertical-ellipsis glyph in platform fonts.
    for y in [674, 690, 706]:
        draw.ellipse(pt((x-3, y-3, x+3, y+3)), fill=COL['gr'])
        svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{COL["gr"]}"/>')
circle(1795, 690, 32, COL['orange'], COL['oe'], 'Σ', 38)
arrow([(1827, 690), (1970, 690)], COL['oe'])
math_label(2260, 520, [atom('φ', sub='ji', italic=True), atom('(x)', italic=True)], 40, COL['oe'])
curve(2010, 584, 500, 212,
      lambda z: (.7*math.cos(z)+.5*math.sin(2*z)-.3*math.cos(3*z))/1.4,
      COL['oe'], 'edge-function')

text(1870, 1040, '(c) Fourier expansion', 38)

svg.append('</g>')
svg.append('</svg>')
PNG.with_suffix('.svg').write_text('\n'.join(svg)+'\n', encoding='utf-8')
img.resize((W, H), Image.Resampling.LANCZOS).save(PNG, dpi=(300, 300))
print(PNG)
