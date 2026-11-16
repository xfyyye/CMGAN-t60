#!/usr/bin/env python3
"""Reproducible vector-first redraw of the shared encoder diagram.

Creates new geometric shapes and text; does not load or modify the old PNG.
Palette values were sampled from the flat fills/strokes of denoise_encoder_v1.png.
Code reference: models/generator.py DenseEncoder / DilatedDenseNet.
"""
from pathlib import Path
from html import escape
import math
import argparse
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('denoise_encoder_v2.png'), help='PNG output path; SVG is saved beside it')
args = parser.parse_args()
PNG = args.output.resolve()
PNG.parent.mkdir(parents=True, exist_ok=True)
SVG = PNG.with_suffix('.svg')
W, H, SCALE = 2400, 1200, 3
COL = {
    'white': '#ffffff', 'ink': '#263442', 'gray': '#eceff3', 'gray_edge': '#657386',
    'blue': '#ddebfa', 'blue_edge': '#3978b8',
    'purple': '#eee4fa', 'purple_edge': '#8062a8', 'panel': '#fbf9fe',
}
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
UNICODE = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
img = Image.new('RGB', (W*SCALE, H*SCALE), COL['white'])
draw = ImageDraw.Draw(img)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="2400" height="1200" fill="#ffffff"/>']

def pt(v):
    return tuple(round(t*SCALE) for t in v)

def rect(x,y,w,h,fill,edge,r=19,stroke=4):
    draw.rounded_rectangle(pt((x,y,x+w,y+h)),radius=round(r*SCALE),fill=fill,outline=edge,width=round(stroke*SCALE))
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{edge}" stroke-width="{stroke}"/>')

def line(points,color=None,width=4,dash=False):
    color=color or COL['ink']
    if dash:
        for a,b in zip(points,points[1:]):
            length=math.dist(a,b)
            for k in range(0,math.ceil(length),19):
                t0=k/length; t1=min(k+10,length)/length
                aa=(a[0]+(b[0]-a[0])*t0,a[1]+(b[1]-a[1])*t0)
                bb=(a[0]+(b[0]-a[0])*t1,a[1]+(b[1]-a[1])*t1)
                draw.line([pt(aa),pt(bb)],fill=color,width=round(width*SCALE))
    else:
        draw.line([pt(p) for p in points],fill=color,width=round(width*SCALE),joint='curve')
    coords=' '.join(f'{x},{y}' for x,y in points)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"'+(' stroke-dasharray="10 9"' if dash else '')+'/>')

def arrow(points,color=None,width=4):
    color=color or COL['ink']
    line(points,color,width)
    a,b=points[-2:]; angle=math.atan2(b[1]-a[1],b[0]-a[0])
    size=16
    p1=(b[0]-size*math.cos(angle)+size*.52*math.sin(angle), b[1]-size*math.sin(angle)-size*.52*math.cos(angle))
    p2=(b[0]-size*math.cos(angle)-size*.52*math.sin(angle), b[1]-size*math.sin(angle)+size*.52*math.cos(angle))
    draw.polygon([pt(b),pt(p1),pt(p2)],fill=color)
    svg.append(f'<polygon points="{b[0]},{b[1]} {p1[0]},{p1[1]} {p2[0]},{p2[1]}" fill="{color}"/>')

def text(x,y,s,size=36,color=None,bold=False,anchor='mm',spacing=9):
    color=color or COL['ink']
    needs_unicode = any(0x2070 <= ord(ch) <= 0x209F or ch in '⌈⌉' for ch in s)
    font_path = UNICODE if needs_unicode else (BOLD if bold else FONT)
    ff=ImageFont.truetype(font_path,round(size*SCALE))
    svg_family = 'Arial Unicode MS, Arial, sans-serif' if needs_unicode else 'Arial, sans-serif'
    # Use metric anchoring for consistent line positions in both outputs.
    lines=s.split('\n'); step=size+spacing; total=(len(lines)-1)*step
    for i,t in enumerate(lines):
        yy=y-total/2+i*step
        draw.text(pt((x,yy)),t,font=ff,fill=color,anchor=anchor)
        svganchor={'mm':'middle','lm':'start','rm':'end'}[anchor]
        svg.append(f'<text x="{x}" y="{yy}" text-anchor="{svganchor}" dominant-baseline="central" font-family="{svg_family}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(t)}</text>')

def dot(x,y,r=6,color=None):
    color=color or COL['purple_edge']
    draw.ellipse(pt((x-r,y-r,x+r,y+r)),fill=color)
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

# Upper panel: unchanged encoder computation order, made dimension-explicit.
text(70,75,'Dense Convolutional Encoder',size=52,bold=True,anchor='lm')
y, bh = 220, 170
nodes=[
    (70,210,COL['gray'],COL['gray_edge'],'Input\n3 × T × F',36),
    (330,260,COL['blue'],COL['blue_edge'],'Conv2d 1 × 1\n3 → C channels',33),
    (640,260,COL['blue'],COL['blue_edge'],'InstanceNorm2d\n+ PReLU',32),
    (950,280,COL['purple'],COL['purple_edge'],'Dilated\nDense Block',38),
    (1280,300,COL['blue'],COL['blue_edge'],'Conv2d 1 × 3\nstride (1, 2)',35),
    (1630,260,COL['blue'],COL['blue_edge'],'InstanceNorm2d\n+ PReLU',32),
    (1940,390,COL['blue'],COL['blue_edge'],'Encoder output\nF⁽⁰⁾: C × T × F′',35),
]
for x,w,fill,edge,label,sz in nodes:
    rect(x,y,w,bh,fill,edge)
    text(x+w/2,y+bh/2,label,sz)
for a,b in zip(nodes,nodes[1:]):
    arrow([(a[0]+a[1],y+bh/2),(b[0],y+bh/2)])
# This label gives the exact manuscript output symbol without ASCII exponents.
text(2135,445,'F′ = ⌈F / 2⌉',size=34)
# Expansion connector is dashed and carries no computational arrow.
line([(1090,y+bh),(1090,520)],COL['purple_edge'],width=3,dash=True)

# Lower panel: three successive accumulated streams, each bypassing one layer.
rect(50,520,2300,610,COL['panel'],COL['purple_edge'],r=27)
text(90,575,'Dilated Dense Block (4 layers)',40,COL['purple_edge'],bold=True,anchor='lm')
text(90,638,'Each layer: padded Conv2d 2 × 3 → InstanceNorm2d → PReLU',34,anchor='lm')
main_y, box_y, box_h = 920, 825, 190
layer_x=[240,800,1360,1920]
cat_x=[590,1150,1710]
layer_w=240; cat_w=120; cat_h=110; cat_y=main_y-cat_h/2
# Plot all bypasses first; branch dots explicitly distinguish forks from crossings.
branches=[(210,650,'C'),(755,1210,'2C'),(1315,1770,'3C')]
for branch_x,target_x,label in branches:
    arrow([(branch_x,main_y),(branch_x,735),(target_x,735),(target_x,cat_y)],COL['purple_edge'])
    text((branch_x+target_x)/2,700,label,34,COL['purple_edge'],bold=True)
# Four feature-generating layers.
for i,x in enumerate(layer_x):
    rect(x,box_y,layer_w,box_h,COL['purple'],COL['purple_edge'])
    cin='C' if i==0 else f'{i+1}C'
    text(x+layer_w/2,main_y,f'Layer {i+1}\ndilation ({2**i}, 1)\n{cin} → C',34,spacing=14)
# The concatenation nodes have two visible incoming edges: left and top.
for x in cat_x:
    rect(x,cat_y,cat_w,cat_h,COL['purple'],COL['purple_edge'],r=16)
    text(x+cat_w/2,main_y,'Concat',31)
# Input, sequential learned feature streams, and final block output.
rect(70,875,110,90,COL['gray'],COL['gray_edge'],r=15)
text(125,main_y,'x₀',39)
arrow([(180,main_y),(layer_x[0],main_y)],COL['purple_edge'])
for i,cat in enumerate(cat_x):
    right=layer_x[i]+layer_w
    arrow([(right,main_y),(cat,main_y)],COL['purple_edge'])
    text((right+cat)/2,main_y-43,f'x{chr(0x2081+i)}',33,COL['purple_edge'])
    arrow([(cat+cat_w,main_y),(layer_x[i+1],main_y)],COL['purple_edge'])
for bx,_,_ in branches:
    dot(bx,main_y)
rect(2220,865,110,110,COL['purple'],COL['purple_edge'],r=15)
arrow([(2160,main_y),(2220,main_y)],COL['purple_edge'])
text(2275,main_y,'Output\nx₄ (C)',28,spacing=12)
# A compact exact expression clarifies concat ordering and the returned tensor.
text(90,1082,'Layer inputs: x₀, [x₁, x₀], [x₂, x₁, x₀], [x₃, x₂, x₁, x₀]     Block output: x₄',32,COL['purple_edge'],anchor='lm')

svg.append('</svg>')
SVG.write_text('\n'.join(svg),encoding='utf-8')
img=img.resize((W,H),Image.Resampling.LANCZOS)
img.save(PNG,dpi=(300,300))
print(PNG)
