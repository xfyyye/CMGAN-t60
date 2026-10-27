#!/usr/bin/env python3
"""Figure 2 v6: balanced text/boxes, smaller circle-C nodes and compact expansion.

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
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('denoise_encoder_v6.png'), help='PNG output path; SVG is saved beside it')
args = parser.parse_args()
PNG = args.output.resolve()
PNG.parent.mkdir(parents=True, exist_ok=True)
SVG = PNG.with_suffix('.svg')
W, H, SCALE = 2200, 620, 3
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
       '<rect width="2200" height="620" fill="#ffffff"/>']

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

def concat_circle(cx,cy,r=28):
    draw.ellipse(pt((cx-r,cy-r,cx+r,cy+r)),fill=COL['panel'],outline=COL['purple_edge'],width=4*SCALE)
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{COL["panel"]}" stroke="{COL["purple_edge"]}" stroke-width="4"/>')
    text(cx,cy,'C',30)

# Upper row: a common centerline; short labels use narrower/shallower boxes.
center_y = 110
nodes=[
    (90,160,100,COL['gray'],COL['gray_edge'],'Input',40),
    (310,190,110,COL['blue'],COL['blue_edge'],'Conv2d',40),
    (560,320,128,COL['blue'],COL['blue_edge'],'Instance Norm\n+ PReLU',38),
    (940,320,128,COL['purple'],COL['purple_edge'],'Dilated\nDense Block',40),
    (1320,190,110,COL['blue'],COL['blue_edge'],'Conv2d',40),
    (1570,320,128,COL['blue'],COL['blue_edge'],'Instance Norm\n+ PReLU',38),
    (1950,160,100,COL['blue'],COL['blue_edge'],'Output',40),
]
for x,w,h,fill,edge,label,sz in nodes:
    rect(x,center_y-h/2,w,h,fill,edge,r=18)
    text(x+w/2,center_y,label,sz,spacing=8)
for a,b in zip(nodes,nodes[1:]):
    arrow([(a[0]+a[1],center_y),(b[0],center_y)])
# Short expansion guide; it has no arrowhead.
line([(1100,174),(1100,235)],COL['purple_edge'],width=3,dash=True)

# Lower panel: unchanged computation, with a shorter accumulated bypass.
rect(40,235,2120,330,COL['panel'],COL['purple_edge'],r=24)
main_y, box_h = 435, 136
box_y = main_y-box_h/2
layer_x=[225,735,1245,1755]
cat_centers=[575,1085,1595]
layer_w=250; cat_r=28
branches=[(200,575),(669,1085),(1179,1595)]
for branch_x,target_x in branches:
    arrow([(branch_x,main_y),(branch_x,295),(target_x,295),(target_x,main_y-cat_r)],COL['purple_edge'])
for i,x in enumerate(layer_x):
    rect(x,box_y,layer_w,box_h,COL['purple'],COL['purple_edge'],r=18)
    text(x+layer_w/2,main_y,f'Dilated Conv\nd = {2**i}',36,spacing=10)
for cx in cat_centers:
    concat_circle(cx,main_y,cat_r)
rect(60,396,110,78,COL['gray'],COL['gray_edge'],r=15)
text(115,main_y,'Input',30)
arrow([(170,main_y),(layer_x[0],main_y)],COL['purple_edge'])
for i,cx in enumerate(cat_centers):
    arrow([(layer_x[i]+layer_w,main_y),(cx-cat_r,main_y)],COL['purple_edge'])
    arrow([(cx+cat_r,main_y),(layer_x[i+1],main_y)],COL['purple_edge'])
for bx,_ in branches:
    dot(bx,main_y,r=5)
# Only the fourth layer is returned.
rect(2030,396,110,78,COL['purple'],COL['purple_edge'],r=15)
arrow([(layer_x[-1]+layer_w,main_y),(2030,main_y)],COL['purple_edge'])
text(2085,main_y,'Output',30)

svg.append('</svg>')
SVG.write_text('\n'.join(svg),encoding='utf-8')
img=img.resize((W,H),Image.Resampling.LANCZOS)
img.save(PNG,dpi=(300,300))
print(PNG)
