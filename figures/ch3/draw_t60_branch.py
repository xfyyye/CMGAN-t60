from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
P=Path(__file__).resolve().parent
im=Image.new('RGB',(3300,1400),'white');d=ImageDraw.Draw(im)
F=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
ink='#263442';oe='#C77825';orange='#FCE7CF';gr='#657386';grey='#ECEFF3';blue='#DDEBFA';be='#3978B8'
def text(x,y,s,n=30,col=ink):
 b=d.multiline_textbbox((0,0),s,font=F(n),spacing=7,align='center');d.multiline_text((x-(b[2]-b[0])/2,y-(b[3]-b[1])/2-b[1]),s,font=F(n),fill=col,spacing=7,align='center')
def box(x,y,w,h,s,col=oe,fill=orange,n=29):
 d.rounded_rectangle((x,y,x+w,y+h),radius=17,fill=fill,outline=col,width=3);text(x+w/2,y+h/2,s,n)
def arrow(p,col=ink):
 d.line(p,fill=col,width=4,joint='curve');a,b=p[-2:];t=math.atan2(b[1]-a[1],b[0]-a[0]);d.polygon([b,(b[0]-18*math.cos(t-.5),b[1]-18*math.sin(t-.5)),(b[0]-18*math.cos(t+.5),b[1]-18*math.sin(t+.5))],fill=col)
text(1650,65,'T60 regression branch',44)
text(440,155,'(a) Regression head',33)
blocks=[(60,245,'Shared TSCB\nfeatures'),(375,245,'Time\naveraging'),(690,350,'Frequency statistics\nMean / Max / Std\nConcatenate'),(1110,280,'Linear projection\nLayerNorm'),(1460,390,'Fourier-KAN layer 1\nLayerNorm + Dropout\nK = 16'),(1920,390,'Fourier-KAN layer 2\nLayerNorm + Dropout\nK = 8'),(2380,230,'Linear'),(2680,230,'Sigmoid'),(2980,250,'T60 output')]
for i,(x,w,s) in enumerate(blocks):box(x,245,w,175,s,col=be if i==0 else oe,fill=blue if i==0 else orange,n=28)
for (x,w,_),(xx,_,__) in zip(blocks,blocks[1:]):arrow([(x+w,332),(xx,332)])
text(1760,465,'Each Fourier-KAN layer uses the edge functions illustrated below.',28,oe)
d.line((65,535,3230,535),fill='#DFE5EB',width=2)
text(900,600,'(b) Fourier expansion of one edge, followed by aggregation',33)
# scalar fanout into cosine and sine basis banks
box(95,835,255,120,'Input feature x_i',gr,grey,27)
arrow([(350,895),(435,895)])
d.line([(435,780),(435,1010)],fill=ink,width=4)
d.ellipse((429,889,441,901),fill=ink)
box(550,720,420,120,'cos(x_i), …, cos(K x_i)')
box(550,950,420,120,'sin(x_i), …, sin(K x_i)')
arrow([(435,780),(550,780)])
arrow([(435,1010),(550,1010)])
box(1080,720,440,120,'Multiply by a_ji1, …, a_jiK')
box(1080,950,440,120,'Multiply by b_ji1, …, b_jiK')
arrow([(970,780),(1080,780)],oe);arrow([(970,1010),(1080,1010)],oe)
box(1680,815,355,160,'Sum over k\nand both basis families\nEdge function φ_ji(x_i)',n=27)
arrow([(1520,780),(1600,780),(1600,855),(1680,855)],oe)
arrow([(1520,1010),(1600,1010),(1600,935),(1680,935)],oe)
box(2220,815,480,160,'Sum edge functions over i\nand add output bias c_j',n=28)
arrow([(2035,895),(2220,895)],oe)
box(2860,835,350,120,'Layer output y_j',n=29)
arrow([(2700,895),(2860,895)],oe)
text(1300,1130,'a_jik and b_jik: learnable coefficients',28,oe)
text(745,1130,'k = 1, …, K',27,gr)
text(2455,1070,'One edge is expanded here;\nall input edges contribute to y_j.',26,gr)
text(1650,1260,'y_j = Σ_i Σ_k [a_jik cos(k x_i) + b_jik sin(k x_i)] + c_j',35)
im.save(P/'t60_branch_v1.png',dpi=(300,300))
