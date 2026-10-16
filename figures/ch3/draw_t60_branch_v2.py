from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
P=Path(__file__).resolve().parent
im=Image.new('RGB',(3300,1550),'white');d=ImageDraw.Draw(im)
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

text(555,600,'(b) Learnable functions on edges',33)
text(2170,600,'(c) Fourier representation of the highlighted edge',33)
# small illustrative KAN, two layers, deliberately not actual widths
xs=[200,620,1040]; ys=[[810,1060,1260],[810,1120],[960]]
for layer in range(2):
 for y in ys[layer]:
  for yy in ys[layer+1]:
   dx=xs[layer+1]-xs[layer];dy=yy-y;L=math.hypot(dx,dy)
   a=(xs[layer]+38*dx/L,y+38*dy/L);b=(xs[layer+1]-38*dx/L,yy-38*dy/L)
   if layer==0 and y==810 and yy==810:continue
   arrow([a,b],'#CBD2DA')
arrow([(238,810),(582,810)],oe)
# mini function drawn over selected edge, no claim of sampled trained function
text(410,750,'φ_ji',34,oe)
for j,x in enumerate(xs):
 for y in ys[j]:
  d.ellipse((x-38,y-38,x+38,y+38),fill=grey if j==0 else orange,outline=gr if j==0 else oe,width=3)
  text(x,y,'x_i' if j==0 and y==810 else ('Σ' if j>0 else 'x'),27)
text(200,710,'Inputs',28)
text(620,710,'Hidden nodes',28)
text(1040,710,'Output',28)
text(410,1375,'Schematic connectivity; actual layer widths omitted.',26,gr)
text(630,1440,'At each node: sum incoming edge functions + bias',26)
# highlighted edge detailed panel: fixed basis -> learned coefficients -> sum -> function
# outline
# dashed inset border
for x in range(1210,3220,24):
 d.line((x,675,min(x+12,3220),675),fill=oe,width=2)
 d.line((x,1370,min(x+12,3220),1370),fill=oe,width=2)
for y in range(675,1370,24):
 d.line((1210,y,1210,min(y+12,1370)),fill=oe,width=2)
 d.line((3220,y,3220,min(y+12,1370)),fill=oe,width=2)
text(1435,720,'Fourier basis',29)
text(1865,720,'Learnable coefficients',29)
# four representative bases
rows=[820,950,1080,1210]
labels=['cos(x)','sin(x)','cos(Kx)','sin(Kx)']
coefs=['× a_ji1','× b_ji1','× a_jiK','× b_jiK']
def plot(x,y,w,h,fn,col):
 d.line((x,y+h/2,x+w,y+h/2),fill='#C5CDD6',width=2)
 d.line((x,y,x,y+h),fill='#C5CDD6',width=2)
 pts=[(x+w*i/160,y+h/2-h*.41*fn(-math.pi+2*math.pi*i/160)) for i in range(161)]
 d.line(pts,fill=col,width=4)
for idx,y in enumerate(rows):
 k=1 if idx<2 else 4
 trig=math.cos if idx%2==0 else math.sin
 plot(1310,y-36,250,72,lambda z,k=k,trig=trig:trig(k*z),be if idx%2==0 else oe)
 text(1435,y+49,labels[idx],24,gr)
 box(1750,y-34,235,68,coefs[idx],n=28)
 arrow([(1580,y),(1750,y)],oe)
# aggregate individual branch weighting into single sum; avoid crossings
sx,sy=2215,1015
for y in rows:
 arrow([(1985,y),(2050,y),(sx-35,sy if y==rows[1] else sy+(y-sy)*.12)],oe)
d.ellipse((sx-38,sy-38,sx+38,sy+38),fill=orange,outline=oe,width=3);text(sx,sy,'Σ',35)
text(2205,1280,'Weighted sum\nover k = 1, …, K',26)
arrow([(sx+38,sy),(2440,sy)],oe)
# resulting representative edge function (chosen illustrative weights)
text(2785,790,'Edge function φ_ji(x)',32,oe)
plot(2480,875,625,265,lambda z:(.7*math.cos(z)+.5*math.sin(2*z)-.3*math.cos(3*z))/1.4,oe)
text(2790,1215,'Illustrative coefficients and curve;\nnot fitted model parameters.',26,gr)
text(2170,1440,'φ_ji(x) = Σ_k [a_jik cos(kx) + b_jik sin(kx)]',32)
im.save(P/'t60_branch_v2.png',dpi=(300,300))
