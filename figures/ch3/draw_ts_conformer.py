from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
ROOT=Path(__file__).resolve().parent
im=Image.new('RGB',(2600,1120),'white'); d=ImageDraw.Draw(im)
fp='/System/Library/Fonts/Supplemental/Arial.ttf'
F=lambda n:ImageFont.truetype(fp,n)
ink='#263442'; edge='#3978B8'; blue='#DDEBFA'; grey='#ECEFF3'
def text(x,y,s,n=30,color=ink):
 b=d.multiline_textbbox((0,0),s,font=F(n),spacing=7,align='center'); d.multiline_text((x-(b[2]-b[0])/2,y-(b[3]-b[1])/2-b[1]),s,font=F(n),fill=color,spacing=7,align='center')
def box(x,y,w,h,s,fill=blue,n=30):
 d.rounded_rectangle((x,y,x+w,y+h),radius=17,fill=fill,outline=edge,width=3);text(x+w/2,y+h/2,s,n)
def arrow(p,col=ink):
 d.line(p,fill=col,width=4,joint='curve');a,b=p[-2:]; theta=math.atan2(b[1]-a[1],b[0]-a[0]);d.polygon([b,(b[0]-17*math.cos(theta-.5),b[1]-17*math.sin(theta-.5)),(b[0]-17*math.cos(theta+.5),b[1]-17*math.sin(theta+.5))],fill=col)
def plus(x,y):
 d.ellipse((x-24,y-24,x+24,y+24),fill='white',outline=edge,width=3);d.line((x-12,y,x+12,y),fill=ink,width=3);d.line((x,y-12,x,y+12),fill=ink,width=3)
text(1300,65,'TS-Conformer block (TSCB)',42)
text(380,150,'(a) Time–frequency processing',32)
# First panel
Y=315
box(65,260,190,110,'Input\nfeatures',grey)
box(320,260,255,110,'Reshape for\ntime modeling',grey)
box(665,260,280,110,'Time\nConformer')
plus(1030,Y)
box(1130,260,275,110,'Reshape for\nfrequency modeling',grey,28)
box(1490,260,280,110,'Frequency\nConformer')
plus(1850,Y)
box(1950,260,255,110,'Restore\nfeature layout',grey)
box(2280,260,250,110,'Output\nfeatures',grey)
for a,b in [(255,320),(575,665),(945,1006),(1054,1130),(1405,1490),(1770,1826),(1874,1950),(2205,2280)]:arrow([(a,Y),(b,Y)])
arrow([(620,Y),(620,210),(1030,210),(1030,291)],edge)
arrow([(1450,Y),(1450,210),(1850,210),(1850,291)],edge)
text(805,425,'Along time, independently\nfor each frequency bin',27)
text(1630,425,'Along frequency, independently\nfor each time frame',27)
d.line((65,500,2530,500),fill='#D9E2ED',width=2)
text(510,560,'(b) Internal structure of each Conformer',32)
# internal block sequence with all four residual connections
Y=760
text(95,Y,'Input',28)
starts=[210,720,1230,1740]
labels=['LayerNorm\nFeed-forward\nScale × 0.5','LayerNorm\nMulti-head\nself-attention','Convolution\nmodule','LayerNorm\nFeed-forward\nScale × 0.5']
arrow([(145,Y),(210,Y)])
for i,(x,s) in enumerate(zip(starts,labels)):
 box(x,690,300,140,s,n=28)
 px=x+375;plus(px,Y);arrow([(x+300,Y),(px-24,Y)])
 arrow([(x-35,Y),(x-35,645),(px,645),(px,736)],edge)
 if i<3:arrow([(px+24,Y),(starts[i+1],Y)])
box(2260,700,225,120,'LayerNorm',n=28)
arrow([(2139,Y),(2260,Y)])
arrow([(2485,Y),(2540,Y)])
text(1300,910,'Time and frequency Conformers have the same structure and separate parameters.',28)
text(1300,980,'Convolution module: LayerNorm → pointwise Conv → GLU → depthwise Conv → BatchNorm → Swish → pointwise Conv → Dropout',25)
text(1300,1040,'Feed-forward: Linear → Swish → Dropout → Linear → Dropout',25)
im.save(ROOT/'ts_conformer_v1.png',dpi=(300,300))
