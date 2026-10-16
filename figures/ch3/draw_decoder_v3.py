from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
P=Path(__file__).resolve().parent
im=Image.new('RGB',(3600,1120),'white');d=ImageDraw.Draw(im)
fp='/System/Library/Fonts/Supplemental/Arial.ttf';F=lambda n:ImageFont.truetype(fp,n)
ink='#263442'; ge='#3D9360';ce='#2E7F92';gr='#657386'
green='#DFF2E3';cyan='#E2F2F7';grey='#ECEFF3'
def text(cx,cy,s,n=30,color=ink):
 b=d.multiline_textbbox((0,0),s,font=F(n),spacing=6,align='center');d.multiline_text((cx-(b[2]-b[0])/2,cy-(b[3]-b[1])/2-b[1]),s,font=F(n),fill=color,spacing=6,align='center')
def box(x,y,w,h,s,fill=green,col=ge,n=29):
 d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill=fill,outline=col,width=3);text(x+w/2,y+h/2,s,n)
def arrow(p,col=ink):
 d.line(p,fill=col,width=5,joint='curve');a,b=p[-2:];t=math.atan2(b[1]-a[1],b[0]-a[0]);d.polygon([b,(b[0]-19*math.cos(t-.5),b[1]-19*math.sin(t-.5)),(b[0]-19*math.cos(t+.5),b[1]-19*math.sin(t+.5))],fill=col)
def node(x,y,s):
 d.ellipse((x-34,y-34,x+34,y+34),fill='white',outline=ink,width=3);text(x,y,s,43)
text(1800,70,'Denoise Decoder',44)
# two lanes inside decoder
d.rounded_rectangle((365,215,2110,890),radius=26,fill='#FBFEFC',outline=ge,width=3)
text(1230,265,'Parallel decoder branches',30,ge)
box(40,500,235,130,'Shared TSCB\nfeatures',grey,gr,29)
arrow([(275,565),(315,565)])
d.line([(315,420),(315,775)],fill=ink,width=5);d.ellipse((308,558,322,572),fill=ink)
# mask row y420
text(560,330,'Mask Decoder',30,ge)
mask=[(405,255,'Dilated\nDenseNet'),(720,230,'Sub-pixel\nupsampling'),(1010,180,'Conv2d'),(1250,275,'InstanceNorm2d\n+ PReLU'),(1585,195,'Conv2d\n1 × 1'),(1840,225,'PReLU\nmask output')]
for x,w,s in mask:box(x,365,w,110,s)
arrow([(315,420),(405,420)],ge)
for (x,w,_),(xx,_,__) in zip(mask,mask[1:]):arrow([(x+w,420),(xx,420)],ge)
# complex row y775
text(580,665,'Complex Decoder',30,ce)
comp=[(405,255,'Dilated\nDenseNet'),(720,230,'Sub-pixel\nupsampling'),(1010,275,'InstanceNorm2d\n+ PReLU'),(1345,215,'Conv2d'),(1660,405,'Complex residual\n(real and imaginary)')]
for x,w,s in comp:box(x,720,w,110,s,cyan,ce)
arrow([(315,775),(405,775)],ce)
for (x,w,_),(xx,_,__) in zip(comp,comp[1:]):arrow([(x+w,775),(xx,775)],ce)
text(1870,865,'No output activation',24,ce)
# reconstruction: magnitude multiplication and phase, all right of decoder
text(2690,110,'Complex-spectrum reconstruction',31)
node(2290,420,'×')
arrow([(2065,420),(2256,420)],ge)
text(2170,385,'Mask',24,ge)
box(2150,175,280,110,'Noisy\nmagnitude',grey,gr)
arrow([(2290,285),(2290,386)],gr)
box(2500,355,350,130,'Recombine magnitude\nwith noisy phase',grey,gr,28)
arrow([(2324,420),(2500,420)],ge)
box(2535,175,280,110,'Noisy phase',grey,gr)
arrow([(2675,285),(2675,355)],gr)
# add at x3040: complex direct right then up with no other routes in its path
node(3040,420,'+')
arrow([(2850,420),(3006,420)],ge)
arrow([(2065,775),(3040,775),(3040,454)],ce)
text(2540,740,'Complex residual',27,ce)
box(3170,355,360,130,'Reconstructed\ncomplex spectrum',green,ge,28)
arrow([(3074,420),(3170,420)])
# stop at spectrum for module detail: waveform conversion lives in concept figure
text(1800,1010,'×  Element-wise multiplication                         +  Element-wise addition',28)
im.save(P/'denoise_decoder_v3.png',dpi=(300,300))
