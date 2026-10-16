from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
P=Path(__file__).resolve().parent
im=Image.new('RGB',(2600,1250),'white');d=ImageDraw.Draw(im)
fp='/System/Library/Fonts/Supplemental/Arial.ttf'
if not Path(fp).exists(): fp='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
F=lambda n:ImageFont.truetype(fp,n)
ink='#263442'; edge='#3D9360'; green='#DFF2E3'; mask='#E7F5EA'; complexc='#E2F2F7'; ce='#2E7F92'; grey='#ECEFF3'; greye='#657386'; orange='#FCE7CF'; oe='#C77825'
def text(cx,cy,s,n=30,color=ink):
 b=d.multiline_textbbox((0,0),s,font=F(n),spacing=6,align='center'); d.multiline_text((cx-(b[2]-b[0])/2,cy-(b[3]-b[1])/2),s,font=F(n),fill=color,spacing=6,align='center')
def box(x,y,w,h,s,fill,outline=edge,n=30):
 d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=fill,outline=outline,width=4);text(x+w/2,y+h/2,s,n)
def arrow(a,b,col=ink,w=5):
 d.line([a,b],fill=col,width=w);ang=math.atan2(b[1]-a[1],b[0]-a[0]);h=18;p1=(b[0]-h*math.cos(ang-.5),b[1]-h*math.sin(ang-.5));p2=(b[0]-h*math.cos(ang+.5),b[1]-h*math.sin(ang+.5));d.polygon([b,p1,p2],fill=col)
d.text((90,45),'Denoise Decoder',font=F(42),fill=ink);d.text((90,105),'Parallel mask and complex-spectrum reconstruction branches',font=F(27),fill='#5C6875')
# input
box(90,500,250,150,'shared\nTSCB features',grey,greye)
# enclosing
d.rounded_rectangle((440,190,2490,980),radius=30,fill='#FBFEFC',outline=edge,width=5)
d.text((490,220),'Denoise Decoder',font=F(34),fill=edge)
# branches labels
# mask
text(600,340,'Mask Decoder',32,edge); y=500
for x,w,s,fill,ol,n in [(500,260,'Dilated\nDenseNet',mask,edge,28),(820,250,'Sub-pixel\nupsampling',mask,edge,27),(1130,230,'Conv2d',mask,edge,29),(1440,270,'InstanceNorm2d\n+ PReLU',mask,edge,27),(1770,220,'Final Conv2d',mask,edge,28),(2050,250,'PReLU\nmask output',mask,edge,28)]: box(x,y,w,150,s,fill,ol,n)
for a,b in [(340,500),(760,820),(1070,1130),(1360,1440),(1710,1770),(1990,2050)]: arrow((a,575),(b,575),edge)
# complex
text(600,760,'Complex Decoder',32,ce); y=830
for x,w,s,fill,ol,n in [(500,260,'Dilated\nDenseNet',complexc,ce,28),(820,250,'Sub-pixel\nupsampling',complexc,ce,27),(1130,270,'InstanceNorm2d\n+ PReLU',complexc,ce,27),(1470,240,'Conv2d',complexc,ce,29),(1770,300,'2-channel\ncomplex output',complexc,ce,27)]: box(x,y,w,120,s,fill,ol,n)
for a,b in [(340,500),(760,820),(1070,1130),(1400,1470),(1710,1770)]: arrow((a,890),(b,890),ce)
# input split routes
arrow((340,575),(420,575),ink);d.line((420,575,420,575),fill=ink,width=4);arrow((420,575),(500,575),edge);arrow((420,575),(420,890),ink);arrow((420,890),(500,890),ce)
# recombination below
d.line((400,575,400,1050),fill=greene if False else '#657386',width=3)
box(520,1050,450,110,'Magnitude mask ×\nnoisy magnitude',orange,oe,27)
box(1100,1050,420,110,'Noisy phase\nrecombination',orange,oe,27)
box(1660,1050,470,110,'Add complex residual\n→ enhanced complex spectrum',orange,oe,25)
# clean orthogonal reconstruction routes
d.line([(2170,575),(2210,575),(2210,1010),(745,1010),(745,1050)],fill=oe,width=5);arrow((745,1010),(745,1050),oe)
arrow((2070,950),(1890,1050),oe);arrow((970,1105),(1100,1105),oe);arrow((1520,1105),(1660,1105),oe)
# note
d.text((90,1195),'Mask branch ends with PReLU; the final complex Conv2d has no activation.',font=F(26),fill='#5C6875')
out=P/'denoise_decoder_v1.png';im.save(out,dpi=(300,300));print(out)
