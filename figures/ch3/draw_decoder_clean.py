from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
P=Path(__file__).resolve().parent
im=Image.new('RGB',(2800,1400),'white'); d=ImageDraw.Draw(im)
fp='/System/Library/Fonts/Supplemental/Arial.ttf'
if not Path(fp).exists(): fp='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
F=lambda n:ImageFont.truetype(fp,n)
ink='#263442'; green='#DFF2E3'; ge='#3D9360'; cyan='#E2F2F7'; ce='#2E7F92'; grey='#ECEFF3'; gr='#657386'; orange='#FCE7CF'; oe='#C77825'; white='white'
def text(cx,cy,s,n=30,color=ink):
 b=d.multiline_textbbox((0,0),s,font=F(n),spacing=6,align='center'); d.multiline_text((cx-(b[2]-b[0])/2,cy-(b[3]-b[1])/2),s,font=F(n),fill=color,spacing=6,align='center')
def box(x,y,w,h,s,fill,outline,n=28):
 d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=fill,outline=outline,width=4); text(x+w/2,y+h/2,s,n)
def line_arrow(points,col=ink,w=5):
 d.line(points,fill=col,width=w,joint='curve');
 a,b=points[-2:]; ang=math.atan2(b[1]-a[1],b[0]-a[0]); h=18
 p1=(b[0]-h*math.cos(ang-.5),b[1]-h*math.sin(ang-.5)); p2=(b[0]-h*math.cos(ang+.5),b[1]-h*math.sin(ang+.5)); d.polygon([b,p1,p2],fill=col)
def node(x,y): d.ellipse((x-16,y-16,x+16,y+16),fill=ink)
d.text((100,55),'Denoise Decoder',font=F(44),fill=ink)
d.text((100,115),'Parallel mask and complex-spectrum reconstruction branches',font=F(28),fill='#5C6875')
# Main decoder container
d.rounded_rectangle((500,200,2680,930),radius=30,fill='#FBFEFC',outline=ge,width=5)
d.text((550,235),'Denoise Decoder',font=F(36),fill=ge)
# input and split
box(80,515,300,150,'Shared\nTSCB features',grey,gr,30); node(440,590)
line_arrow([(380,590),(424,590)])
# branch rows
mask_y=390; comp_y=700
mask=[(520,mask_y,250,130,'Dilated\nDenseNet'),(830,mask_y,250,130,'Sub-pixel\nupsampling'),(1140,mask_y,230,130,'Conv2d'),(1430,mask_y,290,130,'InstanceNorm2d\n+ PReLU'),(1770,mask_y,230,130,'Final Conv2d'),(2070,mask_y,260,130,'PReLU\nmask output')]
for x,y,w,h,s in mask: box(x,y,w,h,s,green,ge,27)
comp=[(520,comp_y,250,130,'Dilated\nDenseNet'),(830,comp_y,250,130,'Sub-pixel\nupsampling'),(1140,comp_y,290,130,'InstanceNorm2d\n+ PReLU'),(1510,comp_y,230,130,'Conv2d'),(1840,comp_y,300,130,'2-channel\ncomplex output')]
for x,y,w,h,s in comp: box(x,y,w,h,s,cyan,ce,27)
# split routes
line_arrow([(440,590),(470,590),(470,455),(520,455)],ge)
line_arrow([(440,590),(470,590),(470,765),(520,765)],ce)
# branch arrows
for arr,col in [(mask,ge),(comp,ce)]:
 for (x,y,w,h,s),(x2,y2,w2,h2,s2) in zip(arr,arr[1:]): line_arrow([(x+w,y+h/2),(x2,y2+h2/2)],col)
# labels
d.text((520,330),'Mask Decoder',font=F(32),fill=ge)
d.text((520,640),'Complex Decoder',font=F(32),fill=ce)
# external signals
box(80,1040,300,120,'Noisy\nmagnitude',orange,oe,28)
box(80,1210,300,120,'Noisy\nphase',orange,oe,28)
# mask multiplication, phase recombination, add residual
box(580,1015,390,150,'Mask × noisy\nmagnitude',orange,oe,27)
box(1130,1015,410,150,'Recombine with\nnoisy phase',orange,oe,27)
box(1700,1015,500,150,'Add complex residual\n→ enhanced complex spectrum',orange,oe,26)
box(2320,1040,300,120,'iSTFT\n(external)',grey,gr,28)
box(2320,1210,300,120,'Denoised\nspeech',green,ge,28)
# mask output down then left to multiply
line_arrow([(2330,455),(2440,455),(2440,970),(775,970),(775,1015)],oe)
# noisy mag to multiply
line_arrow([(380,1100),(580,1100)],oe)
# mask product to recombine
line_arrow([(970,1090),(1130,1090)],oe)
# noisy phase to recombine (route right then up)
line_arrow([(380,1270),(1050,1270),(1050,1140),(1335,1140),(1335,1165)],oe)
# complex output down then to add
line_arrow([(2140,830),(2140,965),(1950,965),(1950,1015)],oe)
# recombine to add
line_arrow([(1540,1090),(1700,1090)],oe)
# add to istft and speech
line_arrow([(2200,1090),(2320,1090)],oe)
line_arrow([(2470,1160),(2470,1210)],ge)
# footer
d.text((100,1360),'The mask branch ends with PReLU. The final complex Conv2d has no activation.',font=F(27),fill='#5C6875')
out=P/'denoise_decoder_v2.png'; im.save(out,dpi=(300,300)); print(out)
