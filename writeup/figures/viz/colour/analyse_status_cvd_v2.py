"""Reproducible token analysis; run with .venv-ml/bin/python. No browser/render sampling."""
from pathlib import Path
import re,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]; OUT=Path(__file__).resolve().parent
css=(ROOT/'services/viz/src/workstation.css').read_text(); theme=(ROOT/'services/viz/src/theme.js').read_text(); ui=(ROOT/'services/viz/src/ui.js').read_text()
names=['safe','warn','danger']; tokens=dict(re.findall(r'--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;',css))
def rgb(h): return np.array([int(h[i:i+2],16)/255 for i in (1,3,5)])
def lin(c): return np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
def enc(c): return np.where(c<=.0031308,12.92*c,1.055*np.maximum(c,0)**(1/2.4)-.055)
# Machado, Oliveira & Fernandes (2009), published linear-RGB matrices.
M={
('protan',1.):[[.152286,1.052583,-.204868],[.114503,.786281,.099216],[-.003882,-.048116,1.051998]],
('deutan',1.):[[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]],
('tritan',1.):[[1.255528,-.076749,-.178779],[-.078411,.930809,.147602],[.004733,.691367,.303900]],
('protan',.6):[[.385450,.769005,-.154455],[.100526,.829802,.069673],[-.007442,-.022190,1.029632]],
('deutan',.6):[[.498864,.674741,-.173604],[.205199,.754872,.039929],[-.011131,.030969,.980162]],
('tritan',.6):[[1.104996,.046633,-.151629],[-.032137,.971635,.060503],[.001336,.317922,.680742]]}
def sim(c,mode): return c if mode=='normal' else enc(np.clip(lin(c)@np.array(M[mode]).T,0,1))
def lab(c):
 xyz=lin(c)@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T / [.95047,1,1.08883]
 f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
 return np.array([116*f[1]-16,500*(f[0]-f[1]),200*(f[1]-f[2])])
def ok(c):
 lms=lin(c)@np.array([[.4122214708,.5363325363,.0514459929],[.2119034982,.6806995451,.1073969566],[.0883024619,.2817188376,.6299787005]]).T
 return np.cbrt(lms)@np.array([[.2104542553,.793617785,-.0040720468],[1.9779984951,-2.428592205,.4505937099],[.0259040371,.7827717662,-.808675766]]).T
def de00(x,y):
 L1,a1,b1=x;L2,a2,b2=y; C1=np.hypot(a1,b1);C2=np.hypot(a2,b2); C=(C1+C2)/2
 G=.5*(1-np.sqrt(C**7/(C**7+25**7))); a1*=1+G;a2*=1+G
 c1=np.hypot(a1,b1);c2=np.hypot(a2,b2);h1=np.degrees(np.arctan2(b1,a1))%360;h2=np.degrees(np.arctan2(b2,a2))%360
 dl=L2-L1;dc=c2-c1;dh=h2-h1
 if c1*c2==0: dh=0
 elif dh>180: dh-=360
 elif dh< -180: dh+=360
 dH=2*np.sqrt(c1*c2)*np.sin(np.radians(dh/2));lm=(L1+L2)/2;cm=(c1+c2)/2
 hm=h1+h2 if c1*c2==0 else ((h1+h2)/2 if abs(h1-h2)<=180 else (h1+h2+360)/2 if h1+h2<360 else (h1+h2-360)/2)
 T=1-.17*np.cos(np.radians(hm-30))+.24*np.cos(np.radians(2*hm))+.32*np.cos(np.radians(3*hm+6))-.20*np.cos(np.radians(4*hm-63))
 sl=1+.015*(lm-50)**2/np.sqrt(20+(lm-50)**2);sc=1+.045*cm;sh=1+.015*cm*T
 rt=-2*np.sqrt(cm**7/(cm**7+25**7))*np.sin(np.radians(60*np.exp(-((hm-275)/25)**2)))
 return np.sqrt((dl/sl)**2+(dc/sc)**2+(dH/sh)**2+rt*(dc/sc)*(dH/sh))
def lum(c): return lin(c)@np.array([.2126,.7152,.0722])
def contrast(a,b):
 x,y=sorted([lum(a),lum(b)]);return (y+.05)/(x+.05)
# Published Sharma et al. (2005) reference pairs, guarding the nontrivial DE00 implementation.
for a,b,expected in [([50,2.6772,-79.7751],[50,0,-82.7485],2.0425),([50,3.1571,-77.2803],[50,0,-82.7485],2.8615),([50,2.49,-.001],[50,-2.49,.0011],7.2195)]:
 assert abs(de00(np.array(a,dtype=float),np.array(b,dtype=float))-expected)<.0001
# Compare the immutable baseline with the current shared theme/CSS tokens.
for n in names: assert tokens[n] in theme
palettes={'Old':['#64c69a','#e7b75b','#f27a80'], 'New':[tokens[n] for n in names]}
modes=['normal',('protan',1.),('deutan',1.)]
pairs=list(itertools.combinations(range(3),2))
surface=rgb('#131b24')
lines=['# Status palette v2', '', 'Machado 2009, severity 1.0; linear sRGB, gamut clipping; CIELAB D65, CIEDE2000. WCAG contrast against #131b24. Model/token analysis, not rendered 3D contrast or clinical validation.', '', '| Palette | Simulation | safe–warn ΔE00 | safe–danger ΔE00 | warn–danger ΔE00 |', '|---|---|---:|---:|---:|']
fig,axes=plt.subplots(6,3,figsize=(10.6,9),facecolor='white')
for pi,(name,hexes) in enumerate(palettes.items()):
 if pi: lines.extend(['| Palette | Simulation | safe–warn ΔE00 | safe–danger ΔE00 | warn–danger ΔE00 |', '|---|---|---:|---:|---:|'])
 pal=np.array([rgb(h) for h in hexes])
 if name=='New': assert all(contrast(c,surface)>=4.5 for c in pal)
 for mi,mode in enumerate(modes):
  q=sim(pal,mode)
  label='Normal' if mode=='normal' else mode[0].capitalize()
  ds=[de00(lab(q[i]),lab(q[j])) for i,j in pairs]
  if name=='New': assert min(ds)>=20
  lines.append('| '+name+' | '+label+' | '+' | '.join(f'{d:.2f}' for d in ds)+' |')
  for i,ax in enumerate(axes[pi*3+mi]):
   ax.set_facecolor(sim(surface,mode));ax.set_xticks([]);ax.set_yticks([])
   ax.text(.5,.65,names[i],color=q[i],ha='center',va='center',fontsize=16,weight='bold')
   ax.text(.5,.30,hexes[i],color=q[i],ha='center',fontsize=12)
   if i==0:ax.set_ylabel(name+' / '+label)
 lines.extend(['',name+' colors: '+', '.join(hexes), 'Normal surface contrast safe / warn / danger: '+' / '.join(f'{contrast(c,surface):.2f}:1' for c in pal), 'Normal inter-status contrast safe–warn / safe–danger / warn–danger: '+' / '.join(f'{contrast(pal[i],pal[j]):.2f}:1' for i,j in pairs),''])
fig.suptitle('Status palette: baseline vs lightness-separated red / amber / teal-green')
fig.tight_layout(rect=(0,0,1,.96))
fig.savefig(OUT/'status_cvd_v2.png',dpi=300)
(OUT/'results_v2.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
