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
for n in names: assert tokens[n] in theme
pal=np.array([rgb(tokens[n]) for n in names]);surface=rgb(tokens['surface']);pairs=list(itertools.combinations(range(3),2));modes=['normal']+[(t,s) for s in (1.,.6) for t in ['protan','deutan','tritan']]
def title(m): return 'Normal' if m=='normal' else f'{m[0].capitalize()} {m[1]:.1f}'
lines=['# Quantitative Farbanalyse der Statusfarben','','## Methode','', 'sRGB-Tokens aus `services/viz/src/workstation.css:4–34`, abgeglichen mit `theme.js:4–6`; Wörter/Formen aus `ui.js:10–27`. Panel #131b24, Inset #10171f, Hintergrund #0c1117. Machado et al. (2009), Schweregrad 1.0 und 0.6: Matrizen auf lineares sRGB, anschließend Gamut-Clipping auf [0,1] und sRGB-Kodierung. CIELAB D65/2°, CIEDE2000 (kL=kC=kH=1); OKLab euklidisch auf nativer 0–1-Skala (kein ×100). Keine chromatische Adaptation. ΔE00 <10: schwer, 10–<20: schwach, ≥20: darüber; heuristische Vergleichsgrenzen, keine klinische Validierung. Kontrast nach WCAG relativer Leuchtdichte; bei CVD sind **auch die Flächen simuliert**. CVD-Kontrastwerte sind modellierte Werte, keine zusätzliche WCAG-Konformitätsprüfung. Opake Chips: Hintergrund beeinflusst den Kontrast, nicht den intrinsischen Farbabstand.','', 'Referenz: Machado, Oliveira & Fernandes, IEEE TVCG 15(6), 1291–1298 (2009), doi:10.1109/TVCG.2009.113; Matrizen https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html. DE00-Code gegen drei Sharma-Referenzpaare geprüft.','', '## Farbabstände','', '| Simulation | Paar | ΔE00 | OKLab ΔE | Einordnung |','|---|---|---:|---:|---|']
for m in modes:
 p=sim(pal,m)
 for i,j in pairs:
  d=de00(lab(p[i]),lab(p[j]));lines.append(f'| {title(m)} | {names[i]}–{names[j]} | {d:.2f} | {np.linalg.norm(ok(p[i])-ok(p[j])):.4f} | {"schwer" if d<10 else "schwach" if d<20 else "≥20"} |')
lines+=['','## Leuchtdichtekontrast (Verhältnisse :1)','','| Simulation | safe/Panel | warn/Panel | danger/Panel | safe–warn | safe–danger | warn–danger |','|---|---:|---:|---:|---:|---:|---:|']
for m in modes:
 p=sim(pal,m);s=sim(surface,m);vals=[contrast(c,s) for c in p]+[contrast(p[i],p[j]) for i,j in pairs];lines.append('| '+title(m)+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
# Chip rendering: unit muted as in CSS; value and distinct symbol/word coloured.
fig,axs=plt.subplots(4,3,figsize=(10.6,6.8));fig.patch.set_facecolor('white')
labels=[re.search(rf'{n}:\s*\{{.*?label: \'([^\']+)',ui,re.S).group(1) for n in names];markers=['o','^','s']
for row,m in enumerate(modes[:4]):
 for col,ax in enumerate(axs[row]):
  ax.set_facecolor(sim(surface,m));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xticks([]);ax.set_yticks([])
  for sp in ax.spines.values(): sp.set_visible(False)
  c=sim(pal[col],m);ax.text(.07,.56,'1.20',color=c,size=30,weight='medium',va='center');ax.text(.59,.53,'mm',color=sim(rgb(tokens['muted']),m),size=13)
  ax.scatter([.09],[.23],s=35,marker=markers[col],color=[c]);ax.text(.15,.23,labels[col],color=c,size=9,weight='semibold',va='center')
  if row==0: ax.set_title(f'{names[col]} · {tokens[names[col]]}',fontsize=11,pad=10)
  if col==0: ax.set_ylabel(['Normal','Protanopie','Deuteranopie','Tritanopie'][row],fontsize=10,labelpad=13)
fig.suptitle('Statusfarben unter simulierter Farbfehlsichtigkeit',fontsize=15,y=.98)
fig.text(.5,.025,'Machado et al. (2009), Schweregrad 1.0 · identischer Beispielwert zur Farbkontrolle; kein konsistenter Statusfall',ha='center',fontsize=9)
fig.subplots_adjust(left=.13,right=.985,bottom=.09,top=.88,hspace=.15,wspace=.08);fig.savefig(OUT/'status_cvd.png',dpi=300);plt.close(fig)
lines+=['','## 3D-Kontext und Grenzen','','Retina #8b3a4a, Opazität 0.13 (`scene/EyeGlobe.jsx:130–137`), Instrumentenschatten #14070a, Opazität 0.6 (`scene/InstrumentShadow.jsx:16–17,64`), Scope-Wand #654451 (`scene/ProximityWall.jsx:117`). Diese Materialwerte sind keine fertigen Pixel: Licht, Tone Mapping, Überlagerung und Hintergrund bestimmen die tatsächlichen 3D-Kontraste. Als **rein illustrative sRGB-Alpha-Mischung** über #0c1117 ergibt Retina '+ '#'+''.join(f'{round(v*255):02x}' for v in (.13*rgb('#8b3a4a')+.87*rgb('#0c1117')))+'. Keine Übertragung der Panel-WCAG-Werte auf 3D.','', '## Neutrale Schlussfolgerung','', 'Die opaken Statusfarben liefern im Normalmodell 6.51–9.36:1 gegen das Panel und überschreiten damit 4.5:1 für Text.', 'Bei vollständiger Rot-Grün-Fehlsichtigkeit rücken mehrere Statuspaare unter die heuristischen Grenzen; die exakten Werte stehen oben.', 'Die Kontraste zwischen Statusfarben sind gering; Leuchtdichte allein bietet keine robuste dreistufige Statuskodierung.', 'Der Hauptstatus besitzt Wörter und verschiedene Formen (●/▲/■); diese Redundanz erhält die semantische Unterscheidung auch bei ähnlichen Farben.', 'L/R-Balken vermitteln Abstand zusätzlich durch Position; dies unterstützt die Interpretation, ersetzt aber keine explizite Statusbezeichnung.', 'Die einzelnen 3D-Tipkugeln und der Statusverlauf der Szenenlegende kodieren die Klassen nur farblich und sind daher besonders betroffen.', 'Der Szenenalarm ergänzt Farbe durch Bannertext/Form, 2-/3-px-Rahmen und bei Critical einen Puls; reduced-motion schaltet den Puls ab.', 'Die Simulation ist eine modellbasierte Prüfung von Tokens, kein Nachweis der Wahrnehmbarkeit im OP oder der tatsächlich gerenderten 3D-Pixel.', '', 'Abbildung: `status_cvd.png`; 3180×2040 px, 300 dpi, weiße Außenfläche, Original-Panel pro Chip. Alle Zeilen zeigen absichtlich denselben Beispielwert; die Statuswörter sind keine daraus abgeleitete Klassifikation.']
(OUT/'results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[10:40]))

# Optional alternative, no UI change: achromatic safe / blue warn / orange danger.
alt_hex=['#ffffff','#56b4e9','#e69f00']
alt=np.array([rgb(h) for h in alt_hex])
extra=['','## Option: farbfehlsichtigkeitsrobustere Dreierpalette','', 'Nur als Option: safe #ffffff, warn #56b4e9, danger #e69f00. In diesem Modell liegen alle Paarabstände bei vollständiger CVD über 25; dafür werden die gewohnten Grün/Gelb/Rot-Assoziationen aufgegeben. Wörter und Formen bleiben erforderlich. Keine Entscheidung oder Änderung an der UI.','', '| Simulation | Paar | ΔE00 | OKLab ΔE |','|---|---|---:|---:|']
for m in modes:
 p=sim(alt,m)
 for i,j in pairs:
  extra.append(f'| {title(m)} | {names[i]}–{names[j]} | {de00(lab(p[i]),lab(p[j])):.2f} | {np.linalg.norm(ok(p[i])-ok(p[j])):.4f} |')
extra+=['', 'Normaler Panelkontrast safe / warn / danger: '+ ' / '.join(f'{contrast(c,surface):.2f}:1' for c in alt)+'.', '', 'Visuelle Prüfung der PNG durchgeführt: alle zwölf Chips, Wörter und Formen lesbar, keine abgeschnittenen Beschriftungen. Matplotlib-Rekonstruktion der UI-Farben und Informationshierarchie; kein Browser-Screenshot und keine pixelgenaue Typografiereplik.']
(OUT/'results.md').write_text('\n'.join(lines+extra)+'\n')
