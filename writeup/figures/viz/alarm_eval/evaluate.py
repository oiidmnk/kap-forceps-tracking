"""Standalone scalar clearance replay; run from repo root with .venv-ml/bin/python."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
RUNS=['10-35-27','10-48-23','11-00-43']
FPS=30

def robust(x):
    x=x[np.isfinite(x)]
    return float(1.4826*np.median(np.abs(x-np.median(x)))) if len(x) else float('nan')

def raw(h):
    return 2 if h<.5 else 1 if h<2 else 0

def replay(h,margin,grace=0,dwell=0):
    state=-1; last=None; pending=None; since=None
    result=[]; dropout_ends=0
    for i,v in enumerate(h):
        t=i*1000/FPS
        if not np.isfinite(v):
            pending=since=None
            if last is None or grace==0 or t-last>grace+1e-7:
                if state>0: dropout_ends+=1
                state=-1
        else:
            last=t
            r=raw(v)
            if state<0 or r>=state:
                state=r; pending=since=None
            else:
                # Exact scalar equivalent of levels.js nextLevel, including clamp.
                held=raw(max(0,v-margin))
                target=state if held>state else held
                if target<state:
                    if pending!=target: pending=target; since=t
                    if t-since>=dwell-1e-7:
                        state=target; pending=since=None
                else: pending=since=None
        result.append(state)
    return np.array(result),dropout_ends

def episodes(x,l):
    return int(np.sum((x==l)&(np.r_[True,x[:-1]!=l])))

def latency(display,truth):
    starts=np.flatnonzero((truth==2)&np.r_[True,truth[:-1]!=2]); delays=[]; missed=0
    for s in starts:
        e=s
        while e<len(truth) and truth[e]==2: e+=1
        hit=np.flatnonzero(display[s:e]==2)
        if len(hit): delays.append(float(hit[0]*1000/FPS))
        else: missed+=1
    return delays,missed,len(starts)

def dump(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

arrays={r:np.genfromtxt(ROOT/f'artifacts/sim_gt_{r}_ml_frames.csv',delimiter=',',names=True) for r in RUNS}
jitters=[]; samples={};
for r,a in arrays.items():
    h=a['height_rec_mm']; gt=a['height_gt_mm']; residual=h-gt
    med=np.array([np.nanmedian(residual[max(0,i-7):i+8]) if np.any(np.isfinite(residual[max(0,i-7):i+8])) else np.nan for i in range(len(h))])
    # Do not bridge missing observations for first differences.
    dh=np.diff(h); dh=dh[np.isfinite(dh)]; high=residual-med; high=high[np.isfinite(high)]
    samples[r]=(dh,high)
for r in RUNS+['pooled']:
    dh,high=samples[r] if r!='pooled' else tuple(np.concatenate([samples[k][i] for k in RUNS]) for i in range(2))
    jitters.append(dict(run=r,pairs=len(dh),diff_std_mm=np.std(dh),diff_robust_sigma_mm=robust(dh),diff_noise_sigma_mm=robust(dh)/np.sqrt(2),residual_highpass_std_mm=np.std(high),residual_highpass_robust_sigma_mm=robust(high)))
margin=float(np.ceil(2*max(jitters[-1]['residual_highpass_robust_sigma_mm'],jitters[-1]['diff_noise_sigma_mm'])*100)/100)
configs={'current':(.05,0,0),'new_015':(.15,400,250),'new_recommended':(margin,400,250)}
rows=[]; displays={}; latencies={k:[] for k in configs}
for r,a in arrays.items():
    h=a['height_rec_mm']; gt=a['height_gt_mm']; truth=np.array([raw(v) for v in gt])
    for name,args in configs.items():
        d,ends=replay(h,*args); displays[r,name]=d
        delays,missed,onsets=latency(d,truth); latencies[name]+=delays
        rows.append(dict(run=r,variant=name,hysteresis_mm=args[0],frames=len(h),duration_s=len(h)/FPS,dropout_pct=100*np.mean(~np.isfinite(h)),changes=int(np.sum(d[1:]!=d[:-1])),changes_per_min=np.sum(d[1:]!=d[:-1])*FPS*60/len(h),critical_episodes=episodes(d,2),warning_episodes=episodes(d,1),critical_s=np.sum(d==2)/FPS,warning_s=np.sum(d==1)/FPS,safe_s=np.sum(d==0)/FPS,unavailable_s=np.sum(d==-1)/FPS,missed_critical_s=np.sum((truth==2)&(d!=2))/FPS,false_critical_s=np.sum((truth!=2)&(d==2))/FPS,truth_critical_s=np.sum(truth==2)/FPS,critical_latency_median_ms=np.median(delays) if delays else np.nan,critical_onsets=onsets,undetected_onsets=missed,dropout_ended_alarms=ends))
for name in configs:
    rr=[r for r in rows if r['variant']==name]; q=dict(rr[0]); q['run']='pooled'
    for k in ['frames','duration_s','changes','critical_episodes','warning_episodes','critical_s','warning_s','safe_s','unavailable_s','missed_critical_s','false_critical_s','truth_critical_s','critical_onsets','undetected_onsets','dropout_ended_alarms']: q[k]=sum(r[k] for r in rr)
    q['changes_per_min']=q['changes']*60/q['duration_s']; q['dropout_pct']=sum(r['dropout_pct']*r['frames'] for r in rr)/q['frames']; q['critical_latency_median_ms']=np.median(latencies[name]) if latencies[name] else np.nan
    rows.append(q)
for q in rows:
    for l in ['critical','warning','safe','unavailable']: q[l+'_pct']=100*q[l+'_s']/q['duration_s']
dump('jitter.csv',jitters); dump('metrics.csv',rows)
# Reproducible per-frame replay, including truth, enables independent inspection.
frame_rows=[]
for r,a in arrays.items():
    for i in range(len(a)):
        frame_rows.append(dict(run=r,frame=int(a['frame'][i]),height_gt_mm=a['height_gt_mm'][i],height_rec_mm=a['height_rec_mm'][i],truth=raw(a['height_gt_mm'][i]),**{k:int(displays[r,k][i]) for k in configs}))
dump('replay.csv',frame_rows)
# Excerpt centred on a true closest approach; avoid crop at edges.
r='11-00-43'; a=arrays[r]; t=a['frame']/FPS; center=t[np.argmin(a['height_gt_mm'])]; start=max(t[0],min(center-15,t[-1]-30)); sel=(t>=start)&(t<=start+30)
fig,ax=plt.subplots(3,1,figsize=(10,4.8),sharex=True,gridspec_kw={'height_ratios':[4,1,1]})
ax[0].plot(t[sel],a['height_gt_mm'][sel],color='black',lw=1.4,label='Simulator-Ground-Truth')
ax[0].plot(t[sel],a['height_rec_mm'][sel],color='#777777',lw=.9,label='Tracker (Lücken = NaN)')
for v in [.5,2]: ax[0].axhline(v,color='black',ls=':',lw=.7)
ax[0].set_ylabel('Abstand Mittelpunkt [mm]'); ax[0].legend(frameon=False,fontsize=8); ax[0].grid(alpha=.15)
colors=['#eeeeee','#3f8c59','#e2ac32','#c64843']
for axis,key,label in [(ax[1],'current','Aktuell'),(ax[2],'new_recommended',f'Neu ({margin:.2f} mm)')]:
    axis.pcolormesh(np.r_[t[sel]-1/60,t[sel][-1]+1/60],[0,1],(displays[r,key][sel]+1)[None,:],cmap=ListedColormap(colors),vmin=0,vmax=3,shading='flat')
    axis.set_yticks([]); axis.set_ylabel(label,rotation=0,ha='right',va='center',fontsize=9)
ax[-1].set_xlabel('Videozeit [s]'); ax[-1].set_xlim(start,start+30)
fig.suptitle(f'Run {r} · 30 s um die größte Annäherung',fontsize=11)
fig.text(.5,.012,'Status: grün = sicher · gelb = Warnung · rot = kritisch · hellgrau = nicht verfügbar',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.04,1,.95]); fig.savefig(OUT/'alarm_timeline.png',dpi=200,facecolor='white'); plt.close(fig)
print('Recommended margin:',margin)
print('Jitter:',jitters)
for q in rows: print(q)

# Boundary checks of the scalar state machine (no browser / application changes).
assert replay([.49,.50,.55,2.06],.05)[0].tolist()==[2,2,1,0]
assert replay([.4,np.nan,.8],.05)[0].tolist()==[2,-1,1]
assert replay([.4]+[np.nan]*13,.08,400,250)[0].tolist()==[2]*13+[-1]
assert replay([.4]+[.8]*9,.08,400,250)[0][-1]==1
assert replay([.4]+[.8]*8,.08,400,250)[0][-1]==2
assert replay([3,.4],.08,400,250)[0].tolist()==[0,2]

labels={'current':'Alt 0,05','new_015':'Neu 0,15','new_recommended':'Neu 0,08'}
text='''# Stabilität der Abstandswarnung – Simulator-Replay

**Empfehlung (vorläufig): Hysterese 0,08 mm, Deeskalations-Dwell 250 ms, Dropout-Grace 400 ms.**
Die Kombination senkt Statuswechsel um 64,5 % (106,5 → 37,8/min) und dropoutbedingte Alarmabbrüche von 126 auf 39.
Sie beseitigt die hier verpassten 0,63 s kritischer Zeit, erhöht aber falsch-kritische Anzeige von 13,83 auf 43,37 s.
Stabilität ist damit deutlich besser; geometrische Genauigkeit wird dadurch nicht besser.

## Methode und Codebezug

11 428 aufeinanderfolgende CSV-Frames, 380,93 s insgesamt (124,40 / 137,90 / 118,63 s), 30 Hz.
Ausgewertet wird jeweils der gesamte CSV-Ausschnitt, nicht die vollständige Aufnahme; keine Filterung nach `visible`.
26,48 % der Frames haben keine rekonstruierte Höhe. Zeitbasis `frame/30` (Logzeit hat konstanten Alignment-Offset).
Schwellen: kritisch <0,5 mm, Warnung <2 mm, sonst sicher; Ground Truth ohne Hysterese/Dwell.

Gelesener Ausgangsstand (andere Agenten können diese Dateien anschließend ändern):
- `services/viz/src/ui.js:30–33`, `services/viz/src/config.js:9–14`: Schwellen und bisher 0,05 mm Hysterese.
- `services/viz/src/feed/levels.js:14–18`: sofortige Eskalation, Deeskalation anhand `distanceLevel(max(0,h−margin))`; `:29–31`: NaN/fehlender Frame setzt Zustand zurück.
- `services/viz/src/feed/pipeline.js:80–102`: Ausfall setzt Level zurück und entfernt Messwerte; Level-/Verfügbarkeitswechsel werden sofort publiziert, sonst 100-ms-Readout (`config.js:22`).
- `services/viz/src/feed/trend.js:49–66`: Lückenmarker, Steigung/TTC; keine Glättung der Level-Eingabe.

`evaluate.py` emuliert diese **skalare** Level-Logik. Neu ist ein explizites vorgeschlagenes Modell, kein Nachweis der parallel entstehenden JS-Implementierung:
400 ms seit letzter gültiger Messung wird das Level gehalten, danach nicht verfügbar; keine Deeskalation ohne Messung.
Ein niedrigeres Ziellevel muss kontinuierlich 250 ms qualifizieren; Zielwechsel, erneute Eskalation oder NaN setzen den Dwell zurück.
Bei 30 Hz sind das mindestens 266,7 ms seit dem ersten qualifizierenden Frame. Eskalation bleibt sofortig.
Nach abgelaufener Grace beginnt die nächste gültige Messung ohne alten Zustand.
Numerische Werte gelten während NaN weiterhin als fehlend; nur die Statusfarbe wird gehalten.

Statuswechsel zählen **alle** Übergänge einschließlich nicht verfügbar, ohne künstliche Übergänge zwischen Läufen oder Initialisierung.
Kritisch-/Warn-Episoden sind zusammenhängende Abschnitte des jeweiligen Levels (kein binärer Alarmzähler).
Dropout-Ende = kritisch/Warnung → nicht verfügbar, bei Neu erst nach Grace; kritisch → Warnung zählt nicht als Alarmende.
Zeitanteile beziehen sich auf alle Frames. Verpasste kritische Zeit zählt auch Ausfälle als nicht kritisch;
falsch-kritisch zählt auch während Grace gehaltene falsche Alarme.
Onset-Latenz: erste kritische Anzeige innerhalb einer wahren kritischen Episode minus deren Beginn, Untergrenze 0;
ein bereits aktiver Fehlalarm ergibt daher 0 ms. Nicht detektierte Episoden werden separat gezählt.

## Jitter (mm)

Robust σ = 1,4826 × MAD. Δh nur zwischen direkt benachbarten gültigen Frames, keine Überbrückung von NaNs.
Rausch-σ aus Δh = robust σ(Δh)/√2 (nur Näherung: reale Bewegung, Korrelation und Ausreißer enthalten).
HP-Residuum = (h_rec−h_gt) minus zentrierter 15-Frame-Median (0,5 s), NaNs ignoriert, Randfenster verkürzt.
Pooled = zusammengeführte Stichproben, keine Differenzen über Laufgrenzen.

| Lauf | σ(Δh) | robust σ(Δh) | Rausch-σ | σ(HP) | robust σ(HP) |
|---|---:|---:|---:|---:|---:|
'''
for j in jitters:
    text+=f"| {j['run']} | {j['diff_std_mm']:.3f} | {j['diff_robust_sigma_mm']:.3f} | {j['diff_noise_sigma_mm']:.3f} | {j['residual_highpass_std_mm']:.3f} | {j['residual_highpass_robust_sigma_mm']:.3f} |\n"
text+='''
**Herleitung:** 2 × max(0,0367; 0,0104) = 0,0734 mm, auf 0,08 mm aufgerundet.
Der HP-Median kann langsame Fehler und an Lücken auch Jitter entfernen; deshalb die größere Δh-basierte Schätzung.
Die viel höheren Standardabweichungen zeigen seltene große Sprünge, die Hysterese allein nicht abfängt.
250 ms Dwell unterdrückt kurze Rückstufungen, ohne Eskalation zu verzögern; 400 ms Grace überbrückt bis zu zwölf fehlende Frames.
Beides ist ein praktikabler Startwert, kein optimierter/klinisch validierter Grenzwert. 0,15 mm reduziert zusätzlich nur
1,6 Wechsel/min (37,8 → 36,2), hält falsch-kritisch etwas länger (43,37 → 43,60 s); deshalb 0,08 mm bevorzugt.

## Replay: Stabilität und Zeitanteile

Neu jeweils Grace 400 ms + Dwell 250 ms. Anteile K/W/S/∅ = kritisch/Warnung/sicher/nicht verfügbar, in %.

| Lauf | Variante | Wechsel/min | Episoden K/W | Anteile K/W/S/∅ (%) | Alarmenden durch Dropout |
|---|---|---:|---:|---:|---:|
'''
for q in rows:
    text+=f"| {q['run']} | {labels[q['variant']]} | {q['changes_per_min']:.1f} | {q['critical_episodes']}/{q['warning_episodes']} | "+'/'.join(f"{q[k+'_pct']:.1f}" for k in ['critical','warning','safe','unavailable'])+f" | {q['dropout_ended_alarms']} |\n"
text+='''
## Gegen Ground Truth

| Lauf | Variante | Kritisch verpasst (s) | Falsch kritisch (s) | Onset-Latenz Median (ms) | Onsets verpasst/gesamt |
|---|---|---:|---:|---:|---:|
'''
for q in rows:
    latency_value='—' if not np.isfinite(q['critical_latency_median_ms']) else f"{q['critical_latency_median_ms']:.0f}"
    text+=f"| {q['run']} | {labels[q['variant']]} | {q['missed_critical_s']:.2f} | {q['false_critical_s']:.2f} | {latency_value} | {q['undetected_onsets']}/{q['critical_onsets']} |\n"
text+='''
Nur Run 11-00-43 enthält einen wahren kritischen Abschnitt (1,80 s): alt fehlen 35,2 %, neu 0 %.
**Ein einziger Onset erlaubt keine belastbare Aussage zur Erkennungslatenz.** In den beiden anderen Läufen sind sämtliche kritischen Anzeigen Fehlalarme.

## Abbildung, Grenzen und Reproduktion

![30-s-Ausschnitt um die größte Annäherung, Run 11-00-43](alarm_timeline.png)

PNG visuell geprüft: lesbare Achsen/Legende, weißer Hintergrund, Abstandskurven schwarz/grau,
Statusfarben ausschließlich in den Statusbändern. Die neue Logik schließt die kurzen Ausfälle am kritischen Abschnitt.

- Clearance ist der **Spitzenmittelpunkt**, nicht `min(L,R)` wie im Dashboard (`levels.js:33–41`); keine Aussage zur näheren Einzelspitze.
- CSV-Höhen beruhen auf Augenradius 12,05 mm; keine Neuberechnung mit dem UI-Radius von 12 mm (`config.js:4`).
- Orthographischer Tracker mit Modellfehler/Bias (siehe `writeup/learnings_trajectories.md`, §4.5); Stabilisierung korrigiert diesen Fehler nicht.
- Nur Simulator, drei Läufe einer Übung, keine unabhängige Validierung; Parameter auf denselben Daten hergeleitet und bewertet.
- Kein Browser-Test, keine Änderung an `services/viz`; vorgeschlagenes Grace-/Dwell-Verhalten ist bei Integration gegen die finale JS-Logik abzugleichen.

Reproduktion aus Repo-Wurzel: `.venv-ml/bin/python writeup/figures/viz/alarm_eval/evaluate.py`
(NumPy + Matplotlib; kein Pandas). Enthält sechs deterministische Checks für Schwellen, Reset, Grace, Dwell und sofortige Eskalation.
Artefakte neben diesem Bericht: `jitter.csv`, `metrics.csv` (ungerundete Werte), `replay.csv` (Frame-Level; −1=∅, 0=sicher, 1=Warnung, 2=kritisch), `alarm_timeline.png`.
Bereit zum Testen und zum Abgleich mit der neuen UI-Implementierung.
'''
(OUT/'results.md').write_text(text)
