"""Run tip/junction detection over the video. Usage: run.py [video] [out_prefix] [step]"""
import sys, csv, time
import cv2, numpy as np
from detector.pipeline import Detector, annotate

VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'data/video/forceps_lev1.mp4'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'artifacts/result'
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 1

det = Detector()
cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS)
w = cv2.VideoWriter(OUT + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps / STEP, (1080, 1080))
f = csv.writer(open(OUT + '.csv', 'w'))
hdr = ['frame']
for t in ('forceps', 'shadow'):
    for k in ('left_tip', 'right_tip', 'junction'): hdr += [f'{t}_{k}_x', f'{t}_{k}_y']
f.writerow(hdr)
i = 0; t0 = time.time()
while True:
    ok, im = cap.read()
    if not ok: break
    if i % STEP == 0:
        d = det.detect(im); row = [i]
        for t in ('forceps', 'shadow'):
            a = getattr(d, t)
            for k in ('left_tip', 'right_tip', 'junction'):
                row += [round(getattr(a, k)[0], 1), round(getattr(a, k)[1], 1)] if a.ok else ['', '']
        f.writerow(row); w.write(annotate(im, d))
        if i % 300 == 0: print(i, round(time.time() - t0), 's', flush=True)
    i += 1
w.release()
