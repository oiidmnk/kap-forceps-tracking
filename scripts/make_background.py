"""Median over frames of a video -> tool-free retina reference. Usage: make_background.py VIDEO OUT.png [step]"""
import sys
import cv2, numpy as np

video, out = sys.argv[1], sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 25
cap, frames, i = cv2.VideoCapture(video), [], 0
while True:                                  # sequential read (seeking mp4 is very slow)
    ok, f = cap.read()
    if not ok: break
    if i % step == 0: frames.append(f)
    i += 1
cv2.imwrite(out, np.median(np.stack(frames), 0).astype(np.uint8))
print(f"{len(frames)} frames -> {out}")
