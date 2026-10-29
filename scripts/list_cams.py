"""List the capture devices OpenCV can open (index, resolution). Run with the ML venv: make cams"""
import cv2

found = False
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ok, f = cap.read()
        if ok:
            found = True
            print(f"  index {i}: {f.shape[1]}x{f.shape[0]}")
    cap.release()
if not found:
    print("  no camera found - macOS: allow the camera for your terminal (System Settings > Privacy & Security > Camera)")
else:
    print("use one with:  make ml-cam CAM=<index>")
