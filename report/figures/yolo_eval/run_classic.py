import json, glob, sys, os, cv2
from detector.config import Settings
from detector.pipeline import Detector
ann_dir, bg, out = sys.argv[1:4]
d = Detector(Settings(background_path=bg))
per = {}
for f in sorted(glob.glob(os.path.join(ann_dir, '*.png'))):
    det = d.detect(cv2.imread(f))
    k = [det.forceps.left_tip, det.forceps.right_tip, det.shadow.left_tip, det.shadow.right_tip] if det.forceps.ok and det.shadow.ok else None
    per[os.path.basename(f)] = None if k is None else {"kpts": [list(map(float, p)) for p in k]}
json.dump({"classical": per}, open(out, 'w'))
print(sum(v is not None for v in per.values()), len(per))
