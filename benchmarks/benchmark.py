import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parent.parent
with (_ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mp = Path(cfg["model"])
if not mp.is_absolute():
    mp = _ROOT / mp
model = YOLO(str(mp))

vp = Path(cfg["input_video"])
if not vp.is_absolute():
    vp = _ROOT / vp
cap = cv2.VideoCapture(str(vp))
n = 0
t0 = time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    model.predict(frame, conf=cfg["confidence_threshold"], verbose=False)
    n += 1
elapsed = time.perf_counter() - t0
cap.release()
fps = n / elapsed if elapsed > 0 else 0.0
print(n)
print(elapsed)
print(fps)
