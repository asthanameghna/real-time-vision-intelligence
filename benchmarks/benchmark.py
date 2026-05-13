import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from app.core.events import load_zone_specs
from app.core.pipeline import VisionPipeline
from run_detection import resolve_path

_ROOT = Path(__file__).resolve().parent.parent
with (_ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mp = Path(cfg["model"])
if not mp.is_absolute():
    mp = _ROOT / mp
model = YOLO(str(mp))

zones_path = resolve_path(str(cfg.get("zones_config", "configs/zones.yaml")))
with zones_path.open(encoding="utf-8") as zf:
    zones_data = yaml.safe_load(zf)
zone_specs, line_specs, occ_zone_ids = load_zone_specs(zones_data)
motion_cfg = cfg.get("motion") or {}
if not isinstance(motion_cfg, dict):
    motion_cfg = {}
VisionPipeline(
    model_path=str(cfg["model"]),
    conf_threshold=float(cfg["confidence_threshold"]),
    max_trajectory_points=int(motion_cfg.get("max_trajectory_points", 64)),
    stationary_speed_pps=float(motion_cfg.get("stationary_speed_pps", 25.0)),
    zone_specs=zone_specs,
    line_specs=line_specs,
    occupancy_zone_ids=occ_zone_ids,
)
print("VisionPipeline initialized successfully")

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
