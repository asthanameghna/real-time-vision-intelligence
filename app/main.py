import json
from pathlib import Path

from fastapi import FastAPI

app = FastAPI()

_EVENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "outputs" / "events.jsonl"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def events():
    if not _EVENTS_PATH.is_file():
        return []

    lines = _EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    parsed = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return parsed[-10:]


@app.get("/metrics")
def metrics():
    return {
        "service_name": "real-time-vision-intelligence",
        "processed_frames": 0,
        "fps": 0.0,
        "total_events": 0,
    }
