"""Read bounded AIMS observation summaries for the dashboard."""

import json
import hashlib
import math
import re
from pathlib import Path
import time

CAMERAS = ("cam_1", "cam_2", "cam_entrance")
VISUAL_PREWARNINGS = {"smoke": "environment.visual_smoke", "flame": "environment.visual_flame"}
VISUAL_FLAME_STAGES = {'visual_initial', 'visual_ongoing'}


def read_hazard_feed(path, now=None, *, notifications=False):
    now = time.time() if now is None else now
    try:
        with Path(path).open("rb") as stream:
            buffer = stream.read(32_769)
        if len(buffer) > 32_768:
            raise ValueError("oversized observation")
        payload = json.loads(buffer)
        updated = payload["observed_at_unix"]
        if type(updated) not in {float, int} or not 0 <= now - updated <= 15:
            return {"state": "stale", "cameras": {}}
        cameras = {}
        recent = []
        for name in CAMERAS:
            raw = payload["cameras"].get(name, {})
            fields = ("frames", "smoke_alarms", "flame_alarms", "spark_candidates",
                      "smoke_samples", "inference_errors")
            clean = {key: value for key in fields if type(value := raw.get(key, 0)) is int and value >= 0}
            if raw.get("online", False) is True and len(clean) == len(fields) and clean["frames"] > 0:
                cameras[name] = clean
                for event in raw.get("recent", [])[-32:]:
                    at = event.get("at")
                    kind = event.get("kind")
                    if notifications:
                        source_at = event.get("source_at")
                        if (kind not in VISUAL_PREWARNINGS or type(source_at) not in {float, int}
                                or not math.isfinite(source_at) or source_at <= 0):
                            continue
                    score = event.get("score")
                    if (kind in {"smoke", "flame", "spark"} and type(at) in {int, float}
                            and 0 <= now - at <= 120):
                        recent.append({"kind": kind, "camera": name, "at": at,
                                       "score": score if type(score) in {int, float}
                                       and math.isfinite(score) and 0 <= score <= 1 else None})
                        if kind == 'flame' and event.get('stage') in VISUAL_FLAME_STAGES:
                            recent[-1]['stage'] = event['stage']
                        if kind == 'flame' and event.get('model') == 'qwen':
                            recent[-1]['visual_model'] = 'qwen'
                        if notifications:
                            key = f"{name}/{kind}/{source_at:.6f}"
                            recent[-1]["event_id"] = "visual-hazard:" + hashlib.sha256(key.encode()).hexdigest()
                            evidence_id = event.get('evidence_id')
                            if isinstance(evidence_id, str) and re.fullmatch('[0-9a-f]{64}', evidence_id):
                                recent[-1]['evidence_id'] = evidence_id
        state = "ready" if len(cameras) == len(CAMERAS) else "partial" if cameras else "unavailable"
        recent.sort(key=lambda event: event["at"], reverse=True)
        return {"state": state, "observed_at_unix": updated, "cameras": cameras,
                "recent": recent if notifications else recent[:12]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        return {"state": "unavailable", "cameras": {}}


def forward_visual_prewarnings(service, path, started_at):
    from management_notifications import iso

    now = service.wall()
    feed = read_hazard_feed(path, now, notifications=True)
    oldest = max(started_at, now - service.config.get("event_max_age_seconds", 15))
    for event in reversed(feed.get("recent", ())):
        qwen = event.get('visual_model') == 'qwen'
        if event["at"] < oldest or (not qwen and event["score"] is None):
            continue
        if qwen and (not event.get('evidence_id') or not event.get('stage')):
            continue
        payload = {"event_id": event["event_id"], "event_type": VISUAL_PREWARNINGS[event["kind"]],
                        "lab_id": service.config["lab_id"], "device_id": event["camera"],
                        "camera_id": event["camera"], "observed_at": iso(event["at"]),
                        "evidence_source": "cluster_visual"}
        if qwen:
            payload['visual_model'] = 'qwen'
        else:
            payload['visual_confidence'] = f'{event["score"]:.1%}'
        if event.get('evidence_id'):
            payload['evidence_id'] = event['evidence_id']
        if event.get('stage'):
            payload['fire_stage'] = event['stage']
        service.record(payload)
