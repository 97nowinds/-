"""Bounded face-stage diagnostics. Never retain images or feature vectors."""
from collections import Counter, deque
from statistics import median
import threading
import time
import logging


def warm_face_models(engine):
    """Initialize fixed-shape face kernels before camera threads start."""
    if not engine.available:
        return
    import numpy as np
    for name, model in engine.analysis.models.items():
        if name not in {"detection", "recognition"}:
            continue
        width, height = engine.det_size if name == "detection" else model.input_size
        value = np.zeros((1, 3, height, width), dtype=np.float32)
        model.session.run(None, {model.session.get_inputs()[0].name: value})


def run_face_attempt(worker, frame, box, track_id):
    """Isolate a failed optional face stage without assigning an identity."""
    started = time.perf_counter()
    try:
        candidate = worker.confirm_face_identity(frame, box, track_id)
    except Exception as error:
        worker.face_recognition_error = type(error).__name__ + ": face attempt failed"
        worker.face_attempt_audit.record(
            track_id=track_id, detection_ms=(time.perf_counter()-started)*1000,
            total_ms=(time.perf_counter()-started)*1000,
            result={"known": False, "reason": "inference_error"})
        logging.error("Face attempt failed camera=%s error_type=%s",
                      worker.camera["id"], type(error).__name__)
        return None
    worker.face_recognition_error = None
    return candidate


class FaceAttemptAudit:
    def __init__(self, capacity=30):
        self.lock = threading.RLock()
        self.reasons = Counter()
        self.recent = deque(maxlen=capacity)

    def record(self, *, track_id, detection_ms, total_ms, result=None, face_box=None):
        reason = ("no_face" if result is None else "known_candidate" if result.get("known")
                  else result.get("reason") or "score_or_margin_rejected")
        distance = (result or {}).get("distance")
        item = {
            "track_id": track_id, "reason": reason,
            "detect_embed_ms": round(detection_ms, 3), "total_ms": round(total_ms, 3),
            "face_size_px": list(face_box[2:]) if face_box else None,
            "similarity": (result or {}).get("similarity", None if distance is None else 1-distance),
            "margin": (result or {}).get("margin"),
            "feature_frames": (result or {}).get("feature_frames"),
        }
        with self.lock:
            self.reasons[reason] += 1
            self.recent.append(item)

    def payload(self):
        with self.lock:
            recent = list(self.recent)
            times = [r["total_ms"] for r in recent]
            return {"reason_counts": dict(self.reasons), "recent": recent,
                    "recent_total_median_ms": median(times) if times else None}
