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


def scale_box(box, from_shape, to_shape):
    """Map an xywh box between full-view images using their actual dimensions."""
    import math
    sx, sy = to_shape[1] / from_shape[1], to_shape[0] / from_shape[0]
    x, y, w, h = box
    left, top = max(0, math.floor(x*sx)), max(0, math.floor(y*sy))
    right = min(to_shape[1], math.ceil((x+w)*sx))
    bottom = min(to_shape[0], math.ceil((y+h)*sy))
    return left, top, max(0, right-left), max(0, bottom-top)


def run_face_attempt(worker, frame, box, track_id, *, source_frame=None, peer_boxes=()):
    """Isolate a failed optional face stage without assigning an identity."""
    started = time.perf_counter()
    try:
        face_frame = source_frame if source_frame is not None else frame
        face_box = scale_box(box, frame.shape, face_frame.shape) if source_frame is not None else box
        worker._face_peer_boxes = [scale_box(peer, frame.shape, face_frame.shape)
                                  if source_frame is not None else peer for peer in peer_boxes]
        candidate = worker.confirm_face_identity(face_frame, face_box, track_id)
        if candidate is not None and source_frame is not None:
            detected_box, result = candidate
            candidate = (scale_box(detected_box, face_frame.shape, frame.shape), result)
    except Exception as error:
        worker.face_recognition_error = type(error).__name__ + ": face attempt failed"
        worker.face_attempt_audit.record(
            track_id=track_id, detection_ms=(time.perf_counter()-started)*1000,
            total_ms=(time.perf_counter()-started)*1000,
            result={"known": False, "reason": "inference_error"})
        logging.error("Face attempt failed camera=%s error_type=%s",
                      worker.camera["id"], type(error).__name__)
        return None
    finally:
        worker._face_peer_boxes = ()
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
            "completed_at": time.time(),
            "track_id": track_id, "reason": reason,
            "person_id": (result or {}).get("person_id") if (result or {}).get("known") else None,
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
