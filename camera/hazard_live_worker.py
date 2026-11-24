"""Continuous cluster visual prewarning from three existing loopback camera relays."""
import base64
from collections import Counter, deque
import json
import math
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import cv2

from fire_detector import FireDetector
from visual_hazard_observer import spark_boxes

CAMERAS = ("cam_1", "cam_2", "cam_entrance")
THRESHOLDS = {"smoke": .60, "flame": .65}
FLAME_REPEAT_SECONDS = 10.0
FACE_MODEL = 'models/face_detection_yunet_2023mar.onnx'
SPARK = SimpleNamespace(spark_luma=205, spark_delta=60, spark_min_area=2,
                        spark_max_area=120, spark_global_fraction=.08)


class SparkAlertWindow:
    """Count a new appearance after a clear frame, and expire it after three seconds."""

    def __init__(self):
        self.recent = deque(maxlen=3)
        self.was_present = False
        self.last_alert = float("-inf")

    def observe(self, present, now):
        while self.recent and now - self.recent[0] > 3.0:
            self.recent.popleft()
        new_appearance = present and not self.was_present
        self.was_present = present
        if not new_appearance:
            return False, False
        self.recent.append(now)
        alert = len(self.recent) >= 3 and now - self.last_alert >= 1.0
        if alert:
            self.recent.clear()
            self.last_alert = now
        return True, alert


class FlameAlertWindow:
    """Require 3 of 5 detections and periodically report continuing flames."""

    def __init__(self):
        self.history = deque(maxlen=5)
        self.last_alert = None

    def observe(self, detected, now):
        self.history.append(detected)
        if not any(self.history):
            self.last_alert = None
        if not detected or sum(self.history) < 3:
            return None
        if self.last_alert is None:
            self.last_alert = now
            return 'visual_initial'
        if now - self.last_alert >= FLAME_REPEAT_SECONDS:
            self.last_alert = now
            return 'visual_ongoing'
        return None


def face_overlap(box, face):
    x, y, w, h = box
    a, b, fw, fh = face
    overlap = max(0, min(x+w, a+fw)-max(x, a)) * max(0, min(y+h, b+fh)-max(y, b))
    return overlap >= .70 * max(w*h, fw*fh) if min(w, h, fw, fh) > 0 else False


class FlameFaceFilter:
    """Reject a flame candidate whose geometry closely matches a detected face."""

    def __init__(self, model):
        self.detector = cv2.FaceDetectorYN.create(str(model), '', (320, 320), .85, .3, 10)

    def rejects(self, frame, box):
        x, y, w, h = box
        top, left = max(0, math.floor(y-h/2)), max(0, math.floor(x-w/2))
        bottom = min(frame.shape[0], math.ceil(y+1.5*h))
        right = min(frame.shape[1], math.ceil(x+1.5*w))
        crop = frame[top:bottom, left:right]
        if min(crop.shape[:2]) < 32:
            return False
        height, width = crop.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            crop = cv2.resize(crop, (max(1, round(width*scale)), max(1, round(height*scale))))
        sy, sx = height / crop.shape[0], width / crop.shape[1]
        self.detector.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self.detector.detect(crop)
        if faces is None:
            return False
        return any(face_overlap(box, (left+fx*sx, top+fy*sy, fw*sx, fh*sy))
                   for fx, fy, fw, fh in faces[:, :4])


def trigger_snapshot(frame, frame_id, box):
    try:
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    except cv2.error:
        return None
    if not ok:
        return None
    return {'jpeg': base64.b64encode(jpeg).decode('ascii'), 'width': frame.shape[1],
            'height': frame.shape[0], 'frame_id': frame_id, 'bbox': box}


class Camera:
    def __init__(self, name):
        self.name = name
        self.lock = threading.Lock()
        self.latest = None
        self.received = 0.0
        self.generation = 0
        self.counts = Counter()
        self.events = deque(maxlen=32)

    def record(self, kind, score=None, snapshot=None, stage=None):
        self.counts[kind] += 1
        event = {"kind": kind, "camera": self.name, "at": time.time(), "score": score,
                 "sequence": self.counts[kind]}
        if snapshot is not None:
            event["snapshot"] = snapshot
        if stage is not None:
            event['stage'] = stage
        self.events.append(event)

    def capture(self):
        url = f"rtsp://127.0.0.1:8554/{self.name}"
        while True:
            stream = None
            try:
                stream = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
                previous = None
                spark_window = SparkAlertWindow()
                while stream.isOpened():
                    ok, frame = stream.read()
                    if not ok:
                        break
                    now = time.monotonic()
                    # The top band contains the changing camera date/time overlays.
                    top = max(1, round(frame.shape[0] * .13))
                    gray = cv2.cvtColor(frame[top:], cv2.COLOR_BGR2GRAY)
                    if previous is not None and previous.shape != gray.shape:
                        spark_window = SparkAlertWindow()
                    candidate = (previous is not None and previous.shape == gray.shape and
                                 bool(spark_boxes(gray, previous, SPARK, cv2)))
                    appearance, spark = spark_window.observe(candidate, now)
                    previous = gray
                    with self.lock:
                        self.latest = frame
                        self.received = now
                        self.counts["frames"] += 1
                        if appearance:
                            self.counts["spark_occurrences"] += 1
                        if spark:
                            self.record("spark")
            except (cv2.error, OSError):
                pass
            finally:
                if stream is not None:
                    stream.release()
                with self.lock:
                    self.latest = None
                    self.generation += 1
            time.sleep(2)

    def infer(self, detector, face_filter):
        generation = -1
        previous_frame = 0
        flame = FlameAlertWindow()
        smoke_history = deque(maxlen=5)
        smoke_active = False
        while True:
            start = time.monotonic()
            with self.lock:
                if self.generation != generation:
                    generation = self.generation
                    flame = FlameAlertWindow()
                    smoke_history.clear()
                    smoke_active = False
                frame, frame_id = self.latest, self.counts["frames"]
            if frame is not None and frame_id != previous_frame:
                previous_frame = frame_id
                try:
                    scores = {name: 0.0 for name in THRESHOLDS}
                    boxes = {}
                    for item in detector.predict(frame, min(THRESHOLDS.values()))["detections"]:
                        name = item["class_name"].casefold()
                        name = "flame" if name == "fire" else name
                        if name in scores and item["confidence"] > scores[name]:
                            if name == 'flame' and item['confidence'] >= THRESHOLDS['flame'] and face_filter:
                                try:
                                    rejected = face_filter.rejects(frame, item['bbox'])
                                except cv2.error:
                                    with self.lock:
                                        self.counts['flame_face_filter_errors'] += 1
                                else:
                                    if rejected:
                                        with self.lock:
                                            self.counts['flame_face_candidates_rejected'] += 1
                                        continue
                            scores[name] = item["confidence"]
                            boxes[name] = item['bbox']
                    with self.lock:
                        if generation == self.generation:
                            self.counts["smoke_samples"] += 1
                            smoke_history.append(scores['smoke'] >= THRESHOLDS['smoke'])
                            if not smoke_active and smoke_history[-1] and sum(smoke_history) >= 3:
                                self.record('smoke', round(scores['smoke'], 4),
                                            trigger_snapshot(frame, frame_id, boxes['smoke']))
                                smoke_active = True
                            elif not any(smoke_history):
                                smoke_active = False

                            stage = flame.observe(scores['flame'] >= THRESHOLDS['flame'], time.monotonic())
                            if stage:
                                self.record('flame', round(scores['flame'], 4),
                                            trigger_snapshot(frame, frame_id, boxes['flame']), stage)
                except (RuntimeError, ValueError, cv2.error):
                    with self.lock:
                        self.counts["inference_errors"] += 1
            time.sleep(max(.01, .25 - (time.monotonic() - start)))

    def status(self):
        with self.lock:
            for event in self.events:
                if time.time() - event['at'] > 15:
                    event.pop('snapshot', None)
            return {
                "online": self.latest is not None and time.monotonic() - self.received <= 8,
                "frames": self.counts["frames"],
                "smoke_samples": self.counts["smoke_samples"],
                "smoke_alarms": self.counts["smoke"],
                "flame_alarms": self.counts["flame"],
                "spark_candidates": self.counts["spark"],
                "spark_occurrences": self.counts["spark_occurrences"],
                "inference_errors": self.counts["inference_errors"],
                'flame_face_candidates_rejected': self.counts['flame_face_candidates_rejected'],
                'flame_face_filter_errors': self.counts['flame_face_filter_errors'],
                "recent": [dict(event) for event in self.events],
            }


def main():
    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parent
    config = json.loads((root / "config/fire_model.json").read_text(encoding="utf-8"))
    cameras = [Camera(name) for name in CAMERAS]
    detectors = [FireDetector(root / config["path"], config["sha256"], threads=2) for _ in cameras]
    try:
        face_filters = [FlameFaceFilter(root / FACE_MODEL) for _ in cameras]
    except (AttributeError, cv2.error):
        face_filters = [None] * len(cameras)
    for camera, detector, face_filter in zip(cameras, detectors, face_filters):
        threading.Thread(target=camera.capture, daemon=True).start()
        threading.Thread(target=camera.infer, args=(detector, face_filter), daemon=True).start()
    while True:
        data = {"observed_at_unix": time.time(), "flame_snapshot": "trigger_frame",
                'smoke_snapshot': 'trigger_frame',
                'flame_repeat_seconds': FLAME_REPEAT_SECONDS,
                'flame_face_filter': 'active' if all(face_filters) else 'unavailable',
                "spark_rule": {"count": 3, "seconds": 3},
                "cameras": {cam.name: cam.status() for cam in cameras}}
        print(json.dumps(data, separators=(",", ":")), flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
