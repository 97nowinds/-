"""Freeze the original detection frame without changing the preview pipeline."""
from copy import deepcopy
import math

import cv2


def encode_detection_snapshot(raw_frame, detection_frame, tracks):
    height, width = raw_frame.shape[:2]
    detection_height, detection_width = detection_frame.shape[:2]
    if min(width, height, detection_width, detection_height) <= 0:
        raise ValueError('invalid frame dimensions')
    # CameraWorker.resize is a full-frame resize, with no crop or letterbox.
    sx, sy = width / detection_width, height / detection_height
    restored = deepcopy(tracks)
    for track in restored:
        box = track.get('box', [])
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('invalid detection box')
        x, y, w, h = map(float, box)
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > detection_width + 1 or y + h > detection_height + 1:
            raise ValueError('detection box outside frame')
        right, bottom = min(width, (x + w) * sx), min(height, (y + h) * sy)
        track['box'] = [x * sx, y * sy, right - x * sx, bottom - y * sy]
    ok, encoded = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError('original JPEG encoding failed')
    return encoded.tobytes(), restored
