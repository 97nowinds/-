import threading
import time

import cv2
import numpy as np


class CamShiftTracker:
    """Short-term color tracker with transferable appearance histogram."""

    def __init__(self, max_failures=10):
        self.max_failures = max(1, int(max_failures))
        self.lock = threading.RLock()
        self.histogram = None
        self.window = None
        self.reference_area = None
        self.reference_aspect = None
        self.rotated_rect = None
        self.target_id = None
        self.status = "idle"
        self.confidence = 0.0
        self.failures = 0
        self.valid = False
        self.updated_at = None
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            12,
            1,
        )

    @staticmethod
    def clamp_box(box, frame_shape):
        height, width = frame_shape[:2]
        x, y, box_width, box_height = (int(round(value)) for value in box)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        box_width = max(0, min(box_width, width - x))
        box_height = max(0, min(box_height, height - y))
        return x, y, box_width, box_height

    @staticmethod
    def hsv_and_mask(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((0, 20, 30), dtype=np.uint8),
            np.array((180, 255, 255), dtype=np.uint8),
        )
        return hsv, mask

    @classmethod
    def create_histogram(cls, frame, box):
        x, y, width, height = cls.clamp_box(box, frame.shape)
        if width < 24 or height < 40:
            raise ValueError("目标框太小，请覆盖人物上半身")
        hsv, mask = cls.hsv_and_mask(frame)
        roi_hsv = hsv[y : y + height, x : x + width]
        roi_mask = mask[y : y + height, x : x + width]
        if cv2.countNonZero(roi_mask) < max(48, int(width * height * 0.025)):
            raise ValueError("目标颜色信息不足，请扩大人物框或改善光照")
        histogram = cv2.calcHist(
            [roi_hsv], [0, 1], roi_mask, [32, 32], [0, 180, 0, 256]
        )
        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        return histogram

    @classmethod
    def back_projection(cls, frame, histogram):
        hsv, mask = cls.hsv_and_mask(frame)
        projection = cv2.calcBackProject(
            [hsv], [0, 1], histogram, [0, 180, 0, 256], 1
        )
        return cv2.bitwise_and(projection, mask)

    def start(self, frame, box, target_id, histogram=None):
        if frame is None or frame.size == 0:
            raise ValueError("摄像头暂时没有画面")
        box = self.clamp_box(box, frame.shape)
        if histogram is None:
            histogram = self.create_histogram(frame, box)
        now = time.time()
        with self.lock:
            self.histogram = histogram.copy()
            self.window = box
            self.reference_area = max(1, box[2] * box[3])
            self.reference_aspect = box[2] / max(box[3], 1)
            self.rotated_rect = None
            self.target_id = str(target_id)
            self.status = "tracking"
            self.confidence = 1.0
            self.failures = 0
            self.valid = True
            self.updated_at = now
        return self.snapshot()

    def stop(self):
        with self.lock:
            self.histogram = None
            self.window = None
            self.reference_area = None
            self.reference_aspect = None
            self.rotated_rect = None
            self.target_id = None
            self.status = "idle"
            self.confidence = 0.0
            self.failures = 0
            self.valid = False
            self.updated_at = time.time()

    def update(self, frame):
        with self.lock:
            if self.status != "tracking" or self.histogram is None:
                return self._snapshot_locked()
            projection = self.back_projection(frame, self.histogram)
            try:
                rotated_rect, next_window = cv2.CamShift(
                    projection, self.window, self.criteria
                )
            except cv2.error:
                return self._fail_locked()

            next_window = self.clamp_box(next_window, frame.shape)
            x, y, width, height = next_window
            area = width * height
            frame_area = frame.shape[0] * frame.shape[1]
            previous_area = max(1, self.window[2] * self.window[3])
            area_to_reference = area / max(1, self.reference_area or area)
            area_to_previous = area / previous_area
            aspect = width / max(height, 1)
            aspect_to_reference = aspect / max(0.01, self.reference_aspect or aspect)
            signal = projection[y : y + height, x : x + width]
            confidence = float(np.mean(signal) / 255.0) if signal.size else 0.0
            if (
                width < 12
                or height < 20
                or area > frame_area * 0.42
                or not 0.30 <= area_to_reference <= 2.6
                or not 0.45 <= area_to_previous <= 1.8
                or not 0.42 <= aspect_to_reference <= 2.4
                or confidence < 0.018
            ):
                return self._fail_locked()

            self.window = next_window
            self.rotated_rect = rotated_rect
            self.confidence = round(confidence, 3)
            self.failures = 0
            self.valid = True
            self.updated_at = time.time()
            return self._snapshot_locked()

    def _fail_locked(self):
        self.failures += 1
        self.confidence = 0.0
        self.valid = False
        self.updated_at = time.time()
        if self.failures >= self.max_failures:
            self.status = "lost"
        return self._snapshot_locked()

    @classmethod
    def find_candidate(cls, frame, histogram, normalized_zone=None):
        projection = cls.back_projection(frame, histogram)
        frame_height, frame_width = frame.shape[:2]
        zone = (0, 0, frame_width, frame_height)
        if normalized_zone:
            zone = cls.clamp_box(
                (
                    normalized_zone[0] * frame_width,
                    normalized_zone[1] * frame_height,
                    normalized_zone[2] * frame_width,
                    normalized_zone[3] * frame_height,
                ),
                frame.shape,
            )
        zx, zy, zw, zh = zone
        zone_projection = projection[zy : zy + zh, zx : zx + zw]
        positive = zone_projection[zone_projection > 0]
        if positive.size < 80:
            return None
        threshold = min(250.0, max(18.0, float(np.percentile(positive, 82))))
        _, binary = cv2.threshold(zone_projection, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame_width * frame_height
        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            x += zx
            y += zy
            area = width * height
            aspect = width / max(height, 1)
            if (
                width < 28
                or height < 48
                or area < frame_area * 0.004
                or area > frame_area * 0.5
                or not 0.18 <= aspect <= 1.6
            ):
                continue
            signal = projection[y : y + height, x : x + width]
            mean_signal = float(np.mean(signal) / 255.0)
            contour_fill = float(cv2.contourArea(contour) / max(area, 1))
            score = mean_signal * 0.75 + min(contour_fill, 1.0) * 0.25
            candidates.append((score, (x, y, width, height)))

        if not candidates:
            return None
        score, box = max(candidates, key=lambda item: item[0])
        if score < 0.08:
            return None
        x, y, width, height = box
        expanded = cls.clamp_box(
            (x - width * 0.12, y - height * 0.18, width * 1.24, height * 1.38),
            frame.shape,
        )
        return {"box": expanded, "score": round(score, 3)}

    def export_histogram(self):
        with self.lock:
            return self.histogram.copy() if self.histogram is not None else None

    def snapshot(self):
        with self.lock:
            return self._snapshot_locked()

    def _snapshot_locked(self):
        return {
            "target_id": self.target_id,
            "status": self.status,
            "box": list(self.window) if self.window is not None else None,
            "confidence": self.confidence,
            "failures": self.failures,
            "valid": self.valid,
            "updated_at": self.updated_at,
        }
