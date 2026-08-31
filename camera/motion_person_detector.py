import threading

import cv2
import numpy as np


class MotionPersonDetector:
    """Fixed-camera motion proposals suitable for seeding a person tracker."""

    def __init__(
        self,
        warmup_frames=20,
        history=300,
        var_threshold=20.0,
        learning_rate=0.002,
        min_area=800,
        min_area_ratio=0.0025,
        max_area_ratio=0.40,
        min_width=25,
        min_height=85,
        min_aspect=0.18,
        max_aspect=0.78,
    ):
        self.warmup_frames = max(1, int(warmup_frames))
        self.history = max(self.warmup_frames * 2, int(history))
        self.var_threshold = float(var_threshold)
        self.learning_rate = float(learning_rate)
        self.min_area = max(1, int(min_area))
        self.min_area_ratio = max(0.0, float(min_area_ratio))
        self.max_area_ratio = float(max_area_ratio)
        self.min_width = max(1, int(min_width))
        self.min_height = max(1, int(min_height))
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)
        self.lock = threading.RLock()
        self.frames_seen = 0
        self._subtractor = self._new_subtractor()

        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11))

    def _new_subtractor(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True,
        )

    @property
    def warmed_up(self):
        return self.frames_seen >= self.warmup_frames

    def reset(self):
        """Forget the learned scene, for example after moving the camera."""
        with self.lock:
            self.frames_seen = 0
            self._subtractor = self._new_subtractor()

    @staticmethod
    def _validate_frame(frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")
        if frame.ndim not in (2, 3):
            raise ValueError("frame must be grayscale or BGR")

    def detect(self, frame):
        """Return motion proposals as ``[(score, (x, y, w, h)), ...]``."""
        self._validate_frame(frame)
        with self.lock:
            learning_rate = -1.0 if not self.warmed_up else self.learning_rate
            foreground = self._subtractor.apply(frame, learningRate=learning_rate)
            self.frames_seen += 1
            if self.frames_seen <= self.warmup_frames:
                return []

            _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_CLOSE, self._close_kernel, iterations=1
            )
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_OPEN, self._open_kernel, iterations=1
            )
            return self._proposals(foreground)

    def update(self, frame):
        """Alias for detect(), convenient for per-frame processing loops."""
        return self.detect(frame)

    def _proposals(self, foreground):
        frame_height, frame_width = foreground.shape[:2]
        frame_area = frame_width * frame_height
        required_area = max(self.min_area, int(round(frame_area * self.min_area_ratio)))

        count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        candidates = []
        for component in range(1, count):
            x = int(stats[component, cv2.CC_STAT_LEFT])
            y = int(stats[component, cv2.CC_STAT_TOP])
            width = int(stats[component, cv2.CC_STAT_WIDTH])
            height = int(stats[component, cv2.CC_STAT_HEIGHT])
            area = int(stats[component, cv2.CC_STAT_AREA])
            box_area = width * height
            aspect = width / max(height, 1)
            touches_horizontal_edge = x <= 2 or x + width >= frame_width - 2

            if (
                width < self.min_width
                or height < self.min_height
                or area < required_area
                or box_area > frame_area * self.max_area_ratio
                or not self.min_aspect <= aspect <= self.max_aspect
                or touches_horizontal_edge
            ):
                continue

            fill_ratio = area / max(box_area, 1)
            if fill_ratio < 0.08:
                continue
            area_score = min(1.0, area / max(frame_area * 0.025, 1.0))
            upright_score = max(0.0, 1.0 - abs(aspect - 0.42) / 0.93)
            score = area_score * 0.55 + min(fill_ratio, 1.0) * 0.30 + upright_score * 0.15
            candidates.append((round(float(score), 4), (x, y, width, height)))

        return sorted(candidates, key=lambda item: item[0], reverse=True)
