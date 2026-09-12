"""Lightweight monocular SLAM-assisted floor localization.

This module is deliberately self-contained and CPU friendly. ArUco markers
provide the metric floor-map anchor; ORB feature matching keeps the image to
map transform alive after the markers leave the view. It is not a replacement
for a calibrated stereo/depth SLAM system, but it is practical for fixed
monocular RTSP cameras and supports camera relocation with a new marker pass.
"""

import threading
import time

import cv2
import numpy as np


class SlamLocalizer:
    def __init__(self, camera_id, config=None):
        config = config or {}
        self.camera_id = camera_id
        self.enabled = bool(config.get("enabled", False))
        self.mode = config.get("mode", "markers")
        self.dictionary_name = config.get("dictionary", "DICT_4X4_50")
        self.min_markers = max(4, int(config.get("min_markers", 4)))
        self.min_orb_matches = max(6, int(config.get("min_orb_matches", 10)))
        self.min_orb_inliers = max(5, int(config.get("min_orb_inliers", 8)))
        self.max_pose_age_seconds = max(0.5, float(config.get("max_pose_age_seconds", 2.0)))
        self.marker_map = {
            int(item["id"]): tuple(float(value) for value in item["map_point"])
            for item in config.get("markers", [])
            if isinstance(item, dict) and "id" in item and len(item.get("map_point", [])) == 2
        }
        self.lock = threading.RLock()
        self.orb = cv2.ORB_create(nfeatures=max(300, int(config.get("orb_features", 800))))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.map_from_reference = None
        self.last_transform = None
        self.last_seen_at = None
        self.last_seen_monotonic = None
        self.last_update_at = None
        self.last_update_monotonic = None
        self.last_inliers = 0
        self.last_matches = 0
        self.marker_count = 0
        self.status = "disabled" if not self.enabled else "awaiting_markers"
        self.error = None

        try:
            dictionary_id = getattr(cv2.aruco, self.dictionary_name)
            self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.detector_parameters = cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "ArucoDetector"):
                self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_parameters)
            else:  # OpenCV 4.6 compatibility
                self.detector = None
        except (AttributeError, cv2.error, TypeError) as exc:
            self.dictionary = None
            self.detector_parameters = None
            self.detector = None
            self.error = f"invalid aruco dictionary: {exc}"
            self.status = "error"

    def reset(self):
        with self.lock:
            self.reference_keypoints = None
            self.reference_descriptors = None
            self.map_from_reference = None
            self.last_transform = None
            self.last_seen_at = None
            self.last_seen_monotonic = None
            self.last_update_at = None
            self.last_update_monotonic = None
            self.last_inliers = 0
            self.last_matches = 0
            self.marker_count = 0
            self.status = "disabled" if not self.enabled else "awaiting_markers"

    def _detect_markers(self, gray):
        if self.dictionary is None:
            return [], []
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.detector_parameters
            )
        if ids is None:
            return [], []
        centers = []
        ids_out = []
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id)
            if marker_id not in self.marker_map:
                continue
            points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            centers.append(points.mean(axis=0))
            ids_out.append(marker_id)
        return ids_out, centers

    def _set_reference(self, gray, map_from_current):
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.min_orb_matches:
            return False
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors
        self.map_from_reference = map_from_current.astype(np.float64)
        self.last_transform = self.map_from_reference.copy()
        return True

    def _calibrate_from_markers(self, gray, marker_ids, centers):
        if len(marker_ids) < self.min_markers:
            return False
        image_points = np.asarray(centers, dtype=np.float32)
        map_points = np.asarray([self.marker_map[item] for item in marker_ids], dtype=np.float32)
        transform, mask = cv2.findHomography(image_points, map_points, cv2.RANSAC, 3.0)
        if transform is None or mask is None or int(mask.sum()) < self.min_markers:
            return False
        return self._set_reference(gray, transform)

    def _track_from_reference(self, gray):
        if self.reference_descriptors is None or self.map_from_reference is None:
            return False
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.min_orb_matches:
            return False
        raw_matches = self.matcher.knnMatch(descriptors, self.reference_descriptors, k=2)
        good = [first for first, second in raw_matches if first.distance < 0.75 * second.distance]
        self.last_matches = len(good)
        if len(good) < self.min_orb_matches:
            return False
        current_points = np.float32([keypoints[item.queryIdx].pt for item in good])
        reference_points = np.float32([self.reference_keypoints[item.trainIdx].pt for item in good])
        current_to_reference, mask = cv2.findHomography(
            current_points, reference_points, cv2.RANSAC, 4.0
        )
        if current_to_reference is None or mask is None:
            return False
        self.last_inliers = int(mask.sum())
        if self.last_inliers < self.min_orb_inliers:
            return False
        self.last_transform = self.map_from_reference @ current_to_reference
        return True

    def update(
        self,
        frame,
        baseline_transform=None,
        *,
        monotonic_time=None,
        unix_time=None,
    ):
        now = time.monotonic() if monotonic_time is None else float(monotonic_time)
        unix_time = time.time() if unix_time is None else float(unix_time)
        with self.lock:
            self.last_update_at = unix_time
            self.last_update_monotonic = now
            if not self.enabled:
                self.status = "disabled"
                return self.status_payload()
            if frame is None or frame.size == 0:
                self.status = "lost"
                return self.status_payload()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_ids, centers = self._detect_markers(gray)
            self.marker_count = len(marker_ids)
            self.last_matches = 0
            if self._calibrate_from_markers(gray, marker_ids, centers):
                self.status = "anchored"
                self.last_seen_at = unix_time
                self.last_seen_monotonic = now
                self.last_inliers = self.marker_count
                return self.status_payload()
            if (
                self.mode == "relative"
                and self.map_from_reference is None
                and baseline_transform is not None
                and self._set_reference(gray, np.asarray(baseline_transform, dtype=np.float64))
            ):
                self.status = "relative_reference"
                self.last_seen_at = unix_time
                self.last_seen_monotonic = now
                return self.status_payload()
            if self._track_from_reference(gray):
                self.status = "tracking"
                self.last_seen_at = unix_time
                self.last_seen_monotonic = now
                return self.status_payload()
            self.status = "lost" if self.map_from_reference is not None else "awaiting_markers"
            return self.status_payload()

    def project_pixel(self, point, monotonic_time=None):
        now = time.monotonic() if monotonic_time is None else float(monotonic_time)
        with self.lock:
            if (
                self.last_transform is None
                or self.last_seen_monotonic is None
                or now - self.last_seen_monotonic > self.max_pose_age_seconds
            ):
                return None
            source = np.asarray([[point]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(source, self.last_transform)[0, 0]
            if not np.isfinite(mapped).all():
                return None
            return {"x": float(mapped[0]), "y": float(mapped[1])}

    def status_payload(self):
        with self.lock:
            return {
                "enabled": self.enabled,
                "engine": "ArUco anchors + ORB monocular SLAM",
                "mode": self.mode,
                "status": self.status,
                "dictionary": self.dictionary_name,
                "configured_markers": len(self.marker_map),
                "visible_markers": self.marker_count,
                "matches": self.last_matches,
                "inliers": self.last_inliers,
                "max_pose_age_seconds": self.max_pose_age_seconds,
                "last_seen_at": self.last_seen_at,
                "timestamp_source": "host_receive",
                "localization_quality": (
                    "high"
                    if self.status == "anchored" and self.marker_count >= self.min_markers
                    else "medium"
                    if self.status == "tracking"
                    else "low"
                ),
                "degraded_reason": (
                    "aruco_marker_map_empty"
                    if self.enabled and not self.marker_map
                    else "pose_unavailable"
                    if self.status in {"awaiting_markers", "lost"}
                    else None
                ),
                "error": self.error,
            }
