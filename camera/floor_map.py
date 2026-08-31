import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class FloorMapError(ValueError):
    pass


OBSERVATION_MAX_AGE_SECONDS = 2.0
FILTER_RETENTION_SECONDS = 4.0


class FloorMapProjector:
    def __init__(self, config):
        self.config = config
        self.width_m = float(config["width_m"])
        self.height_m = float(config["height_m"])
        self.transforms = {}
        self._kalman_filters = {}
        self._kalman_seen_at = {}
        self._filter_lock = threading.RLock()
        for camera_id, camera in config.get("cameras", {}).items():
            image_value = camera.get("image_points")
            map_value = camera.get("map_points")
            if not image_value and not map_value:
                self.transforms[camera_id] = None
                continue
            image_points = self._points(image_value, f"{camera_id}.image_points")
            map_points = self._points(map_value, f"{camera_id}.map_points")
            self.transforms[camera_id] = cv2.getPerspectiveTransform(image_points, map_points)

    @classmethod
    def from_path(cls, path):
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _points(value, name):
        if not isinstance(value, list) or len(value) != 4:
            raise FloorMapError(f"{name} must contain four points")
        points = np.asarray(value, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise FloorMapError(f"{name} contains invalid coordinates")
        return points

    def project(self, camera_id, box, frame_shape):
        transform = self.transforms.get(camera_id)
        if box is None:
            return None
        if transform is None:
            camera = self.config.get("cameras", {}).get(camera_id, {})
            anchor = camera.get("tracking_anchor")
            if (
                isinstance(anchor, list)
                and len(anchor) == 2
                and np.isfinite(np.asarray(anchor, dtype=np.float64)).all()
            ):
                return self.clamp_position({"x": anchor[0], "y": anchor[1]})
            return None
        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return None
        x, y, width, height = box
        foot_x = (x + width * 0.5) / frame_width
        foot_y = (y + height) / frame_height
        source = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, transform)[0, 0]
        if not np.isfinite(mapped).all():
            return None
        return self.clamp_position({"x": float(mapped[0]), "y": float(mapped[1])})

    def pixel_transform(self, camera_id, frame_shape):
        """Return the existing normalized-image transform in pixel coordinates."""
        transform = self.transforms.get(camera_id)
        if transform is None:
            return None
        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return None
        pixel_to_normalized = np.array(
            [[1.0 / frame_width, 0.0, 0.0], [0.0, 1.0 / frame_height, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return transform.astype(np.float64) @ pixel_to_normalized

    def clamp_position(self, position):
        """Keep a SLAM or homography result inside the configured floor map."""
        if position is None:
            return None
        return {
            "x": round(float(np.clip(position["x"], 0.0, self.width_m)), 2),
            "y": round(float(np.clip(position["y"], 0.0, self.height_m)), 2),
        }

    def zone_for(self, x, y):
        zones = sorted(
            self.config.get("zones", []),
            key=lambda zone: zone.get("kind") == "overlap",
            reverse=True,
        )
        for zone in zones:
            if zone["x"] <= x <= zone["x"] + zone["width"] and zone["y"] <= y <= zone["y"] + zone["height"]:
                return zone["name"]
        return "实验室"

    def is_walkable(self, position, margin=0.05):
        """Reject foot points that fall inside solid laboratory fixtures."""
        if position is None:
            return False
        x = float(position["x"])
        y = float(position["y"])
        solid_types = {"cabinet", "bench", "fume_hood", "instrument"}
        for fixture in self.config.get("fixtures", []):
            if fixture.get("type") not in solid_types:
                continue
            left = float(fixture["x"]) - margin
            top = float(fixture["y"]) - margin
            right = float(fixture["x"]) + float(fixture["width"]) + margin
            bottom = float(fixture["y"]) + float(fixture["height"]) + margin
            if left <= x <= right and top <= y <= bottom:
                return False
        return True

    def _new_kalman_filter(self, position):
        """Create a constant-velocity filter for one map track."""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-2
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman.statePost = np.array(
            [[position["x"]], [position["y"]], [0.0], [0.0]], dtype=np.float32
        )
        return kalman

    def _smooth_position(self, key, position, now):
        """Smooth a projected foot point while keeping the map bounded."""
        with self._filter_lock:
            kalman = self._kalman_filters.get(key)
            previous_at = self._kalman_seen_at.get(key)
            if kalman is None:
                kalman = self._new_kalman_filter(position)
                self._kalman_filters[key] = kalman
            elif previous_at is not None:
                delta = max(0.01, min(1.0, float(now) - previous_at))
                kalman.transitionMatrix[0, 2] = delta
                kalman.transitionMatrix[1, 3] = delta

            measurement = np.array(
                [[position["x"]], [position["y"]]], dtype=np.float32
            )
            if previous_at is None:
                corrected = kalman.statePost
            else:
                kalman.predict()
                corrected = kalman.correct(measurement)
            self._kalman_seen_at[key] = float(now)
            return {
                "x": round(float(np.clip(corrected[0, 0], 0.0, self.width_m)), 2),
                "y": round(float(np.clip(corrected[1, 0], 0.0, self.height_m)), 2),
            }

    def _prune_kalman_filters(self, now):
        with self._filter_lock:
            expired = [
                key
                for key, seen_at in self._kalman_seen_at.items()
                if float(now) - seen_at > FILTER_RETENTION_SECONDS
            ]
            for key in expired:
                self._kalman_seen_at.pop(key, None)
                self._kalman_filters.pop(key, None)

    def public_config(self):
        cameras = {}
        for camera_id, camera in self.config.get("cameras", {}).items():
            cameras[camera_id] = {
                key: camera[key]
                for key in ("label", "position", "fov")
                if key in camera
            }
        return {
            "name": self.config.get("name", "实验室平面图"),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "calibrated": bool(self.config.get("calibrated", False)),
            "zones": self.config.get("zones", []),
            "fixtures": self.config.get("fixtures", []),
            "cameras": cameras,
        }

    def state(self, observations, now=None):
        now = time.time() if now is None else float(now)
        self._prune_kalman_filters(now)
        grouped = {}
        for observation in observations:
            if not observation or observation.get("position") is None:
                continue
            try:
                observed_at = float(observation["observed_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if now - observed_at > OBSERVATION_MAX_AGE_SECONDS:
                continue
            person_id = observation.get("person_id")
            key = person_id or observation.get("track_id") or f"unknown:{observation['camera_id']}"
            grouped.setdefault(key, []).append(observation)

        people = []
        for key, matches in grouped.items():
            weights = []
            for match in matches:
                source = match.get("identity_source")
                weights.append(
                    1.0
                    if source == "face"
                    else 0.8
                    if source
                    in {"handoff", "overlap_handoff", "transition_handoff"}
                    else 0.55
                )
            total_weight = sum(weights)
            x = sum(match["position"]["x"] * weight for match, weight in zip(matches, weights)) / total_weight
            y = sum(match["position"]["y"] * weight for match, weight in zip(matches, weights)) / total_weight
            camera_count = len({match["camera_id"] for match in matches})
            if camera_count >= 2 and not self.config.get("calibrated", False):
                overlap = next(
                    (
                        zone
                        for zone in self.config.get("zones", [])
                        if zone.get("kind") == "overlap"
                    ),
                    None,
                )
                if overlap:
                    x = float(np.clip(x, overlap["x"], overlap["x"] + overlap["width"]))
                    y = float(np.clip(y, overlap["y"], overlap["y"] + overlap["height"]))
            position = self._smooth_position(key, {"x": x, "y": y}, now)
            best_index = max(range(len(matches)), key=lambda index: weights[index])
            best = matches[best_index]
            global_track_ids = sorted(
                {
                    str(match.get("global_track_id") or match.get("track_id"))
                    for match in matches
                    if match.get("global_track_id") or match.get("track_id")
                }
            )
            people.append(
                {
                    "track_id": key,
                    "global_track_id": best.get("global_track_id") or best.get("track_id"),
                    "global_track_ids": global_track_ids,
                    "person_id": best.get("person_id"),
                    "person_number": best.get("person_number"),
                    "name": best.get("name") or "未注册",
                    "identified": bool(best.get("person_id")),
                    "identity_source": best.get("identity_source") or "visual",
                    "identity_lock_status": (
                        best.get("identity_lock_status")
                        or ("locked" if best.get("person_id") else "visual")
                    ),
                    "x": position["x"],
                    "y": position["y"],
                    "zone": self.zone_for(position["x"], position["y"]),
                    "cameras": sorted({match["camera_id"] for match in matches}),
                    "observed_at": max(match["observed_at"] for match in matches),
                }
            )

        public = self.public_config()
        public["people"] = people
        return public
