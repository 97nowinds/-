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
FUSION_MAX_SKEW_SECONDS = 0.6
FUSION_MAX_DISTANCE_METERS = 2.0


class FloorMapProjector:
    def __init__(self, config):
        self.config = config
        self.width_m = float(config["width_m"])
        self.height_m = float(config["height_m"])
        calibration = config.get("calibration", {})
        fallback_status = "formal" if config.get("calibrated", False) else "uncalibrated"
        self.calibration_status = calibration.get("status", fallback_status)
        if self.calibration_status not in {"uncalibrated", "approximate", "formal"}:
            raise FloorMapError(
                "calibration.status must be uncalibrated, approximate, or formal"
            )
        self.calibration_reason = calibration.get("reason")
        self.transforms = {}
        self._kalman_filters = {}
        self._kalman_seen_at = {}
        self._position_sources = {}
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

    def has_projection(self, camera_id):
        """Return whether a camera has a calibrated per-pixel map transform."""
        return self.transforms.get(camera_id) is not None

    @property
    def geometry_fusion_allowed(self):
        return self.calibration_status == "formal"

    def localization_status(self, camera_id, slam_state=None, stream_skew_seconds=0.0, max_stream_skew_seconds=0.5):
        camera = self.config.get("cameras", {}).get(camera_id, {})
        mode = camera.get("localization_mode")
        if mode is None:
            mode = "formal_homography" if self.geometry_fusion_allowed and self.has_projection(camera_id) else "approximate_homography" if self.has_projection(camera_id) else "fixed_anchor" if camera.get("tracking_anchor") else "unavailable"
        reasons = []
        if self.calibration_status != "formal":
            reasons.append(self.calibration_reason or f"map calibration is {self.calibration_status}")
        if not self.has_projection(camera_id):
            reasons.append("no per-pixel floor transform")
        if float(stream_skew_seconds or 0.0) > float(max_stream_skew_seconds):
            reasons.append("stream timestamp skew exceeds fusion limit")
        slam_state = slam_state or {}
        slam_status = slam_state.get("status")
        if slam_state.get("enabled") and slam_status not in {"anchored", "tracking"}:
            reasons.append(f"SLAM {slam_status or 'unavailable'}")
        quality = "high" if not reasons and mode == "formal_homography" else "medium" if self.has_projection(camera_id) else "low"
        return {
            "mode": mode,
            "calibration_status": self.calibration_status,
            "geometry_fusion_allowed": bool(
                self.geometry_fusion_allowed
                and self.has_projection(camera_id)
                and float(stream_skew_seconds or 0.0) <= float(max_stream_skew_seconds)
            ),
            "last_quality": quality,
            "degraded_reason": "; ".join(reasons) if reasons else None,
        }

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

    def _smooth_position(self, key, position, now, reset=False):
        """Smooth a projected foot point while keeping the map bounded."""
        with self._filter_lock:
            if reset:
                self._kalman_filters.pop(key, None)
                self._kalman_seen_at.pop(key, None)
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
                self._position_sources.pop(key, None)

    @staticmethod
    def _observation_weight(observation):
        source = observation.get("identity_source")
        if source == "face":
            return 1.0
        if source in {"handoff", "overlap_handoff", "transition_handoff"}:
            return 0.8
        return 0.55

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
            "calibrated": self.geometry_fusion_allowed,
            "calibration_status": self.calibration_status,
            "geometry_fusion_allowed": self.geometry_fusion_allowed,
            "calibration_reason": self.calibration_reason,
            "zones": self.config.get("zones", []),
            "fixtures": self.config.get("fixtures", []),
            "cameras": cameras,
        }

    def state(
        self,
        observations,
        now=None,
        stream_skew_seconds=0.0,
        max_stream_skew_seconds=0.5,
        observation_max_age_seconds=OBSERVATION_MAX_AGE_SECONDS,
        fusion_max_distance_m=FUSION_MAX_DISTANCE_METERS,
    ):
        now = time.time() if now is None else float(now)
        self._prune_kalman_filters(now)
        valid = []
        identity_owners = {}
        for raw in observations:
            if not raw or raw.get("position") is None:
                continue
            try:
                observed_at = float(raw["observed_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if now - observed_at > float(observation_max_age_seconds):
                continue
            observation = dict(raw)
            valid.append(observation)
            person_id = observation.get("person_id")
            if person_id:
                identity_owners.setdefault(
                    (observation.get("camera_id"), person_id), []
                ).append(observation)

        identity_winners = {
            key: max(
                matches,
                key=lambda match: (
                    self._observation_weight(match),
                    float(match.get("confidence") or 0.0),
                    float(match["observed_at"]),
                ),
            )
            for key, matches in identity_owners.items()
            if len(
                {
                    (match.get("track_id"), match.get("local_id"))
                    for match in matches
                }
            ) > 1
        }

        grouped = {}
        for observation in valid:
            person_id = observation.get("person_id")
            owner_key = (observation.get("camera_id"), person_id)
            winner = identity_winners.get(owner_key)
            if winner is not None and winner is not observation:
                # Defensive containment: even if an upstream detector emits
                # the same registered identity for two simultaneous bodies,
                # only the strongest observation may retain that identity.
                observation.update(
                    {
                        "person_id": None,
                        "person_number": None,
                        "name": None,
                        "identity_source": "visual",
                        "identity_lock_status": "conflict_rejected",
                        "global_track_id": None,
                    }
                )
                person_id = None
            fallback_key = observation.get("track_id") or "unknown"
            if winner is not None and person_id is None:
                fallback_key = (
                    f"{fallback_key}:{observation.get('camera_id')}:"
                    f"{observation.get('local_id', 'unknown')}"
                )
            key = person_id or fallback_key or f"unknown:{observation['camera_id']}"
            grouped.setdefault(key, []).append(observation)

        people = []
        for key, matches in grouped.items():
            freshest_at = max(float(match["observed_at"]) for match in matches)
            matches = [
                match
                for match in matches
                if freshest_at - float(match["observed_at"])
                <= float(max_stream_skew_seconds)
            ]
            weights = [self._observation_weight(match) for match in matches]
            best_index = max(
                range(len(matches)),
                key=lambda index: (
                    weights[index],
                    float(matches[index]["observed_at"]),
                    float(matches[index].get("confidence") or 0.0),
                ),
            )
            best = matches[best_index]
            selected = list(matches)
            camera_count = len({match["camera_id"] for match in matches})
            geometry_allowed = bool(
                self.geometry_fusion_allowed
                and float(stream_skew_seconds or 0.0) <= float(max_stream_skew_seconds)
            )
            if camera_count >= 2:
                if not geometry_allowed:
                    selected = [best]
                else:
                    selected = [
                        match
                        for match in matches
                        if np.hypot(
                            match["position"]["x"] - best["position"]["x"],
                            match["position"]["y"] - best["position"]["y"],
                        )
                        <= float(fusion_max_distance_m)
                    ]
            selected_weights = [self._observation_weight(match) for match in selected]
            total_weight = sum(selected_weights)
            x = sum(
                match["position"]["x"] * weight
                for match, weight in zip(selected, selected_weights)
            ) / total_weight
            y = sum(
                match["position"]["y"] * weight
                for match, weight in zip(selected, selected_weights)
            ) / total_weight
            source_camera = best["camera_id"]
            previous_source = self._position_sources.get(key)
            reset_filter = bool(
                not geometry_allowed
                and previous_source is not None
                and previous_source != source_camera
            )
            self._position_sources[key] = source_camera
            position = self._smooth_position(
                key, {"x": x, "y": y}, now, reset=reset_filter
            )
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
                    "cameras": sorted({match["camera_id"] for match in selected}),
                    "position_source_camera": source_camera,
                    "association_conflict": len(selected) < len(matches),
                    "observed_at": max(match["observed_at"] for match in matches),
                }
            )

        public = self.public_config()
        public["stream_skew_seconds"] = round(float(stream_skew_seconds or 0.0), 4)
        public["geometry_fusion_allowed"] = bool(
            self.geometry_fusion_allowed
            and float(stream_skew_seconds or 0.0) <= float(max_stream_skew_seconds)
        )
        public["geometry_degraded_reason"] = (
            "stream timestamp skew exceeds fusion limit"
            if self.geometry_fusion_allowed
            and float(stream_skew_seconds or 0.0) > float(max_stream_skew_seconds)
            else self.calibration_reason
            if not self.geometry_fusion_allowed
            else None
        )
        public["people"] = people
        return public
