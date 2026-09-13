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
        self.projection_domain_margin = float(
            config.get("projection_domain_margin_normalized", 0.03)
        )
        if not 0.0 <= self.projection_domain_margin <= 0.25:
            raise FloorMapError(
                "projection_domain_margin_normalized must be between 0 and 0.25"
            )
        zones = config.get("zones", [])
        zones_by_id = {zone.get("id"): zone for zone in zones if zone.get("id")}
        allowed_zone_ids = config.get("tracking_allowed_zone_ids", [])
        if not isinstance(allowed_zone_ids, list):
            raise FloorMapError("tracking_allowed_zone_ids must be a list")
        missing_zone_ids = [zone_id for zone_id in allowed_zone_ids if zone_id not in zones_by_id]
        if missing_zone_ids:
            raise FloorMapError(
                "tracking_allowed_zone_ids contains unknown zones: "
                + ", ".join(missing_zone_ids)
            )
        self.tracking_allowed_zone_ids = list(dict.fromkeys(allowed_zone_ids))
        self.tracking_zones = [zones_by_id[zone_id] for zone_id in self.tracking_allowed_zone_ids]
        self.transforms = {}
        self.image_domains = {}
        self.tracking_image_regions = {}
        self._projection_status = {}
        self._kalman_filters = {}
        self._kalman_seen_at = {}
        self._filter_positions = {}
        self._position_sources = {}
        self._last_mapped_people = {}
        self._filter_lock = threading.RLock()
        for camera_id, camera in config.get("cameras", {}).items():
            annotation = config.get("annotations", {}).get(camera_id, {})
            self.tracking_image_regions[camera_id] = self._tracking_regions(
                annotation.get("regions", {})
            )
            image_value = camera.get("image_points")
            map_value = camera.get("map_points")
            if not image_value and not map_value:
                self.transforms[camera_id] = None
                self.image_domains[camera_id] = None
                continue
            image_points = self._points(image_value, f"{camera_id}.image_points")
            map_points = self._points(map_value, f"{camera_id}.map_points")
            self.transforms[camera_id] = cv2.getPerspectiveTransform(image_points, map_points)
            self.image_domains[camera_id] = cv2.convexHull(image_points).reshape(-1, 2)

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

    @staticmethod
    def _tracking_regions(regions):
        """Return normalized polygons that are valid places for a person footpoint."""
        polygons = []
        if not isinstance(regions, dict):
            return polygons
        for region_type in ("main_aisle", "secondary_aisle", "rear_service"):
            region = regions.get(region_type)
            if not isinstance(region, dict):
                continue
            points = region.get("points")
            if isinstance(points, list) and len(points) >= 3:
                polygon = np.asarray(points, dtype=np.float32)
            else:
                try:
                    x = float(region["x"])
                    y = float(region["y"])
                    width = float(region["width"])
                    height = float(region["height"])
                except (KeyError, TypeError, ValueError):
                    continue
                polygon = np.asarray(
                    [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                    dtype=np.float32,
                )
            if polygon.ndim == 2 and polygon.shape[1] == 2 and np.isfinite(polygon).all():
                polygons.append(polygon)
        return polygons

    def track_footpoint_allowed(self, camera_id, box, frame_shape):
        """Reject tracks outside both the calibrated image domain and annotated aisles.

        Cameras without either kind of image-space constraint remain unrestricted so
        entrance-camera and older configurations keep their previous behaviour.
        """
        if box is None:
            return False
        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return False
        x, y, width, height = box
        footpoint = (
            float(x + width * 0.5) / float(frame_width),
            float(y + height) / float(frame_height),
        )
        domain = self.image_domains.get(camera_id)
        if domain is not None:
            distance = cv2.pointPolygonTest(domain, footpoint, True)
            if distance < -self.projection_domain_margin:
                return False
        regions = self.tracking_image_regions.get(camera_id, [])
        if regions and not any(
            cv2.pointPolygonTest(region, footpoint, True) >= -self.projection_domain_margin
            for region in regions
        ):
            return False
        return True

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
        signed_distance = cv2.pointPolygonTest(
            self.image_domains[camera_id], (float(foot_x), float(foot_y)), True
        )
        if signed_distance < -self.projection_domain_margin:
            self._projection_status[camera_id] = {
                "quality": "low",
                "reason": "footpoint outside calibrated image domain",
                "inside_domain": False,
                "distance_normalized": round(float(signed_distance), 4),
                "updated_monotonic": time.monotonic(),
            }
            return None
        source = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, transform)[0, 0]
        if not np.isfinite(mapped).all():
            return None
        near_boundary = signed_distance < 0.0
        self._projection_status[camera_id] = {
            "quality": "low" if near_boundary else (
                "high" if self.calibration_status == "formal" else "medium"
            ),
            "reason": "footpoint near calibrated image boundary" if near_boundary else None,
            "inside_domain": not near_boundary,
            "distance_normalized": round(float(signed_distance), 4),
            "updated_monotonic": time.monotonic(),
        }
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
        projection_status = self._projection_status.get(camera_id)
        recent_projection = None
        if projection_status:
            projection_age = time.monotonic() - projection_status["updated_monotonic"]
            if projection_age <= OBSERVATION_MAX_AGE_SECONDS:
                recent_projection = {
                    key: value
                    for key, value in projection_status.items()
                    if key != "updated_monotonic"
                }
                recent_projection["age_seconds"] = round(max(0.0, projection_age), 3)
                if projection_status.get("reason"):
                    reasons.append(projection_status["reason"])
                if projection_status.get("quality") == "low":
                    quality = "low"
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
            "recent_projection": recent_projection,
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
        if self.tracking_zones:
            zones = sorted(
                self.tracking_zones,
                key=lambda zone: float(zone["width"]) * float(zone["height"]),
            )
        else:
            zones = sorted(
                self.config.get("zones", []),
                key=lambda zone: zone.get("kind") == "overlap",
                reverse=True,
            )
        epsilon = 1e-6
        for zone in zones:
            if (
                float(zone["x"]) - epsilon <= x <= float(zone["x"]) + float(zone["width"]) + epsilon
                and float(zone["y"]) - epsilon <= y <= float(zone["y"]) + float(zone["height"]) + epsilon
            ):
                return zone["name"]
        return "实验室"

    def nearest_tracking_position(self, position):
        """Project a map point into the configured union of walkable tracking zones."""
        current = self.clamp_position(position)
        if not self.tracking_zones:
            return current, False
        for zone in self.tracking_zones:
            if (
                float(zone["x"]) <= current["x"] <= float(zone["x"]) + float(zone["width"])
                and float(zone["y"]) <= current["y"] <= float(zone["y"]) + float(zone["height"])
            ):
                return current, False
        candidates = []
        for zone in self.tracking_zones:
            candidates.append(
                {
                    "x": float(np.clip(current["x"], zone["x"], float(zone["x"]) + float(zone["width"]))),
                    "y": float(np.clip(current["y"], zone["y"], float(zone["y"]) + float(zone["height"]))),
                }
            )
        nearest = min(
            candidates,
            key=lambda candidate: np.hypot(
                candidate["x"] - current["x"],
                candidate["y"] - current["y"],
            ),
        )
        return self.clamp_position(nearest), True

    @staticmethod
    def _zone_contains(zone, position, epsilon=1e-6):
        return (
            float(zone["x"]) - epsilon
            <= float(position["x"])
            <= float(zone["x"]) + float(zone["width"]) + epsilon
            and float(zone["y"]) - epsilon
            <= float(position["y"])
            <= float(zone["y"]) + float(zone["height"]) + epsilon
        )

    @staticmethod
    def _zone_portal(first, second):
        left = max(float(first["x"]), float(second["x"]))
        right = min(
            float(first["x"]) + float(first["width"]),
            float(second["x"]) + float(second["width"]),
        )
        top = max(float(first["y"]), float(second["y"]))
        bottom = min(
            float(first["y"]) + float(first["height"]),
            float(second["y"]) + float(second["height"]),
        )
        if right < left or bottom < top:
            return None
        return {"x": (left + right) * 0.5, "y": (top + bottom) * 0.5}

    def _corridor_waypoint(self, previous, target):
        """Route cross-zone motion through connected rectangular corridor portals."""
        if not self.tracking_zones:
            return target, False
        source_indices = [
            index
            for index, zone in enumerate(self.tracking_zones)
            if self._zone_contains(zone, previous)
        ]
        target_indices = [
            index
            for index, zone in enumerate(self.tracking_zones)
            if self._zone_contains(zone, target)
        ]
        if not source_indices or not target_indices or set(source_indices) & set(target_indices):
            return target, False

        adjacency = {index: [] for index in range(len(self.tracking_zones))}
        portals = {}
        for first in range(len(self.tracking_zones)):
            for second in range(first + 1, len(self.tracking_zones)):
                portal = self._zone_portal(
                    self.tracking_zones[first], self.tracking_zones[second]
                )
                if portal is None:
                    continue
                adjacency[first].append(second)
                adjacency[second].append(first)
                portals[(first, second)] = portal
                portals[(second, first)] = portal

        candidates = []
        for source_index in source_indices:
            queue = [(source_index, [source_index])]
            visited = {source_index}
            while queue:
                current, path = queue.pop(0)
                if current in target_indices:
                    waypoints = [
                        portals[(first, second)]
                        for first, second in zip(path, path[1:])
                    ]
                    route = [previous, *waypoints, target]
                    distance = sum(
                        float(
                            np.hypot(
                                second["x"] - first["x"],
                                second["y"] - first["y"],
                            )
                        )
                        for first, second in zip(route, route[1:])
                    )
                    candidates.append((len(path), distance, waypoints))
                    break
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, [*path, neighbor]))
        if not candidates:
            return dict(previous), True
        _, _, waypoints = min(candidates, key=lambda item: (item[0], item[1]))
        return (dict(waypoints[0]) if waypoints else target), True

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

    def nearest_walkable_position(self, position, margin=0.05):
        """Move an approximate point out of solid fixtures by the shortest offset."""
        current, adjusted = self.nearest_tracking_position(position)
        solid_types = {"cabinet", "bench", "fume_hood", "instrument"}
        for _ in range(4):
            containing = None
            for fixture in self.config.get("fixtures", []):
                if fixture.get("type") not in solid_types:
                    continue
                left = max(0.0, float(fixture["x"]) - margin)
                top = max(0.0, float(fixture["y"]) - margin)
                right = min(
                    self.width_m,
                    float(fixture["x"]) + float(fixture["width"]) + margin,
                )
                bottom = min(
                    self.height_m,
                    float(fixture["y"]) + float(fixture["height"]) + margin,
                )
                if left <= current["x"] <= right and top <= current["y"] <= bottom:
                    containing = (left, top, right, bottom)
                    break
            if containing is None:
                break
            left, top, right, bottom = containing
            candidates = [
                {"x": max(0.0, left - 0.01), "y": current["y"]},
                {"x": min(self.width_m, right + 0.01), "y": current["y"]},
                {"x": current["x"], "y": max(0.0, top - 0.01)},
                {"x": current["x"], "y": min(self.height_m, bottom + 0.01)},
            ]
            current = min(
                candidates,
                key=lambda candidate: np.hypot(
                    candidate["x"] - current["x"],
                    candidate["y"] - current["y"],
                ),
            )
            current = self.clamp_position(current)
            adjusted = True
        return current, adjusted

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

    def _smooth_position(
        self,
        key,
        position,
        now,
        max_speed_mps=2.2,
        max_motion_gap_seconds=0.25,
        reset=False,
    ):
        """Smooth a projected foot point and reject physically impossible jumps."""
        with self._filter_lock:
            position, walkability_adjusted = self.nearest_walkable_position(position)
            if reset:
                self._kalman_filters.pop(key, None)
                self._kalman_seen_at.pop(key, None)
                self._filter_positions.pop(key, None)
            kalman = self._kalman_filters.get(key)
            previous_at = self._kalman_seen_at.get(key)
            previous_position = self._filter_positions.get(key)
            requested_position = dict(position)
            corridor_routed = False
            if previous_position is not None:
                position, corridor_routed = self._corridor_waypoint(
                    previous_position, position
                )
            now = float(now)
            if previous_at is not None and now <= previous_at:
                return {
                    **previous_position,
                    "motion_limited": False,
                    "raw_step_m": 0.0,
                    "walkability_adjusted": walkability_adjusted,
                    "corridor_routed": False,
                }
            if kalman is None:
                kalman = self._new_kalman_filter(position)
                self._kalman_filters[key] = kalman
            elif previous_at is not None:
                delta = max(0.01, min(1.0, now - previous_at))
                kalman.transitionMatrix[0, 2] = delta
                kalman.transitionMatrix[1, 3] = delta

            filtered_measurement = dict(position)
            raw_step = 0.0
            motion_limited = False
            if previous_position is not None and previous_at is not None:
                # A held map point remains visible while measurements are absent,
                # but that hidden time must not become a large "movement credit"
                # when another camera resumes the track. Limit the credited gap
                # so the point catches up over successive publications.
                delta_seconds = max(
                    0.01,
                    min(float(max_motion_gap_seconds), now - previous_at),
                )
                dx = float(position["x"]) - previous_position["x"]
                dy = float(position["y"]) - previous_position["y"]
                raw_step = float(
                    np.hypot(
                        requested_position["x"] - previous_position["x"],
                        requested_position["y"] - previous_position["y"],
                    )
                )
                navigation_step = float(np.hypot(dx, dy))
                max_step = max(0.08, float(max_speed_mps) * delta_seconds)
                if navigation_step > max_step:
                    scale = max_step / navigation_step
                    filtered_measurement = {
                        "x": previous_position["x"] + dx * scale,
                        "y": previous_position["y"] + dy * scale,
                    }
                    motion_limited = True
            measurement = np.array(
                [[filtered_measurement["x"]], [filtered_measurement["y"]]],
                dtype=np.float32,
            )
            if previous_at is None:
                corrected = kalman.statePost
            elif corridor_routed:
                corrected = measurement
                kalman.statePost = np.array(
                    [
                        [filtered_measurement["x"]],
                        [filtered_measurement["y"]],
                        [0.0],
                        [0.0],
                    ],
                    dtype=np.float32,
                )
            else:
                kalman.predict()
                corrected = kalman.correct(measurement)
            corrected_position, corrected_adjusted = self.nearest_walkable_position(
                {"x": corrected[0, 0], "y": corrected[1, 0]}
            )
            if previous_position is not None and previous_at is not None:
                delta_seconds = max(
                    0.01,
                    min(float(max_motion_gap_seconds), now - previous_at),
                )
                max_step = max(0.08, float(max_speed_mps) * delta_seconds)
                final_dx = corrected_position["x"] - previous_position["x"]
                final_dy = corrected_position["y"] - previous_position["y"]
                final_step = float(np.hypot(final_dx, final_dy))
                if final_step > max_step:
                    scale = max_step / final_step
                    limited_position, limited_adjusted = self.nearest_walkable_position(
                        {
                            "x": previous_position["x"] + final_dx * scale,
                            "y": previous_position["y"] + final_dy * scale,
                        }
                    )
                    limited_step = float(
                        np.hypot(
                            limited_position["x"] - previous_position["x"],
                            limited_position["y"] - previous_position["y"],
                        )
                    )
                    # Snapping a limited point to another corridor must never
                    # reintroduce the very teleport that the speed gate removed.
                    corrected_position = (
                        dict(previous_position)
                        if limited_step > max_step + 0.005
                        else limited_position
                    )
                    corrected_adjusted = bool(corrected_adjusted or limited_adjusted)
                    motion_limited = True
            result = {
                "x": corrected_position["x"],
                "y": corrected_position["y"],
                "motion_limited": motion_limited,
                "raw_step_m": round(raw_step, 3),
                "walkability_adjusted": bool(
                    walkability_adjusted or corrected_adjusted
                ),
                "corridor_routed": corridor_routed,
            }
            self._kalman_seen_at[key] = now
            self._filter_positions[key] = {"x": result["x"], "y": result["y"]}
            return result

    def _prune_kalman_filters(self, now, retention_seconds=FILTER_RETENTION_SECONDS):
        with self._filter_lock:
            expired = [
                key
                for key, seen_at in self._kalman_seen_at.items()
                if float(now) - seen_at > float(retention_seconds)
            ]
            for key in expired:
                self._kalman_seen_at.pop(key, None)
                self._kalman_filters.pop(key, None)
                self._filter_positions.pop(key, None)
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
            "projection_domain_margin_normalized": self.projection_domain_margin,
            "tracking_allowed_zone_ids": self.tracking_allowed_zone_ids,
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
        max_map_speed_mps=2.2,
        max_motion_gap_seconds=0.25,
        active_tracks=None,
        map_position_hold_seconds=8.0,
    ):
        now = time.time() if now is None else float(now)
        # The continuity filter must live at least as long as the public map
        # hold. Otherwise an ID that resumes during the visible hold interval
        # would be treated as a brand-new point and could teleport.
        self._prune_kalman_filters(
            now,
            retention_seconds=max(
                float(map_position_hold_seconds), FILTER_RETENTION_SECONDS
            ),
        )
        valid = []
        identity_owners = {}
        visual_owners = {}
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
            visual_id = observation.get("track_id")
            if visual_id:
                visual_owners.setdefault(
                    (observation.get("camera_id"), str(visual_id)), set()
                ).add(observation.get("local_id"))

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
            visual_owner_conflict = len(
                visual_owners.get(
                    (observation.get("camera_id"), str(fallback_key)), set()
                )
            ) > 1
            if (winner is not None and person_id is None) or visual_owner_conflict:
                fallback_key = (
                    f"{fallback_key}:{observation.get('camera_id')}:"
                    f"{observation.get('local_id', 'unknown')}"
                )
            if visual_owner_conflict:
                observation["visual_owner_conflict"] = True
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
            geometry_allowed = bool(
                self.geometry_fusion_allowed
                and float(stream_skew_seconds or 0.0) <= float(max_stream_skew_seconds)
            )
            weights = [self._observation_weight(match) for match in matches]
            previous_source = self._position_sources.get(key)
            preferred_indices = [
                index
                for index, match in enumerate(matches)
                if not geometry_allowed and match["camera_id"] == previous_source
            ]
            candidate_indices = preferred_indices or list(range(len(matches)))
            best_index = max(
                candidate_indices,
                key=lambda index: (
                    weights[index],
                    float(matches[index]["observed_at"]),
                    float(matches[index].get("confidence") or 0.0),
                ),
            )
            best = matches[best_index]
            selected = list(matches)
            camera_count = len({match["camera_id"] for match in matches})
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
            source_changed = bool(previous_source and previous_source != source_camera)
            self._position_sources[key] = source_camera
            position = self._smooth_position(
                key,
                {"x": x, "y": y},
                max(float(match["observed_at"]) for match in selected),
                max_speed_mps=max_map_speed_mps,
                max_motion_gap_seconds=max_motion_gap_seconds,
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
                    "position_source_changed": source_changed,
                    "motion_limited": position["motion_limited"],
                    "raw_position_step_m": position["raw_step_m"],
                    "walkability_adjusted": position["walkability_adjusted"],
                    "corridor_routed": position["corridor_routed"],
                    "association_conflict": bool(
                        len(selected) < len(matches)
                        or any(match.get("visual_owner_conflict") for match in matches)
                    ),
                    "observed_at": max(match["observed_at"] for match in matches),
                    "position_estimated": False,
                    "localization_state": "measured",
                    "position_age_seconds": 0.0,
                }
            )

        active_by_track = {}
        for active in active_tracks or []:
            track_id = active.get("track_id")
            if not track_id:
                continue
            entry = active_by_track.setdefault(
                str(track_id), {"cameras": set(), "observed_at": 0.0}
            )
            entry["cameras"].update(active.get("cameras") or [])
            entry["observed_at"] = max(
                entry["observed_at"], float(active.get("observed_at") or now)
            )

        with self._filter_lock:
            measured_keys = set()
            for person in people:
                cache_key = str(person.get("global_track_id") or person["track_id"])
                measured_keys.add(cache_key)
                self._last_mapped_people[cache_key] = {
                    "person": dict(person),
                    "mapped_at": float(person["observed_at"]),
                }
            for cache_key, active in active_by_track.items():
                if cache_key in measured_keys:
                    continue
                cached = self._last_mapped_people.get(cache_key)
                if not cached:
                    continue
                position_age = max(0.0, now - float(cached["mapped_at"]))
                if position_age > float(map_position_hold_seconds):
                    continue
                held = dict(cached["person"])
                held.update(
                    {
                        "cameras": sorted(active["cameras"]),
                        "observed_at": active["observed_at"],
                        "position_observed_at": cached["mapped_at"],
                        "position_age_seconds": round(position_age, 3),
                        "position_estimated": True,
                        "localization_state": "held",
                        "position_source_changed": False,
                        "motion_limited": False,
                        "raw_position_step_m": 0.0,
                    }
                )
                people.append(held)
            expired_cache_keys = [
                cache_key
                for cache_key, cached in self._last_mapped_people.items()
                if now - float(cached["mapped_at"])
                > max(float(map_position_hold_seconds), FILTER_RETENTION_SECONDS)
            ]
            for cache_key in expired_cache_keys:
                self._last_mapped_people.pop(cache_key, None)

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
        public["map_position_hold_seconds"] = float(map_position_hold_seconds)
        public["max_motion_gap_seconds"] = float(max_motion_gap_seconds)
        return public
