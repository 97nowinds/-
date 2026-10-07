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
        self.footpoint_inference_max_distance = float(
            config.get("footpoint_inference_max_distance_normalized", 0.18)
        )
        if not 0.0 <= self.footpoint_inference_max_distance <= 0.5:
            raise FloorMapError(
                "footpoint_inference_max_distance_normalized must be between 0 and 0.5"
            )
        self.zone_switch_confirmation_observations = int(
            config.get("zone_switch_confirmation_observations", 3)
        )
        if not 1 <= self.zone_switch_confirmation_observations <= 30:
            raise FloorMapError(
                "zone_switch_confirmation_observations must be between 1 and 30"
            )
        self.camera_projection_domain_margins = {}
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
        portal_values = config.get("corridor_portals", [])
        if not isinstance(portal_values, list):
            raise FloorMapError("corridor_portals must be a list")
        self.corridor_portals = {}
        self._zone_adjacency = {}
        for portal in portal_values:
            if not isinstance(portal, dict):
                raise FloorMapError("each corridor portal must be an object")
            portal_id = portal.get("zone_id")
            first = portal.get("from_zone_id")
            second = portal.get("to_zone_id")
            if not all(isinstance(value, str) and value for value in (portal_id, first, second)):
                raise FloorMapError("corridor portal needs zone_id/from_zone_id/to_zone_id")
            if portal_id not in zones_by_id or first not in zones_by_id or second not in zones_by_id:
                raise FloorMapError("corridor portal references an unknown zone")
            if portal_id not in self.tracking_allowed_zone_ids:
                raise FloorMapError("corridor portal zone_id must be a tracking_allowed_zone_id")
            if first == second or portal_id in {first, second}:
                raise FloorMapError("corridor portal must connect three distinct zone ids")
            self.corridor_portals[portal_id] = {first, second}
            self._zone_adjacency.setdefault(portal_id, set()).update((first, second))
            self._zone_adjacency.setdefault(first, set()).add(portal_id)
            self._zone_adjacency.setdefault(second, set()).add(portal_id)
        self.transforms = {}
        self.image_domains = {}
        self.tracking_image_regions = {}
        self._projection_status = {}
        self._kalman_filters = {}
        self._kalman_seen_at = {}
        self._filter_positions = {}
        self._filter_velocities = {}
        self._position_sources = {}
        # Per-global-track display ownership while an observation moves from
        # one camera to another.  This intentionally affects *only* map
        # presentation: association remains the coordinator's one-to-one
        # decision and is never weakened by a visual smoothing decision.
        self._handoff_switch_states = {}
        self._zone_switch_states = {}
        self._last_mapped_people = {}
        self._filter_lock = threading.RLock()
        for camera_id, camera in config.get("cameras", {}).items():
            camera_margin = float(
                camera.get(
                    "projection_domain_margin_normalized",
                    self.projection_domain_margin,
                )
            )
            if not 0.0 <= camera_margin <= 0.25:
                raise FloorMapError(
                    f"{camera_id}.projection_domain_margin_normalized must be between 0 and 0.25"
                )
            self.camera_projection_domain_margins[camera_id] = camera_margin
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

    def reset_replay_cycle(self):
        """Clear measured, predicted, and held positions from the last loop."""
        with self._filter_lock:
            for state in (
                self._kalman_filters, self._kalman_seen_at,
                self._filter_positions, self._filter_velocities,
                self._position_sources, self._handoff_switch_states,
                self._zone_switch_states, self._last_mapped_people,
            ):
                state.clear()

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
        margin = self.camera_projection_domain_margins.get(
            camera_id, self.projection_domain_margin
        )
        if domain is not None:
            distance = cv2.pointPolygonTest(domain, footpoint, True)
            if distance < -margin:
                return False
        regions = self.tracking_image_regions.get(camera_id, [])
        if regions and not any(
            cv2.pointPolygonTest(region, footpoint, True) >= -margin
            for region in regions
        ):
            return False
        return True

    def _localization_polygons(self, camera_id):
        """Return image-space polygons where a visible foot can be calibrated."""
        domain = self.image_domains.get(camera_id)
        regions = self.tracking_image_regions.get(camera_id, [])
        if domain is None:
            return [np.asarray(region, dtype=np.float32) for region in regions]
        if not regions:
            return [np.asarray(domain, dtype=np.float32)]
        intersections = []
        domain_hull = cv2.convexHull(np.asarray(domain, dtype=np.float32)).reshape(-1, 2)
        for region in regions:
            region_hull = cv2.convexHull(np.asarray(region, dtype=np.float32)).reshape(-1, 2)
            area, polygon = cv2.intersectConvexConvex(domain_hull, region_hull)
            if area > 1e-8 and polygon is not None and len(polygon) >= 3:
                intersections.append(polygon.reshape(-1, 2))
        return intersections or [domain_hull]

    @staticmethod
    def _nearest_point_on_polygon(point, polygon):
        point = np.asarray(point, dtype=np.float64)
        polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if cv2.pointPolygonTest(
            polygon.astype(np.float32), tuple(point.astype(float)), False
        ) >= 0:
            return point, 0.0
        nearest = None
        nearest_distance = float("inf")
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            segment = end - start
            denominator = float(np.dot(segment, segment))
            ratio = 0.0 if denominator <= 1e-12 else float(
                np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0)
            )
            candidate = start + ratio * segment
            distance = float(np.linalg.norm(candidate - point))
            if distance < nearest_distance:
                nearest = candidate
                nearest_distance = distance
        return nearest, nearest_distance

    @staticmethod
    def _box_intersects_polygon(box, frame_shape, polygon):
        frame_height, frame_width = frame_shape[:2]
        x, y, width, height = box
        left = float(np.clip(x / frame_width, 0.0, 1.0))
        top = float(np.clip(y / frame_height, 0.0, 1.0))
        right = float(np.clip((x + width) / frame_width, 0.0, 1.0))
        bottom = float(np.clip((y + height) / frame_height, 0.0, 1.0))
        if right <= left or bottom <= top:
            return False
        rectangle = np.asarray(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )
        area, _ = cv2.intersectConvexConvex(
            rectangle, cv2.convexHull(np.asarray(polygon, dtype=np.float32)).reshape(-1, 2)
        )
        return area > 1e-8

    def project_track(self, camera_id, box, frame_shape):
        """Project a track without making foot visibility a detection gate.

        A normal visible foot uses the existing homography.  When the detector
        box is clipped or occluded but overlaps a calibrated aisle, the anchor
        is moved to the nearest valid aisle point.  The caller receives an
        explicit low-confidence estimate instead of losing the person track.
        """
        if box is None:
            return {
                "position": None,
                "estimated": False,
                "quality": "unavailable",
                "reason": "missing person box",
                "image_anchor": None,
                "footpoint_visible": False,
            }
        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return {
                "position": None,
                "estimated": False,
                "quality": "unavailable",
                "reason": "invalid frame dimensions",
                "image_anchor": None,
                "footpoint_visible": False,
            }
        x, y, width, height = box
        raw_foot = np.asarray(
            [(x + width * 0.5) / frame_width, (y + height) / frame_height],
            dtype=np.float64,
        )
        footpoint_visible = bool(raw_foot[1] < 0.985)
        polygons = self._localization_polygons(camera_id)
        margin = self.camera_projection_domain_margins.get(
            camera_id, self.projection_domain_margin
        )
        inside_localization_region = not polygons or any(
            cv2.pointPolygonTest(
                np.asarray(polygon, dtype=np.float32),
                tuple(raw_foot.astype(float)),
                True,
            )
            >= -margin
            for polygon in polygons
        )
        position = self.project(camera_id, box, frame_shape)
        recent = self._projection_status.get(camera_id, {})
        if position is not None and inside_localization_region:
            estimated = not bool(recent.get("inside_domain", True))
            return {
                "position": position,
                "estimated": estimated,
                "quality": "low" if estimated else recent.get("quality", "medium"),
                "reason": recent.get("reason"),
                "image_anchor": {"x": float(raw_foot[0]), "y": float(raw_foot[1])},
                "footpoint_visible": footpoint_visible,
            }

        transform = self.transforms.get(camera_id)
        if transform is None or not polygons:
            # Preserve fixed anchors and cameras without a per-pixel map.
            fallback = self.project(camera_id, box, frame_shape)
            return {
                "position": fallback,
                "estimated": fallback is not None,
                "quality": "low" if fallback is not None else "unavailable",
                "reason": "fixed camera anchor" if fallback is not None else "no floor transform",
                "image_anchor": None,
                "footpoint_visible": footpoint_visible,
            }

        candidates = []
        for polygon in polygons:
            anchor, distance = self._nearest_point_on_polygon(raw_foot, polygon)
            if anchor is not None:
                candidates.append((distance, anchor, polygon))
        if not candidates:
            return {
                "position": None,
                "estimated": False,
                "quality": "unavailable",
                "reason": "no valid aisle anchor",
                "image_anchor": None,
                "footpoint_visible": footpoint_visible,
            }
        distance, anchor, polygon = min(candidates, key=lambda item: item[0])
        overlaps_aisle = self._box_intersects_polygon(box, frame_shape, polygon)
        if distance > self.footpoint_inference_max_distance or not overlaps_aisle:
            return {
                "position": None,
                "estimated": False,
                "quality": "unavailable",
                "reason": "person box does not overlap a calibrated aisle",
                "image_anchor": None,
                "footpoint_visible": footpoint_visible,
            }
        # Move a tiny amount toward the polygon centre so floating point noise
        # cannot leave the inferred anchor just outside the valid domain.
        centroid = np.mean(np.asarray(polygon, dtype=np.float64), axis=0)
        anchor = anchor * 0.999 + centroid * 0.001
        source = np.asarray([[[anchor[0], anchor[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, transform)[0, 0]
        if not np.isfinite(mapped).all():
            return {
                "position": None,
                "estimated": False,
                "quality": "unavailable",
                "reason": "inferred homography result is invalid",
                "image_anchor": None,
                "footpoint_visible": footpoint_visible,
            }
        self._projection_status[camera_id] = {
            "quality": "low",
            "reason": "footpoint inferred from overlapping calibrated aisle",
            "inside_domain": True,
            "distance_normalized": round(float(distance), 4),
            "estimated": True,
            "updated_monotonic": time.monotonic(),
        }
        return {
            "position": self.clamp_position({"x": float(mapped[0]), "y": float(mapped[1])}),
            "estimated": True,
            "quality": "low",
            "reason": "footpoint inferred from overlapping calibrated aisle",
            "image_anchor": {"x": float(anchor[0]), "y": float(anchor[1])},
            "footpoint_visible": footpoint_visible,
        }

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
        margin = self.camera_projection_domain_margins.get(
            camera_id, self.projection_domain_margin
        )
        if signed_distance < -margin:
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

    def overlap_zone_for(self, x, y):
        """Return the camera-overlap overlay without changing corridor topology."""
        epsilon = 1e-6
        for zone in self.config.get("zones", []):
            if zone.get("kind") != "overlap":
                continue
            if (
                float(zone["x"]) - epsilon <= x <= float(zone["x"]) + float(zone["width"]) + epsilon
                and float(zone["y"]) - epsilon <= y <= float(zone["y"]) + float(zone["height"]) + epsilon
            ):
                return zone
        return None

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

    def _tracking_zone_candidate(self, position, preferred_zone_id=None):
        """Choose a walkable zone while preserving an overlapping current zone."""
        if not self.tracking_zones:
            return None
        containing = [
            zone for zone in self.tracking_zones if self._zone_contains(zone, position)
        ]
        if preferred_zone_id and any(
            zone.get("id") == preferred_zone_id for zone in containing
        ):
            return preferred_zone_id
        if containing:
            return containing[0].get("id")
        candidates = []
        for zone in self.tracking_zones:
            nearest = {
                "x": float(
                    np.clip(
                        position["x"],
                        float(zone["x"]),
                        float(zone["x"]) + float(zone["width"]),
                    )
                ),
                "y": float(
                    np.clip(
                        position["y"],
                        float(zone["y"]),
                        float(zone["y"]) + float(zone["height"]),
                    )
                ),
            }
            candidates.append(
                (
                    float(
                        np.hypot(
                            nearest["x"] - float(position["x"]),
                            nearest["y"] - float(position["y"]),
                        )
                    ),
                    zone.get("id"),
                )
            )
        return min(candidates, key=lambda item: item[0])[1]

    def _zone_name(self, zone_id):
        zone = next(
            (item for item in self.tracking_zones if item.get("id") == zone_id),
            None,
        )
        return zone.get("name") if zone else None

    def _zone_transition_allowed(self, source_zone_id, target_zone_id):
        """When portal topology is configured, forbid corridor shortcuts."""
        if not self.corridor_portals or source_zone_id == target_zone_id:
            return True
        return target_zone_id in self._zone_adjacency.get(source_zone_id, set())

    def _zone_confirmation_observations(self, zone_id):
        zone = next(
            (item for item in self.tracking_zones if item.get("id") == zone_id),
            {},
        )
        value = zone.get(
            "confirmation_observations", self.zone_switch_confirmation_observations
        )
        try:
            value = int(value)
        except (TypeError, ValueError) as error:
            raise FloorMapError("zone confirmation_observations must be an integer") from error
        if not 1 <= value <= self.zone_switch_confirmation_observations:
            raise FloorMapError(
                "zone confirmation_observations must be between 1 and "
                "zone_switch_confirmation_observations"
            )
        return value

    def _position_in_tracking_zone(self, position, zone_id):
        zone = next(
            (zone for zone in self.tracking_zones if zone.get("id") == zone_id),
            None,
        )
        if zone is None:
            return dict(position)
        return {
            "x": float(
                np.clip(
                    position["x"],
                    float(zone["x"]),
                    float(zone["x"]) + float(zone["width"]),
                )
            ),
            "y": float(
                np.clip(
                    position["y"],
                    float(zone["y"]),
                    float(zone["y"]) + float(zone["height"]),
                )
            ),
        }

    def _apply_zone_switch_hysteresis(self, key, position):
        """Require repeated evidence before moving one track to another corridor."""
        if not self.tracking_zones:
            return dict(position), False, None, None, 0
        state = self._zone_switch_states.get(key)
        current_zone_id = state.get("zone_id") if state else None
        candidate_zone_id = self._tracking_zone_candidate(
            position, preferred_zone_id=current_zone_id
        )
        if current_zone_id is None:
            self._zone_switch_states[key] = {
                "zone_id": candidate_zone_id,
                "pending_zone_id": None,
                "pending_observations": 0,
            }
            return dict(position), False, candidate_zone_id, None, 0
        if candidate_zone_id == current_zone_id:
            state.update(pending_zone_id=None, pending_observations=0)
            return dict(position), False, current_zone_id, None, 0
        if not self._zone_transition_allowed(current_zone_id, candidate_zone_id):
            # A homography outlier must never cut directly through the central
            # bench from one corridor to another.  The track must first be
            # observed in the explicitly configured connecting portal.
            state.update(pending_zone_id=None, pending_observations=0)
            held = self._position_in_tracking_zone(position, current_zone_id)
            return held, True, current_zone_id, candidate_zone_id, 0
        if state.get("pending_zone_id") != candidate_zone_id:
            state["pending_zone_id"] = candidate_zone_id
            state["pending_observations"] = 1
        else:
            state["pending_observations"] = int(state["pending_observations"]) + 1
        if (
            state["pending_observations"]
            >= self._zone_confirmation_observations(candidate_zone_id)
        ):
            state.update(
                zone_id=candidate_zone_id,
                pending_zone_id=None,
                pending_observations=0,
            )
            return dict(position), False, candidate_zone_id, None, 0
        held = self._position_in_tracking_zone(position, current_zone_id)
        return (
            held,
            True,
            current_zone_id,
            candidate_zone_id,
            int(state["pending_observations"]),
        )

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
            if reset:
                self._kalman_filters.pop(key, None)
                self._kalman_seen_at.pop(key, None)
                self._filter_positions.pop(key, None)
                self._filter_velocities.pop(key, None)
                self._zone_switch_states.pop(key, None)
            now = float(now)
            previous_at = self._kalman_seen_at.get(key)
            previous_position = self._filter_positions.get(key)
            if (
                previous_at is not None
                and previous_position is not None
                and now <= previous_at
            ):
                zone_state = self._zone_switch_states.get(key, {})
                return {
                    **previous_position,
                    "motion_limited": False,
                    "raw_step_m": 0.0,
                    "walkability_adjusted": False,
                    "corridor_routed": False,
                    "zone_switch_held": bool(zone_state.get("pending_zone_id")),
                    "tracking_zone_id": zone_state.get("zone_id"),
                    "pending_zone_id": zone_state.get("pending_zone_id"),
                    "pending_zone_observations": int(
                        zone_state.get("pending_observations", 0)
                    ),
                }
            (
                position,
                zone_switch_held,
                tracking_zone_id,
                pending_zone_id,
                pending_zone_observations,
            ) = self._apply_zone_switch_hysteresis(key, position)
            position, walkability_adjusted = self.nearest_walkable_position(position)
            kalman = self._kalman_filters.get(key)
            requested_position = dict(position)
            corridor_routed = False
            if previous_position is not None:
                position, corridor_routed = self._corridor_waypoint(
                    previous_position, position
                )
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
                "zone_switch_held": zone_switch_held,
                "tracking_zone_id": tracking_zone_id,
                "pending_zone_id": pending_zone_id,
                "pending_zone_observations": pending_zone_observations,
            }
            # Keep a bounded, observed velocity for short display-only
            # projection gaps.  It is deliberately derived from the emitted
            # (already speed-gated) map positions, so a bad homography sample
            # cannot turn a held person into a long-distance prediction.
            if previous_position is not None and previous_at is not None:
                velocity_dt = max(0.01, min(1.0, now - previous_at))
                vx = (result["x"] - previous_position["x"]) / velocity_dt
                vy = (result["y"] - previous_position["y"]) / velocity_dt
                speed = float(np.hypot(vx, vy))
                if speed > float(max_speed_mps):
                    scale = float(max_speed_mps) / speed
                    vx *= scale
                    vy *= scale
                previous_velocity = self._filter_velocities.get(key)
                if previous_velocity is not None:
                    # Damp one-frame detector jitter without coupling tracks.
                    vx = previous_velocity["x"] * 0.35 + vx * 0.65
                    vy = previous_velocity["y"] * 0.35 + vy * 0.65
                self._filter_velocities[key] = {"x": vx, "y": vy}
            else:
                self._filter_velocities[key] = {"x": 0.0, "y": 0.0}
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
                self._filter_velocities.pop(key, None)
                self._position_sources.pop(key, None)
                self._zone_switch_states.pop(key, None)

    @staticmethod
    def _observation_weight(observation):
        source = observation.get("identity_source")
        if source == "face":
            weight = 1.0
        elif source in {"handoff", "overlap_handoff", "transition_handoff"}:
            weight = 0.8
        else:
            weight = 0.55
        if observation.get("position_estimated"):
            weight *= 0.35
        return weight

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
            "footpoint_inference_max_distance_normalized": self.footpoint_inference_max_distance,
            "zone_switch_confirmation_observations": self.zone_switch_confirmation_observations,
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
        map_position_predict_seconds=1.2,
        map_position_predict_max_distance_m=0.65,
        # Preserve the historical direct-source behavior for callers that do
        # not opt in.  The application supplies its configured value (2).
        handoff_target_confirmations=1,
        handoff_transition_seconds=0.6,
        handoff_wait_seconds=0.8,
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
        with self._filter_lock:
            self._handoff_switch_states = {
                key: value
                for key, value in self._handoff_switch_states.items()
                if now - float(value.get("updated_at", now)) <= 10.0
            }
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
            observed_source_camera = best["camera_id"]
            source_camera = observed_source_camera
            source_changed = bool(
                previous_source and previous_source != observed_source_camera
            )
            handoff_state = "stable"
            handoff_waiting = False
            handoff_progress = 1.0
            position_discontinuity = False
            previous_person = self._last_mapped_people.get(
                str(best.get("global_track_id") or key), {}
            ).get("person")
            if source_changed and int(handoff_target_confirmations) > 1:
                switch = self._handoff_switch_states.get(key)
                if (
                    not switch
                    or switch.get("from_camera") != previous_source
                    or switch.get("to_camera") != observed_source_camera
                ):
                    switch = {
                        "from_camera": previous_source,
                        "to_camera": observed_source_camera,
                        "confirmations": 1,
                        "started_at": now,
                        "last_observed_at": float(best["observed_at"]),
                    }
                else:
                    # Dashboard polling may reuse the same detection. Count
                    # distinct target-camera measurements, not state calls.
                    observed_at = float(best["observed_at"])
                    if observed_at > float(switch["last_observed_at"]) + 1e-4:
                        switch["confirmations"] = int(switch["confirmations"]) + 1
                        switch["last_observed_at"] = observed_at
                switch["updated_at"] = now
                self._handoff_switch_states[key] = switch
                confirmations = int(switch["confirmations"])
                if confirmations < int(handoff_target_confirmations) and previous_person:
                    # The incoming camera has not yet shown a stable tracklet.
                    # Keep the last source coordinate briefly instead of
                    # drawing a dot that visibly bounces across the corridor.
                    source_camera = previous_source
                    source_changed = False
                    handoff_state = "waiting_target_confirmation"
                    handoff_waiting = True
                    handoff_progress = min(
                        1.0,
                        max(0.0, now - float(switch["started_at"]))
                        / float(handoff_wait_seconds),
                    )
                else:
                    handoff_state = "transitioning"
                    handoff_progress = min(
                        1.0,
                        max(0.0, now - float(switch["started_at"]))
                        / float(handoff_transition_seconds),
                    )
            else:
                self._handoff_switch_states.pop(key, None)

            self._position_sources[key] = source_camera
            raw_position = {"x": x, "y": y}
            position_observed_at = max(
                float(match["observed_at"]) for match in selected
            )
            if handoff_waiting and previous_person:
                raw_position = {
                    "x": float(previous_person["x"]),
                    "y": float(previous_person["y"]),
                }
                position_observed_at = float(previous_person.get(
                    "position_observed_at", previous_person["observed_at"]
                ))
            elif handoff_state == "transitioning" and previous_person:
                cross_camera_gap = float(np.hypot(
                    x - float(previous_person["x"]),
                    y - float(previous_person["y"]),
                ))
                if not geometry_allowed and cross_camera_gap > 1.5:
                    # Approximate camera maps can disagree by several metres.
                    # Continuing the old filter fabricates a walk through the
                    # corridor. Keep identity, but start a new position segment.
                    position_discontinuity = True
                    handoff_state = "relocalized"
                else:
                    raw_position = {
                        "x": float(previous_person["x"])
                        + (x - float(previous_person["x"])) * handoff_progress,
                        "y": float(previous_person["y"])
                        + (y - float(previous_person["y"])) * handoff_progress,
                    }
            position = self._smooth_position(
                key,
                raw_position,
                position_observed_at,
                max_speed_mps=max_map_speed_mps,
                max_motion_gap_seconds=max_motion_gap_seconds,
                reset=position_discontinuity,
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
                    "counted": bool(
                        best.get("person_id")
                        or any(match.get("counted", True) for match in matches)
                    ),
                    "count_status": (
                        "identified"
                        if best.get("person_id")
                        else "confirmed"
                        if any(match.get("counted", True) for match in matches)
                        else best.get("count_status") or "provisional"
                    ),
                    "count_duplicate_of": best.get("count_duplicate_of"),
                    "count_duplicate_reid_score": best.get(
                        "count_duplicate_reid_score"
                    ),
                    "identity_source": best.get("identity_source") or "visual",
                    "identity_lock_status": (
                        best.get("identity_lock_status")
                        or ("locked" if best.get("person_id") else "visual")
                    ),
                    "x": position["x"],
                    "y": position["y"],
                    "zone": (
                        self._zone_name(position["tracking_zone_id"])
                        if self.corridor_portals
                        else None
                    )
                    or self.zone_for(position["x"], position["y"]),
                    "cameras": sorted({match["camera_id"] for match in selected}),
                    "position_source_camera": source_camera,
                    "position_source_changed": source_changed,
                    "position_discontinuity": position_discontinuity,
                    "handoff_state": handoff_state,
                    "handoff_waiting": handoff_waiting,
                    "map_visible": not handoff_waiting,
                    "handoff_target_camera": (
                        observed_source_camera if handoff_state != "stable" else None
                    ),
                    "handoff_progress": round(handoff_progress, 3),
                    "motion_limited": position["motion_limited"],
                    "raw_position_step_m": position["raw_step_m"],
                    "walkability_adjusted": position["walkability_adjusted"],
                    "corridor_routed": position["corridor_routed"],
                    "zone_switch_held": position["zone_switch_held"],
                    "tracking_zone_id": position["tracking_zone_id"],
                    "pending_zone_id": position["pending_zone_id"],
                    "pending_zone_observations": position[
                        "pending_zone_observations"
                    ],
                    "association_conflict": bool(
                        len(selected) < len(matches)
                        or any(match.get("visual_owner_conflict") for match in matches)
                    ),
                    "observed_at": max(match["observed_at"] for match in matches),
                    "position_observed_at": position_observed_at,
                    "position_estimated": bool(
                        handoff_waiting or best.get("position_estimated")
                    ),
                    "localization_state": (
                        "held" if handoff_waiting else
                        "estimated" if best.get("position_estimated") else "measured"
                    ),
                    "localization_quality": best.get("localization_quality"),
                    "localization_reason": best.get("localization_reason"),
                    "footpoint_visible": best.get("footpoint_visible"),
                    "position_age_seconds": (
                        round(max(0.0, now - position_observed_at), 3)
                        if handoff_waiting else 0.0
                    ),
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
                    "mapped_at": float(person["position_observed_at"]),
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
                predicted = False
                if (
                    position_age <= float(map_position_predict_seconds)
                    and float(map_position_predict_max_distance_m) > 0.0
                ):
                    velocity = self._filter_velocities.get(cache_key)
                    if velocity:
                        predicted_distance = min(
                            float(map_position_predict_max_distance_m),
                            position_age
                            * float(np.hypot(velocity["x"], velocity["y"])),
                        )
                        speed = float(np.hypot(velocity["x"], velocity["y"]))
                        if predicted_distance > 0.005 and speed > 0.005:
                            candidate = {
                                "x": held["x"] + velocity["x"] / speed * predicted_distance,
                                "y": held["y"] + velocity["y"] / speed * predicted_distance,
                            }
                            candidate, _ = self.nearest_walkable_position(candidate)
                            # Do not let a projection gap route a dot across a
                            # corridor boundary.  The next real observation
                            # still goes through normal routing and speed gates.
                            if self.zone_for(candidate["x"], candidate["y"]) == held.get("zone"):
                                held.update(candidate)
                                predicted = True
                held.update(
                    {
                        "cameras": sorted(active["cameras"]),
                        "observed_at": active["observed_at"],
                        "position_observed_at": cached["mapped_at"],
                        "position_age_seconds": round(position_age, 3),
                        "position_estimated": True,
                        "localization_state": "predicted" if predicted else "held",
                        "position_source_changed": False,
                        "motion_limited": False,
                        "raw_position_step_m": 0.0,
                    }
                )
                if held.get("position_source_camera") not in active["cameras"]:
                    # The identity is visible in another camera, but its old
                    # map coordinate no longer has a current source. Keep the
                    # person/count while withholding the misleading old dot.
                    held["map_visible"] = False
                    held["handoff_waiting"] = True
                    held["handoff_state"] = "waiting_target_localization"
                    held["handoff_target_camera"] = sorted(active["cameras"])[0]
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
        public["counted_people"] = sum(
            1 for person in people if person.get("counted", True)
        )
        public["provisional_people"] = sum(
            1 for person in people if not person.get("counted", True)
        )
        public["visual_track_count"] = len(people)
        public["map_position_hold_seconds"] = float(map_position_hold_seconds)
        public["map_position_predict_seconds"] = float(map_position_predict_seconds)
        public["map_position_predict_max_distance_m"] = float(
            map_position_predict_max_distance_m
        )
        public["handoff_target_confirmations"] = int(handoff_target_confirmations)
        public["handoff_transition_seconds"] = float(handoff_transition_seconds)
        public["handoff_wait_seconds"] = float(handoff_wait_seconds)
        public["max_motion_gap_seconds"] = float(max_motion_gap_seconds)
        public["max_map_speed_mps"] = float(max_map_speed_mps)
        return public
