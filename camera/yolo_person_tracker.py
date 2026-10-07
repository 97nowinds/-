import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from association import (
    ASSOCIATION_ALGORITHM,
    AssociationObservation,
    AssociationTarget,
    BatchAssociator,
    hungarian_maximize,
)
from mtmc_config import load_mtmc_config
from reid_gallery import BodyQualityScorer, TrackFeatureGallery


class YoloPersonTracker:
    """YOLO person detector with one persistent ByteTrack state per camera."""

    def __init__(
        self,
        model_path,
        *,
        confidence=0.45,
        iou=0.50,
        image_size=640,
        tracker_config="bytetrack.yaml",
        device=None,
        model_factory=YOLO,
    ):
        self.model_path = str(Path(model_path))
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.tracker_config = str(tracker_config)
        self.device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")
        self.model = model_factory(self.model_path)
        self.last_diagnostics = {
            "tracker_result_boxes": 0,
            "valid_track_boxes": 0,
            "deduplicated_tracks": 0,
            "invalid_or_unassigned_boxes": 0,
            "confidences": [],
        }

    @staticmethod
    def _values(tensor):
        if tensor is None:
            return []
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "tolist"):
            return tensor.tolist()
        return list(tensor)

    def track(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            classes=[0],
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            self.last_diagnostics = {
                "tracker_result_boxes": 0,
                "valid_track_boxes": 0,
                "deduplicated_tracks": 0,
                "invalid_or_unassigned_boxes": 0,
                "confidences": [],
            }
            return []

        boxes = results[0].boxes
        track_ids = self._values(boxes.id)
        xyxy = self._values(boxes.xyxy)
        confidences = self._values(boxes.conf)
        if not track_ids:
            self.last_diagnostics = {
                "tracker_result_boxes": len(xyxy),
                "valid_track_boxes": 0,
                "deduplicated_tracks": 0,
                "invalid_or_unassigned_boxes": len(xyxy),
                "confidences": [round(float(value), 4) for value in confidences],
            }
            return []

        tracks = []
        for track_id, corners, confidence in zip(track_ids, xyxy, confidences):
            if track_id is None or len(corners) != 4:
                continue
            left, top, right, bottom = [int(round(value)) for value in corners]
            width = right - left
            height = bottom - top
            if width < 4 or height < 4:
                continue
            tracks.append(
                {
                    "track_id": int(track_id),
                    "box": (left, top, width, height),
                    "confidence": round(float(confidence), 4),
                }
            )
        deduplicated = self.deduplicate_tracks(tracks)
        self.last_diagnostics = {
            # Ultralytics exposes boxes after the tracker callback here.  This
            # is intentionally not labelled as raw detector output.
            "tracker_result_boxes": len(xyxy),
            "valid_track_boxes": len(tracks),
            "deduplicated_tracks": len(deduplicated),
            "invalid_or_unassigned_boxes": max(0, len(xyxy) - len(tracks)),
            "confidences": [round(float(value), 4) for value in confidences],
        }
        return deduplicated

    def diagnostics(self):
        return {
            **self.last_diagnostics,
            "minimum_detector_confidence": self.confidence,
            "nms_iou_threshold": self.iou,
            "image_size": self.image_size,
            "tracker_config": self.tracker_config,
        }

    @staticmethod
    def box_iou(first, second):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / max(union, 1)

    @classmethod
    def deduplicate_tracks(cls, tracks, iou_threshold=0.85):
        """Suppress duplicate tracker boxes while preserving distinct people."""
        kept = []
        for track in sorted(
            tracks, key=lambda item: float(item.get("confidence", 0.0)), reverse=True
        ):
            if any(
                cls.box_iou(track["box"], existing["box"]) >= iou_threshold
                for existing in kept
            ):
                continue
            kept.append(track)
        return sorted(kept, key=lambda item: item["track_id"])

    def reset(self):
        # Ultralytics stores persistent tracker state on the predictor.
        # Dropping it prevents an ID from surviving a camera reconnect.
        self.model.predictor = None


class TrackTrailStore:
    def __init__(self, max_points=90, stale_seconds=3.0, minimum_step=2.0):
        self.max_points = int(max_points)
        self.stale_seconds = float(stale_seconds)
        self.minimum_step = float(minimum_step)
        self.histories = defaultdict(lambda: deque(maxlen=self.max_points))
        self.last_seen = {}

    def update(self, track_id, point, now=None):
        now = time.time() if now is None else float(now)
        point = (int(point[0]), int(point[1]))
        history = self.histories[track_id]
        if not history or math.dist(history[-1], point) >= self.minimum_step:
            history.append(point)
        self.last_seen[track_id] = now
        return list(history)

    def prune(self, now=None):
        now = time.time() if now is None else float(now)
        expired = [
            track_id
            for track_id, seen_at in self.last_seen.items()
            if now - seen_at > self.stale_seconds
        ]
        for track_id in expired:
            self.last_seen.pop(track_id, None)
            self.histories.pop(track_id, None)

    def points(self, track_id):
        return list(self.histories.get(track_id, ()))

    def clear(self):
        self.histories.clear()
        self.last_seen.clear()


class CrossCameraTrackCoordinator:
    """Assign one global ID to camera-local YOLO tracks during handoff."""

    def __init__(
        self,
        feature_extractor,
        floor_map=None,
        ttl_seconds=None,
        similarity_threshold=None,
        match_margin=None,
        feature_refresh_seconds=None,
        same_camera_similarity_threshold=None,
        same_camera_ttl_seconds=2.0,
        mtmc_config=None,
        event_store=None,
        camera_roles=None,
    ):
        self.feature_extractor = feature_extractor
        self.floor_map = floor_map
        self.mtmc_config = mtmc_config or load_mtmc_config()
        reid_config = self.mtmc_config["reid"]
        association_config = self.mtmc_config["association"]
        self.ttl_seconds = float(
            association_config["global_track_ttl_seconds"] if ttl_seconds is None else ttl_seconds
        )
        self.similarity_threshold = float(
            reid_config["medium_similarity"] if similarity_threshold is None else similarity_threshold
        )
        self.match_margin = float(
            reid_config["match_margin"] if match_margin is None else match_margin
        )
        self.feature_refresh_seconds = float(
            reid_config["feature_refresh_seconds"]
            if feature_refresh_seconds is None
            else feature_refresh_seconds
        )
        self.same_camera_similarity_threshold = float(
            reid_config["high_similarity"]
            if same_camera_similarity_threshold is None
            else same_camera_similarity_threshold
        )
        self.same_camera_ttl_seconds = float(same_camera_ttl_seconds)
        self.transitions = list(
            floor_map.config.get("camera_transitions", []) if floor_map else []
        )
        self.retention_seconds = max(
            [self.ttl_seconds]
            + [float(item.get("max_gap_seconds", 0.0)) for item in self.transitions]
        )
        self.identity_retention_seconds = max(
            self.ttl_seconds,
            float(association_config["identity_retention_seconds"]),
        )
        self.lock = __import__("threading").RLock()
        self.next_id = 1
        self.local_bindings = {}
        self.local_features = {}
        self.local_galleries = {}
        self.local_motion = {}
        self.global_galleries = {}
        self.global_tracks = {}
        # A global identity survives the short ByteTrack TTL so it can be
        # handed from cam_1 to cam_2 after the target leaves the overlap.
        self.global_identities = {}
        # One registered person may own only one active global track. This
        # prevents a weak Re-ID match from showing the same identity twice.
        self.person_locks = {}
        self.pending = {}
        self.redirects = {}
        self.redirect_limit = max(64, int(reid_config.get("redirect_limit", 256)))
        self.active_local_ids_by_camera = {}
        self.distinct_global_pairs = {}
        self.stream_timestamps = {}
        self.association_reservations = {}
        self.last_merge_rejection_reason = None
        self.recent_associations = deque(maxlen=300)
        self.event_store = event_store
        self.camera_roles = dict(camera_roles or {})
        calibration_status = getattr(floor_map, "calibration_status", "uncalibrated")
        self.batch_associator = BatchAssociator(
            self.mtmc_config,
            transitions=self.transitions,
            calibration_status=calibration_status,
        )

    def register_stream_timestamp(self, camera_id, monotonic_time, unix_time):
        with self.lock:
            self.stream_timestamps[camera_id] = {
                "monotonic_time": float(monotonic_time),
                "unix_time": float(unix_time),
            }

    def stream_skew_seconds(self, now=None):
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            recent = [
                item["monotonic_time"]
                for item in self.stream_timestamps.values()
                if 0.0 <= now - item["monotonic_time"] <= self.retention_seconds
            ]
        return max(recent) - min(recent) if len(recent) >= 2 else 0.0

    def _geometry_allowed(self):
        if not self.floor_map:
            return False
        if hasattr(self.floor_map, "geometry_fusion_allowed"):
            return bool(self.floor_map.geometry_fusion_allowed)
        return bool(getattr(self.floor_map, "config", {}).get("calibrated", False))

    def _camera_geometry_allowed(self, camera_id):
        if not self._geometry_allowed():
            return False
        checker = getattr(self.floor_map, "has_projection", None)
        return bool(checker(camera_id)) if checker is not None else True

    def _new_gallery(self):
        config = self.mtmc_config["reid"]
        return TrackFeatureGallery(
            capacity=config["gallery_capacity"],
            min_quality=config["min_sample_quality"],
            duplicate_similarity=config["duplicate_similarity"],
            top_k=config["top_k"],
        )

    @staticmethod
    def similarity(first, second):
        if first is None or second is None:
            return -1.0
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.dot(first, second) / denominator) if denominator > 1e-8 else -1.0

    @staticmethod
    def clothing_tone(frame, box):
        """Use only clear dark/light torso evidence to veto an impossible match."""
        if frame is None or getattr(frame, "ndim", 0) != 3:
            return None
        x, y, width, height = (int(value) for value in box)
        if width < 24 or height < 60:
            return None
        x1 = max(0, x + int(width * 0.25))
        x2 = min(frame.shape[1], x + int(width * 0.75))
        y1 = max(0, y + int(height * 0.16))
        y2 = min(frame.shape[0], y + int(height * 0.54))
        if x2 - x1 < 10 or y2 - y1 < 20:
            return None
        torso = frame[y1:y2, x1:x2]
        brightness = float(np.median(np.max(torso, axis=2)))
        if brightness <= 85.0:
            return "dark"
        if brightness >= 165.0:
            return "light"
        return None

    @staticmethod
    def _dominant_clothing_tone(votes):
        dark = int((votes or {}).get("dark", 0))
        light = int((votes or {}).get("light", 0))
        if max(dark, light) < 2:
            return None
        if dark >= 3 * light:
            return "dark"
        if light >= 3 * dark:
            return "light"
        return None

    def _is_overlap(self, position, camera_id=None, other_camera_id=None):
        if (
            not self.floor_map
            or not position
            or not self._geometry_allowed()
            or (camera_id is not None and not self._camera_geometry_allowed(camera_id))
            or (
                other_camera_id is not None
                and not self._camera_geometry_allowed(other_camera_id)
            )
        ):
            return False
        for zone in self.floor_map.config.get("zones", []):
            if zone.get("kind") != "overlap":
                continue
            cameras = zone.get("cameras") or []
            if cameras and camera_id is not None and camera_id not in cameras:
                continue
            if cameras and other_camera_id is not None and other_camera_id not in cameras:
                continue
            if (
                zone["x"] <= position["x"] <= zone["x"] + zone["width"]
                and zone["y"] <= position["y"] <= zone["y"] + zone["height"]
            ):
                return True
        return False

    def is_overlap_position(self, position, camera_id=None, other_camera_id=None):
        """Public overlap check used by camera workers and map aggregation."""
        with self.lock:
            return self._is_overlap(position, camera_id, other_camera_id)

    def _transition_gap(self, source_camera_id, target_camera_id):
        for transition in self.transitions:
            source = transition.get("from")
            target = transition.get("to")
            direct = source == source_camera_id and target == target_camera_id
            reverse = (
                transition.get("bidirectional", False)
                and source == target_camera_id
                and target == source_camera_id
            )
            if direct or reverse:
                return max(0.0, float(transition.get("max_gap_seconds", 0.0)))
        return None

    def _handoff_allowed(
        self, source_camera_id, target_camera_id, position, age_seconds
    ):
        if self._is_overlap(position, target_camera_id, source_camera_id):
            return True
        maximum_gap = self._transition_gap(source_camera_id, target_camera_id)
        return maximum_gap is not None and age_seconds <= maximum_gap

    def _record_event(self, event, **fields):
        if self.event_store is not None:
            if fields.get("unix_time") is None:
                fields.pop("unix_time", None)
            return self.event_store.record(event, **fields)
        return None

    def handoff_gates(self, camera_id):
        """Return configured image-space entry/exit gates for video overlays."""
        gates = []
        seen = set()
        for transition in self.transitions:
            gate = (transition.get("image_handoff_gates") or {}).get(camera_id)
            if not gate:
                continue
            other_camera = (
                transition.get("to")
                if transition.get("from") == camera_id
                else transition.get("from")
            )
            for direction in ("exit", "entry"):
                value = gate.get(direction)
                if isinstance(value, dict) and value.get("points"):
                    signature = tuple(
                        (round(float(point[0]), 5), round(float(point[1]), 5))
                        for point in value["points"]
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    gates.append(
                        {
                            "direction": direction,
                            "points": value["points"],
                            "other_camera": other_camera,
                        }
                    )
        return gates

    def _new_global(
        self,
        now,
        camera_id,
        appearance,
        position,
        *,
        gallery=None,
        local_id=None,
        unix_time=None,
        confidence="low",
    ):
        global_id = f"person_{self.next_id}"
        self.next_id += 1
        self.global_tracks[global_id] = {
            "camera_id": camera_id,
            "appearance": appearance,
            "position": position,
            "box": None,
            "local_id": local_id,
            "first_seen": now,
            "last_seen": now,
            "observation_count": 0,
            "association_confirmed": False,
            "count_confirmed": False,
            "in_overlap": self._is_overlap(position, camera_id),
        }
        if gallery is not None:
            self.global_galleries[global_id] = gallery
        self._record_event(
            "create",
            global_id=global_id,
            target_camera=camera_id,
            target_local_id=local_id,
            confidence=confidence,
            monotonic_time=now,
            unix_time=unix_time,
            algorithm_version=ASSOCIATION_ALGORITHM,
        )
        return global_id

    @staticmethod
    def _box_center_distance(first, second):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        return math.hypot(
            ax + aw * 0.5 - (bx + bw * 0.5),
            ay + ah * 0.5 - (by + bh * 0.5),
        )

    def _same_camera_candidates(
        self,
        camera_id,
        appearance,
        position,
        box,
        now,
        active_local_ids,
    ):
        """Reconnect a recent ByteTrack ID switch without merging live peers."""
        if appearance is None or active_local_ids is None:
            return []
        active_local_ids = {int(item) for item in active_local_ids}
        eligible = []
        for candidate_id, target in self.global_tracks.items():
            if target.get("camera_id") != camera_id:
                continue
            age = now - target.get("last_seen", 0.0)
            if age < 0.0 or age > self.same_camera_ttl_seconds:
                continue
            if any(
                key[0] == camera_id
                and bound_id == candidate_id
                and key[1] in active_local_ids
                for key, bound_id in self.local_bindings.items()
            ):
                continue
            target_position = target.get("position")
            if position is not None and target_position is not None:
                if math.hypot(
                    position["x"] - target_position["x"],
                    position["y"] - target_position["y"],
                ) > 2.0:
                    continue
            target_box = target.get("box")
            if box is not None and target_box is not None:
                iou = YoloPersonTracker.box_iou(box, target_box)
                scale = max(box[2], box[3], target_box[2], target_box[3], 1)
                if iou < 0.05 and self._box_center_distance(box, target_box) > scale * 0.75:
                    continue
            score = self.similarity(target.get("appearance"), appearance)
            if score >= self.same_camera_similarity_threshold:
                eligible.append((score, candidate_id))
        return sorted(eligible, reverse=True)

    def _same_camera_relink_assignments(
        self, camera_id, prepared, active_local_ids, now, unix_time
    ):
        """Reconnect ByteTrack ID switches using conservative pixel continuity.

        Floor-map geometry is intentionally not used here: an approximate map
        must not turn a nearby second person into the previous target.  The
        assignment is one-to-one, requires strong appearance evidence and a
        clear spatial margin, and therefore stays unresolved during ambiguous
        crossings.
        """
        settings = self.mtmc_config["association"]
        unbound = [
            item
            for item in prepared
            if (camera_id, item["local_id"]) not in self.local_bindings
            and item.get("appearance") is not None
        ]
        if not unbound:
            return {}
        active_local_ids = {int(item) for item in active_local_ids}
        candidates = []
        for global_id, target in self.global_tracks.items():
            if target.get("camera_id") != camera_id:
                continue
            age = now - float(target.get("last_seen", 0.0))
            if age < 0.0 or age > settings["same_camera_relink_seconds"]:
                continue
            if any(
                bound_camera == camera_id
                and bound_local_id in active_local_ids
                and bound_id == global_id
                for (bound_camera, bound_local_id), bound_id in self.local_bindings.items()
            ):
                continue
            if target.get("appearance") is None or target.get("box") is None:
                continue
            candidates.append((global_id, target, age))
        if not candidates:
            return {}

        matrix = np.full((len(unbound), len(candidates)), -np.inf, dtype=np.float64)
        details = {}
        maximum_ratio = float(settings["same_camera_relink_max_center_ratio"])
        for row, item in enumerate(unbound):
            box = item.get("box")
            if box is None:
                continue
            for column, (global_id, target, age) in enumerate(candidates):
                if (
                    item.get("clothing_tone") in {"dark", "light"}
                    and target.get("clothing_tone") in {"dark", "light"}
                    and item["clothing_tone"] != target["clothing_tone"]
                ):
                    continue
                target_box = target["box"]
                iou = YoloPersonTracker.box_iou(box, target_box)
                scale = max(box[2], box[3], target_box[2], target_box[3], 1)
                center_ratio = self._box_center_distance(box, target_box) / scale
                if iou < 0.02 and center_ratio > maximum_ratio:
                    continue
                spatial_score = max(
                    iou,
                    1.0 - center_ratio / max(maximum_ratio, 1e-6),
                )
                if spatial_score < settings["same_camera_relink_min_spatial_score"]:
                    continue
                target_gallery = self.global_galleries.get(global_id)
                comparison = (
                    item["gallery"].compare(target_gallery)
                    if item.get("gallery") is not None and target_gallery is not None
                    else {}
                )
                reid_score = comparison.get("score")
                if reid_score is None:
                    reid_score = self.similarity(
                        target.get("appearance"), item.get("appearance")
                    )
                if (
                    reid_score is None
                    or reid_score < self.same_camera_similarity_threshold
                ):
                    continue
                time_score = max(
                    0.0,
                    1.0 - age / settings["same_camera_relink_seconds"],
                )
                score = 0.65 * spatial_score + 0.25 * reid_score + 0.10 * time_score
                matrix[row, column] = score
                details[(row, column)] = {
                    "global_id": global_id,
                    "reid_score": float(reid_score),
                    "spatial_score": float(spatial_score),
                    "time_gap_seconds": float(age),
                    "final_score": float(score),
                }

        assignments = {}
        for row, column in hungarian_maximize(matrix):
            assigned_score = float(matrix[row, column])
            row_competitors = [
                float(value)
                for other_column, value in enumerate(matrix[row])
                if other_column != column and np.isfinite(value)
            ]
            column_competitors = [
                float(matrix[other_row, column])
                for other_row in range(len(unbound))
                if other_row != row and np.isfinite(matrix[other_row, column])
            ]
            margin = min(
                assigned_score - max(row_competitors) if row_competitors else 1.0,
                assigned_score - max(column_competitors) if column_competitors else 1.0,
            )
            if assigned_score < settings["same_camera_relink_min_score"]:
                continue
            if margin < settings["same_camera_relink_match_margin"]:
                continue
            item = unbound[row]
            detail = details[(row, column)]
            global_id = detail["global_id"]
            # ByteTrack can resurrect an old local ID after assigning a new
            # one.  Once continuity is transferred, remove inactive aliases
            # so the old ID cannot later reactivate as a second live owner of
            # the same global person.
            for binding_key, bound_id in list(self.local_bindings.items()):
                if (
                    binding_key[0] == camera_id
                    and binding_key[1] != item["local_id"]
                    and binding_key[1] not in active_local_ids
                    and bound_id == global_id
                ):
                    self.local_bindings.pop(binding_key, None)
                    self.local_features.pop(binding_key, None)
                    self.local_galleries.pop(binding_key, None)
                    self.local_motion.pop(binding_key, None)
            self.local_bindings[(camera_id, item["local_id"])] = global_id
            assignments[item["local_id"]] = global_id
            self._record_event(
                "associate",
                camera_id=camera_id,
                target_camera=camera_id,
                target_local_id=item["local_id"],
                global_id=global_id,
                candidate_global_id=global_id,
                accepted=True,
                confidence="high",
                reason="same_camera_spatial_relink",
                reid_score=round(detail["reid_score"], 6),
                geometry_score=round(detail["spatial_score"], 6),
                final_score=round(detail["final_score"], 6),
                match_margin=round(float(margin), 6),
                time_gap_seconds=round(detail["time_gap_seconds"], 6),
                unix_time=unix_time,
                algorithm_version=ASSOCIATION_ALGORITHM,
            )
        return assignments

    @staticmethod
    def _number(global_id):
        return int(global_id.rsplit("_", 1)[-1])

    def _matching_candidates(
        self, camera_id, appearance, position, now, exclude=None
    ):
        eligible = []
        for candidate_id, target in self.global_tracks.items():
            if candidate_id == exclude or target["camera_id"] == camera_id:
                continue
            age = now - target["last_seen"]
            if not self._handoff_allowed(
                target["camera_id"], camera_id, position, age
            ):
                continue
            target_position = target.get("position")
            if (
                self._geometry_allowed()
                and position
                and target_position
            ):
                distance = math.hypot(
                    position["x"] - target_position["x"],
                    position["y"] - target_position["y"],
                )
                # The map is only an approximate projection before full
                # calibration. The overlap zone itself is the hard gate; a
                # wider distance tolerance prevents small projection errors
                # from breaking an otherwise valid handoff.
                if distance > 4.5:
                    continue
            score = self.similarity(target["appearance"], appearance)
            eligible.append((score, candidate_id))
        candidates = [item for item in eligible if item[0] >= self.similarity_threshold]
        return sorted(candidates, reverse=True)

    def _has_unique_best_match(self, candidates):
        return bool(candidates) and (
            len(candidates) == 1
            or candidates[0][0] - candidates[1][0] >= self.match_margin
        )

    def _identity_candidates(
        self,
        camera_id,
        appearance,
        position,
        now,
        minimum_similarity=None,
        exclude=None,
    ):
        """Find identities allowed by overlap or a directed camera transition."""
        if appearance is None:
            return []
        threshold = (
            self.similarity_threshold
            if minimum_similarity is None
            else float(minimum_similarity)
        )
        eligible = []
        for global_id, record in self.global_identities.items():
            if global_id == exclude:
                continue
            person_id = record.get("identity", {}).get("person_id")
            if person_id and self.person_locks.get(person_id) != global_id:
                continue
            if record.get("camera_id") == camera_id:
                continue
            if not self._handoff_allowed(
                record.get("camera_id"),
                camera_id,
                position,
                now - record["updated_at"],
            ):
                continue
            score = self.similarity(record.get("appearance"), appearance)
            if score >= threshold:
                eligible.append((score, global_id))
        return sorted(eligible, reverse=True)

    def _directed_spatial_identity_candidates(
        self, camera_id, appearance, position, now, exclude=None
    ):
        """Recover a late indoor track from one recent entrance confirmation."""
        if (
            appearance is None
            or position is None
            or not self._geometry_allowed()
        ):
            return []
        active_here = {
            global_id
            for global_id, target in self.global_tracks.items()
            if target.get("camera_id") == camera_id
            and now - target.get("last_seen", 0.0) <= self.ttl_seconds
        }
        if len(active_here) != 1:
            return []

        eligible = []
        entrance_cameras = {
            item
            for item, role in self.camera_roles.items()
            if role in {"entrance", "entrance_identity"}
        }
        if not entrance_cameras:
            entrance_cameras = {"cam_entrance"}  # legacy configuration fallback
        for global_id, record in self.global_identities.items():
            if global_id == exclude or record.get("camera_id") not in entrance_cameras:
                continue
            source_camera_id = record.get("camera_id")
            transition = next(
                (
                    item
                    for item in self.transitions
                    if item.get("from") == source_camera_id
                    and item.get("to") == camera_id
                ),
                None,
            )
            if transition is None:
                continue
            age = now - record.get("updated_at", 0.0)
            maximum_age = min(
                float(transition.get("max_gap_seconds", 0.0)),
                float(transition.get("spatial_handoff_max_age_seconds", 3.0)),
            )
            if age < 0.0 or age > maximum_age:
                continue
            source_position = record.get("position")
            if source_position is None:
                continue
            distance = math.hypot(
                position["x"] - source_position["x"],
                position["y"] - source_position["y"],
            )
            if distance > float(
                transition.get("spatial_handoff_max_distance_m", 2.5)
            ):
                continue
            person_id = record.get("identity", {}).get("person_id")
            if person_id and self.person_locks.get(person_id) != global_id:
                continue
            score = self.similarity(record.get("appearance"), appearance)
            if score >= float(
                transition.get("spatial_handoff_similarity_threshold", 0.35)
            ):
                eligible.append((score, global_id))
        return sorted(eligible, reverse=True)

    def _local_appearance(self, binding_key, frame, box, now):
        cached = self.local_features.get(binding_key)
        if cached and now - cached[1] < self.feature_refresh_seconds:
            return cached[0]
        appearance = self.feature_extractor.extract(frame, box)
        if appearance is not None:
            self.local_features[binding_key] = (appearance, now)
        return appearance

    def _release_identity(self, global_id):
        record = self.global_identities.pop(global_id, None)
        person_id = (record or {}).get("identity", {}).get("person_id")
        if person_id and self.person_locks.get(person_id) == global_id:
            self.person_locks.pop(person_id, None)
        target = self.global_tracks.get(global_id)
        if target is not None:
            target.pop("identity", None)
        return record

    def _identity_is_active(self, record, now):
        return bool(
            record
            and now - record.get("updated_at", 0.0)
            <= self.identity_retention_seconds
        )

    def _merge(self, first_id, second_id, prefer=None, unix_time=None):
        self.last_merge_rejection_reason = None
        if frozenset((first_id, second_id)) in self.distinct_global_pairs:
            self.last_merge_rejection_reason = "previously_distinct_same_camera_people"
            return None
        if prefer in (first_id, second_id):
            keep_id = prefer
            remove_id = second_id if prefer == first_id else first_id
        else:
            keep_id, remove_id = sorted((first_id, second_id), key=self._number)
        if keep_id == remove_id:
            return keep_id
        keep_identity = self.global_identities.get(keep_id)
        remove_identity = self.global_identities.get(remove_id)
        keep_person = (keep_identity or {}).get("identity", {}).get("person_id")
        remove_person = (remove_identity or {}).get("identity", {}).get("person_id")
        if keep_person and remove_person and keep_person != remove_person:
            self.last_merge_rejection_reason = "different_known_identities_never_merge"
            return None
        # Final invariant: merging must never make two live ByteTrack owners in
        # the same camera share one global ID. This remains authoritative even
        # if an earlier Re-ID decision or a late merge was over-confident.
        active_owners = {}
        for global_id in (keep_id, remove_id):
            owners = defaultdict(set)
            for (bound_camera, local_id), bound_id in self.local_bindings.items():
                if bound_id != global_id:
                    continue
                if local_id in self.active_local_ids_by_camera.get(bound_camera, set()):
                    owners[bound_camera].add(local_id)
            active_owners[global_id] = owners
        for camera_id in set(active_owners[keep_id]) & set(active_owners[remove_id]):
            if len(
                active_owners[keep_id][camera_id]
                | active_owners[remove_id][camera_id]
            ) > 1:
                self.last_merge_rejection_reason = "active_same_camera_owner_conflict"
                return None
        kept = self.global_tracks.get(keep_id)
        removed = self.global_tracks.pop(remove_id, None)
        if removed and kept is None:
            self.global_tracks[keep_id] = removed
            kept = removed
        elif removed and kept is not None:
            kept["first_seen"] = min(
                float(kept.get("first_seen", kept.get("last_seen", 0.0))),
                float(removed.get("first_seen", removed.get("last_seen", 0.0))),
            )
            kept["observation_count"] = int(kept.get("observation_count", 0)) + int(
                removed.get("observation_count", 0)
            )
            kept["association_confirmed"] = bool(
                kept.get("association_confirmed")
                or removed.get("association_confirmed")
            )
            kept["count_confirmed"] = bool(
                kept.get("count_confirmed") or removed.get("count_confirmed")
            )
            votes = kept.setdefault("clothing_tone_votes", {"dark": 0, "light": 0})
            for tone, count in (removed.get("clothing_tone_votes") or {}).items():
                if tone in votes:
                    votes[tone] += int(count)
            kept["clothing_tone"] = self._dominant_clothing_tone(votes)
        removed_identity = self.global_identities.pop(remove_id, None)
        if removed_identity:
            current = self.global_identities.get(keep_id)
            if current is None or removed_identity["updated_at"] > current["updated_at"]:
                self.global_identities[keep_id] = removed_identity
            kept_identity = self.global_identities.get(keep_id)
            if kept_identity is not None:
                kept_identity["identity"]["global_track_id"] = keep_id
            person_id = removed_identity.get("identity", {}).get("person_id")
            if person_id and self.person_locks.get(person_id) == remove_id:
                self.person_locks[person_id] = keep_id
        for key, bound_id in list(self.local_bindings.items()):
            if bound_id == remove_id:
                self.local_bindings[key] = keep_id
        keep_gallery = self.global_galleries.get(keep_id)
        remove_gallery = self.global_galleries.pop(remove_id, None)
        if keep_gallery is None and remove_gallery is not None:
            self.global_galleries[keep_id] = remove_gallery
        elif keep_gallery is not None and remove_gallery is not None:
            keep_gallery.merge(remove_gallery)
        self.redirects[remove_id] = keep_id
        while len(self.redirects) > self.redirect_limit:
            self.redirects.pop(next(iter(self.redirects)))
        for pair, seen_at in list(self.distinct_global_pairs.items()):
            if remove_id in pair:
                self.distinct_global_pairs.pop(pair)
                replacement = frozenset(keep_id if item == remove_id else item for item in pair)
                if len(replacement) == 2:
                    self.distinct_global_pairs[replacement] = max(
                        seen_at, self.distinct_global_pairs.get(replacement, seen_at)
                    )
        self.pending.pop(remove_id, None)
        self._record_event(
            "merge",
            global_id=keep_id,
            source_global_id=remove_id,
            target_global_id=keep_id,
            unix_time=unix_time,
            algorithm_version=ASSOCIATION_ALGORITHM,
        )
        self._record_event(
            "redirect",
            global_id=remove_id,
            redirect_to=keep_id,
            unix_time=unix_time,
            algorithm_version=ASSOCIATION_ALGORITHM,
        )
        return keep_id

    def merge_global_ids(self, global_ids):
        """Merge already-bound IDs after a frame-level one-person decision."""
        global_ids = [item for item in dict.fromkeys(global_ids) if item]
        if not global_ids:
            return None
        with self.lock:
            keep_id = min(global_ids, key=self._number)
            for global_id in global_ids:
                if global_id != keep_id:
                    merged_id = self._merge(keep_id, global_id)
                    if merged_id is not None:
                        keep_id = merged_id
            return keep_id

    def set_identity(self, global_id, identity, camera_id, position=None, now=None):
        """Attach a confirmed face identity to a cross-camera global target."""
        if not global_id or not identity or not identity.get("known"):
            return None
        person_id = identity.get("person_id")
        if not person_id:
            return None
        now = time.time() if now is None else float(now)
        with self.lock:
            existing = self.global_identities.get(global_id)
            existing_person = (existing or {}).get("identity", {}).get("person_id")
            if existing_person and existing_person != person_id:
                if self._identity_is_active(existing, now):
                    return None
                self._release_identity(global_id)

            lock_owner = self.person_locks.get(person_id)
            if lock_owner and lock_owner != global_id:
                owner_record = self.global_identities.get(lock_owner)
                if self._identity_is_active(owner_record, now):
                    # A face result is not sufficient evidence to merge two
                    # simultaneously active bodies. Occlusion can put another
                    # worker's face crop on the wrong ByteTrack box; merging
                    # here would then label every involved local track as the
                    # same registered person. Cross-camera continuity must be
                    # established by the spatial/Re-ID coordinator first.
                    return None
                else:
                    self._release_identity(lock_owner)

            locked_identity = dict(identity)
            locked_identity.update(
                {
                    "identity_lock_status": "locked",
                    "global_track_id": global_id,
                }
            )
            record = {
                "identity": locked_identity,
                "camera_id": camera_id,
                "position": dict(position) if position else None,
                "appearance": (
                    self.global_tracks.get(global_id, {}).get("appearance")
                ),
                "updated_at": now,
            }
            self.global_identities[global_id] = record
            self.person_locks[person_id] = global_id
            target = self.global_tracks.get(global_id)
            if target is not None:
                target["identity"] = dict(locked_identity)
            return dict(locked_identity)

    def identity_for(self, global_id, camera_id, position=None, now=None):
        """Return identity after a spatial overlap or configured transition."""
        if not global_id:
            return None
        now = time.time() if now is None else float(now)
        with self.lock:
            record = self.global_identities.get(global_id)
            if record is None:
                return None
            if not self._identity_is_active(record, now):
                self._release_identity(global_id)
                return None
            person_id = record.get("identity", {}).get("person_id")
            if person_id and self.person_locks.get(person_id) != global_id:
                return None
            if record.get("camera_id") == camera_id:
                # Restore an already-established global owner after local ID loss.
                if global_id not in self.global_tracks:
                    return None
                return {**record["identity"], "identity_source": "same_camera_reacquisition"}
            if not self._handoff_allowed(
                record.get("camera_id"),
                camera_id,
                position,
                now - record["updated_at"],
            ):
                return None
            in_overlap = self._is_overlap(
                position, camera_id, record.get("camera_id")
            )
            result = dict(record["identity"])
            result.update(
                {
                    "identity_source": (
                        "overlap_handoff" if in_overlap else "transition_handoff"
                    ),
                    "handoff_from_camera": record.get("camera_id"),
                    "appearance_score": record.get("last_match_score"),
                    "handoff_match_source": record.get("last_match_source"),
                }
            )
            return result

    def update_batch(
        self,
        camera_id,
        tracks,
        frame,
        *,
        active_local_ids=None,
        monotonic_time=None,
        unix_time=None,
        stream_skew_seconds=0.0,
    ):
        """Associate one camera frame as a deterministic, one-to-one batch.

        The worker calls this once per inference result. Feature extraction and
        gallery admission happen per local track, while all newly observed
        tracks share one Hungarian assignment against the same target snapshot.
        """
        monotonic_time = (
            time.monotonic() if monotonic_time is None else float(monotonic_time)
        )
        unix_time = time.time() if unix_time is None else float(unix_time)
        active_local_ids = {
            int(item)
            for item in (
                active_local_ids
                if active_local_ids is not None
                else [track["track_id"] for track in tracks]
            )
        }
        prepared = []
        track_inputs = []
        refresh_inputs = []
        for track in tracks:
            local_id = int(track["track_id"])
            box = tuple(track["box"])
            key = (camera_id, local_id)
            gallery = self.local_galleries.setdefault(key, self._new_gallery())
            cached = self.local_features.get(key)
            occluded = any(
                int(other["track_id"]) != local_id
                and YoloPersonTracker.box_iou(box, other["box"])
                >= self.mtmc_config["reid"]["occlusion_iou_threshold"]
                for other in tracks
            )
            # Crossing bodies produce mixed crops. Keep the last clean gallery
            # until separation, including the legacy single-vector fallback.
            needs_refresh = not occluded and (
                cached is None or monotonic_time - cached[1] >= self.feature_refresh_seconds
            )
            quality = BodyQualityScorer.assess(frame, box) if needs_refresh else None
            track_inputs.append((track, local_id, box, key, gallery, quality))
            if needs_refresh:
                refresh_inputs.append((local_id, box))

        refreshed = {}
        if refresh_inputs:
            boxes = [box for _local_id, box in refresh_inputs]
            extract_batch = getattr(self.feature_extractor, "extract_batch", None)
            if callable(extract_batch):
                features = extract_batch(frame, boxes)
            else:
                # Test doubles and external embedders may still expose only the
                # original one-person API.
                features = [self.feature_extractor.extract(frame, box) for box in boxes]
            refreshed = {
                local_id: feature
                for (local_id, _box), feature in zip(refresh_inputs, features)
            }

        for track, local_id, box, key, gallery, quality in track_inputs:
            if quality is not None:
                appearance = refreshed.get(local_id)
                if appearance is not None:
                    gallery.add(
                        appearance,
                        monotonic_time=monotonic_time,
                        unix_time=unix_time,
                        quality=quality["score"],
                        camera_id=camera_id,
                        quality_details=quality,
                    )
                    representative = gallery.representative
                    # Preserve the historical single-vector contract even if
                    # the sample was below the gallery quality threshold.
                    self.local_features[key] = (
                        representative if representative is not None else appearance,
                        monotonic_time,
                    )
            appearance = gallery.representative
            if appearance is None and self.local_features.get(key):
                appearance = self.local_features[key][0]
            position = track.get("position")
            direction = track.get("direction")
            previous_motion = self.local_motion.get(key)
            if direction is None and position is not None and previous_motion is not None:
                previous_position, previous_at = previous_motion
                elapsed = monotonic_time - previous_at
                if elapsed > 1e-6:
                    direction = (
                        (position["x"] - previous_position["x"]) / elapsed,
                        (position["y"] - previous_position["y"]) / elapsed,
                    )
            if position is not None:
                self.local_motion[key] = (dict(position), monotonic_time)
            prepared.append(
                {
                    "local_id": local_id,
                    "box": box,
                    "position": position,
                    "geometry_trusted": not bool(track.get("position_estimated")),
                    "direction": direction,
                    "gallery": gallery,
                    "appearance": appearance,
                    "image_anchor": track.get("image_anchor"),
                    "clothing_tone": self.clothing_tone(frame, box),
                }
            )
        prepared.sort(key=lambda item: (str(camera_id), int(item["local_id"])))

        with self.lock:
            self.active_local_ids_by_camera[camera_id] = set(active_local_ids)
            for pair, seen_at in list(self.distinct_global_pairs.items()):
                if monotonic_time - seen_at > self.retention_seconds:
                    self.distinct_global_pairs.pop(pair, None)
            for global_id, target in list(self.global_tracks.items()):
                if monotonic_time - target["last_seen"] > self.retention_seconds:
                    self.global_tracks.pop(global_id, None)
                    self.global_galleries.pop(global_id, None)
                    self.pending.pop(global_id, None)
                    self._record_event(
                        "expire",
                        global_id=global_id,
                        unix_time=unix_time,
                        algorithm_version=ASSOCIATION_ALGORITHM,
                    )
            for global_id, record in list(self.global_identities.items()):
                if not self._identity_is_active(record, monotonic_time):
                    self._release_identity(global_id)
            for temporary_id, pending in list(self.pending.items()):
                if monotonic_time > pending["expires_at"]:
                    self.pending.pop(temporary_id, None)
            batch_window = self.mtmc_config["association"]["window_seconds"]
            for reserved_id, reserved_at in list(self.association_reservations.items()):
                if monotonic_time - reserved_at > batch_window:
                    self.association_reservations.pop(reserved_id, None)

            relinked_results = self._same_camera_relink_assignments(
                camera_id,
                prepared,
                active_local_ids,
                monotonic_time,
                unix_time,
            )

            active_by_global = defaultdict(set)
            for (bound_camera, local_id), global_id in self.local_bindings.items():
                if local_id in self.active_local_ids_by_camera.get(bound_camera, set()):
                    active_by_global[global_id].add(bound_camera)

            observations = []
            observation_by_key = {}
            existing_results = dict(relinked_results)
            for item in prepared:
                if item["local_id"] in relinked_results:
                    continue
                key = (camera_id, item["local_id"])
                bound_id = self.local_bindings.get(key)
                if bound_id in self.redirects:
                    bound_id = self.redirects[bound_id]
                    self.local_bindings[key] = bound_id
                observation = AssociationObservation(
                    camera_id=camera_id,
                    local_id=item["local_id"],
                    monotonic_time=monotonic_time,
                    gallery=item["gallery"],
                    appearance=item["appearance"],
                    position=item["position"],
                    direction=item["direction"],
                    geometry_trusted=(
                        self._camera_geometry_allowed(camera_id)
                        and item["geometry_trusted"]
                    ),
                    image_anchor=item.get("image_anchor"),
                    clothing_tone=item.get("clothing_tone"),
                    handoff_exit_grace=(
                        {
                            self.global_tracks[bound_id]["handoff_grace_source"]:
                            self.global_tracks[bound_id]["handoff_grace_until"]
                        }
                        if bound_id in self.global_tracks
                        and self.global_tracks[bound_id].get("handoff_grace_source")
                        else {}
                    ),
                    excluded_global_ids={
                        other for pair in self.distinct_global_pairs
                        if bound_id in pair for other in pair if other != bound_id
                    },
                )
                observations.append(observation)
                observation_by_key[(camera_id, item["local_id"])] = item

            targets = []
            for global_id, target in self.global_tracks.items():
                if global_id in self.association_reservations:
                    continue
                targets.append(
                    AssociationTarget(
                        global_id=global_id,
                        camera_id=target["camera_id"],
                        last_seen=target["last_seen"],
                        gallery=self.global_galleries.get(global_id),
                        appearance=target.get("appearance"),
                        position=target.get("position"),
                        direction=target.get("direction"),
                        active_camera_ids=active_by_global.get(global_id, set()),
                        person_id=(
                            self.global_identities.get(global_id, {})
                            .get("identity", {})
                            .get("person_id")
                        ),
                        geometry_trusted=(
                            self._camera_geometry_allowed(target["camera_id"])
                            and target.get("geometry_trusted", True)
                        ),
                        source_local_id=target.get("local_id"),
                        image_anchor=target.get("image_anchor"),
                        recent_exit_gates=dict(target.get("exit_gate_seen_at") or {}),
                        clothing_tone=target.get("clothing_tone"),
                    )
                )
            decisions, evaluations = self.batch_associator.associate(
                observations,
                targets,
                stream_skew_seconds=stream_skew_seconds,
            )
            overlap_sources_by_local = defaultdict(set)
            for evaluation in evaluations:
                if (
                    evaluation.get("reason")
                    == "simultaneous_cross_camera_without_formal_overlap"
                    and evaluation.get("target_in_handoff_gate")
                    and (
                        evaluation.get("source_in_handoff_gate")
                        or evaluation.get("source_exit_gate_recent")
                    )
                ):
                    overlap_sources_by_local[int(evaluation["target_local_id"])].add(
                        evaluation["source_global_id"]
                    )
            for evaluation in evaluations:
                # Persist cross-camera evidence, not the expected same-camera
                # owner conflicts produced on every normal tracking frame.
                # The latter can evict useful handoff events from the bounded
                # diagnostic history without adding association information.
                if evaluation.get("source_camera") == evaluation.get("target_camera"):
                    continue
                self._record_event(
                    "associate",
                    **evaluation,
                    accepted=False,
                    confidence="low",
                    unix_time=unix_time,
                )

            decision_by_local = {item["local_id"]: item for item in decisions}
            results = dict(existing_results)
            for observation in observations:
                item = observation_by_key[(camera_id, observation.local_id)]
                key = (camera_id, observation.local_id)
                decision = decision_by_local[observation.local_id]
                previous_id = self.local_bindings.get(key)
                candidate_id = decision.get("candidate_global_id")
                created_global_id = False
                if decision.get("accepted") and candidate_id in self.global_tracks:
                    global_id = candidate_id
                    self.global_tracks[global_id]["association_confirmed"] = True
                    self.global_tracks[global_id]["count_confirmed"] = True
                    if previous_id in self.global_tracks and previous_id != candidate_id:
                        merged_id = self._merge(
                            previous_id,
                            candidate_id,
                            unix_time=unix_time,
                        )
                        if merged_id is not None:
                            global_id = merged_id
                        else:
                            global_id = previous_id
                            decision = {
                                **decision,
                                "accepted": False,
                                "confidence": "low",
                                "reason": self.last_merge_rejection_reason
                                or "merge_invariant_rejected",
                            }
                    # A same-camera reacquisition is already protected by the
                    # active-local-owner invariant. Reserving it here can hide
                    # the target from another camera exactly during a handoff.
                    if (
                        decision.get("accepted")
                        and decision.get("source_camera") != camera_id
                    ):
                        self.association_reservations[global_id] = monotonic_time
                elif decision.get("confidence") == "medium" and candidate_id:
                    if previous_id in self.global_tracks:
                        global_id = previous_id
                    else:
                        global_gallery = self._new_gallery()
                        global_gallery.merge(item["gallery"])
                        global_id = self._new_global(
                            monotonic_time,
                            camera_id,
                            item["appearance"],
                            item["position"],
                            gallery=global_gallery,
                            local_id=observation.local_id,
                            unix_time=unix_time,
                            confidence="medium",
                        )
                    self.pending[global_id] = {
                        "candidate_global_id": candidate_id,
                        "expires_at": monotonic_time
                        + self.mtmc_config["association"]["pending_ttl_seconds"],
                        "last_score": decision.get("final_score"),
                    }
                elif previous_id in self.global_tracks:
                    global_id = previous_id
                else:
                    global_gallery = self._new_gallery()
                    global_gallery.merge(item["gallery"])
                    global_id = self._new_global(
                        monotonic_time,
                        camera_id,
                        item["appearance"],
                        item["position"],
                        gallery=global_gallery,
                        local_id=observation.local_id,
                        unix_time=unix_time,
                        confidence="low",
                    )
                    created_global_id = True
                blocked_sources = overlap_sources_by_local.get(observation.local_id)
                if created_global_id and blocked_sources:
                    # Keep an ambiguous one-frame ID off the map while the
                    # active source camera finishes its handoff. The detection
                    # box remains visible and association rules stay strict.
                    self.global_tracks[global_id]["handoff_wait_until"] = monotonic_time + 2.0
                    if len(blocked_sources) == 1:
                        self.global_tracks[global_id]["handoff_grace_source"] = next(
                            iter(blocked_sources)
                        )
                        self.global_tracks[global_id]["handoff_grace_until"] = (
                            monotonic_time + 2.0
                        )
                self.local_bindings[key] = global_id
                results[observation.local_id] = global_id
                recorded = {**decision, "global_id": global_id}
                self.recent_associations.append(recorded)
                # Keep per-track diagnostics current in memory, but do not
                # persist routine continuity frames for an already-bound local
                # track when there was no association candidate to evaluate.
                if candidate_id is not None or previous_id is None:
                    self._record_event("associate", **recorded, unix_time=unix_time)

            visible_global_ids = sorted(set(results.values()))
            for index, first in enumerate(visible_global_ids):
                for second in visible_global_ids[index + 1:]:
                    self.distinct_global_pairs[frozenset((first, second))] = monotonic_time

            for item in prepared:
                global_id = results[item["local_id"]]
                target = self.global_tracks[global_id]
                global_gallery = self.global_galleries.setdefault(global_id, self._new_gallery())
                global_gallery.merge(item["gallery"])
                appearance = global_gallery.representative
                if appearance is None:
                    appearance = item["appearance"]
                target.update(
                    {
                        "camera_id": camera_id,
                        "appearance": appearance,
                        "position": item["position"],
                        "geometry_trusted": item["geometry_trusted"],
                        "box": item["box"],
                        "image_anchor": item.get("image_anchor"),
                        "local_id": item["local_id"],
                        "direction": item["direction"],
                        "last_seen": monotonic_time,
                        "in_overlap": self._is_overlap(item["position"], camera_id),
                    }
                )
                if item.get("clothing_tone") in {"dark", "light"}:
                    votes = target.setdefault(
                        "clothing_tone_votes", {"dark": 0, "light": 0}
                    )
                    votes[item["clothing_tone"]] += 1
                    target["clothing_tone"] = self._dominant_clothing_tone(votes)
                exit_gate_seen_at = target.setdefault("exit_gate_seen_at", {})
                for destination in self.batch_associator.exit_gate_destinations(
                    camera_id, item.get("image_anchor")
                ):
                    exit_gate_seen_at[destination] = monotonic_time
                for destination, seen_at in list(exit_gate_seen_at.items()):
                    if monotonic_time - float(seen_at) > float(
                        self.mtmc_config["association"]["handoff_exit_memory_seconds"]
                    ):
                        exit_gate_seen_at.pop(destination, None)
                target["observation_count"] = int(target.get("observation_count", 0)) + 1
            return results

    def _count_duplicate_groups(self):
        """Group only near-certain cross-camera duplicates for counting.

        This does not merge identities or bindings. It only prevents two active
        views of the same person from inflating the public occupancy count while
        conservative MTMC rules keep their global IDs separate.
        """
        active_cameras = defaultdict(set)
        for (camera_id, local_id), bound_id in self.local_bindings.items():
            if local_id in self.active_local_ids_by_camera.get(camera_id, set()):
                active_cameras[bound_id].add(camera_id)
        active_ids = sorted(
            (global_id for global_id in active_cameras if global_id in self.global_tracks),
            key=self._number,
        )
        parent = {global_id: global_id for global_id in active_ids}
        edge_scores = {}

        def find(global_id):
            while parent[global_id] != global_id:
                parent[global_id] = parent[parent[global_id]]
                global_id = parent[global_id]
            return global_id

        def union(first_id, second_id):
            first_root, second_root = find(first_id), find(second_id)
            if first_root != second_root:
                parent[second_root] = first_root

        threshold = float(
            self.mtmc_config["counting"]["duplicate_reid_similarity"]
        )
        for index, first_id in enumerate(active_ids):
            first_gallery = self.global_galleries.get(first_id)
            if first_gallery is None:
                continue
            for second_id in active_ids[index + 1 :]:
                first_tone = self.global_tracks[first_id].get("clothing_tone")
                second_tone = self.global_tracks[second_id].get("clothing_tone")
                if (
                    first_tone in {"dark", "light"}
                    and second_tone in {"dark", "light"}
                    and first_tone != second_tone
                ):
                    continue
                if not any(
                    self._transition_gap(first_camera, second_camera) is not None
                    for first_camera in active_cameras[first_id]
                    for second_camera in active_cameras[second_id]
                    if first_camera != second_camera
                ):
                    continue
                first_person = (
                    self.global_identities.get(first_id, {})
                    .get("identity", {})
                    .get("person_id")
                )
                second_person = (
                    self.global_identities.get(second_id, {})
                    .get("identity", {})
                    .get("person_id")
                )
                if first_person and second_person and first_person != second_person:
                    continue
                comparison = first_gallery.compare(self.global_galleries.get(second_id))
                score = comparison.get("score")
                if score is None or float(score) < threshold:
                    continue
                union(first_id, second_id)
                edge_scores[(first_id, second_id)] = float(score)

        members_by_root = defaultdict(list)
        for global_id in active_ids:
            members_by_root[find(global_id)].append(global_id)
        groups = []
        for members in members_by_root.values():
            if len(members) < 2:
                continue
            representative = min(
                members,
                key=lambda item: (
                    float(
                        self.global_tracks[item].get(
                            "first_seen", self.global_tracks[item].get("last_seen", 0.0)
                        )
                    ),
                    self._number(item),
                ),
            )
            scores = [
                score
                for pair, score in edge_scores.items()
                if pair[0] in members and pair[1] in members
            ]
            groups.append(
                {
                    "representative_global_id": representative,
                    "member_global_ids": sorted(members, key=self._number),
                    "reid_score": round(max(scores), 6) if scores else None,
                    "threshold": threshold,
                    "reason": "count_only_cross_camera_duplicate",
                }
            )
        return groups

    def _count_state(self, global_id):
        target = self.global_tracks.get(global_id)
        if target is None:
            return {
                "counted": False,
                "count_status": "expired",
                "track_age_seconds": 0.0,
                "observation_count": 0,
            }
        first_seen = float(target.get("first_seen", target.get("last_seen", 0.0)))
        last_seen = float(target.get("last_seen", first_seen))
        age_seconds = max(0.0, last_seen - first_seen)
        observation_count = int(target.get("observation_count", 0))
        identified = global_id in self.global_identities
        stable = bool(
            age_seconds >= float(self.mtmc_config["counting"]["confirmation_seconds"])
            and observation_count
            >= int(self.mtmc_config["counting"]["minimum_observations"])
        )
        if identified or target.get("association_confirmed") or stable:
            target["count_confirmed"] = True
        counted = bool(target.get("count_confirmed"))
        duplicate_of = None
        duplicate_score = None
        for group in self._count_duplicate_groups():
            if global_id not in group["member_global_ids"]:
                continue
            representative = group["representative_global_id"]
            if global_id != representative:
                counted = False
                duplicate_of = representative
                duplicate_score = group["reid_score"]
            break
        if identified:
            status = "identified"
        elif duplicate_of is not None:
            status = "duplicate_suppressed"
        elif counted:
            status = "confirmed"
        elif global_id in self.pending:
            status = "pending_association"
        else:
            status = "provisional"
        return {
            "counted": counted,
            "count_status": status,
            "track_age_seconds": round(age_seconds, 3),
            "observation_count": observation_count,
            "count_duplicate_of": duplicate_of,
            "count_duplicate_reid_score": duplicate_score,
        }

    def track_diagnostics(self, camera_id, local_id, now=None):
        key = (camera_id, int(local_id))
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            global_id = self.local_bindings.get(key)
            target = self.global_tracks.get(global_id) or {}
            gallery = self.local_galleries.get(key)
            recent = next(
                (
                    item
                    for item in reversed(self.recent_associations)
                    if item.get("camera_id") == camera_id
                    and int(item.get("local_id", -1)) == int(local_id)
                ),
                None,
            )
            return {
                "global_id": global_id,
                "gallery": gallery.snapshot() if gallery is not None else self._new_gallery().snapshot(),
                "confidence": (recent or {}).get("confidence", "existing"),
                "final_match_score": (recent or {}).get("final_score"),
                "pending": global_id in self.pending,
                "handoff_waiting": now < float(target.get("handoff_wait_until", 0.0)),
                **self._count_state(global_id),
            }

    def diagnostics(self):
        with self.lock:
            recent_redirects = dict(list(self.redirects.items())[-50:])
            return {
                "algorithm_version": self.mtmc_config["algorithm_version"],
                "association_algorithm": ASSOCIATION_ALGORITHM,
                "active_global_tracks": len(self.global_tracks),
                "pending_associations": len(self.pending),
                "batch_window_reservations": len(self.association_reservations),
                "redirect_count": len(self.redirects),
                "recent_redirects": recent_redirects,
                "counting_duplicate_groups": self._count_duplicate_groups(),
                "recent_associations": list(self.recent_associations)[-30:],
                "event_summary": self.event_store.summary() if self.event_store else None,
            }

    def update(
        self,
        camera_id,
        local_id,
        frame,
        box,
        position,
        active_local_ids=None,
        now=None,
    ):
        now = time.time() if now is None else float(now)
        binding_key = (camera_id, int(local_id))
        with self.lock:
            appearance = self._local_appearance(binding_key, frame, box, now)
            for global_id, target in list(self.global_tracks.items()):
                if now - target["last_seen"] > self.retention_seconds:
                    self.global_tracks.pop(global_id, None)
            for global_id, record in list(self.global_identities.items()):
                if not self._identity_is_active(record, now):
                    self._release_identity(global_id)
            global_id = self.local_bindings.get(binding_key)
            if global_id not in self.global_tracks:
                global_id = None

            current_overlap = self._is_overlap(position, camera_id)
            if global_id is None:
                same_camera_candidates = self._same_camera_candidates(
                    camera_id,
                    appearance,
                    position,
                    box,
                    now,
                    active_local_ids,
                )
                if self._has_unique_best_match(same_camera_candidates):
                    global_id = same_camera_candidates[0][1]
                else:
                    candidates = self._matching_candidates(
                        camera_id, appearance, position, now
                    )
                    if self._has_unique_best_match(candidates):
                        global_id = candidates[0][1]
                    else:
                        # The old global track may have expired while the person
                        # crossed the blind zone. Rehydrate it from the confirmed
                        # identity registry, still gated by the overlap zone.
                        identity_candidates = self._identity_candidates(
                            camera_id, appearance, position, now
                        )
                        if self._has_unique_best_match(identity_candidates):
                            global_id = identity_candidates[0][1]
                            self.global_tracks[global_id] = {
                                "camera_id": camera_id,
                                "appearance": appearance,
                                "position": position,
                                "box": box,
                                "last_seen": now,
                                "in_overlap": current_overlap,
                            }
                        else:
                            global_id = self._new_global(
                                now, camera_id, appearance, position
                            )
                self.local_bindings[binding_key] = global_id
                if active_local_ids is not None:
                    for key, bound_id in list(self.local_bindings.items()):
                        if (
                            key != binding_key
                            and key[0] == camera_id
                            and bound_id == global_id
                            and key[1] not in active_local_ids
                        ):
                            self.local_bindings.pop(key, None)
                            self.local_features.pop(key, None)

            # Existing camera-local tracks may have received separate IDs before
            # reaching the overlap. Merge them once both views agree there.
            candidates = self._matching_candidates(
                camera_id,
                appearance,
                position,
                now,
                exclude=global_id,
            )
            if self._has_unique_best_match(candidates):
                merged_id = self._merge(global_id, candidates[0][1])
                if merged_id is not None:
                    global_id = merged_id
                    self.local_bindings[binding_key] = global_id
            else:
                # ByteTrack can establish the indoor ID before ArcFace reaches
                # its entrance vote threshold. Re-evaluate that existing ID
                # using direction, time, distance, crowd and weak Re-ID gates.
                late_candidates = self._directed_spatial_identity_candidates(
                    camera_id,
                    appearance,
                    position,
                    now,
                    exclude=global_id,
                )
                if self._has_unique_best_match(late_candidates):
                    score, identity_global_id = late_candidates[0]
                    merged_id = self._merge(global_id, identity_global_id)
                    if merged_id is not None:
                        global_id = merged_id
                        self.local_bindings[binding_key] = global_id
                        identity_record = self.global_identities.get(global_id)
                        if identity_record is not None:
                            identity_record["last_match_score"] = score
                            identity_record["last_match_source"] = (
                                "directed_spatial_reid"
                            )

            target = self.global_tracks[global_id]
            if appearance is not None:
                previous = target.get("appearance")
                updated = appearance if previous is None else previous * 0.85 + appearance * 0.15
                norm = float(np.linalg.norm(updated))
                target["appearance"] = updated / norm if norm > 1e-8 else updated
            target.update(
                {
                    "camera_id": camera_id,
                    "position": position,
                    "box": box,
                    "last_seen": now,
                    "in_overlap": current_overlap,
                }
            )
            target.setdefault("first_seen", now)
            target["observation_count"] = int(target.get("observation_count", 0)) + 1
            return global_id

    def forget_camera(self, camera_id):
        with self.lock:
            for key in [key for key in self.local_bindings if key[0] == camera_id]:
                self.local_bindings.pop(key, None)
                self.local_features.pop(key, None)
                self.local_galleries.pop(key, None)
                self.local_motion.pop(key, None)
            self.active_local_ids_by_camera.pop(camera_id, None)

    def reset_replay_cycle(self):
        """Discard all cross-camera evidence at a recording loop boundary."""
        with self.lock:
            for state in (
                self.local_bindings, self.local_features, self.local_galleries,
                self.local_motion, self.global_galleries, self.global_tracks,
                self.global_identities, self.person_locks, self.pending,
                self.redirects, self.active_local_ids_by_camera,
                self.distinct_global_pairs, self.stream_timestamps,
                self.association_reservations,
            ):
                state.clear()
            self.recent_associations.clear()
            self.last_merge_rejection_reason = None
            # Keep next_id monotonic so a new loop cannot reuse an old ID.

    def forget_local(self, camera_id, local_id):
        """Release one camera-local binding without disturbing its peers."""
        key = (camera_id, int(local_id))
        with self.lock:
            self.local_bindings.pop(key, None)
            self.local_features.pop(key, None)
            self.local_galleries.pop(key, None)
            self.local_motion.pop(key, None)
