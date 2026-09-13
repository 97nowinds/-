import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from association import ASSOCIATION_ALGORITHM, AssociationObservation, AssociationTarget, BatchAssociator
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
        device=None,
        model_factory=YOLO,
    ):
        self.model_path = str(Path(model_path))
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")
        self.model = model_factory(self.model_path)

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
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        track_ids = self._values(boxes.id)
        xyxy = self._values(boxes.xyxy)
        confidences = self._values(boxes.conf)
        if not track_ids:
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
        return self.deduplicate_tracks(tracks)

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
        self.active_local_ids_by_camera = {}
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
            "last_seen": now,
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
        removed = self.global_tracks.pop(remove_id, None)
        if removed and keep_id not in self.global_tracks:
            self.global_tracks[keep_id] = removed
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
                return None
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
        for track in tracks:
            local_id = int(track["track_id"])
            box = tuple(track["box"])
            key = (camera_id, local_id)
            gallery = self.local_galleries.setdefault(key, self._new_gallery())
            cached = self.local_features.get(key)
            if cached is None or monotonic_time - cached[1] >= self.feature_refresh_seconds:
                quality = BodyQualityScorer.assess(frame, box)
                appearance = self.feature_extractor.extract(frame, box)
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
                    "direction": direction,
                    "gallery": gallery,
                    "appearance": appearance,
                }
            )
        prepared.sort(key=lambda item: (str(camera_id), int(item["local_id"])))

        with self.lock:
            self.active_local_ids_by_camera[camera_id] = set(active_local_ids)
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

            active_by_global = defaultdict(set)
            for (bound_camera, local_id), global_id in self.local_bindings.items():
                if local_id in self.active_local_ids_by_camera.get(bound_camera, set()):
                    active_by_global[global_id].add(bound_camera)

            observations = []
            observation_by_key = {}
            existing_results = {}
            for item in prepared:
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
                    geometry_trusted=self._camera_geometry_allowed(camera_id),
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
                        geometry_trusted=self._camera_geometry_allowed(target["camera_id"]),
                        source_local_id=target.get("local_id"),
                    )
                )
            decisions, evaluations = self.batch_associator.associate(
                observations,
                targets,
                stream_skew_seconds=stream_skew_seconds,
            )
            for evaluation in evaluations:
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
                if decision.get("accepted") and candidate_id in self.global_tracks:
                    global_id = candidate_id
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
                    if decision.get("accepted"):
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
                self.local_bindings[key] = global_id
                results[observation.local_id] = global_id
                recorded = {**decision, "global_id": global_id}
                self.recent_associations.append(recorded)
                self._record_event("associate", **recorded, unix_time=unix_time)

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
                        "box": item["box"],
                        "local_id": item["local_id"],
                        "direction": item["direction"],
                        "last_seen": monotonic_time,
                        "in_overlap": self._is_overlap(item["position"], camera_id),
                    }
                )
            return results

    def track_diagnostics(self, camera_id, local_id):
        key = (camera_id, int(local_id))
        with self.lock:
            global_id = self.local_bindings.get(key)
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
            }

    def diagnostics(self):
        with self.lock:
            return {
                "algorithm_version": self.mtmc_config["algorithm_version"],
                "association_algorithm": ASSOCIATION_ALGORITHM,
                "active_global_tracks": len(self.global_tracks),
                "pending_associations": len(self.pending),
                "batch_window_reservations": len(self.association_reservations),
                "redirects": dict(self.redirects),
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
            return global_id

    def forget_camera(self, camera_id):
        with self.lock:
            for key in [key for key in self.local_bindings if key[0] == camera_id]:
                self.local_bindings.pop(key, None)
                self.local_features.pop(key, None)
                self.local_galleries.pop(key, None)
                self.local_motion.pop(key, None)
            self.active_local_ids_by_camera.pop(camera_id, None)

    def forget_local(self, camera_id, local_id):
        """Release one camera-local binding without disturbing its peers."""
        key = (camera_id, int(local_id))
        with self.lock:
            self.local_bindings.pop(key, None)
            self.local_features.pop(key, None)
            self.local_galleries.pop(key, None)
            self.local_motion.pop(key, None)
