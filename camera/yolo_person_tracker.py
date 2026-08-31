import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


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
        return sorted(tracks, key=lambda item: item["track_id"])

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
        ttl_seconds=6.0,
        similarity_threshold=0.72,
        match_margin=0.05,
        feature_refresh_seconds=1.0,
    ):
        self.feature_extractor = feature_extractor
        self.floor_map = floor_map
        self.ttl_seconds = float(ttl_seconds)
        self.similarity_threshold = float(similarity_threshold)
        self.match_margin = float(match_margin)
        self.feature_refresh_seconds = float(feature_refresh_seconds)
        self.transitions = list(
            floor_map.config.get("camera_transitions", []) if floor_map else []
        )
        self.retention_seconds = max(
            [self.ttl_seconds]
            + [float(item.get("max_gap_seconds", 0.0)) for item in self.transitions]
        )
        self.identity_retention_seconds = max(self.ttl_seconds, 90.0)
        self.lock = __import__("threading").RLock()
        self.next_id = 1
        self.local_bindings = {}
        self.local_features = {}
        self.global_tracks = {}
        # A global identity survives the short ByteTrack TTL so it can be
        # handed from cam_1 to cam_2 after the target leaves the overlap.
        self.global_identities = {}
        # One registered person may own only one active global track. This
        # prevents a weak Re-ID match from showing the same identity twice.
        self.person_locks = {}

    @staticmethod
    def similarity(first, second):
        if first is None or second is None:
            return -1.0
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.dot(first, second) / denominator) if denominator > 1e-8 else -1.0

    def _is_overlap(self, position, camera_id=None, other_camera_id=None):
        if not self.floor_map or not position:
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

    def _new_global(self, now, camera_id, appearance, position):
        global_id = f"person_{self.next_id}"
        self.next_id += 1
        self.global_tracks[global_id] = {
            "camera_id": camera_id,
            "appearance": appearance,
            "position": position,
            "last_seen": now,
            "in_overlap": self._is_overlap(position, camera_id),
        }
        return global_id

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
            if position and target_position:
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

    def _identity_candidates(self, camera_id, appearance, position, now):
        """Find identities allowed by overlap or a directed camera transition."""
        if appearance is None:
            return []
        eligible = []
        for global_id, record in self.global_identities.items():
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
            if score >= self.similarity_threshold:
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

    def _merge(self, first_id, second_id):
        keep_id, remove_id = sorted((first_id, second_id), key=self._number)
        if keep_id == remove_id:
            return keep_id
        keep_identity = self.global_identities.get(keep_id)
        remove_identity = self.global_identities.get(remove_id)
        keep_person = (keep_identity or {}).get("identity", {}).get("person_id")
        remove_person = (remove_identity or {}).get("identity", {}).get("person_id")
        if keep_person and remove_person and keep_person != remove_person:
            return None
        removed = self.global_tracks.pop(remove_id, None)
        if removed and keep_id not in self.global_tracks:
            self.global_tracks[keep_id] = removed
        removed_identity = self.global_identities.pop(remove_id, None)
        if removed_identity:
            current = self.global_identities.get(keep_id)
            if current is None or removed_identity["updated_at"] > current["updated_at"]:
                self.global_identities[keep_id] = removed_identity
            person_id = removed_identity.get("identity", {}).get("person_id")
            if person_id and self.person_locks.get(person_id) == remove_id:
                self.person_locks[person_id] = keep_id
        for key, bound_id in list(self.local_bindings.items()):
            if bound_id == remove_id:
                self.local_bindings[key] = keep_id
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
                    return None
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
                }
            )
            return result

    def update(self, camera_id, local_id, frame, box, position, now=None):
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
                            "last_seen": now,
                            "in_overlap": current_overlap,
                        }
                    else:
                        global_id = self._new_global(now, camera_id, appearance, position)
                self.local_bindings[binding_key] = global_id

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
