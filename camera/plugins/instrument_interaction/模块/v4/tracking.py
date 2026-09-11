from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _blend_bbox(previous: list[float], current: list[float], alpha: float) -> list[float]:
    return [alpha * float(now) + (1.0 - alpha) * float(old) for old, now in zip(previous, current)]


@dataclass
class StableTrack:
    track_id: str
    class_name: str
    display_name: str
    raw_class_name: str
    detector_source: str
    bbox: list[float]
    confidence: float
    first_seen: float
    last_seen: float
    first_frame: int
    last_frame: int
    hits: int = 1
    misses: int = 0
    recoveries: int = 0
    history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=120))
    full_history: list[dict[str, Any]] = field(default_factory=list)
    hands: dict[str, dict[str, Any]] = field(default_factory=dict)

    def velocity(self) -> tuple[float, float]:
        observed = [item for item in self.history if item.get("observed")]
        if len(observed) < 2:
            return 0.0, 0.0
        previous, current = observed[-2], observed[-1]
        dt = float(current["timestamp"]) - float(previous["timestamp"])
        if dt <= 1e-6:
            return 0.0, 0.0
        return (
            (float(current["center"][0]) - float(previous["center"][0])) / dt,
            (float(current["center"][1]) - float(previous["center"][1])) / dt,
        )

    def predicted_bbox(self, timestamp: float, max_prediction_seconds: float) -> list[float]:
        dt = min(max(0.0, float(timestamp) - self.last_seen), max_prediction_seconds)
        vx, vy = self.velocity()
        return [
            self.bbox[0] + vx * dt,
            self.bbox[1] + vy * dt,
            self.bbox[2] + vx * dt,
            self.bbox[3] + vy * dt,
        ]

    def observe(
        self,
        detection: dict[str, Any],
        *,
        timestamp: float,
        frame_index: int,
        smooth_alpha: float,
    ) -> None:
        recovered = self.misses > 0
        if recovered:
            self.recoveries += 1
        self.bbox = _blend_bbox(self.bbox, list(detection["bbox"]), smooth_alpha)
        self._update_hands(detection.get("hands") or {}, timestamp, smooth_alpha)
        self.confidence = float(detection["confidence"])
        self.detector_source = str(detection.get("detector_source") or self.detector_source)
        self.raw_class_name = str(detection.get("raw_class_name") or self.raw_class_name)
        self.last_seen = float(timestamp)
        self.last_frame = int(frame_index)
        self.hits += 1
        self.misses = 0
        self._append_history(timestamp, frame_index, self.bbox, observed=True, recovered=recovered)

    def _update_hands(self, hands: dict[str, Any], timestamp: float, alpha: float) -> None:
        for side in ("left", "right"):
            value = hands.get(side) or {}
            point = value.get("point")
            confidence = float(value.get("confidence") or 0.0)
            if not point or confidence < 0.12:
                continue
            previous = self.hands.get(side)
            if previous:
                blended = [
                    alpha * float(point[index]) + (1.0 - alpha) * float(previous["point"][index])
                    for index in (0, 1)
                ]
                dt = max(1e-6, float(timestamp) - float(previous["timestamp"]))
                velocity = [
                    (blended[index] - float(previous["point"][index])) / dt for index in (0, 1)
                ]
            else:
                blended = [float(point[0]), float(point[1])]
                velocity = [0.0, 0.0]
            updated = {
                "point": blended,
                "confidence": confidence,
                "timestamp": float(timestamp),
                "velocity": velocity,
            }
            elbow_point = value.get("elbow_point")
            if elbow_point:
                previous_elbow = (previous or {}).get("elbow_point")
                if previous_elbow:
                    updated["elbow_point"] = [
                        alpha * float(elbow_point[index])
                        + (1.0 - alpha) * float(previous_elbow[index])
                        for index in (0, 1)
                    ]
                else:
                    updated["elbow_point"] = [float(elbow_point[0]), float(elbow_point[1])]
                updated["elbow_confidence"] = float(value.get("elbow_confidence") or confidence)
            self.hands[side] = updated

    def hand_snapshot(self, timestamp: float, hold_seconds: float = 0.65) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for side, value in self.hands.items():
            age = max(0.0, float(timestamp) - float(value["timestamp"]))
            if age > hold_seconds:
                continue
            output[side] = {
                "point": [round(float(item), 3) for item in value["point"]],
                "confidence": round(float(value["confidence"]), 6),
                "velocity": [round(float(item), 3) for item in value.get("velocity") or [0.0, 0.0]],
                "observed": age <= 0.12,
                "seconds_since_observed": round(age, 4),
            }
            if value.get("elbow_point"):
                output[side]["elbow_point"] = [
                    round(float(item), 3) for item in value["elbow_point"]
                ]
                output[side]["elbow_confidence"] = round(
                    float(value.get("elbow_confidence") or 0.0), 6
                )
        return output

    def miss(self) -> None:
        self.misses += 1

    def _append_history(
        self,
        timestamp: float,
        frame_index: int,
        bbox: list[float],
        *,
        observed: bool,
        recovered: bool = False,
    ) -> dict[str, Any]:
        center = bbox_center(bbox)
        previous = self.history[-1] if self.history else None
        dx = center[0] - float(previous["center"][0]) if previous else 0.0
        dy = center[1] - float(previous["center"][1]) if previous else 0.0
        dt = float(timestamp) - float(previous["timestamp"]) if previous else 0.0
        speed = math.hypot(dx, dy) / dt if dt > 1e-6 else 0.0
        record = {
            "timestamp": round(float(timestamp), 6),
            "frame_index": int(frame_index),
            "bbox": [round(float(value), 3) for value in bbox],
            "center": [round(center[0], 3), round(center[1], 3)],
            "dx": round(dx, 3),
            "dy": round(dy, 3),
            "velocity_px_per_second": round(speed, 3),
            "confidence": round(float(self.confidence), 6),
            "observed": bool(observed),
            "recovered_after_miss": bool(recovered),
            "hands": self.hand_snapshot(timestamp),
        }
        self.history.append(record)
        if observed:
            self.full_history.append(record)
        return record

    def snapshot(
        self,
        *,
        timestamp: float,
        frame_index: int,
        observed: bool,
        max_prediction_seconds: float,
    ) -> dict[str, Any]:
        bbox = list(self.bbox) if observed else self.predicted_bbox(timestamp, max_prediction_seconds)
        vx, vy = self.velocity()
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "display_name": self.display_name,
            "raw_class_name": self.raw_class_name,
            "detector_source": self.detector_source,
            "bbox": [round(float(value), 3) for value in bbox],
            "center": [round(value, 3) for value in bbox_center(bbox)],
            "confidence": round(float(self.confidence), 6),
            "observed": bool(observed),
            "predicted": not observed,
            "seconds_since_observed": round(max(0.0, float(timestamp) - self.last_seen), 4),
            "hits": self.hits,
            "misses": self.misses,
            "confirmed": self.hits >= 2,
            "recoveries": self.recoveries,
            "velocity": [round(vx, 3), round(vy, 3)],
            "velocity_px_per_second": round(math.hypot(vx, vy), 3),
            "frame_index": int(frame_index),
            "timestamp": round(float(timestamp), 6),
            "hands": self.hand_snapshot(timestamp),
        }


class StableTracker:
    """Class-aware tracker with smoothing, short-gap prediction and recent-track revival."""

    SLENDER_CLASSES = {"burette", "measuring_cylinder", "glass_rod", "dropper"}

    def __init__(
        self,
        *,
        detection_fps: float,
        hold_seconds: float = 0.8,
        slender_hold_seconds: float = 1.4,
        archive_seconds: float = 3.0,
        smooth_alpha: float = 0.72,
    ) -> None:
        self.detection_fps = max(1.0, float(detection_fps))
        self.hold_seconds = float(hold_seconds)
        self.slender_hold_seconds = float(slender_hold_seconds)
        self.archive_seconds = float(archive_seconds)
        self.smooth_alpha = float(smooth_alpha)
        self.active: dict[str, StableTrack] = {}
        self.archived: dict[str, StableTrack] = {}
        self.counters: defaultdict[str, int] = defaultdict(int)
        self.events: list[dict[str, Any]] = []
        self.last_observed_ids: set[str] = set()

    def _hold_for(self, class_name: str) -> float:
        return self.slender_hold_seconds if class_name in self.SLENDER_CLASSES else self.hold_seconds

    @staticmethod
    def _association_score(
        track: StableTrack,
        detection: dict[str, Any],
        timestamp: float,
        frame_diagonal: float,
    ) -> tuple[float, float, float]:
        predicted = track.predicted_bbox(timestamp, 0.5)
        candidate = list(detection["bbox"])
        overlap = max(bbox_iou(track.bbox, candidate), bbox_iou(predicted, candidate))
        tc = bbox_center(predicted)
        dc = bbox_center(candidate)
        center_distance = math.hypot(tc[0] - dc[0], tc[1] - dc[1]) / max(frame_diagonal, 1.0)
        track_area = max(bbox_area(track.bbox), 1.0)
        candidate_area = max(bbox_area(candidate), 1.0)
        size_score = min(track_area, candidate_area) / max(track_area, candidate_area)
        center_limit = 0.24 if track.class_name in StableTracker.SLENDER_CLASSES else 0.16
        center_score = max(0.0, 1.0 - center_distance / center_limit)
        score = 0.45 * overlap + 0.40 * center_score + 0.15 * size_score
        return score, overlap, center_distance

    def _new_track(self, detection: dict[str, Any], timestamp: float, frame_index: int) -> StableTrack:
        class_name = str(detection["class_name"])
        self.counters[class_name] += 1
        track = StableTrack(
            track_id=f"{class_name}_{self.counters[class_name]:02d}",
            class_name=class_name,
            display_name=str(detection.get("display_name") or class_name),
            raw_class_name=str(detection.get("raw_class_name") or class_name),
            detector_source=str(detection.get("detector_source") or "unknown"),
            bbox=list(detection["bbox"]),
            confidence=float(detection["confidence"]),
            first_seen=float(timestamp),
            last_seen=float(timestamp),
            first_frame=int(frame_index),
            last_frame=int(frame_index),
        )
        track._update_hands(detection.get("hands") or {}, timestamp, 1.0)
        track._append_history(timestamp, frame_index, track.bbox, observed=True)
        self.active[track.track_id] = track
        return track

    def update(
        self,
        detections: list[dict[str, Any]],
        *,
        timestamp: float,
        frame_index: int,
        frame_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        width, height = frame_size
        diagonal = math.hypot(width, height)
        candidates: list[tuple[float, str, int, float, float]] = []
        for track_id, track in self.active.items():
            for index, detection in enumerate(detections):
                if str(detection["class_name"]) != track.class_name:
                    continue
                score, overlap, distance = self._association_score(track, detection, timestamp, diagonal)
                center_limit = 0.24 if track.class_name in self.SLENDER_CLASSES else 0.16
                if score >= 0.24 and (overlap >= 0.015 or distance <= center_limit):
                    candidates.append((score, track_id, index, overlap, distance))

        assigned_tracks: set[str] = set()
        assigned_detections: set[int] = set()
        assignments: dict[int, str] = {}
        for score, track_id, index, _, _ in sorted(candidates, key=lambda item: item[0], reverse=True):
            if track_id in assigned_tracks or index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(index)
            assignments[index] = track_id

        for track_id, track in list(self.active.items()):
            if track_id not in assigned_tracks:
                track.miss()

        observed_ids: set[str] = set()
        for index, detection in enumerate(detections):
            track_id = assignments.get(index)
            if track_id is None:
                revived = self._revive(detection, timestamp, frame_index, diagonal)
                track = revived or self._new_track(detection, timestamp, frame_index)
            else:
                track = self.active[track_id]
                track.observe(
                    detection,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    smooth_alpha=self.smooth_alpha,
                )
            observed_ids.add(track.track_id)

        for track_id, track in list(self.active.items()):
            if timestamp - track.last_seen > self.archive_seconds:
                self.archived[track_id] = self.active.pop(track_id)
                self.events.append(
                    {"type": "track_archived", "track_id": track_id, "timestamp": round(timestamp, 6)}
                )
        self.last_observed_ids = observed_ids
        return self.snapshot(timestamp=timestamp, frame_index=frame_index)

    def _revive(
        self,
        detection: dict[str, Any],
        timestamp: float,
        frame_index: int,
        frame_diagonal: float,
    ) -> StableTrack | None:
        choices: list[tuple[float, str]] = []
        for track_id, track in self.archived.items():
            if track.class_name != str(detection["class_name"]):
                continue
            if timestamp - track.last_seen > self.archive_seconds * 2.0:
                continue
            score, overlap, distance = self._association_score(track, detection, timestamp, frame_diagonal)
            if score >= 0.34 and (overlap >= 0.02 or distance <= 0.12):
                choices.append((score, track_id))
        if not choices:
            return None
        _, track_id = max(choices)
        track = self.archived.pop(track_id)
        self.active[track_id] = track
        track.observe(
            detection,
            timestamp=timestamp,
            frame_index=frame_index,
            smooth_alpha=self.smooth_alpha,
        )
        self.events.append(
            {"type": "track_revived", "track_id": track_id, "timestamp": round(timestamp, 6)}
        )
        return track

    def snapshot(self, *, timestamp: float, frame_index: int) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for track in self.active.values():
            elapsed = max(0.0, timestamp - track.last_seen)
            if elapsed > self._hold_for(track.class_name):
                continue
            observed = track.track_id in self.last_observed_ids and elapsed <= 1e-4
            if track.hits < 2 and not observed:
                continue
            output.append(
                track.snapshot(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    observed=observed,
                    max_prediction_seconds=self._hold_for(track.class_name),
                )
            )
        return sorted(output, key=lambda item: (item["class_name"], item["track_id"]))

    def recent_history(self, track_id: str, seconds: float = 5.0) -> list[dict[str, Any]]:
        track = self.active.get(track_id) or self.archived.get(track_id)
        if track is None or not track.history:
            return []
        cutoff = float(track.history[-1]["timestamp"]) - float(seconds)
        return [item for item in track.history if float(item["timestamp"]) >= cutoff]

    def export(self) -> dict[str, Any]:
        tracks = {**self.archived, **self.active}
        by_class: defaultdict[str, list[str]] = defaultdict(list)
        for track in tracks.values():
            by_class[track.class_name].append(track.track_id)
        return {
            "tracker": "V4 StableTracker (class-aware smoothing + short-gap prediction + revival)",
            "detection_fps": self.detection_fps,
            "hold_seconds": self.hold_seconds,
            "slender_hold_seconds": self.slender_hold_seconds,
            "archive_seconds": self.archive_seconds,
            "tracks_by_class": {key: sorted(value) for key, value in sorted(by_class.items())},
            "events": self.events,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "display_name": track.display_name,
                    "first_seen": round(track.first_seen, 6),
                    "last_seen": round(track.last_seen, 6),
                    "hits": track.hits,
                    "recoveries": track.recoveries,
                    "last_confidence": round(float(track.confidence), 6),
                    "mean_confidence": round(
                        sum(float(item.get("confidence") or 0.0) for item in track.full_history)
                        / max(1, len(track.full_history)),
                        6,
                    ),
                    "history": track.full_history,
                }
                for track in sorted(tracks.values(), key=lambda item: item.track_id)
            ],
        }
