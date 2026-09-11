from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from settings import PROJECT_ROOT


DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "configs" / "fixed_instrument_calibration_v0_1.json"


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _segment_intersects_mask(
    start: tuple[float, float] | None,
    end: tuple[float, float],
    mask: np.ndarray,
) -> bool:
    if start is None:
        return False
    height, width = mask.shape[:2]
    steps = max(2, int(math.ceil(max(abs(end[0] - start[0]), abs(end[1] - start[1])))))
    xs = np.rint(np.linspace(start[0], end[0], steps + 1)).astype(np.int32)
    ys = np.rint(np.linspace(start[1], end[1], steps + 1)).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return bool(valid.any() and np.any(mask[ys[valid], xs[valid]] > 0))


@dataclass(frozen=True)
class InstrumentRegion:
    instrument_id: str
    display_name: str
    class_key: str
    interaction_margin_pixels: int


class FixedInstrumentRegions:
    """Load fixed instrument masks and query wrist/forearm spatial relations."""

    def __init__(
        self,
        *,
        camera_id: str,
        frame_size: tuple[int, int],
        camera_config: dict[str, Any],
        temporal_rules: dict[str, Any] | None = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.reference_size = tuple(int(value) for value in camera_config["frame_size"])
        self.temporal_rules = dict(temporal_rules or {})
        self.transform, self.registration_method = self._resolve_transform(camera_config)
        self.instruments = [
            InstrumentRegion(
                instrument_id=str(item["id"]),
                display_name=str(item["display_name"]),
                class_key=str(item.get("class_key") or item["id"]),
                interaction_margin_pixels=int(item.get("interaction_margin_pixels") or 0),
            )
            for item in camera_config.get("instruments") or []
        ]
        self.instrument_lookup = {item.instrument_id: item for item in self.instruments}
        self.core_masks, self.interaction_masks = self._build_masks(camera_config)
        self.distance_maps = {
            instrument_id: cv2.distanceTransform(
                np.where(mask > 0, 0, 255).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            for instrument_id, mask in self.core_masks.items()
        }

    @classmethod
    def from_file(
        cls,
        *,
        camera_id: str,
        frame_size: tuple[int, int] | None = None,
        calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    ) -> "FixedInstrumentRegions":
        payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        cameras = payload.get("cameras") or {}
        if camera_id not in cameras:
            raise KeyError(f"Unknown fixed-instrument camera_id: {camera_id}")
        camera = dict(cameras[camera_id])
        selected_size = frame_size or tuple(int(value) for value in camera["frame_size"])
        return cls(
            camera_id=camera_id,
            frame_size=selected_size,
            camera_config=camera,
            temporal_rules=payload.get("interaction_temporal_rules") or {},
        )

    def _resolve_transform(self, camera_config: dict[str, Any]) -> tuple[np.ndarray, str]:
        width, height = self.frame_size
        reference_width, reference_height = self.reference_size
        if (width, height) == (reference_width, reference_height):
            return np.eye(3, dtype=np.float64), "identity_reference_coordinates"
        key = f"{width}x{height}"
        registration = (camera_config.get("video_registrations") or {}).get(key)
        if registration:
            matrix = np.asarray(registration["homography_reference_to_video"], dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(f"Invalid homography for {self.camera_id}/{key}")
            return matrix, str(registration.get("method") or "configured_homography")
        scale_x = width / max(1.0, float(reference_width))
        scale_y = height / max(1.0, float(reference_height))
        return (
            np.asarray([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]]),
            "fallback_axis_scaling_unvalidated",
        )

    def _build_masks(
        self,
        camera_config: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        reference_width, reference_height = self.reference_size
        target_width, target_height = self.frame_size
        raw_items = camera_config.get("instruments") or []
        nominal_reference: dict[str, np.ndarray] = {}
        for item in raw_items:
            mask = np.zeros((reference_height, reference_width), dtype=np.uint8)
            for polygon in item.get("polygons") or []:
                cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
            nominal_reference[str(item["id"])] = mask

        visible_reference: dict[str, np.ndarray] = {}
        for item in raw_items:
            instrument_id = str(item["id"])
            mask = nominal_reference[instrument_id].copy()
            for blocker in item.get("occluded_by") or []:
                mask[nominal_reference[str(blocker)] > 0] = 0
            visible_reference[instrument_id] = mask

        core_masks: dict[str, np.ndarray] = {}
        interaction_masks: dict[str, np.ndarray] = {}
        for item in raw_items:
            instrument_id = str(item["id"])
            visible = visible_reference[instrument_id]
            margin = max(0, int(item.get("interaction_margin_pixels") or 0))
            if margin:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
                interaction = cv2.dilate(visible, kernel)
            else:
                interaction = visible.copy()
            core_masks[instrument_id] = cv2.warpPerspective(
                visible,
                self.transform,
                (target_width, target_height),
                flags=cv2.INTER_NEAREST,
            )
            interaction_masks[instrument_id] = cv2.warpPerspective(
                interaction,
                self.transform,
                (target_width, target_height),
                flags=cv2.INTER_NEAREST,
            )
        return core_masks, interaction_masks

    def _distance_at(self, instrument_id: str, wrist: tuple[float, float]) -> float:
        width, height = self.frame_size
        x = int(round(min(max(wrist[0], 0.0), width - 1.0)))
        y = int(round(min(max(wrist[1], 0.0), height - 1.0)))
        outside = math.hypot(wrist[0] - x, wrist[1] - y)
        return float(self.distance_maps[instrument_id][y, x]) + outside

    def spatial_relations(
        self,
        *,
        wrist: tuple[float, float] | list[float],
        elbow: tuple[float, float] | list[float] | None = None,
    ) -> list[dict[str, Any]]:
        wrist_point = _point(wrist)
        if wrist_point is None:
            return []
        elbow_point = _point(elbow)
        width, height = self.frame_size
        diagonal = max(1.0, math.hypot(width, height))
        x, y = int(round(wrist_point[0])), int(round(wrist_point[1]))
        in_frame = 0 <= x < width and 0 <= y < height
        output: list[dict[str, Any]] = []
        for instrument in self.instruments:
            instrument_id = instrument.instrument_id
            core = self.core_masks[instrument_id]
            zone = self.interaction_masks[instrument_id]
            inside_core = bool(in_frame and core[y, x] > 0)
            inside_zone = bool(in_frame and zone[y, x] > 0)
            distance = self._distance_at(instrument_id, wrist_point)
            margin = max(1.0, float(instrument.interaction_margin_pixels))
            forearm_distance_limit = max(
                12.0,
                margin * float(self.temporal_rules.get("forearm_max_distance_factor") or 2.0),
            )
            forearm_intersects = bool(
                distance <= forearm_distance_limit
                and _segment_intersects_mask(elbow_point, wrist_point, zone)
            )
            if inside_core:
                score = 1.0
            elif inside_zone:
                score = 0.58 + 0.32 * max(0.0, 1.0 - distance / margin)
            elif forearm_intersects:
                score = 0.48
            else:
                score = max(0.0, 0.40 * (1.0 - distance / max(1.0, margin * 3.0)))
            output.append(
                {
                    "instrument_id": instrument_id,
                    "instrument_name": instrument.display_name,
                    "inside_core": inside_core,
                    "inside_interaction_zone": inside_zone,
                    "forearm_intersects_zone": forearm_intersects,
                    "spatial_hit": bool(inside_core or inside_zone or forearm_intersects),
                    "distance_to_core_pixels": round(distance, 4),
                    "distance_to_core_normalized": round(distance / diagonal, 7),
                    "score": round(min(1.0, score), 4),
                }
            )
        return sorted(
            output,
            key=lambda item: (
                not item["inside_core"],
                not item["inside_interaction_zone"],
                not item["forearm_intersects_zone"],
                float(item["distance_to_core_pixels"]),
                str(item["instrument_id"]),
            ),
        )

    @staticmethod
    def best_spatial_hit(relations: list[dict[str, Any]]) -> dict[str, Any] | None:
        return next((item for item in relations if item.get("spatial_hit")), None)

    def metadata(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "reference_size": list(self.reference_size),
            "frame_size": list(self.frame_size),
            "registration_method": self.registration_method,
            "instrument_count": len(self.instruments),
            "instrument_ids": [item.instrument_id for item in self.instruments],
        }


@dataclass
class _HandMemory:
    state: str = "idle"
    target_id: str | None = None
    candidate_since: float | None = None
    interaction_start: float | None = None
    interaction_id: str | None = None
    last_hit: float | None = None
    last_point: tuple[float, float] | None = None
    last_timestamp: float | None = None
    hit_count: int = 0
    strong_hit_count: int = 0
    path_normalized: float = 0.0
    approach_seen: bool = False
    max_score: float = 0.0
    evidence: set[str] = field(default_factory=set)


class FixedInstrumentInteractionEngine:
    """Temporal VRM state machine over pose-enriched person observations.

    Each person needs a stable ``track_id`` and a ``hands`` mapping. A hand
    observation contains ``point``, ``confidence`` and optionally
    ``elbow_point``. The public video adapter may derive these pose fields from
    the raw frame, so an external identity module does not need to provide them.
    """

    def __init__(self, regions: FixedInstrumentRegions, rules: dict[str, Any] | None = None) -> None:
        self.regions = regions
        self.rules = {**regions.temporal_rules, **(rules or {})}
        self.memory: dict[tuple[str, str], _HandMemory] = {}
        self.distance_history: dict[tuple[str, str, str], tuple[float, float]] = {}
        self.last_timestamp: float | None = None
        self.event_counter = 0
        self.transitions: list[dict[str, Any]] = []
        self.completed_interactions: list[dict[str, Any]] = []

    def _rule(self, key: str, default: float) -> float:
        return float(self.rules.get(key, default))

    @staticmethod
    def _clear(memory: _HandMemory) -> None:
        memory.state = "idle"
        memory.target_id = None
        memory.candidate_since = None
        memory.interaction_start = None
        memory.interaction_id = None
        memory.last_hit = None
        memory.last_point = None
        memory.last_timestamp = None
        memory.hit_count = 0
        memory.strong_hit_count = 0
        memory.path_normalized = 0.0
        memory.approach_seen = False
        memory.max_score = 0.0
        memory.evidence.clear()

    def _start_candidate(
        self,
        memory: _HandMemory,
        *,
        timestamp: float,
        relation: dict[str, Any],
        point: tuple[float, float],
        approach_seen: bool,
    ) -> None:
        memory.state = "candidate"
        memory.target_id = str(relation["instrument_id"])
        memory.candidate_since = timestamp
        memory.last_hit = timestamp
        memory.last_point = point
        memory.last_timestamp = timestamp
        memory.hit_count = 1
        memory.strong_hit_count = int(bool(relation.get("inside_core")))
        memory.path_normalized = 0.0
        memory.approach_seen = approach_seen
        memory.max_score = float(relation.get("score") or 0.0)
        memory.evidence = self._evidence(relation)

    @staticmethod
    def _evidence(relation: dict[str, Any]) -> set[str]:
        evidence: set[str] = set()
        if relation.get("inside_core"):
            evidence.add("wrist_inside_instrument_core")
        if relation.get("inside_interaction_zone"):
            evidence.add("wrist_inside_interaction_zone")
        if relation.get("forearm_intersects_zone"):
            evidence.add("forearm_intersects_interaction_zone")
        return evidence

    def _confirm(
        self,
        memory: _HandMemory,
        *,
        person_id: str,
        hand: str,
        timestamp: float,
    ) -> None:
        self.event_counter += 1
        memory.state = "interacting"
        memory.interaction_start = float(
            memory.candidate_since if memory.candidate_since is not None else timestamp
        )
        memory.interaction_id = f"fixed_interaction_{self.event_counter:04d}"
        instrument = self.regions.instrument_lookup[str(memory.target_id)]
        self.transitions.append(
            {
                "type": "interaction_started",
                "interaction_id": memory.interaction_id,
                "timestamp": round(timestamp, 6),
                "start_time": round(memory.interaction_start, 6),
                "person_id": person_id,
                "hand": hand,
                "instrument_id": instrument.instrument_id,
                "instrument_name": instrument.display_name,
            }
        )

    def _close(
        self,
        memory: _HandMemory,
        *,
        person_id: str,
        hand: str,
        timestamp: float,
    ) -> None:
        if memory.state != "interacting" or not memory.target_id:
            self._clear(memory)
            return
        end_time = float(memory.last_hit if memory.last_hit is not None else timestamp)
        start_time = float(memory.interaction_start if memory.interaction_start is not None else end_time)
        instrument = self.regions.instrument_lookup[memory.target_id]
        result = {
            "interaction_id": memory.interaction_id,
            "person_id": person_id,
            "hand": hand,
            "instrument_id": instrument.instrument_id,
            "instrument_name": instrument.display_name,
            "start_time": round(start_time, 6),
            "end_time": round(end_time, 6),
            "duration_seconds": round(max(0.0, end_time - start_time), 4),
            "confidence": round(memory.max_score, 4),
            "evidence": sorted(memory.evidence),
        }
        self.completed_interactions.append(result)
        self.transitions.append({"type": "interaction_ended", "timestamp": round(timestamp, 6), **result})
        self._clear(memory)

    def _miss(self, key: tuple[str, str], timestamp: float) -> None:
        memory = self.memory[key]
        if memory.last_hit is None:
            return
        elapsed = timestamp - memory.last_hit
        if memory.state == "candidate" and elapsed > self._rule("candidate_timeout_seconds", 0.8):
            self._clear(memory)
        elif memory.state == "interacting" and elapsed > self._rule("end_grace_seconds", 0.45):
            self._close(memory, person_id=key[0], hand=key[1], timestamp=timestamp)

    def _update_hand(
        self,
        *,
        person_id: str,
        hand: str,
        timestamp: float,
        point: tuple[float, float],
        elbow: tuple[float, float] | None,
    ) -> list[dict[str, Any]]:
        key = (person_id, hand)
        memory = self.memory.setdefault(key, _HandMemory())
        relations = self.regions.spatial_relations(wrist=point, elbow=elbow)
        previous_distances = {
            item["instrument_id"]: self.distance_history.get((person_id, hand, item["instrument_id"]))
            for item in relations
        }
        for item in relations:
            self.distance_history[(person_id, hand, item["instrument_id"])] = (
                timestamp,
                float(item["distance_to_core_normalized"]),
            )

        selected = None
        if memory.target_id:
            selected = next(
                (
                    item
                    for item in relations
                    if item["instrument_id"] == memory.target_id and item.get("spatial_hit")
                ),
                None,
            )
        selected = selected or self.regions.best_spatial_hit(relations)
        if selected is None:
            self._miss(key, timestamp)
            return relations

        target_id = str(selected["instrument_id"])
        previous_distance = previous_distances.get(target_id)
        approach_seen = bool(
            previous_distance
            and timestamp - previous_distance[0] <= self._rule("max_observation_gap_seconds", 0.35)
            and previous_distance[1] - float(selected["distance_to_core_normalized"])
            >= self._rule("min_approach_normalized", 0.0025)
        )

        if memory.target_id != target_id:
            if memory.state == "interacting":
                self._miss(key, timestamp)
                if memory.state == "interacting":
                    return relations
            self._clear(memory)
            self._start_candidate(
                memory,
                timestamp=timestamp,
                relation=selected,
                point=point,
                approach_seen=approach_seen,
            )
        elif memory.last_hit is not None and (
            timestamp - memory.last_hit > self._rule("max_observation_gap_seconds", 0.35)
            and memory.state == "candidate"
        ):
            self._clear(memory)
            self._start_candidate(
                memory,
                timestamp=timestamp,
                relation=selected,
                point=point,
                approach_seen=approach_seen,
            )
        else:
            diagonal = max(1.0, math.hypot(*self.regions.frame_size))
            if memory.last_point is not None and memory.last_timestamp is not None:
                if timestamp - memory.last_timestamp <= self._rule("max_observation_gap_seconds", 0.35):
                    memory.path_normalized += math.dist(memory.last_point, point) / diagonal
            memory.last_point = point
            memory.last_timestamp = timestamp
            memory.last_hit = timestamp
            memory.hit_count += 1
            memory.strong_hit_count += int(bool(selected.get("inside_core")))
            memory.approach_seen = memory.approach_seen or approach_seen
            memory.max_score = max(memory.max_score, float(selected.get("score") or 0.0))
            memory.evidence.update(self._evidence(selected))

        if memory.state == "candidate":
            duration = timestamp - float(
                memory.candidate_since if memory.candidate_since is not None else timestamp
            )
            strong_confirmed = memory.strong_hit_count >= int(
                self._rule("min_strong_core_samples", 2)
            )
            ordinary_confirmed = bool(
                memory.hit_count >= int(self._rule("min_hit_samples", 3))
                and duration >= self._rule("min_hit_duration_seconds", 0.2)
                and (
                    memory.approach_seen
                    or memory.path_normalized >= self._rule("min_path_normalized", 0.006)
                    or duration >= self._rule("min_dwell_seconds", 0.45)
                )
            )
            if strong_confirmed or ordinary_confirmed:
                self._confirm(memory, person_id=person_id, hand=hand, timestamp=timestamp)
        return relations

    def update(self, *, timestamp: float, people: list[dict[str, Any]]) -> dict[str, Any]:
        timestamp = float(timestamp)
        if self.last_timestamp is not None and timestamp < self.last_timestamp - 1e-6:
            raise ValueError("FixedInstrumentInteractionEngine requires monotonically increasing timestamps")
        self.last_timestamp = timestamp
        observed_keys: set[tuple[str, str]] = set()
        observed_people: set[str] = set()
        relation_records: list[dict[str, Any]] = []
        min_confidence = self._rule("min_hand_confidence", 0.12)
        for person in people:
            person_id = str(person.get("track_id") or person.get("person_id") or "")
            if not person_id:
                continue
            observed_people.add(person_id)
            pose_joints = person.get("pose_joints") or {}
            for hand, value in (person.get("hands") or {}).items():
                point = _point((value or {}).get("point"))
                confidence = float((value or {}).get("confidence") or 0.0)
                if point is None or confidence < min_confidence or not (value or {}).get("observed", True):
                    continue
                elbow = _point((value or {}).get("elbow_point"))
                if elbow is None:
                    elbow = _point((pose_joints.get(f"{hand}_elbow") or {}).get("point"))
                key = (person_id, str(hand))
                observed_keys.add(key)
                relations = self._update_hand(
                    person_id=person_id,
                    hand=str(hand),
                    timestamp=timestamp,
                    point=point,
                    elbow=elbow,
                )
                relation_records.extend(
                    {"person_id": person_id, "hand": str(hand), **item} for item in relations
                )
        for key in list(self.memory):
            if key not in observed_keys:
                self._miss(key, timestamp)

        person_ids = observed_people | {
            key[0] for key, memory in self.memory.items() if memory.state != "idle"
        }
        person_states: list[dict[str, Any]] = []
        for person_id in sorted(person_ids):
            interactions: list[dict[str, Any]] = []
            candidates: list[dict[str, Any]] = []
            for (memory_person, hand), memory in self.memory.items():
                if memory_person != person_id or memory.state == "idle" or not memory.target_id:
                    continue
                instrument = self.regions.instrument_lookup[memory.target_id]
                item = {
                    "hand": hand,
                    "instrument_id": instrument.instrument_id,
                    "instrument_name": instrument.display_name,
                    "confidence": round(memory.max_score, 4),
                    "evidence": sorted(memory.evidence),
                }
                if memory.state == "interacting":
                    item["interaction_id"] = memory.interaction_id
                    item["start_time"] = round(
                        float(
                            memory.interaction_start
                            if memory.interaction_start is not None
                            else timestamp
                        ),
                        6,
                    )
                    interactions.append(item)
                else:
                    item["candidate_since"] = round(
                        float(
                            memory.candidate_since
                            if memory.candidate_since is not None
                            else timestamp
                        ),
                        6,
                    )
                    candidates.append(item)
            person_states.append(
                {
                    "person_id": person_id,
                    "interacting": bool(interactions),
                    "state": "interacting" if interactions else "not_interacting",
                    "interactions": interactions,
                    "candidates": candidates,
                }
            )
        return {
            "timestamp": round(timestamp, 6),
            "camera_id": self.regions.camera_id,
            "people": person_states,
            "relations": relation_records,
        }

    def finalize(self, timestamp: float | None = None) -> dict[str, Any]:
        final_time = float(timestamp if timestamp is not None else (self.last_timestamp or 0.0))
        for (person_id, hand), memory in self.memory.items():
            if memory.state == "interacting":
                self._close(memory, person_id=person_id, hand=hand, timestamp=final_time)
            elif memory.state == "candidate":
                self._clear(memory)
        return self.export()

    def export(self) -> dict[str, Any]:
        return {
            "engine": "Fixed instrument wrist/forearm temporal VRM",
            "camera": self.regions.metadata(),
            "rules": self.rules,
            "transitions": self.transitions,
            "interactions": self.completed_interactions,
            "summary": {
                "interaction_count": len(self.completed_interactions),
                "person_ids": sorted(
                    {item["person_id"] for item in self.completed_interactions}
                ),
                "instrument_ids": sorted(
                    {item["instrument_id"] for item in self.completed_interactions}
                ),
            },
        }
