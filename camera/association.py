"""Deterministic, one-to-one association for small multi-camera batches."""

from dataclasses import dataclass, field
import math

import numpy as np


ASSOCIATION_ALGORITHM = "hungarian-mtmc-v1"


@dataclass
class AssociationObservation:
    camera_id: str
    local_id: int
    monotonic_time: float
    gallery: object = None
    appearance: object = None
    position: dict = None
    direction: tuple = None
    person_class: str = "person"
    geometry_trusted: bool = True
    image_anchor: dict = None
    excluded_global_ids: set = field(default_factory=set)
    handoff_exit_grace: dict = field(default_factory=dict)
    clothing_tone: str = None


@dataclass
class AssociationTarget:
    global_id: str
    camera_id: str
    last_seen: float
    gallery: object = None
    appearance: object = None
    position: dict = None
    direction: tuple = None
    active_camera_ids: set = field(default_factory=set)
    person_id: str = None
    geometry_trusted: bool = True
    source_local_id: int = None
    image_anchor: dict = None
    recent_exit_gates: dict = field(default_factory=dict)
    clothing_tone: str = None


def _cosine(first, second):
    if first is None or second is None:
        return None
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-8 else None


def hungarian_maximize(scores):
    """Return row/column pairs maximizing a rectangular score matrix."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        return []
    rows, columns = matrix.shape
    size = max(rows, columns)
    finite = matrix[np.isfinite(matrix)]
    ceiling = float(np.max(finite)) if finite.size else 0.0
    padded = np.full((size, size), ceiling, dtype=np.float64)
    padded[:rows, :columns] = np.where(
        np.isfinite(matrix), ceiling - matrix, 1e6
    )

    # Shortest augmenting path implementation of the Hungarian algorithm.
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = padded[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = []
    for column in range(1, size + 1):
        row = p[column] - 1
        col = column - 1
        if row < rows and col < columns and np.isfinite(matrix[row, col]):
            pairs.append((row, col))
    return sorted(pairs)


class BatchAssociator:
    def __init__(self, config, transitions=None, calibration_status="uncalibrated"):
        self.config = config
        self.reid = config["reid"]
        self.association = config["association"]
        self.calibration = config["calibration"]
        self.transitions = list(transitions or [])
        self.calibration_status = calibration_status
        for transition in self.transitions:
            if "simultaneous_reid_threshold" in transition:
                threshold = float(transition["simultaneous_reid_threshold"])
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError(
                        "camera transition simultaneous_reid_threshold must be between 0 and 1"
                    )
            for camera_id, gate in (transition.get("image_handoff_gates") or {}).items():
                if not isinstance(camera_id, str) or not isinstance(gate, dict):
                    raise ValueError("image_handoff_gates must map camera ids to objects")
                for direction in ("entry", "exit"):
                    value = gate.get(direction)
                    if value is None:
                        continue
                    points = value.get("points") if isinstance(value, dict) else None
                    if not isinstance(points, list) or len(points) < 3:
                        raise ValueError("handoff gate must contain at least three normalized points")
                    for point in points:
                        if (
                            not isinstance(point, (list, tuple))
                            or len(point) != 2
                            or not all(isinstance(item, (int, float)) and 0.0 <= item <= 1.0 for item in point)
                        ):
                            raise ValueError("handoff gate points must be normalized x/y pairs")

    def _transition(self, source, target):
        if source == target:
            return {
                "max_gap_seconds": self.association["same_camera_relink_seconds"],
                "same_camera": True,
            }
        for transition in self.transitions:
            direct = transition.get("from") == source and transition.get("to") == target
            reverse = (
                transition.get("bidirectional", False)
                and transition.get("from") == target
                and transition.get("to") == source
            )
            if direct or reverse:
                return transition
        return None

    @staticmethod
    def _point_in_polygon(point, points):
        if point is None:
            return False
        x, y = float(point.get("x", -1.0)), float(point.get("y", -1.0))
        inside = False
        previous = points[-1]
        for current in points:
            current_x, current_y = float(current[0]), float(current[1])
            previous_x, previous_y = float(previous[0]), float(previous[1])
            crosses = (current_y > y) != (previous_y > y)
            if crosses and x < (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x:
                inside = not inside
            previous = current
        return inside

    def _inside_handoff_gate(self, transition, camera_id, direction, anchor):
        gate = (transition.get("image_handoff_gates") or {}).get(camera_id, {}).get(direction)
        if gate is None:
            return True
        return self._point_in_polygon(anchor, gate["points"])

    def exit_gate_destinations(self, camera_id, anchor):
        """Return adjacent cameras whose source exit was just observed."""
        destinations = set()
        for transition in self.transitions:
            if transition.get("from") == camera_id:
                other = transition.get("to")
            elif transition.get("bidirectional", False) and transition.get("to") == camera_id:
                other = transition.get("from")
            else:
                continue
            if other and self._inside_handoff_gate(
                transition, camera_id, "exit", anchor
            ):
                destinations.add(other)
        return destinations

    def _gallery_score(self, observation, target):
        if observation.gallery is not None and target.gallery is not None:
            result = observation.gallery.compare(target.gallery)
            if result.get("score") is not None:
                return float(result["score"]), result
        score = _cosine(observation.appearance, target.appearance)
        return score, {
            "score": score,
            "best_score": score,
            "pair_count": 1 if score is not None else 0,
            "top_k": 1 if score is not None else 0,
            "quality": 0.0,
        }

    def evaluate(self, observation, target, stream_skew_seconds=0.0):
        evaluation = {
            "source_camera": target.camera_id,
            "target_camera": observation.camera_id,
            "source_global_id": target.global_id,
            "source_local_id": target.source_local_id,
            "target_local_id": observation.local_id,
            "algorithm_version": ASSOCIATION_ALGORITHM,
        }
        if observation.person_class != "person":
            return {**evaluation, "eligible": False, "reason": "class_not_person"}
        if target.global_id in observation.excluded_global_ids:
            return {**evaluation, "eligible": False, "reason": "previously_distinct_same_camera_people"}
        if (
            observation.clothing_tone in {"dark", "light"}
            and target.clothing_tone in {"dark", "light"}
            and observation.clothing_tone != target.clothing_tone
        ):
            return {
                **evaluation,
                "eligible": False,
                "reason": "clothing_tone_mismatch",
                "source_clothing_tone": target.clothing_tone,
                "target_clothing_tone": observation.clothing_tone,
            }
        transition = self._transition(target.camera_id, observation.camera_id)
        if transition is None:
            return {**evaluation, "eligible": False, "reason": "topology_rejected"}
        source_in_gate = self._inside_handoff_gate(
            transition, target.camera_id, "exit", target.image_anchor
        )
        recent_exit_at = (target.recent_exit_gates or {}).get(
            observation.camera_id
        )
        source_exit_recent = bool(
            recent_exit_at is not None
            and 0.0
            <= float(observation.monotonic_time) - float(recent_exit_at)
            <= float(self.association["handoff_exit_memory_seconds"])
        )
        source_exit_grace = bool(
            target.global_id in observation.handoff_exit_grace
            and float(observation.monotonic_time)
            <= float(observation.handoff_exit_grace[target.global_id])
        )
        source_exit_recent = source_exit_recent or source_exit_grace
        target_in_gate = self._inside_handoff_gate(
            transition, observation.camera_id, "entry", observation.image_anchor
        )
        evaluation.update(
            source_in_handoff_gate=source_in_gate,
            source_exit_gate_recent=source_exit_recent,
            source_exit_gate_grace=source_exit_grace,
            target_in_handoff_gate=target_in_gate,
        )
        if not source_in_gate and not source_exit_recent:
            return {**evaluation, "eligible": False, "reason": "source_not_in_handoff_exit_gate"}
        if not target_in_gate:
            return {**evaluation, "eligible": False, "reason": "target_not_in_handoff_entry_gate"}
        if observation.camera_id in target.active_camera_ids:
            return {**evaluation, "eligible": False, "reason": "simultaneous_same_camera_conflict"}
        simultaneous = bool(target.active_camera_ids)
        validated_overlap = bool(transition.get("simultaneous_overlap_validated", False))
        if simultaneous and self.calibration_status != "formal" and not validated_overlap:
            return {
                **evaluation,
                "eligible": False,
                "reason": "simultaneous_cross_camera_without_formal_overlap",
            }
        age = float(observation.monotonic_time) - float(target.last_seen)
        max_gap = float(transition.get("max_gap_seconds", 0.0))
        if age < 0.0:
            if simultaneous and abs(age) <= float(self.calibration["max_stream_skew_seconds"]):
                age = 0.0
            else:
                return {**evaluation, "eligible": False, "reason": "negative_time_gap"}
        if age > max_gap:
            return {**evaluation, "eligible": False, "reason": "candidate_expired", "time_gap_seconds": age}
        reid_score, reid_details = self._gallery_score(observation, target)
        if reid_score is None or reid_score < self.reid["medium_similarity"]:
            return {
                **evaluation,
                "eligible": False,
                "reason": "reid_below_medium",
                "reid_score": reid_score,
                "reid_details": reid_details,
                "time_gap_seconds": age,
            }
        simultaneous_reid_threshold = float(
            transition.get("simultaneous_reid_threshold", self.reid["high_similarity"])
        )
        if (
            simultaneous
            and self.calibration_status != "formal"
            and validated_overlap
            and reid_score < simultaneous_reid_threshold
        ):
            return {
                **evaluation,
                "eligible": False,
                "reason": "validated_overlap_reid_below_high",
                "reid_score": reid_score,
                "reid_details": reid_details,
                "time_gap_seconds": age,
                "simultaneous_overlap_validated": True,
                "simultaneous_reid_threshold": simultaneous_reid_threshold,
            }
        components = {"reid": float(np.clip(reid_score, 0.0, 1.0))}
        components["time"] = float(np.clip(1.0 - age / max(max_gap, 1e-6), 0.0, 1.0))
        geometry_used = False
        if (
            self.calibration_status == "formal"
            and float(stream_skew_seconds) <= self.calibration["max_stream_skew_seconds"]
            and observation.geometry_trusted
            and target.geometry_trusted
            and observation.position is not None
            and target.position is not None
        ):
            distance = math.hypot(
                observation.position["x"] - target.position["x"],
                observation.position["y"] - target.position["y"],
            )
            maximum = float(transition.get("max_distance_m", self.calibration["fusion_max_distance_m"]))
            if distance > maximum:
                return {
                    **evaluation,
                    "eligible": False,
                    "reason": "geometry_distance_rejected",
                    "reid_score": reid_score,
                    "time_gap_seconds": age,
                    "geometry_distance_m": distance,
                }
            components["geometry"] = float(np.clip(1.0 - distance / maximum, 0.0, 1.0))
            geometry_used = True
        direction_score = _cosine(observation.direction, target.direction)
        if direction_score is not None:
            components["motion"] = (direction_score + 1.0) * 0.5
        weights = self.association["weights"]
        denominator = sum(weights[name] for name in components)
        final_score = sum(components[name] * weights[name] for name in components) / denominator
        return {
            **evaluation,
            "eligible": True,
            "reason": "candidate_scored",
            "reid_score": reid_score,
            "reid_details": reid_details,
            "time_gap_seconds": age,
            "time_score": components.get("time"),
            "geometry_score": components.get("geometry"),
            "motion_score": components.get("motion"),
            "geometry_used": geometry_used,
            "simultaneous_overlap_validated": validated_overlap,
            "final_score": round(float(final_score), 6),
        }

    def associate(self, observations, targets, stream_skew_seconds=0.0):
        observations = sorted(observations, key=lambda item: (item.camera_id, str(item.local_id)))
        targets = sorted(targets, key=lambda item: item.global_id)
        evaluations = []
        matrix = np.full((len(observations), len(targets)), -np.inf, dtype=np.float64)
        by_pair = {}
        for row, observation in enumerate(observations):
            for column, target in enumerate(targets):
                result = self.evaluate(observation, target, stream_skew_seconds)
                evaluations.append(result)
                by_pair[(row, column)] = result
                if result.get("eligible"):
                    matrix[row, column] = result["final_score"]

        assigned = {row: column for row, column in hungarian_maximize(matrix)}
        decisions = []
        for row, observation in enumerate(observations):
            column = assigned.get(row)
            if column is None:
                decisions.append(
                    {
                        "camera_id": observation.camera_id,
                        "local_id": observation.local_id,
                        "global_id": None,
                        "candidate_global_id": None,
                        "confidence": "low",
                        "accepted": False,
                        "reason": "no_eligible_candidate",
                        "algorithm_version": ASSOCIATION_ALGORITHM,
                    }
                )
                continue
            result = by_pair[(row, column)]
            assigned_score = float(matrix[row, column])
            row_competitors = [
                float(value)
                for other_column, value in enumerate(matrix[row])
                if other_column != column and np.isfinite(value)
            ]
            column_competitors = [
                float(matrix[other_row, column])
                for other_row in range(len(observations))
                if other_row != row and np.isfinite(matrix[other_row, column])
            ]
            row_margin = (
                assigned_score - max(row_competitors) if row_competitors else 1.0
            )
            column_margin = (
                assigned_score - max(column_competitors) if column_competitors else 1.0
            )
            margin = min(row_margin, column_margin)
            high = bool(
                result["final_score"] >= self.association["high_confidence_score"]
                and result["reid_score"] >= self.reid["high_similarity"]
                and margin >= self.reid["match_margin"]
            )
            medium = bool(
                result["final_score"] >= self.association["medium_confidence_score"]
                and result["reid_score"] >= self.reid["medium_similarity"]
            )
            confidence = "high" if high else "medium" if medium else "low"
            decisions.append(
                {
                    **result,
                    "camera_id": observation.camera_id,
                    "local_id": observation.local_id,
                    "global_id": targets[column].global_id if high else None,
                    "candidate_global_id": targets[column].global_id,
                    "confidence": confidence,
                    "accepted": high,
                    "reason": "high_confidence_match" if high else "pending_more_evidence" if medium else "score_below_threshold",
                    "match_margin": round(float(margin), 6),
                }
            )
        return decisions, evaluations
