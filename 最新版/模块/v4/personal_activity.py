from __future__ import annotations

import math
from typing import Any


DEFAULT_PERSONAL_OBJECT_RULES = {
    "personal_object_filter_enabled": True,
    "personal_object_max_hand_separation_width_ratio": 0.30,
    "personal_object_max_hand_separation_height_ratio": 0.10,
    "personal_object_max_hand_height_difference_ratio": 0.10,
    "personal_object_min_wrist_height_ratio": 0.36,
    "personal_object_max_wrist_height_ratio": 0.62,
    "personal_object_max_gap_seconds": 0.40,
    "personal_object_min_duration_seconds": 1.00,
    "personal_object_min_interaction_overlap_seconds": 1.00,
    "personal_object_min_interaction_overlap_ratio": 0.55,
    "external_interface_publish_delay_seconds": 2.80,
}


def personal_object_rules(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**DEFAULT_PERSONAL_OBJECT_RULES, **(overrides or {})}


def is_two_hand_personal_object_posture(
    person: dict[str, Any],
    *,
    rules: dict[str, Any] | None = None,
) -> bool:
    """Return whether the pose resembles sustained two-hand personal-object use.

    This is deliberately phrased as a posture cue rather than a phone
    detector: a small phone is not reliably detectable in the current CCTV
    resolution. Temporal confirmation is required by the caller.
    """
    selected = personal_object_rules(rules)
    if not bool(selected["personal_object_filter_enabled"]):
        return False
    bbox = person.get("bbox")
    hands = person.get("hands") or {}
    left = hands.get("left") or {}
    right = hands.get("right") or {}
    left_point = left.get("point")
    right_point = right.get("point")
    if not bbox or not left_point or not right_point:
        return False
    if not left.get("observed", True) or not right.get("observed", True):
        return False
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    separation = math.dist(left_point, right_point)
    left_y = (float(left_point[1]) - float(bbox[1])) / height
    right_y = (float(right_point[1]) - float(bbox[1])) / height
    minimum_y = float(selected["personal_object_min_wrist_height_ratio"])
    maximum_y = float(selected["personal_object_max_wrist_height_ratio"])
    return bool(
        separation / width
        <= float(selected["personal_object_max_hand_separation_width_ratio"])
        and separation / height
        <= float(selected["personal_object_max_hand_separation_height_ratio"])
        and abs(float(left_point[1]) - float(right_point[1])) / height
        <= float(selected["personal_object_max_hand_height_difference_ratio"])
        and minimum_y <= left_y <= maximum_y
        and minimum_y <= right_y <= maximum_y
    )


def build_personal_object_intervals(
    frame_records: list[dict[str, Any]],
    *,
    rules: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    selected = personal_object_rules(rules)
    maximum_gap = float(selected["personal_object_max_gap_seconds"])
    minimum_duration = float(selected["personal_object_min_duration_seconds"])
    active: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []

    def close(person_id: str) -> None:
        memory = active.pop(person_id, None)
        if memory is None:
            return
        duration = float(memory["last_time"]) - float(memory["start_time"])
        if duration + 1e-6 < minimum_duration:
            return
        completed.append(
            {
                "person_id": person_id,
                "start_time": round(float(memory["start_time"]), 6),
                "end_time": round(float(memory["last_time"]), 6),
                "duration_seconds": round(max(0.0, duration), 4),
                "sample_count": int(memory["sample_count"]),
                "type": "possible_personal_handheld_activity",
                "evidence": [
                    "two_hands_close_together",
                    "both_wrists_held_at_mid_torso",
                    "posture_sustained_over_time",
                ],
            }
        )

    for record in sorted(frame_records, key=lambda item: float(item["timestamp"])):
        timestamp = float(record["timestamp"])
        seen: set[str] = set()
        for person in record.get("people") or []:
            person_id = str(person.get("track_id") or person.get("person_id") or "")
            if not person_id or not is_two_hand_personal_object_posture(person, rules=selected):
                continue
            seen.add(person_id)
            memory = active.get(person_id)
            if memory is None or timestamp - float(memory["last_time"]) > maximum_gap + 1e-6:
                close(person_id)
                active[person_id] = {
                    "start_time": timestamp,
                    "last_time": timestamp,
                    "sample_count": 1,
                }
            else:
                memory["last_time"] = timestamp
                memory["sample_count"] += 1
        for person_id, memory in list(active.items()):
            if person_id not in seen and timestamp - float(memory["last_time"]) > maximum_gap + 1e-6:
                close(person_id)
    for person_id in list(active):
        close(person_id)
    return sorted(completed, key=lambda item: (item["start_time"], item["person_id"]))


def suppress_personal_object_false_interactions(
    interactions: list[dict[str, Any]],
    personal_object_intervals: list[dict[str, Any]],
    *,
    rules: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = personal_object_rules(rules)
    minimum_overlap = float(selected["personal_object_min_interaction_overlap_seconds"])
    minimum_ratio = float(selected["personal_object_min_interaction_overlap_ratio"])
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for interaction in interactions:
        start = float(interaction["start_time"])
        end = float(interaction["end_time"])
        duration = max(0.2, end - start)
        matching = None
        for interval in personal_object_intervals:
            if str(interval["person_id"]) != str(interaction["person_id"]):
                continue
            overlap = max(
                0.0,
                min(end, float(interval["end_time"]))
                - max(start, float(interval["start_time"])),
            )
            if overlap >= minimum_overlap and overlap / duration >= minimum_ratio:
                matching = (interval, overlap)
                break
        if matching is None:
            kept.append(interaction)
            continue
        interval, overlap = matching
        suppressed.append(
            {
                **interaction,
                "suppressed": True,
                "suppression_reason": "sustained_two_hand_personal_object_posture",
                "overlap_seconds": round(overlap, 4),
                "personal_object_interval": interval,
            }
        )
    return kept, suppressed
