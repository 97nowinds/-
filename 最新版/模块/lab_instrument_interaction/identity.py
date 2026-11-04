from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any, Iterable


IDENTITY_ASSOCIATION_VERSION = "1.0"


def _as_bbox(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} 必须是 [x1, y1, x2, y2]")
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 中必须全部是数字") from exc
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError(f"{field} 中不能包含无穷大或 NaN")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{field} 的右下角必须位于左上角右下方")
    return bbox


def _read_identity_payload(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        return json.loads(path.read_text(encoding="utf-8"))
    return value


def normalize_identity_timeline(
    value: Any,
    *,
    video_fps: float | None = None,
) -> list[dict[str, Any]]:
    """Validate the public identity timeline and return a canonical form."""
    payload = _read_identity_payload(value)
    if isinstance(payload, dict):
        payload = payload.get("frames", payload.get("observations"))
    if not isinstance(payload, list):
        raise ValueError("identity_timeline 必须是列表，或包含 frames 列表的对象")

    output: list[dict[str, Any]] = []
    for frame_index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"identity_timeline[{frame_index}] 必须是对象")
        timestamp = item.get("timestamp", item.get("timestamp_seconds"))
        if timestamp is None and item.get("frame_index") is not None:
            if not video_fps or video_fps <= 0:
                raise ValueError("使用 frame_index 时必须提供有效 video_fps")
            timestamp = float(item["frame_index"]) / float(video_fps)
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"identity_timeline[{frame_index}] 缺少有效 timestamp"
            ) from exc
        if timestamp < 0 or not math.isfinite(timestamp):
            raise ValueError(
                f"identity_timeline[{frame_index}].timestamp 必须是非负数"
            )

        people = item.get("persons", item.get("people", []))
        if not isinstance(people, list):
            raise ValueError(
                f"identity_timeline[{frame_index}].persons 必须是列表"
            )
        normalized_people: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for person_index, person in enumerate(people):
            if not isinstance(person, dict):
                raise ValueError(
                    f"identity_timeline[{frame_index}].persons[{person_index}] 必须是对象"
                )
            person_id = str(
                person.get("person_id", person.get("track_id", ""))
            ).strip()
            if not person_id:
                raise ValueError(
                    f"identity_timeline[{frame_index}].persons[{person_index}] 缺少 person_id"
                )
            if person_id in seen_ids:
                raise ValueError(
                    f"同一时间戳出现重复 person_id：{person_id}"
                )
            seen_ids.add(person_id)
            person_name = person.get("person_name", person.get("name"))
            normalized_people.append(
                {
                    "person_id": person_id,
                    "person_name": (
                        str(person_name).strip() if person_name is not None else None
                    ),
                    "bbox": _as_bbox(
                        person.get("bbox"),
                        field=(
                            f"identity_timeline[{frame_index}].persons"
                            f"[{person_index}].bbox"
                        ),
                    ),
                    "confidence": (
                        float(person["confidence"])
                        if person.get("confidence") is not None
                        else None
                    ),
                }
            )
        output.append(
            {
                "timestamp": round(timestamp, 6),
                "persons": normalized_people,
            }
        )
    output.sort(key=lambda item: float(item["timestamp"]))
    return output


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_match_score(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = _area(a)
    area_b = _area(b)
    union = area_a + area_b - intersection
    iou = intersection / max(1.0, union)
    intersection_over_smaller = intersection / max(1.0, min(area_a, area_b))
    center_a = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    center_b = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    diagonal = max(1.0, math.hypot(b[2] - b[0], b[3] - b[1]))
    center_similarity = max(
        0.0,
        1.0 - math.hypot(
            center_a[0] - center_b[0], center_a[1] - center_b[1]
        )
        / diagonal,
    )
    return max(iou, 0.85 * intersection_over_smaller, 0.55 * center_similarity)


def _nearest_identity_frame(
    timeline: list[dict[str, Any]],
    timestamps: list[float],
    timestamp: float,
    *,
    tolerance_seconds: float,
) -> dict[str, Any] | None:
    if not timeline:
        return None
    position = bisect.bisect_left(timestamps, timestamp)
    candidates = [
        timeline[index]
        for index in (position - 1, position)
        if 0 <= index < len(timeline)
    ]
    nearest = min(
        candidates,
        key=lambda item: abs(float(item["timestamp"]) - timestamp),
    )
    if abs(float(nearest["timestamp"]) - timestamp) > tolerance_seconds:
        return None
    return nearest


def _iter_rule_person_boxes(
    rule_result: dict[str, Any],
    *,
    internal_person_id: str,
    start_time: float,
    end_time: float,
) -> Iterable[tuple[float, list[float], float]]:
    for frame in rule_result.get("frames") or []:
        timestamp = float(frame.get("timestamp") or 0.0)
        if timestamp < start_time - 0.35 or timestamp > end_time + 0.35:
            continue
        for person in frame.get("people") or []:
            if str(person.get("track_id")) != internal_person_id:
                continue
            yield timestamp, _as_bbox(person.get("bbox"), field="internal bbox"), 1.0


def _iter_vlm_person_boxes(
    event: dict[str, Any],
) -> Iterable[tuple[float, list[float], float]]:
    for observation in event.get("target_person_observations") or []:
        if observation.get("bbox") is None:
            continue
        yield (
            float(observation.get("timestamp") or 0.0),
            _as_bbox(observation["bbox"], field="VLM target bbox"),
            2.0,
        )


def attach_external_identities(
    events: list[dict[str, Any]],
    *,
    identity_timeline: Any,
    rule_result: dict[str, Any],
    video_fps: float | None = None,
    timestamp_tolerance_seconds: float = 0.5,
    minimum_box_match_score: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach external IDs to decisions without changing interaction booleans."""
    timeline = normalize_identity_timeline(
        identity_timeline,
        video_fps=video_fps,
    )
    if not timeline:
        return (
            [
                {
                    **event,
                    "internal_person_id": str(
                        event.get("person_id") or "person_01"
                    ),
                    "person_name": event.get("person_name"),
                    "identity_source": "internal_tracker",
                    "identity_association": {
                        "matched": False,
                        "reason": "identity_timeline_not_provided",
                    },
                }
                for event in events
            ],
            {
                "version": IDENTITY_ASSOCIATION_VERSION,
                "source": "internal_tracker",
                "input_frame_count": 0,
                "matched_event_count": 0,
                "unmatched_event_count": len(events),
            },
        )

    timestamps = [float(item["timestamp"]) for item in timeline]
    enriched: list[dict[str, Any]] = []
    matched_count = 0
    for event in events:
        internal_person_id = str(event.get("person_id") or "person_01")
        start_time = float(event.get("start_time") or 0.0)
        end_time = float(event.get("end_time") or start_time)
        observations = list(_iter_vlm_person_boxes(event))
        observations.extend(
            _iter_rule_person_boxes(
                rule_result,
                internal_person_id=internal_person_id,
                start_time=start_time,
                end_time=end_time,
            )
        )
        votes: dict[str, dict[str, Any]] = {}
        for timestamp, internal_bbox, source_weight in observations:
            identity_frame = _nearest_identity_frame(
                timeline,
                timestamps,
                timestamp,
                tolerance_seconds=timestamp_tolerance_seconds,
            )
            if identity_frame is None:
                continue
            ranked: list[tuple[float, dict[str, Any]]] = []
            for person in identity_frame["persons"]:
                ranked.append(
                    (
                        _box_match_score(internal_bbox, person["bbox"]),
                        person,
                    )
                )
            if not ranked:
                continue
            score, person = max(ranked, key=lambda item: item[0])
            if score < minimum_box_match_score:
                continue
            person_id = str(person["person_id"])
            vote = votes.setdefault(
                person_id,
                {
                    "person_id": person_id,
                    "person_name": person.get("person_name"),
                    "score_sum": 0.0,
                    "support_count": 0,
                    "maximum_box_match_score": 0.0,
                },
            )
            vote["score_sum"] += float(score) * source_weight
            vote["support_count"] += 1
            vote["maximum_box_match_score"] = max(
                float(vote["maximum_box_match_score"]), float(score)
            )
            if person.get("person_name"):
                vote["person_name"] = person["person_name"]

        selected = (
            max(
                votes.values(),
                key=lambda item: (
                    float(item["score_sum"]),
                    int(item["support_count"]),
                    str(item["person_id"]),
                ),
            )
            if votes
            else None
        )
        if selected is None:
            enriched.append(
                {
                    **event,
                    "internal_person_id": internal_person_id,
                    "person_name": None,
                    "identity_source": "internal_tracker",
                    "identity_association": {
                        "matched": False,
                        "reason": "no_external_bbox_match",
                        "observation_count": len(observations),
                    },
                }
            )
            continue

        matched_count += 1
        enriched.append(
            {
                **event,
                "internal_person_id": internal_person_id,
                "person_id": selected["person_id"],
                "person_name": selected.get("person_name"),
                "identity_source": "external_person_recognition",
                "identity_association": {
                    "matched": True,
                    "support_count": int(selected["support_count"]),
                    "score_sum": round(float(selected["score_sum"]), 4),
                    "maximum_box_match_score": round(
                        float(selected["maximum_box_match_score"]), 4
                    ),
                    "candidate_person_ids": sorted(votes),
                },
            }
        )

    return enriched, {
        "version": IDENTITY_ASSOCIATION_VERSION,
        "source": "external_person_recognition",
        "input_frame_count": len(timeline),
        "matched_event_count": matched_count,
        "unmatched_event_count": len(events) - matched_count,
        "timestamp_tolerance_seconds": timestamp_tolerance_seconds,
        "minimum_box_match_score": minimum_box_match_score,
    }
