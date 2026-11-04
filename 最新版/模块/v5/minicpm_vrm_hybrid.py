from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import video_utils

from settings import PROJECT_ROOT
from v4.fixed_instrument_video import analyze_fixed_instrument_video
from v5.minicpm_multi_instrument_realtime import (
    MultiInstrumentPersonDetector,
    _decode_video,
    _render_multi_result_video,
    analyze_video_multi_realtime,
)
from v5.minicpm_interaction_judge import MiniCPMInteractionJudge


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "minicpm_v46_vrm_hybrid"


def _strong_rule_event(event: dict[str, Any]) -> tuple[bool, list[str]]:
    evidence = set(event.get("evidence") or [])
    hands = set(event.get("hands") or [])
    duration = float(event.get("duration_seconds") or 0.0)
    confidence = float(event.get("confidence") or 0.0)
    reasons: list[str] = []
    if confidence < 0.90:
        reasons.append("rule_confidence_below_0.90")
    if duration < 0.80:
        reasons.append("rule_duration_below_0.80s")
    if "wrist_inside_instrument_core" not in evidence:
        reasons.append("no_wrist_inside_instrument_core")
    if not (
        "wrist_inside_interaction_zone" in evidence
        or "forearm_intersects_interaction_zone" in evidence
    ):
        reasons.append("no_zone_or_forearm_evidence")
    if len(hands) < 2 and duration < 1.20:
        reasons.append("single_hand_evidence_too_brief")
    return not reasons, reasons


def _overlap_seconds(a: dict[str, Any], b: dict[str, Any]) -> float:
    if str(a.get("instrument_id")) != str(b.get("instrument_id")):
        return 0.0
    return max(
        0.0,
        min(float(a["end_time"]), float(b["end_time"]))
        - max(float(a["start_time"]), float(b["start_time"])),
    )


def _merge_final_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            str(event["instrument_id"]),
            str(event.get("person_id") or "person_01"),
        )
        grouped.setdefault(key, []).append(event)
    merged: list[dict[str, Any]] = []
    for (instrument_id, person_id), items in grouped.items():
        current: dict[str, Any] | None = None
        for item in sorted(items, key=lambda value: float(value["start_time"])):
            if (
                current is None
                or float(item["start_time"]) - float(current["end_time"]) > 1.5
            ):
                if current is not None:
                    merged.append(current)
                current = {
                    "instrument_id": instrument_id,
                    "instrument_name": str(item["instrument_name"]),
                    "person_id": person_id,
                    "start_time": float(item["start_time"]),
                    "end_time": float(item["end_time"]),
                    "confidence": float(item.get("confidence") or 0.0),
                    "decision_sources": set(item.get("decision_sources") or []),
                    "source_event_ids": set(item.get("source_event_ids") or []),
                    "evidence": set(item.get("evidence") or []),
                    "target_person_observations": list(
                        item.get("target_person_observations") or []
                    ),
                }
            else:
                current["end_time"] = max(
                    float(current["end_time"]), float(item["end_time"])
                )
                current["confidence"] = max(
                    float(current["confidence"]),
                    float(item.get("confidence") or 0.0),
                )
                current["decision_sources"].update(item.get("decision_sources") or [])
                current["source_event_ids"].update(item.get("source_event_ids") or [])
                current["evidence"].update(item.get("evidence") or [])
                current["target_person_observations"].extend(
                    item.get("target_person_observations") or []
                )
        if current is not None:
            merged.append(current)
    output: list[dict[str, Any]] = []
    for index, item in enumerate(
        sorted(merged, key=lambda value: (value["start_time"], value["instrument_id"])),
        start=1,
    ):
        start = float(item["start_time"])
        end = float(item["end_time"])
        output.append(
            {
                "interaction_id": f"hybrid_interaction_{index:04d}",
                "person_id": item["person_id"],
                "instrument_id": item["instrument_id"],
                "instrument_name": item["instrument_name"],
                "start_time": round(start, 4),
                "end_time": round(end, 4),
                "duration_seconds": round(max(0.0, end - start), 4),
                "confidence": round(float(item["confidence"]), 4),
                "decision_sources": sorted(item["decision_sources"]),
                "source_event_ids": sorted(item["source_event_ids"]),
                "evidence": sorted(item["evidence"]),
                "target_person_observations": item[
                    "target_person_observations"
                ],
            }
        )
    return output


def _fuse_results(
    rule_result: dict[str, Any],
    vlm_result: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    interaction_result = rule_result["interaction_result"]
    rule_events = list(interaction_result.get("person_instrument_interactions") or [])
    suppressed_events = list(interaction_result.get("suppressed_interactions") or [])
    vlm_events = list(vlm_result.get("interactions") or [])

    candidates: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for event in vlm_events:
        candidates.append(
            {
                **event,
                "decision_sources": ["minicpm_v46_positive"],
                "source_event_ids": [str(event.get("interaction_id") or "")],
                "evidence": ["MiniCPM连续画面判定为交互"],
            }
        )
        decisions.append(
            {
                "source": "minicpm_v46",
                "instrument_id": event["instrument_id"],
                "instrument_name": event["instrument_name"],
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "accepted": True,
                "reason": "MiniCPM positive",
            }
        )

    for event in rule_events:
        strong, rejection_reasons = _strong_rule_event(event)
        overlaps_vlm = any(_overlap_seconds(event, item) > 0 for item in vlm_events)
        if strong:
            candidates.append(
                {
                    **event,
                    "decision_sources": [
                        "strong_vrm_recovery"
                        if not overlaps_vlm
                        else "minicpm_and_strong_vrm"
                    ],
                    "source_event_ids": [str(event.get("interaction_id") or "")],
                    "evidence": list(event.get("evidence") or []),
                }
            )
        decisions.append(
            {
                "source": "fixed_instrument_vrm",
                "instrument_id": event["instrument_id"],
                "instrument_name": event["instrument_name"],
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "accepted": strong,
                "overlaps_minicpm_positive": overlaps_vlm,
                "reason": (
                    "strong spatial-temporal relation"
                    if strong
                    else ", ".join(rejection_reasons)
                ),
            }
        )

    for event in suppressed_events:
        decisions.append(
            {
                "source": "fixed_instrument_vrm_suppressed",
                "instrument_id": event["instrument_id"],
                "instrument_name": event["instrument_name"],
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "accepted": False,
                "reason": str(
                    event.get("suppression_reason")
                    or "suppressed by personal-object filter"
                ),
            }
        )

    return _merge_final_events(candidates), {
        "candidate_decisions": decisions,
        "rule_event_count": len(rule_events),
        "rule_suppressed_event_count": len(suppressed_events),
        "vlm_event_count": len(vlm_events),
    }


def analyze_video_hybrid(
    video_path: str | Path,
    *,
    camera_id: str = "lab_camera_view_2",
    output_dir: str | Path | None = None,
    judge_instance: MiniCPMInteractionJudge | None = None,
    person_detector_instance: MultiInstrumentPersonDetector | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(video_path).resolve()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else OUTPUT_ROOT / f"{source.stem}_{stamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    # The demo contains clips slightly over the older web-upload limit. This
    # changes only the local offline analyzer limit, not any model decision.
    video_utils.MAX_CLIP_SECONDS = max(float(video_utils.MAX_CLIP_SECONDS), 600.0)
    print("[1/3] 正在提取手腕/手臂与仪器区域的时序关系证据……", flush=True)
    rule_result = analyze_fixed_instrument_video(
        source,
        camera_id=camera_id,
        detection_fps=5.0,
        min_person_area_ratio=0.018,
        output_dir=result_dir / "rule_evidence",
        decision_backend="rule",
    )
    print("[2/3] 正在由MiniCPM审核候选交互画面……", flush=True)
    vlm_result = analyze_video_multi_realtime(
        source,
        camera_id=camera_id,
        output_dir=result_dir / "vlm_review",
        judge_instance=judge_instance,
        person_detector_instance=person_detector_instance,
    )
    events, fusion = _fuse_results(rule_result, vlm_result)
    print("[3/3] 正在融合两类证据并生成最终视频……", flush=True)
    _, fps, width, height = _decode_video(source)
    annotated = result_dir / "annotated.mp4"
    _render_multi_result_video(
        source,
        annotated,
        fps=fps,
        width=width,
        height=height,
        events=events,
    )

    monitored = list(vlm_result["monitored_instruments"])
    active_ids = {str(item["instrument_id"]) for item in events}
    instrument_status = {
        str(item["instrument_name"]): str(item["instrument_id"]) in active_ids
        for item in monitored
    }
    recovered = [
        item
        for item in events
        if "strong_vrm_recovery" in item.get("decision_sources", [])
    ]
    output = {
        "schema_version": "0.4-minicpm-vrm-hybrid",
        "module": "minicpm_v46_constrained_vrm_hybrid",
        "source_video": str(source),
        "camera_id": camera_id,
        "decision_policy": {
            "minicpm_positive_is_accepted": True,
            "minicpm_negative_can_be_recovered_only_by_strong_vrm": True,
            "strong_vrm_requirements": {
                "minimum_confidence": 0.90,
                "minimum_duration_seconds": 0.80,
                "requires_wrist_inside_instrument_core": True,
                "requires_zone_or_forearm_evidence": True,
                "brief_single_hand_evidence_rejected": True,
                "personal_object_suppressed_evidence_rejected": True,
            },
            "wrist_relation_alone_is_not_sufficient": True,
        },
        "monitored_instruments": monitored,
        "interactions": events,
        "fusion": fusion,
        "summary": {
            "is_interacting_video": bool(events),
            "active_instrument_count": len(active_ids),
            "interaction_count": len(events),
            "minicpm_event_count": len(vlm_result.get("interactions") or []),
            "strong_vrm_recovery_count": len(recovered),
            "instrument_status": instrument_status,
        },
        "components": {
            "minicpm": {
                "result_json": vlm_result["artifacts"]["result_json"],
                "model": vlm_result.get("model"),
                "summary": vlm_result.get("summary"),
            },
            "vrm": {
                "result_json": rule_result["artifacts"]["result_json"],
                "pose_model": rule_result["detector"]["pose_model"],
                "summary": rule_result["interaction_result"]["summary"],
            },
        },
        "artifacts": {
            "annotated_video": str(annotated),
            "result_json": str(result_dir / "result.json"),
            "vlm_review_dir": str(result_dir / "vlm_review"),
            "rule_evidence_dir": str(result_dir / "rule_evidence"),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    (result_dir / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM-V 4.6 + constrained wrist/forearm VRM hybrid"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--camera-id", default="lab_camera_view_2")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze_video_hybrid(
        args.video,
        camera_id=args.camera_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(result["artifacts"]["annotated_video"])
    print(result["artifacts"]["result_json"])


if __name__ == "__main__":
    main()
