from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from settings import PROJECT_ROOT
from v2.media_utils import transcode_to_browser_mp4
from v4.fixed_instrument_interaction import (
    DEFAULT_CALIBRATION_PATH,
    FixedInstrumentRegions,
    InstrumentRegion,
)
from v5.minicpm_interaction_judge import MiniCPMInteractionJudge
from v5.minicpm_realtime_video import (
    FAST_PERSON_WEIGHTS,
    _current_smoothed_state,
    _motion_region,
    _motion_score,
    _skipped_result,
    _smooth_realtime_reviews,
    _target_crop_with_cached_detections,
)
from v5.minicpm_video_interaction import (
    PersonTargetDetector,
    _decode_video,
    _font,
    _mask_bounds,
    _rect_intersection_area,
    _sample_frame_indices,
    _save_storyboard,
    _select_near_person,
    _window_starts,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "minicpm_v46_multi_realtime"
INSTRUMENT_HINTS_PATH = (
    PROJECT_ROOT / "prompts" / "minicpm_v46_instrument_hints.json"
)


@dataclass
class InstrumentRuntime:
    instrument: InstrumentRegion
    mask: np.ndarray
    motion_region: tuple[int, int, int, int]
    reviews: list[dict[str, Any]] = field(default_factory=list)
    last_invoked: float = float("-inf")
    boost_until: float = float("-inf")
    burst_until: float = float("-inf")
    last_burst_start: float = float("-inf")
    previous_activity_triggered: bool = False
    previous_motion_score: float = 0.0
    attribution_until: float = float("-inf")


class MultiInstrumentPersonDetector(PersonTargetDetector):
    """Higher-resolution person gate with a small-edge-fragment filter."""

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        detections = super().detect(frame)
        minimum_height = max(48.0, frame.shape[0] * 0.14)
        return [
            item
            for item in detections
            if float(item["bbox"][3]) - float(item["bbox"][1]) >= minimum_height
        ]


def _target_crop_multi(
    frames_bgr: list[np.ndarray],
    frame_indices: list[int],
    *,
    person_detector: PersonTargetDetector,
    detection_cache: dict[int, list[dict[str, Any]]],
    instrument_mask: np.ndarray,
) -> tuple[
    list[Image.Image],
    list[dict[str, Any] | None],
    tuple[int, int, int, int],
]:
    """Tighter crop for small distant instruments while retaining the person."""
    height, width = frames_bgr[0].shape[:2]
    instrument_bounds = _mask_bounds(instrument_mask)
    local_index = len(frames_bgr) // 2
    source_index = frame_indices[local_index]
    if source_index not in detection_cache:
        detection_cache[source_index] = person_detector.detect(frames_bgr[local_index])
    selected = _select_near_person(
        detection_cache[source_index],
        instrument_bounds=instrument_bounds,
        frame_size=(width, height),
    )
    selected_people = [selected]
    person_boxes = [list(selected["bbox"])] if selected is not None else []
    ix1, iy1, ix2, iy2 = instrument_bounds
    x1, y1 = ix1 - 75, iy1 - 115
    x2, y2 = ix2 + 60, iy2 + 85
    for px1, py1, px2, py2 in person_boxes:
        x1, y1 = min(x1, px1 - 12), min(y1, py1 - 12)
        x2, y2 = max(x2, px2 + 12), max(y2, py2 + 12)
    roi = (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )
    contours, _ = cv2.findContours(
        instrument_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    target_bbox = list(selected["bbox"]) if selected is not None else None
    rx1, ry1, rx2, ry2 = roi
    output: list[Image.Image] = []
    for index, frame in enumerate(frames_bgr):
        marked = frame.copy()
        if index == 0:
            cv2.drawContours(marked, contours, -1, (0, 255, 0), 3, cv2.LINE_AA)
            if target_bbox is not None:
                px1, py1, px2, py2 = [int(round(value)) for value in target_bbox]
                cv2.rectangle(marked, (px1, py1), (px2, py2), (255, 120, 20), 3)
            cv2.putText(
                marked,
                "TARGET INSTRUMENT / PERSON",
                (max(4, rx1 + 6), max(24, ry1 + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        crop = marked[ry1:ry2, rx1:rx2]
        output.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    return output, selected_people, roi


def _person_key(person: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
    if person is None:
        return None
    return tuple(int(round(value / 8.0)) for value in person["bbox"])


def _association_score(
    person: dict[str, Any],
    *,
    instrument_mask: np.ndarray,
    motion_region: tuple[int, int, int, int],
    motion_score: float,
    frame_size: tuple[int, int],
    active_state: bool,
) -> float:
    """Rank target instruments for one person; never make the final decision."""
    width, height = frame_size
    bbox = [float(value) for value in person["bbox"]]
    bounds = _mask_bounds(instrument_mask)
    person_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    instrument_center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
    distance = float(
        np.hypot(
            person_center[0] - instrument_center[0],
            person_center[1] - instrument_center[1],
        )
    )
    proximity = max(0.0, 1.0 - distance / max(1.0, float(np.hypot(width, height))))
    person_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    overlap = _rect_intersection_area(bbox, motion_region) / person_area
    return (
        5.0 * float(motion_score)
        + proximity
        + 2.0 * overlap
        + (3.0 if active_state else 0.0)
    )


def _gate_candidate(
    runtime: InstrumentRuntime,
    *,
    window_start: float,
    has_person: bool,
    motion_score: float,
    activity_threshold: float,
    activity_change_threshold: float,
    activity_burst_interval: float,
    activity_burst_duration: float,
    idle_review_interval: float,
    schedule_mode: str,
) -> tuple[bool, list[str], bool]:
    active_state = _current_smoothed_state(runtime.reviews)
    activity_triggered = motion_score >= activity_threshold
    already_dense = window_start <= runtime.boost_until or active_state
    if schedule_mode == "normal2_active1":
        activity_due = activity_triggered
        periodic_due = window_start - runtime.last_invoked >= 2.0
        boosted = already_dense
        reasons = []
        if activity_due:
            reasons.append("activity_1s")
        if periodic_due:
            reasons.append("normal_2s")
        if boosted:
            reasons.append("follow_up")
    else:
        significant_change = (
            motion_score >= activity_threshold
            and motion_score - runtime.previous_motion_score
            >= activity_change_threshold
        )
        new_burst = (
            activity_triggered
            and bool(runtime.reviews)
            and not already_dense
            and window_start > runtime.burst_until
            and (
                not runtime.previous_activity_triggered
                or significant_change
                or window_start - runtime.last_burst_start >= activity_burst_interval
            )
        )
        if new_burst:
            runtime.burst_until = window_start + activity_burst_duration
            runtime.last_burst_start = window_start
        activity_due = window_start <= runtime.burst_until
        periodic_due = window_start - runtime.last_invoked >= idle_review_interval
        boosted = already_dense
        reasons = []
        if activity_due:
            reasons.append("activity")
        if periodic_due:
            reasons.append("periodic")
        if boosted:
            reasons.append("follow_up")
    return has_person and (activity_due or periodic_due or boosted), reasons, active_state


def _render_multi_result_video(
    source: Path,
    target: Path,
    *,
    fps: float,
    width: int,
    height: int,
    events: list[dict[str, Any]],
) -> None:
    raw = target.with_name("annotated_raw.mp4")
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("无法创建多仪器MiniCPM结果视频")
    font = _font(20)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            active_names = sorted(
                {
                    (
                        f"{item.get('person_name') or item.get('person_id')} → "
                        f"{item['instrument_name']}"
                        if item.get("person_name") or item.get("person_id")
                        else str(item["instrument_name"])
                    )
                    for item in events
                    if float(item["start_time"])
                    <= timestamp
                    <= float(item["end_time"])
                }
            )
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            if active_names:
                label = "MiniCPM：正在交互：" + "、".join(active_names)
            else:
                label = "MiniCPM：未判定仪器交互"
            panel_width = min(width - 16, max(410, 34 + len(label) * 21))
            draw.rectangle((8, 8, panel_width, 47), fill=(18, 20, 24))
            draw.text((16, 13), label, font=font, fill=(255, 255, 255))
            writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    transcode_to_browser_mp4(raw, target, fps)
    raw.unlink(missing_ok=True)


def _instrument_events(runtime: InstrumentRuntime) -> list[dict[str, Any]]:
    events = _smooth_realtime_reviews(runtime.reviews)
    for index, event in enumerate(events, start=1):
        event["interaction_id"] = (
            f"{runtime.instrument.instrument_id}_interaction_{index:04d}"
        )
        event["instrument_id"] = runtime.instrument.instrument_id
        event["instrument_name"] = runtime.instrument.display_name
        positive_reviews = [
            review
            for review in runtime.reviews
            if bool(review.get("is_interacting"))
            and float(review["window_end"]) >= float(event["start_time"])
            and float(review["window_start"]) <= float(event["end_time"])
        ]
        event["confidence"] = round(
            max(
                (float(review.get("confidence") or 0.0) for review in positive_reviews),
                default=0.0,
            ),
            4,
        )
        event["model_evidence"] = [
            str(review["evidence"])
            for review in positive_reviews
            if review.get("evidence")
        ]
        # Preserve the person box that was actually shown to MiniCPM. This is
        # metadata only; it does not participate in the interaction decision.
        # The public API uses it to attach identities supplied by another
        # system without changing the validated VLM logic.
        event["target_person_observations"] = [
            {
                "timestamp": round(
                    (
                        float(review["window_start"])
                        + float(review["window_end"])
                    )
                    / 2.0,
                    4,
                ),
                "bbox": list(review["selected_person_bbox"]),
                "detector_confidence": review.get(
                    "selected_person_confidence"
                ),
            }
            for review in positive_reviews
            if review.get("selected_person_bbox") is not None
        ]
    return events


def analyze_video_multi_realtime(
    video_path: str | Path,
    *,
    camera_id: str = "lab_camera_view_2",
    instrument_ids: list[str] | None = None,
    window_seconds: float = 2.0,
    step_seconds: float = 1.0,
    frame_count: int = 8,
    activity_threshold: float = 0.04,
    activity_change_threshold: float = 0.05,
    activity_burst_interval: float = 6.0,
    activity_burst_duration: float = 1.0,
    idle_review_interval: float = 6.0,
    schedule_mode: str = "adaptive",
    max_calls_per_window: int = 1,
    output_dir: str | Path | None = None,
    gate_only: bool = False,
    judge_instance: MiniCPMInteractionJudge | None = None,
    person_detector_instance: PersonTargetDetector | None = None,
) -> dict[str, Any]:
    if schedule_mode not in {"adaptive", "normal2_active1"}:
        raise ValueError(f"Unsupported schedule_mode: {schedule_mode}")
    if max_calls_per_window < 1:
        raise ValueError("max_calls_per_window must be at least 1")
    started = time.perf_counter()
    source = Path(video_path).resolve()
    frames, fps, width, height = _decode_video(source)
    duration = (len(frames) - 1) / fps
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else OUTPUT_ROOT / f"{source.stem}_{stamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    instrument_hints = (
        json.loads(INSTRUMENT_HINTS_PATH.read_text(encoding="utf-8"))
        if INSTRUMENT_HINTS_PATH.is_file()
        else {}
    )

    regions = FixedInstrumentRegions.from_file(
        camera_id=camera_id,
        frame_size=(width, height),
        calibration_path=DEFAULT_CALIBRATION_PATH,
    )
    selected_ids = instrument_ids or [item.instrument_id for item in regions.instruments]
    unknown = [item for item in selected_ids if item not in regions.instrument_lookup]
    if unknown:
        raise KeyError(f"Unknown instruments for {camera_id}: {unknown}")
    runtimes: dict[str, InstrumentRuntime] = {}
    for instrument_id in selected_ids:
        instrument = regions.instrument_lookup[instrument_id]
        mask = regions.core_masks[instrument_id]
        runtimes[instrument_id] = InstrumentRuntime(
            instrument=instrument,
            mask=mask,
            motion_region=_motion_region(mask, frame_size=(width, height)),
        )

    if not FAST_PERSON_WEIGHTS.is_file():
        raise FileNotFoundError(FAST_PERSON_WEIGHTS)
    detector_reused = person_detector_instance is not None
    detector = person_detector_instance or MultiInstrumentPersonDetector(
        weights=FAST_PERSON_WEIGHTS, image_size=768
    )
    judge_reused = judge_instance is not None
    judge = (
        None
        if gate_only
        else judge_instance or MiniCPMInteractionJudge(max_new_tokens=64)
    )

    detection_cache: dict[int, list[dict[str, Any]]] = {}
    starts = _window_starts(duration, window_seconds, step_seconds)
    total_model_calls = 0
    for number, window_start in enumerate(starts, start=1):
        indices = _sample_frame_indices(
            start=window_start,
            window_seconds=window_seconds,
            frame_count=frame_count,
            fps=fps,
            maximum_index=len(frames) - 1,
        )
        sampled = [frames[index] for index in indices]
        prepared: dict[str, dict[str, Any]] = {}

        for instrument_id, runtime in runtimes.items():
            motion_score, pair_scores = _motion_score(sampled, runtime.motion_region)
            crop_function = (
                _target_crop_with_cached_detections
                if instrument_id == "instrument_006"
                else _target_crop_multi
            )
            target_frames, selected_people, roi = crop_function(
                sampled,
                indices,
                person_detector=detector,
                detection_cache=detection_cache,
                instrument_mask=runtime.mask,
            )
            person = next((item for item in selected_people if item is not None), None)
            has_person = person is not None
            should_invoke, reasons, active_state = _gate_candidate(
                runtime,
                window_start=window_start,
                has_person=has_person,
                motion_score=motion_score,
                activity_threshold=activity_threshold,
                activity_change_threshold=activity_change_threshold,
                activity_burst_interval=activity_burst_interval,
                activity_burst_duration=activity_burst_duration,
                idle_review_interval=idle_review_interval,
                schedule_mode=schedule_mode,
            )
            association = (
                _association_score(
                    person,
                    instrument_mask=runtime.mask,
                    motion_region=runtime.motion_region,
                    motion_score=motion_score,
                    frame_size=(width, height),
                    active_state=active_state,
                )
                if person is not None
                else 0.0
            )
            prepared[instrument_id] = {
                "motion_score": motion_score,
                "pair_scores": pair_scores,
                "target_frames": target_frames,
                "selected_people": selected_people,
                "person": person,
                "roi": roi,
                "gate_candidate": should_invoke,
                "gate_reasons": reasons,
                "association_score": association,
                "active_state": active_state,
            }
        # Attribute each detected person before scheduling periodic reviews. If
        # scheduling happened first, the same stationary person would rotate
        # through every nearby instrument as their review timers became due.
        best_by_person: dict[tuple[int, int, int, int] | None, dict[str, Any]] = {}
        for instrument_id, item in prepared.items():
            if item["person"] is None:
                continue
            runtime = runtimes[instrument_id]
            candidate = {
                "instrument_id": instrument_id,
                "person_key": _person_key(item["person"]),
                "attribution_priority": item["association_score"]
                + (0.30 if window_start <= runtime.attribution_until else 0.0),
            }
            key = candidate["person_key"]
            current = best_by_person.get(key)
            if (
                current is None
                or candidate["attribution_priority"]
                > current["attribution_priority"]
            ):
                best_by_person[key] = candidate
        attributed_ids = {
            item["instrument_id"] for item in best_by_person.values()
        }
        for instrument_id in attributed_ids:
            runtimes[instrument_id].attribution_until = window_start + 2.0

        candidates: list[dict[str, Any]] = []
        for instrument_id in attributed_ids:
            item = prepared[instrument_id]
            if not item["gate_candidate"]:
                continue
            candidates.append(
                {
                    "instrument_id": instrument_id,
                    "priority": item["association_score"]
                    + (5.0 if item["active_state"] else 0.0)
                    + (
                        2.0
                        if "activity" in " ".join(item["gate_reasons"])
                        else 0.0
                    ),
                }
            )
        ranked = sorted(
            candidates, key=lambda item: item["priority"], reverse=True
        )
        invoked_ids = {
            item["instrument_id"] for item in ranked[:max_calls_per_window]
        }

        status_parts: list[str] = []
        for instrument_id, runtime in runtimes.items():
            item = prepared[instrument_id]
            person_count = sum(
                selected is not None for selected in item["selected_people"]
            )
            storyboard: Path | None = None
            if gate_only:
                if instrument_id in invoked_ids:
                    result_data = _skipped_result(
                        "触发器建议调用MiniCPM（诊断模式未实际调用）"
                    )
                    runtime.last_invoked = window_start
                elif not item["person"]:
                    result_data = _skipped_result("目标仪器附近未检测到人员")
                else:
                    result_data = _skipped_result("本窗口未获得MiniCPM审核优先级")
            elif not item["person"]:
                result_data = _skipped_result(
                    "目标仪器附近未检测到人员，跳过MiniCPM"
                )
            elif instrument_id not in invoked_ids:
                reason = (
                    "同一人员更接近另一台活动仪器，本窗口跳过MiniCPM"
                    if item["gate_candidate"]
                    else "活动量低且未到周期复核时间，跳过MiniCPM"
                )
                result_data = _skipped_result(reason)
            else:
                storyboard = (
                    result_dir
                    / "review_inputs"
                    / instrument_id
                    / f"window_{window_start:07.2f}.jpg"
                )
                _save_storyboard(item["target_frames"], storyboard)
                result_data = {
                    **judge.judge(
                        item["target_frames"],
                        instrument_name=runtime.instrument.display_name,
                        person_description="第一帧蓝色矩形指定的人员",
                        instrument_hint=str(
                            instrument_hints.get(instrument_id) or ""
                        ),
                    ).as_dict(),
                    "model_invoked": True,
                }
                runtime.last_invoked = window_start
                total_model_calls += 1
                if result_data["is_interacting"]:
                    runtime.boost_until = max(
                        runtime.boost_until, window_start + 2.0
                    )

            runtime.previous_activity_triggered = (
                item["motion_score"] >= activity_threshold
            )
            runtime.previous_motion_score = item["motion_score"]
            review = {
                "window_start": round(window_start, 4),
                "window_end": round(min(duration, window_start + window_seconds), 4),
                "frame_indices": indices,
                "instrument_id": instrument_id,
                "instrument_name": runtime.instrument.display_name,
                "person_detected_anchor_count": person_count,
                "roi": list(item["roi"]),
                "activity_region": list(runtime.motion_region),
                "activity_score": item["motion_score"],
                "activity_pair_scores": item["pair_scores"],
                "association_score": round(float(item["association_score"]), 6),
                "selected_person_bbox": (
                    list(item["person"]["bbox"])
                    if item["person"] is not None
                    else None
                ),
                "selected_person_confidence": (
                    float(item["person"].get("confidence") or 0.0)
                    if item["person"] is not None
                    else None
                ),
                "gate_candidate": bool(item["gate_candidate"]),
                "gate_reasons": item["gate_reasons"],
                "storyboard": str(storyboard) if storyboard else None,
                **result_data,
            }
            runtime.reviews.append(review)
            status = (
                "WOULD_CALL"
                if gate_only and instrument_id in invoked_ids
                else "CALL"
                if result_data["model_invoked"]
                else "SKIP"
            )
            status_parts.append(
                f"{runtime.instrument.display_name}:{status}/{result_data['is_interacting']}"
            )
        print(
            f"[{number}/{len(starts)}] {window_start:.2f}-{window_start + window_seconds:.2f}s "
            + " | ".join(status_parts),
            flush=True,
        )

    instrument_results: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_reviews: list[dict[str, Any]] = []
    for runtime in runtimes.values():
        events = _instrument_events(runtime)
        all_events.extend(events)
        all_reviews.extend(runtime.reviews)
        instrument_results.append(
            {
                "instrument_id": runtime.instrument.instrument_id,
                "instrument_name": runtime.instrument.display_name,
                "is_interacting_video": bool(events),
                "interaction_count": len(events),
                "model_invocation_count": sum(
                    item["model_invoked"] for item in runtime.reviews
                ),
                "raw_positive_window_count": sum(
                    item["is_interacting"] for item in runtime.reviews
                ),
                "interactions": events,
                "reviews": runtime.reviews,
            }
        )
    all_events.sort(key=lambda item: (item["start_time"], item["instrument_id"]))
    annotated = result_dir / "annotated.mp4"
    if not gate_only:
        _render_multi_result_video(
            source,
            annotated,
            fps=fps,
            width=width,
            height=height,
            events=all_events,
        )

    output = {
        "schema_version": "0.3-multi-realtime",
        "module": "minicpm_v46_multi_fixed_instrument_interaction",
        "source_video": str(source),
        "camera_id": camera_id,
        "person_id": "person_01",
        "monitored_instruments": [
            {
                "instrument_id": runtime.instrument.instrument_id,
                "instrument_name": runtime.instrument.display_name,
            }
            for runtime in runtimes.values()
        ],
        "decision_backend": "minicpm_v46_binary_vlm",
        "gate": {
            "purpose": "invocation scheduling and target attribution only; never an interaction decision",
            "schedule_mode": schedule_mode,
            "uses_person_presence": True,
            "uses_activity_score": True,
            "uses_instrument_regions": True,
            "uses_wrist_keypoints": False,
            "max_calls_per_window": max_calls_per_window,
            "activity_threshold": activity_threshold,
            "idle_review_interval": idle_review_interval,
            "gate_only": gate_only,
        },
        "uses_wrist_interaction_rules": False,
        "uses_old_interaction_engine": False,
        "uses_qwen_vlm": False,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": len(frames),
            "duration_seconds": round(duration, 4),
        },
        "sampling": {
            "window_seconds": window_seconds,
            "step_seconds": step_seconds,
            "frame_count": frame_count,
        },
        "model": judge.metadata() if judge is not None else None,
        "instrument_hints_path": str(INSTRUMENT_HINTS_PATH),
        "model_instance_reused": judge_reused,
        "person_detector": {
            "weights": str(FAST_PERSON_WEIGHTS),
            "image_size": detector.image_size,
            "purpose": "shared person presence and VLM invocation gate only",
            "instance_reused": detector_reused,
        },
        "instrument_results": instrument_results,
        "interactions": all_events,
        "summary": {
            "is_interacting_video": bool(all_events),
            "active_instrument_count": sum(
                item["is_interacting_video"] for item in instrument_results
            ),
            "interaction_count": len(all_events),
            "monitored_instrument_count": len(runtimes),
            "window_count": len(starts),
            "instrument_window_count": len(all_reviews),
            "model_invocation_count": total_model_calls,
            "skipped_instrument_window_count": sum(
                not item["model_invoked"] for item in all_reviews
            ),
            "invalid_response_count": sum(
                not item["valid_response"] for item in all_reviews
            ),
            "instrument_status": {
                item["instrument_name"]: item["is_interacting_video"]
                for item in instrument_results
            },
        },
        "artifacts": {
            "annotated_video": str(annotated) if not gate_only else None,
            "result_json": str(result_dir / "result.json"),
            "review_inputs": str(result_dir / "review_inputs"),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    (result_dir / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM-V 4.6 multi-instrument adaptive real-time prototype"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--camera-id", default="lab_camera_view_2")
    parser.add_argument("--instrument-id", action="append", dest="instrument_ids")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--activity-threshold", type=float, default=0.04)
    parser.add_argument("--activity-change-threshold", type=float, default=0.05)
    parser.add_argument("--activity-burst-interval", type=float, default=6.0)
    parser.add_argument("--activity-burst-duration", type=float, default=1.0)
    parser.add_argument("--idle-review-interval", type=float, default=6.0)
    parser.add_argument(
        "--schedule-mode",
        choices=["adaptive", "normal2_active1"],
        default="adaptive",
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--max-calls-per-window", type=int, default=1)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    result = analyze_video_multi_realtime(
        args.video,
        camera_id=args.camera_id,
        instrument_ids=args.instrument_ids,
        output_dir=args.output_dir,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        activity_threshold=args.activity_threshold,
        activity_change_threshold=args.activity_change_threshold,
        activity_burst_interval=args.activity_burst_interval,
        activity_burst_duration=args.activity_burst_duration,
        idle_review_interval=args.idle_review_interval,
        schedule_mode=args.schedule_mode,
        frame_count=args.frame_count,
        max_calls_per_window=args.max_calls_per_window,
        gate_only=args.gate_only,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(result["artifacts"]["annotated_video"])
    print(result["artifacts"]["result_json"])


if __name__ == "__main__":
    main()
