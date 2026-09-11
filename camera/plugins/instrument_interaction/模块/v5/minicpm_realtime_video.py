from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from settings import PROJECT_ROOT
from v4.fixed_instrument_interaction import (
    DEFAULT_CALIBRATION_PATH,
    FixedInstrumentRegions,
)
from v5.minicpm_interaction_judge import MiniCPMInteractionJudge
from v5.minicpm_video_interaction import (
    PersonTargetDetector,
    _decode_video,
    _mask_bounds,
    _rect_intersection_area,
    _render_result_video,
    _sample_frame_indices,
    _save_storyboard,
    _select_near_person,
    _window_starts,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "minicpm_v46_realtime"
FAST_PERSON_WEIGHTS = PROJECT_ROOT / "models" / "detectors" / "yolo11n.pt"


def _motion_region(
    instrument_mask: np.ndarray,
    *,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = frame_size
    x1, y1, x2, y2 = _mask_bounds(instrument_mask)
    return (
        max(0, x1 - 55),
        max(0, y1 - 75),
        min(width, x2 + 90),
        min(height, y2 + 105),
    )


def _motion_score(
    frames: list[np.ndarray],
    region: tuple[int, int, int, int],
) -> tuple[float, list[float]]:
    """Cheap activity trigger only; this score is never an interaction decision."""
    x1, y1, x2, y2 = region
    grays: list[np.ndarray] = []
    for frame in frames:
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > 320:
            scale = 320.0 / gray.shape[1]
            gray = cv2.resize(
                gray,
                (320, max(1, int(round(gray.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        grays.append(cv2.GaussianBlur(gray, (5, 5), 0))
    ratios: list[float] = []
    kernel = np.ones((3, 3), np.uint8)
    for previous, current in zip(grays, grays[1:]):
        delta = cv2.absdiff(previous, current)
        moving = cv2.threshold(delta, 16, 255, cv2.THRESH_BINARY)[1]
        moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, kernel)
        ratios.append(float(np.count_nonzero(moving)) / max(1, moving.size))
    if not ratios:
        return 0.0, []
    # A brief lid/hand movement must survive, so use the upper-middle pair
    # rather than averaging it away across the whole two-second window.
    score = float(np.percentile(np.asarray(ratios), 70))
    return round(score, 6), [round(item, 6) for item in ratios]


def _anchor_positions(frame_count: int) -> list[int]:
    # One center-frame person check per second is sufficient for this fixed
    # camera gate and avoids running the detector on all eight VLM frames.
    return [frame_count // 2]


def _target_crop_with_cached_detections(
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
    height, width = frames_bgr[0].shape[:2]
    instrument_bounds = _mask_bounds(instrument_mask)
    selected_people: list[dict[str, Any] | None] = []
    for local_index in _anchor_positions(len(frames_bgr)):
        source_index = frame_indices[local_index]
        if source_index not in detection_cache:
            detection_cache[source_index] = person_detector.detect(frames_bgr[local_index])
        selected_people.append(
            _select_near_person(
                detection_cache[source_index],
                instrument_bounds=instrument_bounds,
                frame_size=(width, height),
            )
        )

    person_boxes = [list(item["bbox"]) for item in selected_people if item is not None]
    ix1, iy1, ix2, iy2 = instrument_bounds
    x1, y1 = ix1 - 90, iy1 - 125
    x2, y2 = ix2 + 165, iy2 + 155
    for px1, py1, px2, py2 in person_boxes:
        x1, y1 = min(x1, px1 - 20), min(y1, py1 - 20)
        x2, y2 = max(x2, px2 + 20), max(y2, py2 + 20)
    roi = (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )

    contours, _ = cv2.findContours(
        instrument_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    target_bbox = next(
        (list(item["bbox"]) for item in selected_people if item is not None), None
    )
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
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        crop = marked[ry1:ry2, rx1:rx2]
        output.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    return output, selected_people, roi


def _skipped_result(reason: str) -> dict[str, Any]:
    return {
        "is_interacting": False,
        "confidence": 1.0,
        "evidence": reason,
        "raw_response": "",
        "valid_response": True,
        "validation_error": None,
        "latency_seconds": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "model_invoked": False,
    }


def _smooth_realtime_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enter conservatively; exit on the next explicit MiniCPM negative review."""
    state = False
    event_start: float | None = None
    events: list[dict[str, Any]] = []
    for index, review in enumerate(reviews):
        recent = reviews[max(0, index - 2) : index + 1]
        positive_count = sum(bool(item["is_interacting"]) for item in recent)
        if not state and len(recent) >= 2 and positive_count >= 2:
            state = True
            event_start = min(
                float(item["window_start"])
                for item in recent
                if item["is_interacting"]
            )
        elif state and review["model_invoked"] and not review["is_interacting"]:
            events.append(
                {
                    "start_time": round(float(event_start or 0.0), 4),
                    "end_time": round(float(review["window_start"]), 4),
                }
            )
            state = False
            event_start = None
        review["smoothed_is_interacting"] = state
    if state and reviews:
        events.append(
            {
                "start_time": round(float(event_start or 0.0), 4),
                "end_time": round(float(reviews[-1]["window_end"]), 4),
            }
        )
    for index, event in enumerate(events, start=1):
        event["interaction_id"] = f"minicpm_interaction_{index:04d}"
        event["person_id"] = "person_01"
        event["instrument_id"] = "instrument_006"
        event["instrument_name"] = "高速离心机"
        event["duration_seconds"] = round(event["end_time"] - event["start_time"], 4)
    return events


def _current_smoothed_state(reviews: list[dict[str, Any]]) -> bool:
    if not reviews:
        return False
    copied = [dict(item) for item in reviews]
    _smooth_realtime_reviews(copied)
    return bool(copied[-1].get("smoothed_is_interacting", False))


def create_persistent_resources() -> tuple[MiniCPMInteractionJudge, PersonTargetDetector]:
    """Load shared GPU models once for a long-running monitoring process."""
    if not FAST_PERSON_WEIGHTS.is_file():
        raise FileNotFoundError(FAST_PERSON_WEIGHTS)
    detector = PersonTargetDetector(weights=FAST_PERSON_WEIGHTS, image_size=512)
    judge = MiniCPMInteractionJudge(max_new_tokens=64)
    return judge, detector


def analyze_video_realtime(
    video_path: str | Path,
    *,
    camera_id: str = "lab_camera_view_2",
    instrument_id: str = "instrument_006",
    window_seconds: float = 2.0,
    step_seconds: float = 1.0,
    frame_count: int = 8,
    activity_threshold: float = 0.04,
    activity_change_threshold: float = 0.05,
    activity_burst_interval: float = 6.0,
    activity_burst_duration: float = 1.0,
    idle_review_interval: float = 6.0,
    schedule_mode: str = "adaptive",
    output_dir: str | Path | None = None,
    gate_only: bool = False,
    judge_instance: MiniCPMInteractionJudge | None = None,
    person_detector_instance: PersonTargetDetector | None = None,
) -> dict[str, Any]:
    if schedule_mode not in {"adaptive", "normal2_active1"}:
        raise ValueError(f"Unsupported schedule_mode: {schedule_mode}")
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

    regions = FixedInstrumentRegions.from_file(
        camera_id=camera_id,
        frame_size=(width, height),
        calibration_path=DEFAULT_CALIBRATION_PATH,
    )
    instrument = regions.instrument_lookup[instrument_id]
    instrument_mask = regions.core_masks[instrument_id]
    activity_region = _motion_region(instrument_mask, frame_size=(width, height))
    weights = FAST_PERSON_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(weights)
    person_detector_reused = person_detector_instance is not None
    person_detector = person_detector_instance or PersonTargetDetector(
        weights=weights, image_size=512
    )
    judge_reused = judge_instance is not None
    judge = (
        None
        if gate_only
        else judge_instance or MiniCPMInteractionJudge(max_new_tokens=64)
    )

    detection_cache: dict[int, list[dict[str, Any]]] = {}
    starts = _window_starts(duration, window_seconds, step_seconds)
    reviews: list[dict[str, Any]] = []
    last_invoked = float("-inf")
    boost_until = float("-inf")
    burst_until = float("-inf")
    last_burst_start = float("-inf")
    previous_activity_triggered = False
    previous_motion_score = 0.0

    for number, window_start in enumerate(starts, start=1):
        indices = _sample_frame_indices(
            start=window_start,
            window_seconds=window_seconds,
            frame_count=frame_count,
            fps=fps,
            maximum_index=len(frames) - 1,
        )
        sampled = [frames[index] for index in indices]
        motion_score, pair_scores = _motion_score(sampled, activity_region)
        target_frames, selected_people, roi = _target_crop_with_cached_detections(
            sampled,
            indices,
            person_detector=person_detector,
            detection_cache=detection_cache,
            instrument_mask=instrument_mask,
        )
        person_count = sum(item is not None for item in selected_people)
        has_person = person_count > 0
        active_state = _current_smoothed_state(reviews)
        activity_triggered = motion_score >= activity_threshold
        already_dense = window_start <= boost_until or active_state
        if schedule_mode == "normal2_active1":
            activity_due = activity_triggered
            periodic_due = window_start - last_invoked >= 2.0
            boosted = already_dense
            should_invoke = has_person and (activity_due or periodic_due or boosted)
            trigger_reasons = []
            if activity_due:
                trigger_reasons.append("activity_1s")
            if periodic_due:
                trigger_reasons.append("normal_2s")
            if boosted:
                trigger_reasons.append("follow_up")
        else:
            significant_activity_change = (
                motion_score >= activity_threshold
                and motion_score - previous_motion_score >= activity_change_threshold
            )
            new_activity_burst = (
                activity_triggered
                and bool(reviews)
                and not already_dense
                and window_start > burst_until
                and (
                    not previous_activity_triggered
                    or significant_activity_change
                    or window_start - last_burst_start >= activity_burst_interval
                )
            )
            if new_activity_burst:
                burst_until = window_start + activity_burst_duration
                last_burst_start = window_start
            activity_due = window_start <= burst_until
            periodic_due = window_start - last_invoked >= idle_review_interval
            boosted = already_dense
            should_invoke = has_person and (activity_due or periodic_due or boosted)
            trigger_reasons = []
            if activity_due:
                trigger_reasons.append("activity")
            if periodic_due:
                trigger_reasons.append("periodic")
            if boosted:
                trigger_reasons.append("follow_up")

        storyboard: Path | None = None
        if gate_only:
            reason = (
                "触发器建议调用MiniCPM（诊断模式未实际调用）"
                if should_invoke
                else "触发器建议跳过MiniCPM"
            )
            result_data = _skipped_result(reason)
            if should_invoke:
                last_invoked = window_start
        elif not has_person:
            result_data = _skipped_result("目标仪器附近未检测到人员，跳过MiniCPM")
        elif not should_invoke:
            result_data = _skipped_result("活动量低且未到周期复核时间，跳过MiniCPM")
        else:
            storyboard = result_dir / "review_inputs" / f"window_{window_start:07.2f}.jpg"
            _save_storyboard(target_frames, storyboard)
            result_data = {
                **judge.judge(
                    target_frames,
                    instrument_name=instrument.display_name,
                    person_description="第一帧蓝色矩形指定的人员",
                ).as_dict(),
                "model_invoked": True,
            }
            last_invoked = window_start
            if result_data["is_interacting"]:
                # A first positive temporarily requests dense follow-up reviews.
                # MiniCPM still decides every final true/false value.
                boost_until = max(boost_until, window_start + 2.0)

        previous_activity_triggered = activity_triggered
        previous_motion_score = motion_score

        review = {
            "window_start": round(window_start, 4),
            "window_end": round(min(duration, window_start + window_seconds), 4),
            "frame_indices": indices,
            "person_detected_anchor_count": person_count,
            "roi": list(roi),
            "activity_region": list(activity_region),
            "activity_score": motion_score,
            "activity_pair_scores": pair_scores,
            "activity_threshold": activity_threshold,
            "activity_change_threshold": activity_change_threshold,
            "activity_burst_interval": activity_burst_interval,
            "activity_burst_duration": activity_burst_duration,
            "gate_candidate": should_invoke,
            "gate_reasons": trigger_reasons,
            "storyboard": str(storyboard) if storyboard else None,
            **result_data,
        }
        reviews.append(review)
        status = "CALL" if result_data["model_invoked"] else "SKIP"
        print(
            f"[{number}/{len(starts)}] {window_start:.2f}-{window_start + window_seconds:.2f}s "
            f"motion={motion_score:.4f} person={person_count}/{len(selected_people)} {status} "
            f"result={result_data['is_interacting']}",
            flush=True,
        )

    events = _smooth_realtime_reviews(reviews)
    annotated = result_dir / "annotated.mp4"
    if not gate_only:
        _render_result_video(
            source,
            annotated,
            fps=fps,
            width=width,
            height=height,
            events=events,
        )

    output = {
        "schema_version": "0.2-realtime",
        "module": "minicpm_v46_realtime_fixed_instrument_interaction",
        "source_video": str(source),
        "camera_id": camera_id,
        "person_id": "person_01",
        "instrument_id": instrument_id,
        "instrument_name": instrument.display_name,
        "decision_backend": "minicpm_v46_binary_vlm",
        "gate": {
            "purpose": "invocation scheduling only; never an interaction decision",
            "schedule_mode": schedule_mode,
            "uses_person_presence": True,
            "uses_activity_score": True,
            "activity_threshold": activity_threshold,
            "activity_change_threshold": activity_change_threshold,
            "activity_burst_interval": activity_burst_interval,
            "activity_burst_duration": activity_burst_duration,
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
        "model_instance_reused": judge_reused,
        "person_detector": {
            "weights": str(weights),
            "image_size": 512,
            "purpose": "person presence, target selection and VLM invocation gate only",
            "instance_reused": person_detector_reused,
        },
        "reviews": reviews,
        "interactions": events,
        "summary": {
            "is_interacting_video": bool(events),
            "interaction_count": len(events),
            "window_count": len(reviews),
            "model_invocation_count": sum(item["model_invoked"] for item in reviews),
            "gate_candidate_count": sum(item["gate_candidate"] for item in reviews),
            "skipped_window_count": sum(not item["model_invoked"] for item in reviews),
            "raw_positive_window_count": sum(item["is_interacting"] for item in reviews),
            "invalid_response_count": sum(not item["valid_response"] for item in reviews),
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
        description="MiniCPM-V 4.6 adaptive real-time interaction prototype"
    )
    parser.add_argument("video", type=Path)
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
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    result = analyze_video_realtime(
        args.video,
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
        gate_only=args.gate_only,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(result["artifacts"]["annotated_video"])
    print(result["artifacts"]["result_json"])


if __name__ == "__main__":
    main()
