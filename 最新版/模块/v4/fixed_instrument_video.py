from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from settings import PROJECT_ROOT
from v2.media_utils import transcode_to_browser_mp4
from v4.fixed_instrument_interaction import (
    DEFAULT_CALIBRATION_PATH,
    FixedInstrumentInteractionEngine,
    FixedInstrumentRegions,
)
from v4.pose import get_hand_pose_estimator
from v4.personal_activity import (
    build_personal_object_intervals,
    suppress_personal_object_false_interactions,
)
from v4.tracking import StableTracker
from video_utils import inspect_video


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "fixed_instrument_interaction"
INSTRUMENT_COLORS = [
    (62, 196, 255),
    (100, 220, 120),
    (255, 160, 80),
    (220, 120, 255),
    (255, 210, 80),
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_over_smaller(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(1.0, min(_bbox_area(a), _bbox_area(b)))


def monitoring_pose_detections(
    poses: list[dict[str, Any]],
    *,
    frame_size: tuple[int, int],
    min_person_area_ratio: float = 0.018,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert raw poses into monitoring people while rejecting small pose fragments."""
    width, height = frame_size
    frame_area = max(1.0, float(width * height))
    eligible = [
        pose
        for pose in poses
        if _bbox_area(list(pose["bbox"])) / frame_area >= float(min_person_area_ratio)
    ]
    # Split-pose fragments often have a slightly higher confidence than the
    # full body. Rank by confidence and covered area together so the full
    # operator survives suppression.
    eligible.sort(
        key=lambda item: float(item.get("confidence") or 0.0)
        * math.sqrt(max(1.0, _bbox_area(list(item["bbox"])))),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    duplicate_count = 0
    for pose in eligible:
        if any(
            _intersection_over_smaller(list(pose["bbox"]), list(kept["bbox"])) >= 0.25
            for kept in selected
        ):
            duplicate_count += 1
            continue
        selected.append(pose)

    detections: list[dict[str, Any]] = []
    for pose in selected:
        hands: dict[str, dict[str, Any]] = {}
        joints = pose.get("joints") or {}
        for side, wrist in (pose.get("hands") or {}).items():
            value = dict(wrist)
            elbow = joints.get(f"{side}_elbow") or {}
            if elbow.get("point"):
                value["elbow_point"] = list(elbow["point"])
                value["elbow_confidence"] = float(elbow.get("confidence") or 0.0)
            hands[str(side)] = value
        detections.append(
            {
                "bbox": list(pose["bbox"]),
                "confidence": float(pose.get("confidence") or 0.0),
                "class_id": 0,
                "class_name": "person",
                "raw_class_name": "person",
                "display_name": "人员",
                "detector_source": "yolo11n_pose_fixed_camera_monitoring",
                "tracking_enabled": True,
                "hands": hands,
            }
        )
    return detections, {
        "raw_pose_count": len(poses),
        "small_pose_fragments_ignored": len(poses) - len(eligible),
        "duplicate_pose_fragments_ignored": duplicate_count,
        "monitoring_person_count": len(detections),
    }


def _mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _draw_overlay(
    frame_bgr: np.ndarray,
    *,
    regions: FixedInstrumentRegions,
    tracks: list[dict[str, Any]],
    interaction_frame: dict[str, Any],
    timestamp: float,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    # Instrument masks remain active in the interaction engine. They are
    # intentionally not drawn on the exported video so the result stays clean.
    image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    small_font = _font(16)
    large_font = _font(22)
    for index, instrument in enumerate(regions.instruments):
        mask = regions.core_masks[instrument.instrument_id]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        point = (int(xs.min()), max(0, int(ys.min()) - 19))
        draw.text(
            point,
            f"{instrument.instrument_id[-3:]} {instrument.display_name}",
            fill=INSTRUMENT_COLORS[index % len(INSTRUMENT_COLORS)],
            font=small_font,
            stroke_width=2,
            stroke_fill=(20, 20, 20),
        )

    state_by_person = {
        str(item["person_id"]): item for item in interaction_frame.get("people") or []
    }
    for track in tracks:
        x1, y1, x2, y2 = [int(round(value)) for value in track["bbox"]]
        state = state_by_person.get(str(track["track_id"])) or {}
        active = bool(state.get("interacting"))
        color = (255, 80, 70) if active else (80, 220, 255)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        active_names = [item["instrument_name"] for item in state.get("interactions") or []]
        label = (
            f"{track['track_id']} 正在操作：{'、'.join(active_names)}"
            if active_names
            else f"{track['track_id']} 未判定操作"
        )
        label_y = max(0, y1 - 28)
        draw.text(
            (x1, label_y),
            label,
            fill=color,
            font=small_font,
            stroke_width=2,
            stroke_fill=(15, 15, 15),
        )
        # Wrist/elbow points remain available to the interaction engine but are
        # intentionally hidden from the exported video.

    interacting_people = [item for item in state_by_person.values() if item.get("interacting")]
    global_text = "当前：有人正在操作仪器" if interacting_people else "当前：未判定有人操作仪器"
    draw.rectangle((8, 8, 350, 43), fill=(15, 18, 22, 210))
    draw.text((16, 11), global_text, fill=(255, 255, 255), font=large_font)
    draw.text(
        (frame_bgr.shape[1] - 135, frame_bgr.shape[0] - 28),
        f"{timestamp:6.2f} 秒",
        fill=(255, 255, 255),
        font=small_font,
        stroke_width=2,
        stroke_fill=(20, 20, 20),
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _output_dir(video_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = OUTPUT_ROOT / f"{video_path.stem}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def aggregate_person_instrument_interactions(
    hand_interactions: list[dict[str, Any]],
    *,
    merge_gap_seconds: float = 1.5,
) -> list[dict[str, Any]]:
    """Merge hand-level evidence into the person-instrument events used by the UI."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in hand_interactions:
        key = (str(item["person_id"]), str(item["instrument_id"]))
        groups.setdefault(key, []).append(item)

    merged: list[dict[str, Any]] = []
    counter = 0
    for (person_id, instrument_id), episodes in sorted(groups.items()):
        current: dict[str, Any] | None = None
        for episode in sorted(episodes, key=lambda item: (float(item["start_time"]), float(item["end_time"]))):
            start = float(episode["start_time"])
            end = float(episode["end_time"])
            if current is None or start - float(current["end_time"]) > float(merge_gap_seconds):
                if current is not None:
                    merged.append(current)
                counter += 1
                current = {
                    "interaction_id": f"person_instrument_interaction_{counter:04d}",
                    "person_id": person_id,
                    "instrument_id": instrument_id,
                    "instrument_name": str(episode["instrument_name"]),
                    "start_time": start,
                    "end_time": end,
                    "confidence": float(episode.get("confidence") or 0.0),
                    "hands": {str(episode["hand"])},
                    "evidence": set(episode.get("evidence") or []),
                    "hand_interaction_ids": {str(episode["interaction_id"])},
                }
            else:
                current["end_time"] = max(float(current["end_time"]), end)
                current["confidence"] = max(
                    float(current["confidence"]), float(episode.get("confidence") or 0.0)
                )
                current["hands"].add(str(episode["hand"]))
                current["evidence"].update(episode.get("evidence") or [])
                current["hand_interaction_ids"].add(str(episode["interaction_id"]))
        if current is not None:
            merged.append(current)

    output: list[dict[str, Any]] = []
    for item in sorted(merged, key=lambda value: (value["start_time"], value["person_id"])):
        start = float(item["start_time"])
        end = float(item["end_time"])
        output.append(
            {
                **item,
                "start_time": round(start, 6),
                "end_time": round(end, 6),
                "duration_seconds": round(max(0.0, end - start), 4),
                "confidence": round(float(item["confidence"]), 4),
                "hands": sorted(item["hands"]),
                "evidence": sorted(item["evidence"]),
                "hand_interaction_ids": sorted(item["hand_interaction_ids"]),
            }
        )
    return output


def _interaction_frame_at(
    timestamp: float,
    tracks: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    camera_id: str,
) -> dict[str, Any]:
    people: list[dict[str, Any]] = []
    for track in tracks:
        person_id = str(track["track_id"])
        active = [
            item
            for item in interactions
            if item["person_id"] == person_id
            and float(item["start_time"]) <= timestamp <= float(item["end_time"])
        ]
        people.append(
            {
                "person_id": person_id,
                "interacting": bool(active),
                "state": "interacting" if active else "not_interacting",
                "interactions": active,
                "candidates": [],
            }
        )
    return {
        "timestamp": round(timestamp, 6),
        "camera_id": camera_id,
        "people": people,
        "relations": [],
    }


def analyze_fixed_instrument_video(
    video_path: str | Path,
    *,
    camera_id: str,
    detection_fps: float = 5.0,
    min_person_area_ratio: float = 0.018,
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    output_dir: str | Path | None = None,
    decision_backend: str = "rule",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the standalone fixed-camera interaction module on one video."""
    started = time.perf_counter()
    source = Path(video_path).resolve()
    info = inspect_video(source)
    sample_fps = max(2.0, min(float(detection_fps), float(info.fps)))
    result_dir = Path(output_dir).resolve() if output_dir else _output_dir(source)
    result_dir.mkdir(parents=True, exist_ok=True)
    regions = FixedInstrumentRegions.from_file(
        camera_id=camera_id,
        frame_size=(info.width, info.height),
        calibration_path=calibration_path,
    )
    pose_estimator = get_hand_pose_estimator()
    tracker = StableTracker(detection_fps=sample_fps, hold_seconds=0.7, archive_seconds=2.0)
    engine = FixedInstrumentInteractionEngine(regions)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{source}")
    frame_records: list[dict[str, Any]] = []
    detector_stats = {
        "sample_count": 0,
        "raw_pose_count": 0,
        "small_pose_fragments_ignored": 0,
        "duplicate_pose_fragments_ignored": 0,
        "monitoring_person_count": 0,
        "pose_inference_seconds": 0.0,
    }
    next_sample_time = 0.0
    last_tracks: list[dict[str, Any]] = []
    last_interaction_frame: dict[str, Any] = {
        "timestamp": 0.0,
        "camera_id": camera_id,
        "people": [],
        "relations": [],
    }
    frame_index = 0
    last_timestamp = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / max(1.0, float(info.fps))
            last_timestamp = timestamp
            if timestamp + 1e-7 >= next_sample_time:
                poses, elapsed, _ = pose_estimator.detect(
                    frame,
                    imgsz=640,
                    suppress_split_people=False,
                )
                detections, counts = monitoring_pose_detections(
                    poses,
                    frame_size=(info.width, info.height),
                    min_person_area_ratio=min_person_area_ratio,
                )
                last_tracks = tracker.update(
                    detections,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    frame_size=(info.width, info.height),
                )
                last_interaction_frame = engine.update(timestamp=timestamp, people=last_tracks)
                detector_stats["sample_count"] += 1
                detector_stats["pose_inference_seconds"] += elapsed
                for key, value in counts.items():
                    detector_stats[key] += int(value)
                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": round(timestamp, 6),
                        "people": last_tracks,
                        "interaction": last_interaction_frame,
                        "pose_counts": counts,
                    }
                )
                next_sample_time += 1.0 / sample_fps
                if progress_callback and detector_stats["sample_count"] % max(1, round(sample_fps * 2)) == 0:
                    progress_callback(
                        f"已分析 {timestamp:.1f}/{info.duration_seconds:.1f} 秒"
                    )
            frame_index += 1
    finally:
        cap.release()

    engine_result = engine.finalize(last_timestamp)
    raw_hand_interactions = list(engine_result["interactions"])
    personal_object_intervals = build_personal_object_intervals(
        frame_records,
        rules=regions.temporal_rules,
    )
    rule_hand_interactions, rule_suppressed_interactions = suppress_personal_object_false_interactions(
        raw_hand_interactions,
        personal_object_intervals,
        rules=regions.temporal_rules,
    )
    selected_backend = str(decision_backend).strip().lower()
    if selected_backend not in {"rule", "qwen_hybrid"}:
        raise ValueError(f"Unknown decision_backend: {decision_backend}")
    qwen_review: dict[str, Any] | None = None
    if selected_backend == "qwen_hybrid":
        from v4.qwen_hybrid_review import review_candidate_interactions_with_qwen

        qwen_review = review_candidate_interactions_with_qwen(
            source,
            regions=regions,
            frame_records=frame_records,
            raw_hand_interactions=raw_hand_interactions,
            known_suppressed_interactions=rule_suppressed_interactions,
            output_dir=result_dir,
            # Container duration points just past the final decodable frame.
            # Keep Qwen's last sample on the actual final frame instead of
            # seeking to EOF (for example 5.96 s for a 25 fps/149-frame clip).
            video_duration=max(
                0.0,
                float(info.duration_seconds) - 1.0 / max(1.0, float(info.fps)),
            ),
            progress_callback=progress_callback,
        )
        hand_interactions = list(qwen_review["approved_hand_interactions"])
        suppressed_interactions = list(qwen_review["rejected_candidates"])
    else:
        hand_interactions = rule_hand_interactions
        suppressed_interactions = rule_suppressed_interactions
    person_instrument_interactions = aggregate_person_instrument_interactions(hand_interactions)
    engine_result["raw_hand_interactions"] = raw_hand_interactions
    engine_result["hand_interactions"] = hand_interactions
    engine_result["suppressed_interactions"] = suppressed_interactions
    engine_result["personal_object_intervals"] = personal_object_intervals
    engine_result["decision_backend"] = selected_backend
    engine_result["rule_hand_interactions"] = rule_hand_interactions
    engine_result["rule_suppressed_interactions"] = rule_suppressed_interactions
    engine_result["qwen_review"] = qwen_review
    engine_result["person_instrument_interactions"] = person_instrument_interactions
    engine_result["interactions"] = person_instrument_interactions
    engine_result["summary"] = {
        "interaction_count": len(person_instrument_interactions),
        "hand_evidence_episode_count": len(hand_interactions),
        "suppressed_false_interaction_count": len(suppressed_interactions),
        "person_ids": sorted({item["person_id"] for item in person_instrument_interactions}),
        "instrument_ids": sorted(
            {item["instrument_id"] for item in person_instrument_interactions}
        ),
        "decision_backend": selected_backend,
        "qwen_review_count": (
            int(qwen_review["summary"]["review_count"]) if qwen_review else 0
        ),
        "qwen_approved_window_count": (
            int(qwen_review["summary"]["approved_window_count"]) if qwen_review else 0
        ),
    }

    raw_video_path = result_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(info.fps),
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建标注视频：{raw_video_path}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"无法重新打开视频进行标注：{source}")
    record_index = 0
    annotation_frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = annotation_frame_index / max(1.0, float(info.fps))
            while (
                record_index + 1 < len(frame_records)
                and float(frame_records[record_index + 1]["timestamp"]) <= timestamp
            ):
                record_index += 1
            tracks = frame_records[record_index]["people"] if frame_records else []
            interaction_frame = _interaction_frame_at(
                timestamp,
                tracks,
                person_instrument_interactions,
                camera_id,
            )
            writer.write(
                _draw_overlay(
                    frame,
                    regions=regions,
                    tracks=tracks,
                    interaction_frame=interaction_frame,
                    timestamp=timestamp,
                )
            )
            annotation_frame_index += 1
    finally:
        cap.release()
        writer.release()

    browser_video_path = result_dir / "annotated.mp4"
    video_info = transcode_to_browser_mp4(raw_video_path, browser_video_path, info.fps)
    raw_video_path.unlink(missing_ok=True)
    detector_stats["pose_inference_seconds"] = round(
        float(detector_stats["pose_inference_seconds"]), 4
    )
    output = {
        "schema_version": "0.1",
        "module": "fixed_instrument_interaction",
        "source_video": str(source),
        "camera_id": camera_id,
        "video": {
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "duration_seconds": round(info.duration_seconds, 4),
        },
        "sampling": {
            "detection_fps": sample_fps,
            "min_person_area_ratio": min_person_area_ratio,
        },
        "detector": {
            **detector_stats,
            "pose_model": pose_estimator.metadata(),
        },
        "regions": regions.metadata(),
        "interaction_result": engine_result,
        "tracker": tracker.export(),
        "frames": frame_records,
        "artifacts": {
            "annotated_video": str(browser_video_path),
            "result_json": str(result_dir / "result.json"),
            "video_codec": video_info.get("codec"),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    (result_dir / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="固定摄像头实验仪器交互分析")
    parser.add_argument("video", type=Path)
    parser.add_argument("--camera-id", default="lab_camera_view_2")
    parser.add_argument("--detection-fps", type=float, default=5.0)
    parser.add_argument("--min-person-area-ratio", type=float, default=0.018)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--decision-backend",
        choices=("rule", "qwen_hybrid"),
        default="rule",
    )
    args = parser.parse_args()
    result = analyze_fixed_instrument_video(
        args.video,
        camera_id=args.camera_id,
        detection_fps=args.detection_fps,
        min_person_area_ratio=args.min_person_area_ratio,
        output_dir=args.output_dir,
        decision_backend=args.decision_backend,
        progress_callback=print,
    )
    print(json.dumps(result["interaction_result"]["summary"], ensure_ascii=False, indent=2))
    print(result["artifacts"]["annotated_video"])


if __name__ == "__main__":
    main()
