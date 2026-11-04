from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from settings import PROJECT_ROOT
from v2.media_utils import transcode_to_browser_mp4
from v4.fixed_instrument_interaction import (
    DEFAULT_CALIBRATION_PATH,
    FixedInstrumentRegions,
)
from v5.minicpm_interaction_judge import MiniCPMInteractionJudge


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "minicpm_v46_interaction"
PERSON_WEIGHTS = PROJECT_ROOT / "models" / "detectors" / "yolo11n-pose.pt"


def _font(size: int) -> ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _decode_video(path: Path) -> tuple[list[np.ndarray], float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"视频元数据或帧无效：{path}")
    return frames, fps, width, height


def _window_starts(duration: float, window_seconds: float, step_seconds: float) -> list[float]:
    maximum = max(0.0, duration - window_seconds)
    starts: list[float] = []
    value = 0.0
    while value <= maximum + 1e-7:
        starts.append(round(value, 3))
        value += step_seconds
    if not starts or maximum - starts[-1] > 0.05:
        starts.append(round(maximum, 3))
    return sorted(set(starts))


def _sample_frame_indices(
    *,
    start: float,
    window_seconds: float,
    frame_count: int,
    fps: float,
    maximum_index: int,
) -> list[int]:
    return [
        min(
            maximum_index,
            max(0, round((start + window_seconds * index / max(1, frame_count - 1)) * fps)),
        )
        for index in range(frame_count)
    ]


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("目标仪器标定掩膜为空")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _rect_intersection_area(a: list[float], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(float(a[0]), b[0]), max(float(a[1]), b[1])
    x2, y2 = min(float(a[2]), b[2]), min(float(a[3]), b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class PersonTargetDetector:
    """Person boxes only. Pose keypoints are intentionally ignored."""

    def __init__(
        self,
        weights: Path = PERSON_WEIGHTS,
        *,
        image_size: int = 640,
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(weights)
        self.weights = weights.resolve()
        self.image_size = int(image_size)
        self.model = YOLO(str(self.weights))

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        result = self.model.predict(
            frame,
            conf=0.15,
            imgsz=self.image_size,
            classes=[0],
            device=0,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        output: list[dict[str, Any]] = []
        for box in result.boxes:
            output.append(
                {
                    "bbox": [float(value) for value in box.xyxy[0].detach().cpu().tolist()],
                    "confidence": float(box.conf[0].item()),
                }
            )
        return output


def _select_near_person(
    detections: list[dict[str, Any]],
    *,
    instrument_bounds: tuple[int, int, int, int],
    frame_size: tuple[int, int],
) -> dict[str, Any] | None:
    width, height = frame_size
    ix1, iy1, ix2, iy2 = instrument_bounds
    trigger = (
        max(0, ix1 - 180),
        max(0, iy1 - 170),
        min(width, ix2 + 190),
        min(height, iy2 + 190),
    )
    instrument_center = ((ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for item in detections:
        bbox = list(item["bbox"])
        intersection = _rect_intersection_area(bbox, trigger)
        if intersection <= 0:
            continue
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        distance = float(np.hypot(center[0] - instrument_center[0], center[1] - instrument_center[1]))
        score = intersection / max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        score += 0.35 * max(0.0, 1.0 - distance / max(width, height))
        score += 0.10 * float(item["confidence"])
        candidates.append((score, item))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def _target_crop(
    frames_bgr: list[np.ndarray],
    *,
    person_detector: PersonTargetDetector,
    instrument_mask: np.ndarray,
) -> tuple[list[Image.Image], list[dict[str, Any] | None], tuple[int, int, int, int]]:
    height, width = frames_bgr[0].shape[:2]
    bounds = _mask_bounds(instrument_mask)
    selected_people: list[dict[str, Any] | None] = []
    for frame in frames_bgr:
        selected_people.append(
            _select_near_person(
                person_detector.detect(frame),
                instrument_bounds=bounds,
                frame_size=(width, height),
            )
        )
    person_boxes = [list(item["bbox"]) for item in selected_people if item is not None]
    ix1, iy1, ix2, iy2 = bounds
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
    contours, _ = cv2.findContours(instrument_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_bbox = next((list(item["bbox"]) for item in selected_people if item is not None), None)
    output: list[Image.Image] = []
    rx1, ry1, rx2, ry2 = roi
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


def _save_storyboard(frames: list[Image.Image], target: Path) -> None:
    thumb_width = 240
    thumbs: list[Image.Image] = []
    for frame in frames:
        thumb = frame.copy()
        thumb.thumbnail((thumb_width, 240), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    height = max(frame.height for frame in thumbs)
    canvas = Image.new("RGB", (thumb_width * len(thumbs), height), "black")
    for index, frame in enumerate(thumbs):
        canvas.paste(frame, (index * thumb_width, 0))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=90)


def _smooth_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = False
    negative_streak = 0
    event_start: float | None = None
    events: list[dict[str, Any]] = []
    for index, review in enumerate(reviews):
        recent = reviews[max(0, index - 2) : index + 1]
        positive_count = sum(bool(item["is_interacting"]) for item in recent)
        if not state and len(recent) >= 2 and positive_count >= 2:
            state = True
            positive_starts = [
                float(item["window_start"]) for item in recent if item["is_interacting"]
            ]
            event_start = min(positive_starts)
            negative_streak = 0
        elif state:
            if review["is_interacting"]:
                negative_streak = 0
            else:
                negative_streak += 1
                if negative_streak >= 2:
                    end_time = float(review["window_start"])
                    events.append(
                        {
                            "start_time": round(float(event_start or 0.0), 4),
                            "end_time": round(end_time, 4),
                        }
                    )
                    state = False
                    event_start = None
                    negative_streak = 0
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


def _render_result_video(
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
        raise RuntimeError("无法创建MiniCPM结果视频")
    font = _font(20)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            active = any(
                float(item["start_time"]) <= timestamp <= float(item["end_time"])
                for item in events
            )
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            draw.rectangle((8, 8, 410, 47), fill=(18, 20, 24))
            label = "MiniCPM：正在与高速离心机交互" if active else "MiniCPM：未判定仪器交互"
            draw.text((16, 13), label, font=font, fill=(255, 255, 255))
            writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    transcode_to_browser_mp4(raw, target, fps)
    raw.unlink(missing_ok=True)


def analyze_video(
    video_path: str | Path,
    *,
    camera_id: str = "lab_camera_view_2",
    instrument_id: str = "instrument_006",
    window_seconds: float = 2.0,
    step_seconds: float = 1.0,
    frame_count: int = 8,
    output_dir: str | Path | None = None,
    only_window_start: float | None = None,
) -> dict[str, Any]:
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
    person_detector = PersonTargetDetector()
    judge = MiniCPMInteractionJudge()
    starts = _window_starts(duration, window_seconds, step_seconds)
    if only_window_start is not None:
        starts = [max(0.0, min(float(only_window_start), max(0.0, duration - window_seconds)))]
    reviews: list[dict[str, Any]] = []
    for number, window_start in enumerate(starts, start=1):
        print(
            f"[{number}/{len(starts)}] {window_start:.2f}-"
            f"{window_start + window_seconds:.2f}s",
            flush=True,
        )
        indices = _sample_frame_indices(
            start=window_start,
            window_seconds=window_seconds,
            frame_count=frame_count,
            fps=fps,
            maximum_index=len(frames) - 1,
        )
        sampled = [frames[index] for index in indices]
        target_frames, selected_people, roi = _target_crop(
            sampled,
            person_detector=person_detector,
            instrument_mask=instrument_mask,
        )
        person_frame_count = sum(item is not None for item in selected_people)
        storyboard = result_dir / "review_inputs" / f"window_{window_start:07.2f}.jpg"
        _save_storyboard(target_frames, storyboard)
        if person_frame_count == 0:
            result_data = {
                "is_interacting": False,
                "confidence": 1.0,
                "evidence": "目标仪器附近未检测到人员",
                "raw_response": "",
                "valid_response": True,
                "validation_error": None,
                "latency_seconds": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "model_invoked": False,
            }
        else:
            result_data = {
                **judge.judge(
                    target_frames,
                    instrument_name=instrument.display_name,
                    person_description="第一帧蓝色矩形指定的人员",
                ).as_dict(),
                "model_invoked": True,
            }
        reviews.append(
            {
                "window_start": round(window_start, 4),
                "window_end": round(min(duration, window_start + window_seconds), 4),
                "frame_indices": indices,
                "person_detected_frame_count": person_frame_count,
                "roi": list(roi),
                "storyboard": str(storyboard),
                **result_data,
            }
        )
        print(
            f"  result={result_data['is_interacting']} "
            f"confidence={result_data['confidence']:.2f} "
            f"evidence={result_data['evidence']}",
            flush=True,
        )
    events = _smooth_reviews(reviews)
    annotated = result_dir / "annotated.mp4"
    _render_result_video(
        source,
        annotated,
        fps=fps,
        width=width,
        height=height,
        events=events,
    )
    output = {
        "schema_version": "0.1",
        "module": "minicpm_v46_fixed_instrument_interaction",
        "source_video": str(source),
        "camera_id": camera_id,
        "person_id": "person_01",
        "instrument_id": instrument_id,
        "instrument_name": instrument.display_name,
        "decision_backend": "minicpm_v46_binary_vlm",
        "uses_person_detection": True,
        "uses_wrist_interaction_rules": False,
        "uses_old_interaction_engine": False,
        "uses_action_classification": False,
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
        "model": judge.metadata(),
        "person_detector": {
            "weights": str(person_detector.weights),
            "purpose": "person box and target selection only; pose keypoints ignored",
        },
        "reviews": reviews,
        "interactions": events,
        "summary": {
            "is_interacting_video": bool(events),
            "interaction_count": len(events),
            "window_count": len(reviews),
            "raw_positive_window_count": sum(item["is_interacting"] for item in reviews),
            "model_invocation_count": sum(item["model_invoked"] for item in reviews),
            "invalid_response_count": sum(not item["valid_response"] for item in reviews),
        },
        "artifacts": {
            "annotated_video": str(annotated),
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
    parser = argparse.ArgumentParser(description="MiniCPM-V 4.6 binary instrument interaction")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only-window-start", type=float)
    args = parser.parse_args()
    result = analyze_video(
        args.video,
        output_dir=args.output_dir,
        only_window_start=args.only_window_start,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(result["artifacts"]["annotated_video"])
    print(result["artifacts"]["result_json"])


if __name__ == "__main__":
    main()
