from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab_instrument_interaction import InstrumentInteractionModule


VIDEO_ROOT = Path("D:/输入视频")
OUTPUT_ROOT = Path("D:/视觉模型/最新版/开发测试结果/stream_api_regression")


def run_stream(module: InstrumentInteractionModule, video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream = module.create_stream(
        camera_id="lab_camera_view_2",
        frame_size=(width, height),
    )
    cursor = 0
    events: list[dict] = []
    frame_index = 0
    started = time.perf_counter()
    try:
        next_deadline = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            response = stream.process_frame(
                frame,
                timestamp=timestamp,
                persons=[
                    {
                        "person_id": "employee_001",
                        "person_name": "实时接口测试人员",
                        "bbox": [150, 0, min(width, 470), height],
                        "confidence": 0.99,
                    }
                ],
                after_event_sequence=cursor,
            )
            events.extend(response["events"])
            cursor = response["last_event_sequence"]
            frame_index += 1
            next_deadline += 1.0 / fps
            delay = next_deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        capture.release()

        deadline = time.time() + 20.0
        while time.time() < deadline:
            response = stream.poll(after_event_sequence=cursor)
            events.extend(response["events"])
            cursor = response["last_event_sequence"]
            if not response["analysis_pending"]:
                break
            time.sleep(0.1)
        final = stream.close()
        events.extend(
            item
            for item in final["events"]
            if int(item["event_sequence"]) > cursor
        )
        return {
            "video": str(video_path),
            "wall_seconds": round(time.perf_counter() - started, 3),
            "events": events,
            "final": final,
        }
    finally:
        capture.release()
        stream.close()


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    module = InstrumentInteractionModule(preload_models=True)
    try:
        positive = run_stream(
            module, VIDEO_ROOT / "微信视频2026-08-20_174426_295.mp4"
        )
        negative = run_stream(
            module, VIDEO_ROOT / "微信视频2026-08-20_161614_942.mp4"
        )
        phone_negative = run_stream(
            module, VIDEO_ROOT / "微信视频2026-08-20_162706_056.mp4"
        )
        positive_started = [
            item
            for item in positive["events"]
            if item.get("type") == "interaction_started"
            and item.get("instrument_id") == "instrument_006"
            and item.get("person_id") == "employee_001"
        ]
        negative_started = [
            item
            for item in negative["events"]
            if item.get("type") == "interaction_started"
        ]
        phone_negative_started = [
            item
            for item in phone_negative["events"]
            if item.get("type") == "interaction_started"
        ]
        summary = {
            "passed": (
                1 <= len(positive_started) <= 2
                and not negative_started
                and not phone_negative_started
            ),
            "positive_started_count": len(positive_started),
            "negative_started_count": len(negative_started),
            "phone_negative_started_count": len(phone_negative_started),
            "positive": positive,
            "negative": negative,
            "phone_negative": phone_negative,
        }
        (output_dir / "stream_regression.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["passed"]:
            raise AssertionError("流式正负样本回归失败")
    finally:
        module.close()


if __name__ == "__main__":
    main()
