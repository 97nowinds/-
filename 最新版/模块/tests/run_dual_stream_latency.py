"""双路实时流压力测试：验证共享模型时不会长期积压旧帧。

本脚本故意让两个流同时播放同一段正样本，以制造比普通现场更高的
MiniCPM 调用竞争。它只验证实时队列与并发调度，不代表两个摄像头的
识别准确率测试。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from lab_instrument_interaction import InstrumentInteractionModule


DEFAULT_VIDEO = Path(r"D:\输入视频\微信视频2026-08-20_174426_295.mp4")
DEFAULT_OUTPUT_ROOT = Path(r"D:\视觉模型\最新版\开发测试结果\dual_stream_latency")


def _run_stream(
    *,
    stream,
    video_path: Path,
    barrier: threading.Barrier,
    results: dict[str, dict],
    result_key: str,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        results[result_key] = {"error": f"无法打开视频：{video_path}"}
        return

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_interval = 1.0 / fps
    max_lag = 0.0
    max_queued = 0
    latest_response = None
    frames = 0
    started = time.monotonic()
    barrier.wait()
    stream_start = time.monotonic()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frames * frame_interval
            response = stream.process_frame(
                frame,
                timestamp=timestamp,
                persons=[
                    {
                        "person_id": f"stress_{result_key}",
                        "person_name": result_key,
                        "bbox": [110, 0, 500, frame.shape[0]],
                        "confidence": 0.99,
                    }
                ],
            )
            latest_response = response
            max_lag = max(
                max_lag,
                float(response["analysis_lag_seconds"] or 0.0),
            )
            max_queued = max(max_queued, int(response["metrics"]["queued_frames"]))
            frames += 1

            target = stream_start + frames * frame_interval
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        drain_deadline = time.monotonic() + 8.0
        while time.monotonic() < drain_deadline:
            latest_response = stream.poll()
            max_lag = max(
                max_lag,
                float(latest_response["analysis_lag_seconds"] or 0.0),
            )
            max_queued = max(
                max_queued,
                int(latest_response["metrics"]["queued_frames"]),
            )
            if not latest_response["analysis_pending"]:
                break
            time.sleep(0.1)

        results[result_key] = {
            "video": str(video_path),
            "frames": frames,
            "wall_seconds": round(time.monotonic() - started, 3),
            "max_analysis_lag_seconds": round(max_lag, 3),
            "max_queued_frames": max_queued,
            "final": latest_response,
        }
    except Exception as exc:  # noqa: BLE001 - 测试报告必须保存完整错误
        results[result_key] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-lag", type=float, default=5.0)
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    capture = cv2.VideoCapture(str(args.video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("无法读取压力测试视频尺寸")

    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    module = InstrumentInteractionModule(preload_models=True)
    streams = [
        module.create_stream(camera_id="lab_camera_view_2", frame_size=(width, height)),
        module.create_stream(camera_id="lab_camera_view_2", frame_size=(width, height)),
    ]
    barrier = threading.Barrier(2)
    threads = [
        threading.Thread(
            target=_run_stream,
            kwargs={
                "stream": stream,
                "video_path": args.video,
                "barrier": barrier,
                "results": results,
                "result_key": f"stream_{index + 1}",
            },
            daemon=True,
        )
        for index, stream in enumerate(streams)
    ]

    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        for stream in streams:
            stream.close()
        module.close()

    errors = [item.get("error") for item in results.values() if item.get("error")]
    lags = [
        float(item.get("max_analysis_lag_seconds", 999.0))
        for item in results.values()
        if not item.get("error")
    ]
    passed = len(results) == 2 and not errors and all(lag <= args.max_lag for lag in lags)
    report = {
        "passed": passed,
        "purpose": "双路共享 MiniCPM 时的实时队列压力测试，不是摄像头准确率测试",
        "thresholds": {"max_analysis_lag_seconds": args.max_lag},
        "streams": results,
    }
    report_path = run_dir / "dual_stream_latency.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
