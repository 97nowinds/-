from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from v5.minicpm_realtime_video import (
    analyze_video_realtime,
    create_persistent_resources,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEMO_VIDEO_ROOT = PACKAGE_ROOT / "演示视频"
DEFAULT_CASES = [
    ("operation_positive", DEMO_VIDEO_ROOT / "微信视频2026-08-20_174426_295.mp4", True),
    ("standing_negative", DEMO_VIDEO_ROOT / "微信视频2026-08-20_161614_942.mp4", False),
    ("phone_negative", DEMO_VIDEO_ROOT / "微信视频2026-08-20_162706_056.mp4", False),
]


def run_benchmark(
    *,
    output_root: Path,
    schedule_mode: str,
    no_person_video: Path | None = None,
) -> dict:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = list(DEFAULT_CASES)
    if no_person_video is not None:
        cases.append(("no_person_negative", no_person_video.resolve(), False))
    for _, source, _ in cases:
        if not source.is_file():
            raise FileNotFoundError(source)

    load_started = time.perf_counter()
    judge, detector = create_persistent_resources()
    shared_load_seconds = time.perf_counter() - load_started

    run_started = time.perf_counter()
    case_results = []
    for name, source, expected in cases:
        print(f"\n=== {name}: {source.name} ===", flush=True)
        result = analyze_video_realtime(
            source,
            output_dir=output_root / name,
            schedule_mode=schedule_mode,
            judge_instance=judge,
            person_detector_instance=detector,
        )
        actual = bool(result["summary"]["is_interacting_video"])
        case_results.append(
            {
                "name": name,
                "source": str(source),
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
                "duration_seconds": result["video"]["duration_seconds"],
                "elapsed_seconds": result["elapsed_seconds"],
                "model_invocation_count": result["summary"]["model_invocation_count"],
                "window_count": result["summary"]["window_count"],
                "interaction_count": result["summary"]["interaction_count"],
                "result_json": result["artifacts"]["result_json"],
            }
        )

    processing_seconds = time.perf_counter() - run_started
    total_video_seconds = sum(item["duration_seconds"] for item in case_results)
    output = {
        "schedule_mode": schedule_mode,
        "model_loaded_once": True,
        "person_detector_loaded_once": True,
        "shared_load_seconds": round(shared_load_seconds, 4),
        "processing_seconds_excluding_shared_load": round(processing_seconds, 4),
        "wall_seconds_including_shared_load": round(
            shared_load_seconds + processing_seconds, 4
        ),
        "total_video_seconds": round(total_video_seconds, 4),
        "processing_realtime_factor": round(
            processing_seconds / max(0.001, total_video_seconds), 4
        ),
        "all_cases_passed": all(item["passed"] for item in case_results),
        "total_model_invocations": sum(
            item["model_invocation_count"] for item in case_results
        ),
        "cases": case_results,
    }
    target = output_root / "benchmark.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    print(target, flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent MiniCPM real-time benchmark")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--schedule-mode",
        choices=["adaptive", "normal2_active1"],
        default="normal2_active1",
    )
    parser.add_argument("--no-person-video", type=Path)
    args = parser.parse_args()
    run_benchmark(
        output_root=args.output_root,
        schedule_mode=args.schedule_mode,
        no_person_video=args.no_person_video,
    )


if __name__ == "__main__":
    main()
