from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab_instrument_interaction import InstrumentInteractionModule


VIDEO_ROOT = Path("D:/输入视频")
OUTPUT_ROOT = Path("D:/视觉模型/最新版/开发测试结果/public_api_regression")


def _single_person_timeline(duration_seconds: int) -> list[dict]:
    return [
        {
            "timestamp": float(second),
            "persons": [
                {
                    "person_id": "employee_001",
                    "person_name": "接口测试人员",
                    "bbox": [250, 0, 650, 432],
                    "confidence": 0.99,
                }
            ],
        }
        for second in range(duration_seconds + 1)
    ]


def main() -> None:
    run_root = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    module = InstrumentInteractionModule(preload_models=True)
    judge_identity = id(module._judge)
    try:
        positive = module.analyze_video(
            VIDEO_ROOT / "微信视频2026-08-20_174426_295.mp4",
            camera_id="lab_camera_view_2",
            identity_timeline=_single_person_timeline(90),
            output_dir=run_root / "positive",
        )
        assert any(
            event["instrument_id"] == "instrument_006"
            and event["person_id"] == "employee_001"
            and event["person_name"] == "接口测试人员"
            for event in positive["interactions"]
        ), positive["interactions"]
        assert id(module._judge) == judge_identity

        negative = module.analyze_video(
            VIDEO_ROOT / "微信视频2026-08-20_161614_942.mp4",
            camera_id="lab_camera_view_2",
            identity_timeline=_single_person_timeline(30),
            output_dir=run_root / "negative",
        )
        assert not negative["interactions"], negative["interactions"]
        assert id(module._judge) == judge_identity

        summary = {
            "passed": True,
            "model_reused": True,
            "positive_summary": positive["summary"],
            "negative_summary": negative["summary"],
            "positive_result": positive["artifacts"]["result_json"],
            "negative_result": negative["artifacts"]["result_json"],
        }
        (run_root / "regression_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        module.close()


if __name__ == "__main__":
    main()
