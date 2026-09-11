from __future__ import annotations

import unittest
import threading
import time
from unittest.mock import patch

import numpy as np

from lab_instrument_interaction import InstrumentInteractionModule
from lab_instrument_interaction.identity import (
    attach_external_identities,
    normalize_identity_timeline,
)
from lab_instrument_interaction.stream import InstrumentInteractionStream


class _FakePose:
    def detect(self, frame, *, imgsz=640, suppress_split_people=False):
        return [], 0.0, 0


class _FakeJudge:
    pass


class _FakePersonDetector:
    def detect(self, frame):
        return []


class IdentityTimelineTests(unittest.TestCase):
    def test_normalizes_aliases_and_sorts_frames(self) -> None:
        frames = normalize_identity_timeline(
            [
                {
                    "timestamp_seconds": 1.0,
                    "people": [
                        {
                            "track_id": "employee-7",
                            "name": "张三",
                            "bbox": [100, 20, 220, 320],
                        }
                    ],
                },
                {"timestamp": 0.0, "persons": []},
            ]
        )
        self.assertEqual([item["timestamp"] for item in frames], [0.0, 1.0])
        self.assertEqual(frames[1]["persons"][0]["person_id"], "employee-7")
        self.assertEqual(frames[1]["persons"][0]["person_name"], "张三")

    def test_matches_external_identity_without_changing_event(self) -> None:
        events = [
            {
                "interaction_id": "event-1",
                "person_id": "person_01",
                "instrument_id": "instrument_006",
                "instrument_name": "高速离心机",
                "start_time": 1.0,
                "end_time": 2.0,
                "target_person_observations": [
                    {"timestamp": 1.5, "bbox": [100, 20, 220, 320]}
                ],
            }
        ]
        timeline = [
            {
                "timestamp": 1.5,
                "persons": [
                    {
                        "person_id": "employee-7",
                        "person_name": "张三",
                        "bbox": [102, 22, 219, 318],
                    },
                    {
                        "person_id": "employee-8",
                        "person_name": "李四",
                        "bbox": [400, 30, 520, 330],
                    },
                ],
            }
        ]
        enriched, metadata = attach_external_identities(
            events,
            identity_timeline=timeline,
            rule_result={"frames": []},
        )
        self.assertEqual(enriched[0]["person_id"], "employee-7")
        self.assertEqual(enriched[0]["person_name"], "张三")
        self.assertEqual(enriched[0]["internal_person_id"], "person_01")
        self.assertEqual(enriched[0]["instrument_id"], "instrument_006")
        self.assertEqual(metadata["matched_event_count"], 1)

    def test_missing_timeline_keeps_internal_identity(self) -> None:
        events = [
            {
                "person_id": "person_02",
                "instrument_id": "instrument_009",
                "start_time": 3.0,
                "end_time": 4.0,
            }
        ]
        enriched, metadata = attach_external_identities(
            events,
            identity_timeline=None,
            rule_result={"frames": []},
        )
        self.assertEqual(enriched[0]["person_id"], "person_02")
        self.assertEqual(enriched[0]["identity_source"], "internal_tracker")
        self.assertEqual(metadata["source"], "internal_tracker")


class PublicFacadeTests(unittest.TestCase):
    def test_facade_reports_supported_cameras_without_loading_models(self) -> None:
        module = InstrumentInteractionModule(preload_models=False)
        try:
            self.assertEqual(module.interface_version, "1.1")
            self.assertIn("lab_camera_view_1", module.supported_cameras())
            self.assertIn("lab_camera_view_2", module.supported_cameras())
            self.assertTrue(module.health_check()["ok"])
        finally:
            module.close()


class StreamInterfaceTests(unittest.TestCase):
    def test_process_frame_is_non_blocking_and_worker_reaches_running(self) -> None:
        with patch(
            "lab_instrument_interaction.stream.get_hand_pose_estimator",
            return_value=_FakePose(),
        ):
            stream = InstrumentInteractionStream(
                camera_id="lab_camera_view_2",
                judge=_FakeJudge(),
                person_detector=_FakePersonDetector(),
                inference_lock=threading.RLock(),
                frame_size=(768, 432),
            )
        frame = np.zeros((432, 768, 3), dtype=np.uint8)
        started = time.perf_counter()
        for index in range(60):
            response = stream.process_frame(
                frame,
                timestamp=index / 25.0,
                persons=[],
            )
            self.assertTrue(response["accepted_frame"])
        self.assertLess(time.perf_counter() - started, 1.0)
        deadline = time.time() + 3.0
        snapshot = stream.poll()
        while snapshot["status"] not in {"running", "error"} and time.time() < deadline:
            time.sleep(0.02)
            snapshot = stream.poll()
        self.assertEqual(snapshot["status"], "running")
        self.assertIsNone(snapshot["error"])
        self.assertEqual(snapshot["metrics"]["received_frames"], 60)
        closed = stream.close()
        self.assertEqual(closed["status"], "closed")


if __name__ == "__main__":
    unittest.main()
