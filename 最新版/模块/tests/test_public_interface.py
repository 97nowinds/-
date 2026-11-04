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
from lab_instrument_interaction.stream import _masked_motion_score


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

    def test_busy_stream_keeps_recent_bounded_pending_frames(self) -> None:
        inference_lock = threading.RLock()
        inference_lock.acquire()
        try:
            with patch(
                "lab_instrument_interaction.stream.get_hand_pose_estimator",
                return_value=_FakePose(),
            ):
                stream = InstrumentInteractionStream(
                    camera_id="lab_camera_view_2",
                    judge=_FakeJudge(),
                    person_detector=_FakePersonDetector(),
                    inference_lock=inference_lock,
                    frame_size=(768, 432),
                    max_queue_frames=24,
                    pending_sample_fps=8.0,
                    max_pending_seconds=2.5,
                )
            frame = np.zeros((432, 768, 3), dtype=np.uint8)
            for index in range(250):
                stream.process_frame(
                    frame,
                    timestamp=index / 25.0,
                    persons=[],
                )
            snapshot = stream.poll()
            self.assertLessEqual(snapshot["metrics"]["queued_frames"], 24)
            self.assertGreater(snapshot["metrics"]["dropped_frames"], 0)
            self.assertGreater(snapshot["metrics"]["coalesced_frames"], 0)
        finally:
            inference_lock.release()
        closed = stream.close()
        self.assertEqual(closed["status"], "closed")

    def test_strong_vrm_event_is_hint_not_public_interaction(self) -> None:
        with patch(
            "lab_instrument_interaction.stream.get_hand_pose_estimator",
            return_value=_FakePose(),
        ):
            stream = InstrumentInteractionStream(
                camera_id="lab_camera_view_1",
                judge=_FakeJudge(),
                person_detector=_FakePersonDetector(),
                inference_lock=threading.RLock(),
                frame_size=(760, 388),
            )
        frame = np.zeros((388, 760, 3), dtype=np.uint8)
        stream._initialize_for_frame(frame)
        event = {
            "person_id": "person_03",
            "instrument_id": "instrument_004",
            "instrument_name": "电子分析天平",
            "start_time": 1.0,
            "end_time": 2.0,
            "confidence": 1.0,
            "evidence": ["wrist_inside_instrument_core"],
        }
        stream._vrm.completed_interactions.append(event)
        with (
            patch(
                "lab_instrument_interaction.stream.aggregate_person_instrument_interactions",
                return_value=[event],
            ),
            patch(
                "lab_instrument_interaction.stream.build_personal_object_intervals",
                return_value=[],
            ),
            patch(
                "lab_instrument_interaction.stream.suppress_personal_object_false_interactions",
                return_value=([event], []),
            ),
            patch(
                "lab_instrument_interaction.stream._strong_rule_event",
                return_value=(True, "test"),
            ),
        ):
            stream._consume_vrm_events(2.1)
        snapshot = stream.poll(after_event_sequence=0)
        self.assertEqual(snapshot["current_interactions"], [])
        self.assertEqual(snapshot["events"], [])
        self.assertIn(("person_03", "instrument_004"), stream._vrm_hints)
        stream.close()

    def test_target_mask_motion_ignores_neighbour_activity(self) -> None:
        first = np.zeros((80, 120, 3), dtype=np.uint8)
        second = first.copy()
        second[20:40, 15:35] = 255
        target = np.zeros((80, 120), dtype=np.uint8)
        target[18:42, 13:37] = 255
        neighbour = np.zeros((80, 120), dtype=np.uint8)
        neighbour[18:42, 75:99] = 255
        target_score = _masked_motion_score([first, second], target)
        neighbour_score = _masked_motion_score([first, second], neighbour)
        self.assertGreater(target_score, neighbour_score)
        self.assertEqual(neighbour_score, 0.0)


if __name__ == "__main__":
    unittest.main()
