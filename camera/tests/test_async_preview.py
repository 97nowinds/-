import threading
import unittest
from collections import deque

import numpy as np

from app import CameraManager, CameraWorker


class AsyncPreviewTests(unittest.TestCase):
    @staticmethod
    def worker(detected_at=10.0):
        worker = CameraWorker.__new__(CameraWorker)
        worker.camera = {
            "id": "cam_2",
            "track_hold_seconds": 1.0,
        }
        worker.lock = threading.RLock()
        worker.yolo_tracks = [
            {
                "track_id": "global_1",
                "local_id": 7,
                "box": (20, 60, 40, 80),
                "confidence": 0.9,
                "counted": True,
            }
        ]
        worker.track_identities = {
            7: {"person_number": "2", "name": "TEST PERSON"}
        }
        worker.last_yolo_tracks_at = detected_at
        worker.preview_overlay_age_seconds = None
        worker.preview_overlay_status = "waiting"
        return worker

    def test_fresh_ai_snapshot_is_drawn_on_newer_capture(self):
        worker = self.worker()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)

        preview = worker.render_async_preview(frame, captured_monotonic=10.4)

        self.assertEqual(worker.preview_overlay_status, "fresh")
        self.assertEqual(worker.preview_overlay_age_seconds, 0.4)
        self.assertTrue(np.any(preview[60, 20] != 0))

    def test_stale_ai_snapshot_is_not_drawn_as_current_person(self):
        worker = self.worker()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)

        preview = worker.render_async_preview(frame, captured_monotonic=12.0)

        self.assertEqual(worker.preview_overlay_status, "stale")
        self.assertEqual(worker.preview_overlay_age_seconds, 2.0)
        self.assertTrue(np.all(preview[60, 20] == 0))

    def test_jpeg_profiles_are_counted_by_active_subscriber(self):
        worker = self.worker()
        worker.jpeg_condition = threading.Condition()
        worker.jpeg_subscribers = {False: 0, True: 0}

        worker.add_jpeg_subscriber(True)
        worker.add_jpeg_subscriber(True)
        worker.remove_jpeg_subscriber(True)
        worker.remove_jpeg_subscriber(False)

        self.assertEqual(worker.jpeg_subscribers, {False: 0, True: 1})

    def test_synchronized_batch_reuses_one_atomic_camera_snapshot(self):
        class FakeWorker:
            def __init__(self, captured_monotonic, history=None):
                self.capture_condition = threading.Condition()
                self.latest_capture = (
                    1,
                    1,
                    1000.0 + captured_monotonic,
                    captured_monotonic,
                    np.zeros((36, 64, 3), dtype=np.uint8),
                )
                self.capture_history = deque(
                    [
                        (
                            index,
                            1,
                            1000.0 + value,
                            value,
                            np.zeros((36, 64, 3), dtype=np.uint8),
                        )
                        for index, value in enumerate(
                            history or [captured_monotonic], start=1
                        )
                    ],
                    maxlen=16,
                )
                self.preview_frames = 0

            @staticmethod
            def resize(frame):
                return frame

            @staticmethod
            def render_async_preview(frame, _captured_monotonic):
                return frame

            def record_preview_frame(self):
                self.preview_frames += 1

        manager = CameraManager.__new__(CameraManager)
        manager.workers = {
            "cam_1": FakeWorker(10.20, [10.00, 10.10, 10.20]),
            "cam_2": FakeWorker(10.22, [10.02, 10.12, 10.22]),
            "cam_entrance": FakeWorker(10.11, [10.01, 10.11]),
        }
        manager.synchronized_preview_lock = threading.Lock()
        manager.synchronized_preview_batches = {}
        manager.synchronized_preview_order = deque(maxlen=4)

        first = manager.synchronized_preview_batch("batch_1")
        second = manager.synchronized_preview_batch("batch_1")

        self.assertIs(first, second)
        self.assertEqual(set(first["frames"]), set(manager.workers))
        self.assertAlmostEqual(first["skew_seconds"], 0.02)
        chosen_times = {
            round(item["captured_monotonic"], 2)
            for item in first["frames"].values()
        }
        self.assertEqual(chosen_times, {10.10, 10.11, 10.12})
        self.assertTrue(
            all(worker.preview_frames == 1 for worker in manager.workers.values())
        )


if __name__ == "__main__":
    unittest.main()
