import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
import numpy as np


def write_timestamps(path, unix_times):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame_index", "unix_time", "local_time"))
        for index, timestamp in enumerate(unix_times):
            writer.writerow((index, timestamp, ""))


def create_session(root, status="complete"):
    directory = root / "worker_001" / "20260912_120000"
    directory.mkdir(parents=True)
    streams = {
        "cam_1": {
            "camera_id": "cam_1",
            "video_file": "cam_1.mp4",
            "timestamps_file": "cam_1_timestamps.csv",
            "fps": 10.0,
            "frames": 4,
        },
        "cam_entrance": {
            "camera_id": "cam_entrance",
            "video_file": "cam_entrance.mp4",
            "timestamps_file": "cam_entrance_timestamps.csv",
            "fps": 10.0,
            "frames": 4,
        },
    }
    metadata = {
        "status": status,
        "subject_id": "worker_001",
        "session_id": directory.name,
        "started_at": "2026-09-12T12:00:00+08:00",
        "elapsed_seconds": 4.0,
        "streams": streams,
    }
    (directory / "info.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    write_timestamps(directory / "cam_1_timestamps.csv", [100.0, 100.1, 100.5, 100.6])
    write_timestamps(
        directory / "cam_entrance_timestamps.csv", [102.0, 102.1, 102.2, 102.3]
    )
    (directory / "cam_1.mp4").write_bytes(b"video")
    (directory / "cam_entrance.mp4").write_bytes(b"video")
    return directory


class RecordingReviewTests(unittest.TestCase):
    def test_video_time_maps_to_frame_before_capture_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = create_session(Path(temporary))
            result = app_module.recording_frame_at(directory, "cam_1", 0.21)

        self.assertEqual(result["frame_index"], 2)
        self.assertEqual(result["video_seconds"], 0.2)
        self.assertEqual(result["capture_relative_seconds"], 0.5)
        self.assertEqual(result["unix_time"], 100.5)

    def test_invalid_video_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = create_session(Path(temporary))
            with self.assertRaisesRegex(ValueError, "finite non-negative"):
                app_module.recording_frame_at(directory, "cam_1", float("nan"))

    def test_api_saves_real_gap_and_aggregates_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = create_session(root)
            with patch.object(app_module, "RECORDINGS_DIR", root):
                client = app_module.app.test_client()
                response = client.post(
                    "/api/recordings/worker_001/20260912_120000/handoffs",
                    json={
                        "person_id": "worker_001",
                        "from_camera": "cam_1",
                        "to_camera": "cam_entrance",
                        "source_end_seconds": 0.21,
                        "target_start_seconds": 0.31,
                        "notes": "normal pace",
                    },
                )

                self.assertEqual(response.status_code, 201)
                payload = response.get_json()
                annotation = payload["annotations"][0]
                self.assertEqual(annotation["source_end"]["frame_index"], 2)
                self.assertEqual(annotation["target_start"]["frame_index"], 3)
                self.assertAlmostEqual(annotation["gap_seconds"], 1.8)
                suggestion = payload["suggestions"]["cam_1->cam_entrance"]
                self.assertEqual(suggestion["samples"], 1)
                self.assertFalse(suggestion["automatic_config_update"])
                self.assertTrue((directory / "manual_handoffs.json").is_file())

                deleted = client.delete(
                    "/api/recordings/worker_001/20260912_120000/handoffs/1"
                )
                self.assertEqual(deleted.status_code, 200)
                self.assertEqual(deleted.get_json()["annotations"], [])
                self.assertIn(
                    "DELETE", deleted.headers["Access-Control-Allow-Methods"]
                )

    def test_recording_list_only_returns_completed_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_session(root)
            active = root / "worker_002" / "20260912_120100"
            active.mkdir(parents=True)
            (active / "info.json").write_text(
                json.dumps(
                    {
                        "status": "recording",
                        "started_at": "2026-09-12T12:01:00+08:00",
                        "streams": {},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(app_module, "RECORDINGS_DIR", root):
                response = app_module.app.test_client().get("/api/recordings")

        self.assertEqual(response.status_code, 200)
        sessions = response.get_json()["sessions"]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["subject_id"], "worker_001")

    def test_frame_endpoint_returns_browser_compatible_jpeg(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_session(root)
            fake_capture = unittest.mock.MagicMock()
            fake_capture.isOpened.return_value = True
            fake_capture.read.return_value = (
                True,
                np.zeros((32, 24, 3), dtype=np.uint8),
            )
            with (
                patch.object(app_module, "RECORDINGS_DIR", root),
                patch.object(app_module.cv2, "VideoCapture", return_value=fake_capture),
            ):
                response = app_module.app.test_client().get(
                    "/api/recordings/worker_001/20260912_120000/frame/cam_1?index=2"
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertEqual(response.headers["X-Frame-Index"], "2")
        fake_capture.set.assert_called_with(app_module.cv2.CAP_PROP_POS_FRAMES, 2)
        fake_capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
