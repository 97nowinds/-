import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from dataset_recorder import DatasetRecorder, RecordingError, validate_subject_id


class DatasetRecorderTests(unittest.TestCase):
    def test_writes_two_raw_video_streams_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recordings"
            recorder = DatasetRecorder(root, ("cam_1", "cam_2"), min_free_bytes=0)
            started = recorder.start("person_003", "overlap route")
            frame = np.full((96, 128, 3), 80, dtype=np.uint8)
            captured_at = time.time()

            for index in range(8):
                recorder.submit("cam_1", frame, fps=20.0, captured_at=captured_at + index / 20)
                recorder.submit("cam_2", frame + 20, fps=20.0, captured_at=captured_at + 0.01 + index / 20)

            stopped = recorder.stop()
            directory = root.parent / stopped["directory"]
            metadata = json.loads((directory / "info.json").read_text(encoding="utf-8"))

            self.assertTrue(started["active"])
            self.assertFalse(stopped["active"])
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["streams"]["cam_1"]["frames"], 8)
            self.assertEqual(metadata["streams"]["cam_2"]["frames"], 8)
            for stream in metadata["streams"].values():
                self.assertTrue((directory / stream["video_file"]).stat().st_size > 0)
                timestamps = (directory / stream["timestamps_file"]).read_text(encoding="utf-8")
                self.assertEqual(len(timestamps.strip().splitlines()), 9)

    def test_rejects_path_traversal_and_duplicate_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = DatasetRecorder(temporary, ("cam_1", "cam_2"), min_free_bytes=0)
            with self.assertRaises(RecordingError):
                recorder.start("../outside")

            recorder.start("empty_scene")
            with self.assertRaises(RecordingError):
                recorder.start("person_003")
            recorder.stop()

    def test_subject_validation_accepts_registered_number_format(self):
        self.assertEqual(validate_subject_id("person_003"), "person_003")
        self.assertEqual(validate_subject_id("stranger-01"), "stranger-01")


if __name__ == "__main__":
    unittest.main()
