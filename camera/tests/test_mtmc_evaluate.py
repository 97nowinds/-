import unittest
import json
import tempfile
from pathlib import Path

from mtmc_evaluate import (
    FORMAT_VERSION,
    apply_session_timestamps,
    evaluate,
    load_session_timestamps,
    markdown_report,
    validate_ground_truth,
)


class MTMCEvaluationTests(unittest.TestCase):
    def sample(self):
        annotations = []
        predictions = []
        for camera_id, frames in (("cam_entrance", (1, 2)), ("cam_1", (3, 4))):
            for frame in frames:
                annotations.append(
                    {
                        "camera_id": camera_id,
                        "frame_index": frame,
                        "unix_time": float(frame),
                        "local_detection": {"local_id": 7, "box": [10, 10, 20, 60]},
                        "person_id": "worker_1",
                        "attributes": {
                            "occluded": False,
                            "registered": True,
                            "crowd_size": 1,
                        },
                    }
                )
                predictions.append(
                    {
                        "camera_id": camera_id,
                        "frame_index": frame,
                        "local_id": 7,
                        "global_id": "person_9",
                    }
                )
        return (
            {
                "format_version": FORMAT_VERSION,
                "annotations": annotations,
                "handoffs": [
                    {
                        "person_id": "worker_1",
                        "from_camera": "cam_entrance",
                        "to_camera": "cam_1",
                        "start_time": 2.0,
                        "end_time": 3.0,
                    }
                ],
            },
            {"predictions": predictions},
        )

    def test_perfect_synthetic_handoff(self):
        ground_truth, predictions = self.sample()
        result = evaluate(ground_truth, predictions)

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["metrics"]["idf1"], 1.0)
        self.assertEqual(result["metrics"]["cross_camera_handoff_success_rate"], 1.0)
        self.assertIn("cam_entrance->cam_1", result["groups"]["camera_pair"])

    def test_switch_and_missed_handoff_are_counted(self):
        ground_truth, predictions = self.sample()
        predictions["predictions"][-2]["global_id"] = "person_10"
        predictions["predictions"][-1]["global_id"] = "person_10"
        result = evaluate(ground_truth, predictions)

        self.assertEqual(result["metrics"]["id_switches"], 1)
        self.assertEqual(result["metrics"]["missed_handoff_rate"], 1.0)

    def test_empty_annotations_report_no_real_data(self):
        result = evaluate(
            {"format_version": FORMAT_VERSION, "annotations": [], "handoffs": []},
            {"predictions": []},
        )

        self.assertEqual(result["message"], "尚无真实评测数据")
        self.assertIn("尚无真实评测数据", markdown_report(result))

    def test_invalid_annotation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "local_detection.local_id"):
            validate_ground_truth(
                {
                    "format_version": FORMAT_VERSION,
                    "annotations": [
                        {"camera_id": "cam_1", "frame_index": 1, "person_id": "p1"}
                    ],
                }
            )

    def test_dataset_recorder_timestamps_are_loaded_and_applied(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "info.json").write_text(
                json.dumps(
                    {
                        "streams": {
                            "cam_1": {"timestamps_file": "cam_1_timestamps.csv"}
                        }
                    }
                ),
                encoding="utf-8",
            )
            (directory / "cam_1_timestamps.csv").write_text(
                "frame_index,unix_time,local_time\n4,100.25,ignored\n",
                encoding="utf-8",
            )
            timestamps = load_session_timestamps(directory)
            payload = apply_session_timestamps(
                {
                    "format_version": FORMAT_VERSION,
                    "annotations": [
                        {
                            "camera_id": "cam_1",
                            "frame_index": 4,
                            "local_detection": {"local_id": 1},
                            "person_id": "p1",
                        }
                    ],
                },
                timestamps,
            )

            self.assertEqual(payload["annotations"][0]["unix_time"], 100.25)


if __name__ == "__main__":
    unittest.main()
