import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from app import _validate_annotation_payload


class AnnotationValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"width_m": 12.0, "height_m": 7.0, "cameras": {"cam_1": {}}}

    def test_accepts_main_aisle_parallelogram(self):
        result = _validate_annotation_payload(
            {
                "camera_id": "cam_1",
                "regions": {
                    "main_aisle": {
                        "points": [[0.1, 0.2], [0.7, 0.2], [0.8, 0.8], [0.2, 0.8]]
                    }
                },
            },
            self.config,
        )
        self.assertEqual(len(result["regions"]["main_aisle"]["points"]), 4)

    def test_rejects_non_parallelogram_main_aisle(self):
        with self.assertRaisesRegex(ValueError, "must form a parallelogram"):
            _validate_annotation_payload(
                {
                    "camera_id": "cam_1",
                    "regions": {
                        "main_aisle": {
                            "points": [[0.1, 0.2], [0.7, 0.2], [0.9, 0.8], [0.2, 0.8]]
                        }
                    },
                },
                self.config,
            )

    def test_legacy_rectangle_remains_supported(self):
        result = _validate_annotation_payload(
            {
                "camera_id": "cam_1",
                "regions": {
                    "main_aisle": {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.4}
                },
            },
            self.config,
        )
        self.assertEqual(result["regions"]["main_aisle"]["width"], 0.5)

    def test_region_save_preserves_canonical_camera_control_points(self):
        self.config["cameras"]["cam_1"] = {
            "image_points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "map_points": [[1, 1], [9, 1], [9, 5], [1, 5]],
        }
        result = _validate_annotation_payload(
            {
                "camera_id": "cam_1",
                "regions": {"rear_service": {"x": 0.8, "y": 0.1, "width": 0.1, "height": 0.8}},
                "image_points": [[0.2, 0.2]] * 4,
                "map_points": [[2, 2]] * 4,
                "calibration_points_changed": False,
            },
            self.config,
        )

        self.assertEqual(result["image_points"], self.config["cameras"]["cam_1"]["image_points"])
        self.assertEqual(result["map_points"], self.config["cameras"]["cam_1"]["map_points"])
        self.assertFalse(result["calibration_points_changed"])

    def test_explicit_reference_edit_uses_submitted_control_points(self):
        result = _validate_annotation_payload(
            {
                "camera_id": "cam_1",
                "image_points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "map_points": [[1, 1], [9, 1], [9, 5], [1, 5]],
                "calibration_points_changed": True,
            },
            self.config,
        )

        self.assertTrue(result["calibration_points_changed"])
        self.assertEqual(result["map_points"][2], [9.0, 5.0])

    def test_stale_annotation_revision_cannot_overwrite_newer_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "floor_map.json"
            path.write_text(json.dumps(self.config), encoding="utf-8")
            with patch.object(app_module, "FLOOR_MAP_PATH", path):
                client = app_module.app.test_client()
                revision = client.get("/api/annotation/config").get_json()["revision"]
                newer = {**self.config, "name": "newer"}
                path.write_text(json.dumps(newer), encoding="utf-8")
                response = client.post(
                    "/api/annotation/save",
                    json={"camera_id": "cam_1", "base_revision": revision},
                )

        self.assertEqual(response.status_code, 409)
        self.assertIn("更新", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
