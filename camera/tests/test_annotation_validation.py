import unittest

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


if __name__ == "__main__":
    unittest.main()
