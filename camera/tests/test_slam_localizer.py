import unittest

import cv2
import numpy as np

from slam_localizer import SlamLocalizer


def marker_scene():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = np.full((600, 900, 3), 255, dtype=np.uint8)
    placements = {1: (80, 80), 2: (700, 80), 3: (700, 420), 4: (80, 420)}
    for marker_id, (x, y) in placements.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 100)
        frame[y : y + 100, x : x + 100] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


class SlamLocalizerTests(unittest.TestCase):
    def config(self, enabled=True):
        return {
            "enabled": enabled,
            "dictionary": "DICT_4X4_50",
            "min_markers": 4,
            "markers": [
                {"id": 1, "map_point": [1, 1]},
                {"id": 2, "map_point": [9, 1]},
                {"id": 3, "map_point": [9, 5]},
                {"id": 4, "map_point": [1, 5]},
            ],
        }

    def test_disabled_localizer_is_safe(self):
        localizer = SlamLocalizer("cam_1", self.config(enabled=False))
        state = localizer.update(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(state["status"], "disabled")
        self.assertIsNone(localizer.project_pixel((50, 50)))

    def test_four_aruco_anchors_create_metric_map_pose(self):
        localizer = SlamLocalizer("cam_1", self.config())
        state = localizer.update(marker_scene())
        position = localizer.project_pixel((450, 300))

        self.assertEqual(state["status"], "anchored")
        self.assertEqual(state["visible_markers"], 4)
        self.assertAlmostEqual(position["x"], 5.0, delta=0.15)
        self.assertAlmostEqual(position["y"], 3.0, delta=0.15)

    def test_status_reports_waiting_for_markers_without_reference(self):
        localizer = SlamLocalizer("cam_1", self.config())
        state = localizer.update(np.zeros((600, 900, 3), dtype=np.uint8))

        self.assertEqual(state["status"], "awaiting_markers")
        self.assertEqual(state["visible_markers"], 0)

    def test_relative_mode_bootstraps_from_existing_floor_transform(self):
        config = self.config(enabled=True)
        config["mode"] = "relative"
        localizer = SlamLocalizer("cam_1", config)
        baseline = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 255, (600, 900, 3), dtype=np.uint8)

        state = localizer.update(frame, baseline_transform=baseline)
        position = localizer.project_pixel((450, 300))

        self.assertEqual(state["status"], "relative_reference")
        self.assertEqual(position, {"x": 450.0, "y": 300.0})


if __name__ == "__main__":
    unittest.main()
