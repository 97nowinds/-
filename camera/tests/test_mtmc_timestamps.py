import unittest

import cv2
import numpy as np

from floor_map import FloorMapProjector
from slam_localizer import SlamLocalizer


class MTMCTimestampTests(unittest.TestCase):
    def test_slam_uses_supplied_frame_times_and_monotonic_expiry(self):
        localizer = SlamLocalizer(
            "cam_1",
            {
                "enabled": True,
                "mode": "relative",
                "markers": [],
                "min_orb_matches": 6,
                "max_pose_age_seconds": 2.0,
            },
        )
        rng = np.random.default_rng(4)
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        state = localizer.update(
            frame,
            baseline_transform=np.eye(3),
            monotonic_time=10.0,
            unix_time=1000.0,
        )

        self.assertEqual(state["last_seen_at"], 1000.0)
        self.assertEqual(state["timestamp_source"], "host_receive")
        self.assertIsNotNone(localizer.project_pixel((10, 10), monotonic_time=11.9))
        self.assertIsNone(localizer.project_pixel((10, 10), monotonic_time=12.1))

    def test_stream_skew_disables_formal_geometry_fusion(self):
        config = {
            "width_m": 10,
            "height_m": 6,
            "calibration": {"status": "formal"},
            "zones": [],
            "fixtures": [],
            "cameras": {
                "cam_1": {
                    "localization_mode": "formal_homography",
                    "image_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
                    "map_points": [[0, 0], [10, 0], [0, 6], [10, 6]],
                }
            },
        }
        projector = FloorMapProjector(config)
        localization = projector.localization_status(
            "cam_1", stream_skew_seconds=0.8, max_stream_skew_seconds=0.5
        )

        self.assertFalse(localization["geometry_fusion_allowed"])
        self.assertIn("timestamp skew", localization["degraded_reason"])


if __name__ == "__main__":
    unittest.main()
