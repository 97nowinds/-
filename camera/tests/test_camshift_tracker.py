import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camshift_tracker import CamShiftTracker


def frame_with_target(x, y=48):
    frame = np.zeros((220, 340, 3), dtype=np.uint8)
    cv2.rectangle(frame, (x, y), (x + 62, y + 118), (30, 80, 225), -1)
    cv2.circle(frame, (x + 20, y + 34), 12, (40, 210, 80), -1)
    return frame


class CamShiftTrackerTests(unittest.TestCase):
    def test_local_tracker_keeps_target_id(self):
        tracker = CamShiftTracker()
        initial = tracker.start(frame_with_target(44), (40, 42, 72, 130), "target_007")
        moved = tracker.update(frame_with_target(58))

        self.assertEqual(initial["target_id"], "target_007")
        self.assertEqual(moved["target_id"], "target_007")
        self.assertEqual(moved["status"], "tracking")
        self.assertGreater(moved["confidence"], 0)
        self.assertTrue(moved["valid"])

    def test_transferred_histogram_finds_target_in_other_camera(self):
        source = frame_with_target(42)
        histogram = CamShiftTracker.create_histogram(source, (38, 42, 74, 130))
        destination = frame_with_target(218)

        candidate = CamShiftTracker.find_candidate(destination, histogram)

        self.assertIsNotNone(candidate)
        x, _, width, _ = candidate["box"]
        self.assertGreater(x + width // 2, 190)

    def test_search_zone_rejects_target_outside_overlap(self):
        source = frame_with_target(42)
        histogram = CamShiftTracker.create_histogram(source, (38, 42, 74, 130))
        destination = frame_with_target(218)

        candidate = CamShiftTracker.find_candidate(
            destination, histogram, normalized_zone=[0.0, 0.0, 0.4, 1.0]
        )

        self.assertIsNone(candidate)

    def test_tracker_rejects_sudden_expansion_onto_background(self):
        tracker = CamShiftTracker()
        initial = tracker.start(frame_with_target(44), (40, 42, 72, 130), "target_007")
        expanded = ((170.0, 110.0), (300.0, 200.0), 0.0)

        with patch("camshift_tracker.cv2.CamShift", return_value=(expanded, (5, 5, 300, 200))):
            result = tracker.update(frame_with_target(44))

        self.assertEqual(result["box"], initial["box"])
        self.assertEqual(result["failures"], 1)
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
