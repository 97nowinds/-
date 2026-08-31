import unittest
import time

from person_tracking import PersonTrack


class PersonTrackTests(unittest.TestCase):
    def test_face_box_expands_to_body_box_inside_frame(self):
        box = PersonTrack.face_to_person((280, 80, 100, 100), (480, 640, 3))

        x, y, width, height = box
        self.assertGreater(width, 100)
        self.assertGreater(height, 100)
        self.assertLess(width, 250)
        self.assertLess(height, 300)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 640)
        self.assertLessEqual(y + height, 480)

    def test_track_expires_without_recent_detector_evidence(self):
        track = PersonTrack(evidence_seconds=1.0)
        track.box = (20, 20, 80, 160)
        track.last_evidence_at = time.time() - 2.0

        self.assertTrue(track.evidence_expired())

    def test_healthy_visual_track_can_continue_after_detector_evidence_expires(self):
        track = PersonTrack(evidence_seconds=1.0)
        track.box = (20, 20, 80, 160)
        track.last_evidence_at = time.time() - 2.0
        track.tracker.status = "tracking"
        track.tracker.confidence = 0.08
        track.tracker.valid = True

        self.assertTrue(track.evidence_expired())
        self.assertTrue(track.visual_track_healthy())

    def test_failed_visual_track_is_not_healthy(self):
        track = PersonTrack()
        track.box = (20, 20, 80, 160)
        track.tracker.status = "lost"
        track.tracker.confidence = 0.0
        track.tracker.valid = False

        self.assertFalse(track.visual_track_healthy())

    def test_box_smoothing_reduces_detector_jitter(self):
        previous = (100, 80, 180, 260)
        detected = (120, 70, 220, 300)

        smoothed = PersonTrack.smooth_box(previous, detected)

        self.assertGreater(smoothed[0], previous[0])
        self.assertLess(smoothed[0], detected[0])
        self.assertGreater(smoothed[2], previous[2])
        self.assertLess(smoothed[2], detected[2])

    def test_box_iou_detects_matching_observation(self):
        self.assertGreater(
            PersonTrack.box_iou((100, 80, 180, 260), (110, 90, 175, 250)),
            0.8,
        )
        self.assertEqual(
            PersonTrack.box_iou((0, 0, 50, 50), (100, 100, 50, 50)),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
