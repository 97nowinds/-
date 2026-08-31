import unittest

import cv2
import numpy as np

from motion_person_detector import MotionPersonDetector


FRAME_HEIGHT = 360
FRAME_WIDTH = 640


def background():
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 35, dtype=np.uint8)
    cv2.line(frame, (0, 250), (FRAME_WIDTH - 1, 250), (48, 48, 48), 2)
    cv2.rectangle(frame, (410, 70), (600, 180), (42, 42, 42), -1)
    return frame


def warmed_detector(warmup_frames=6):
    detector = MotionPersonDetector(warmup_frames=warmup_frames)
    frame = background()
    for _ in range(warmup_frames):
        detector.detect(frame)
    return detector


class MotionPersonDetectorTests(unittest.TestCase):
    def test_static_background_does_not_report_candidates(self):
        detector = warmed_detector()

        for _ in range(8):
            self.assertEqual(detector.detect(background()), [])

    def test_moving_distant_upright_rectangle_is_detected(self):
        detector = warmed_detector()
        frame = background()
        cv2.rectangle(frame, (120, 115), (159, 214), (210, 170, 90), -1)

        candidates = detector.detect(frame)

        self.assertTrue(candidates)
        scores = [score for score, _ in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
        score, (x, y, width, height) = candidates[0]
        self.assertGreater(score, 0)
        self.assertLessEqual(x, 120)
        self.assertLessEqual(y, 115)
        self.assertGreaterEqual(x + width, 159)
        self.assertGreaterEqual(y + height, 214)
        self.assertGreaterEqual(height, 90)

    def test_small_pixel_noise_is_filtered(self):
        detector = warmed_detector()
        frame = background()
        rng = np.random.default_rng(7)
        for x, y in rng.integers([0, 0], [FRAME_WIDTH - 2, FRAME_HEIGHT - 2], size=(80, 2)):
            frame[y : y + 2, x : x + 2] = 255

        self.assertEqual(detector.detect(frame), [])

    def test_reset_requires_a_new_warmup_period(self):
        detector = warmed_detector()
        moving = background()
        cv2.rectangle(moving, (120, 115), (159, 214), (210, 170, 90), -1)
        self.assertTrue(detector.detect(moving))

        detector.reset()

        self.assertFalse(detector.warmed_up)
        self.assertEqual(detector.detect(moving), [])


if __name__ == "__main__":
    unittest.main()
