from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke_guard import SmokeGuard


class SmokeGuardTests(unittest.TestCase):
    def test_person_overlap_leaves_independent_smoke_candidate(self):
        guard = SmokeGuard.__new__(SmokeGuard)
        guard.reflection = None
        guard.person = SimpleNamespace(predict=lambda image, score: {
            'detections': [{'bbox': [80, 40, 110, 210]}],
        })
        frame = np.random.default_rng(1).integers(30, 230, (256, 256, 3), dtype=np.uint8)
        foreground = {'bbox': [95, 45, 70, 60]}
        separate = {'bbox': [125, 20, 100, 45]}
        kept, rejected = guard.screen(frame, [foreground, separate])
        self.assertEqual(kept, [separate])
        self.assertEqual(rejected, {'person': 1})

    def test_gray_failure_is_filtered_without_running_person_detector(self):
        guard = SmokeGuard.__new__(SmokeGuard)
        frame = np.full((256, 256, 3), 128, dtype=np.uint8)
        kept, rejected = guard.screen(frame, [{'bbox': [0, 0, 250, 250]}])
        self.assertEqual(kept, [])
        self.assertEqual(rejected, {'gray_frame': 1})


if __name__ == '__main__':
    unittest.main()
