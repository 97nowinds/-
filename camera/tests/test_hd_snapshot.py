import unittest
from copy import deepcopy
import cv2
import numpy as np
from hd_snapshot import encode_detection_snapshot


class HdSnapshotTests(unittest.TestCase):
    def test_original_pixels_and_box_restore_without_mutating_preview(self):
        raw = np.zeros((1440, 2560, 3), dtype=np.uint8)
        raw[400:800, 200:600] = (20, 190, 80)
        original = raw.copy()
        detector = cv2.resize(raw, (640, 360))
        tracks = [dict(track_id='stable-7', global_id='stable-7', box=[50, 100, 100, 100])]
        previous = deepcopy(tracks)
        jpeg, restored = encode_detection_snapshot(raw, detector, tracks)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, raw.shape)
        self.assertEqual(restored[0]['box'], [200, 400, 400, 400])
        self.assertEqual(restored[0]['track_id'], 'stable-7')
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(tracks, previous)
        self.assertLess(np.abs(decoded[500, 300].astype(float) - raw[500, 300]).max(), 5)

    def test_invalid_box_never_produces_evidence(self):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        for box in ([0, 0, 100, 10], [0, 0, float('nan'), 5], [0, 0, -1, 4]):
            with self.assertRaises(ValueError):
                encode_detection_snapshot(frame, frame, [dict(box=box)])


if __name__ == '__main__':
    unittest.main()
