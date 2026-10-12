import hashlib
import unittest
from continuous_handoff import prepare_live_event
import cv2
import numpy as np
import test_handoff_sample


class ContinuousHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ok, encoded = cv2.imencode('.jpg', np.zeros((1440, 2560, 3), dtype=np.uint8))
        assert ok
        cls.jpeg = encoded.tobytes()

    def frozen(self):
        frozen = test_handoff_sample.SampleTests().sample()
        frozen.update(camera_id='cam_2', frame_width=2560, frame_height=1440,
            source_dimensions=[2560, 1440], image_sha256=hashlib.sha256(self.jpeg).hexdigest(),
            map_id='lab-floor', map_revision='d9ea86e4bd80d086')
        frozen['camera']['yolo_tracks'][0]['box'] = [1, 2, 3, 4]
        return frozen

    def test_empty_scene_is_transmitted_without_inventing_person_or_map(self):
        frozen = self.frozen()
        frozen['camera']['yolo_tracks'] = []
        frozen['map_observations'] = []
        event = prepare_live_event(frozen, self.jpeg)
        self.assertEqual(event['camera']['tracks'], [])
        self.assertEqual(event['camera']['timing']['sync_status'], 'unavailable')
        self.assertEqual(event['camera']['geometry']['hd_alignment_status'], 'pending')

    def test_bound_frame_and_restored_box_are_kept(self):
        frozen = self.frozen()
        event = prepare_live_event(frozen, self.jpeg)
        self.assertEqual(event['snapshot_id'], frozen['snapshot_id'])
        self.assertEqual(event['camera']['tracks'][0]['bbox'], [1, 2, 3, 4])
        self.assertEqual(event['camera']['tracks'][0]['track_id'], 'g1')
        self.assertIn('location', event['camera']['tracks'][0])

    def test_other_camera_hash_mismatch_and_dimensions_mismatch_are_rejected(self):
        for field, value in [('camera_id', 'cam_1'), ('image_sha256', 'a'*64), ('frame_width', 2561),
                             ('map_revision', 'unconfirmed-map'), ('source_frame_id', None)]:
            frozen = self.frozen()
            frozen[field] = value
            with self.assertRaises(ValueError):
                prepare_live_event(frozen, self.jpeg)


if __name__ == '__main__':
    unittest.main()
