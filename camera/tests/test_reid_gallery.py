import unittest

import numpy as np

from reid_gallery import BodyQualityScorer, TrackFeatureGallery


class ReIDGalleryTests(unittest.TestCase):
    def gallery(self, capacity=4, min_quality=0.4, top_k=3):
        return TrackFeatureGallery(
            capacity=capacity,
            min_quality=min_quality,
            duplicate_similarity=0.99,
            top_k=top_k,
        )

    def test_low_quality_frame_is_rejected(self):
        gallery = self.gallery()
        accepted = gallery.add(np.ones(4), quality=0.1, camera_id="cam_1")
        self.assertFalse(accepted)
        self.assertEqual(gallery.snapshot()["gallery_size"], 0)

    def test_capacity_is_bounded(self):
        gallery = self.gallery(capacity=3)
        for index in range(8):
            vector = np.zeros(8, dtype=np.float32)
            vector[index] = 1
            gallery.add(vector, quality=0.6 + index * 0.01, camera_id="cam_1")
        self.assertEqual(len(gallery.samples), 3)

    def test_duplicate_feature_is_not_added_twice(self):
        gallery = self.gallery()
        gallery.add(np.asarray([1, 0, 0]), quality=0.7, camera_id="cam_1")
        accepted = gallery.add(np.asarray([1, 0.001, 0]), quality=0.6, camera_id="cam_1")
        self.assertFalse(accepted)
        self.assertEqual(len(gallery.samples), 1)
        self.assertEqual(gallery.duplicate_samples, 1)

    def test_multiple_consistent_frames_outvote_one_anomaly(self):
        first = self.gallery(top_k=3)
        second = self.gallery(top_k=3)
        for vector in ([1, 0, 0], [0.95, 0.3, 0], [0.95, -0.3, 0]):
            first.add(np.asarray(vector), quality=0.9, camera_id="cam_1")
        for vector in ([1, 0.02, 0], [0.94, 0.32, 0], [0.94, -0.32, 0], [0, 0, 1]):
            second.add(np.asarray(vector), quality=0.9, camera_id="cam_2")
        result = first.compare(second)
        self.assertGreater(result["score"], 0.95)
        self.assertEqual(result["top_k"], 3)

    def test_empty_gallery_has_safe_fallback(self):
        result = self.gallery().compare(self.gallery())
        self.assertIsNone(result["score"])
        self.assertEqual(result["pair_count"], 0)

    def test_body_quality_rejects_tiny_dark_crop(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        quality = BodyQualityScorer.assess(frame, (0, 0, 5, 8))
        self.assertLess(quality["score"], 0.4)


if __name__ == "__main__":
    unittest.main()
