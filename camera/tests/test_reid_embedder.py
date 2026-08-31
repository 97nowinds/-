import unittest

import numpy as np

from reid_embedder import ReIDEmbedder, ReIDNetwork


class ReIDEmbedderTests(unittest.TestCase):
    def test_crop_person_clips_box_to_frame(self):
        frame = np.zeros((100, 80, 3), dtype=np.uint8)

        crop = ReIDEmbedder.crop_person(frame, (-10, 20, 50, 90))

        self.assertEqual(crop.shape, (80, 40, 3))

    def test_preprocess_returns_reid_input_shape(self):
        crop = np.full((200, 80, 3), 127, dtype=np.uint8)

        tensor = ReIDEmbedder.preprocess(crop)

        self.assertEqual(tuple(tensor.shape), (1, 3, 256, 128))
        self.assertTrue(np.isfinite(tensor.numpy()).all())

    def test_reid_backbone_uses_last_stride_one(self):
        model = ReIDNetwork()

        self.assertEqual(model.base.layer4[0].conv2.stride, (1, 1))
        self.assertEqual(model.base.layer4[0].downsample[0].stride, (1, 1))


if __name__ == "__main__":
    unittest.main()
