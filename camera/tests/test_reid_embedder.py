import unittest
import threading

import numpy as np
import torch

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

    def test_extract_batch_runs_one_model_call_for_multiple_people(self):
        class FakeModel:
            def __init__(self):
                self.batch_sizes = []

            def __call__(self, inputs):
                self.batch_sizes.append(int(inputs.shape[0]))
                return torch.ones((inputs.shape[0], 4), dtype=torch.float32)

        embedder = ReIDEmbedder.__new__(ReIDEmbedder)
        embedder.device = torch.device("cpu")
        embedder.lock = threading.RLock()
        embedder.model = FakeModel()
        frame = np.full((160, 200, 3), 127, dtype=np.uint8)

        features = embedder.extract_batch(
            frame,
            [(0, 0, 50, 120), (60, 10, 50, 120), (130, 20, 5, 20)],
        )

        self.assertEqual(embedder.model.batch_sizes, [2])
        self.assertEqual(len(features), 3)
        self.assertEqual(features[0].shape, (4,))
        self.assertEqual(features[1].shape, (4,))
        self.assertIsNone(features[2])


if __name__ == "__main__":
    unittest.main()
