import unittest

import numpy as np

from identity_handoff import IdentityHandoff


class FakeFeatureExtractor:
    def extract(self, frame, box):
        return np.asarray(frame, dtype=np.float32).mean(axis=(0, 1))


class IdentityHandoffTests(unittest.TestCase):
    def setUp(self):
        self.handoff = IdentityHandoff(
            FakeFeatureExtractor(), ttl_seconds=10, similarity_threshold=0.55
        )
        self.identity = {
            "known": True,
            "person_id": "person_003",
            "person_number": 3,
            "name": "yxq",
            "identity_source": "face",
        }

    def test_other_camera_inherits_same_registered_identity(self):
        appearance = np.array([0.8, 0.2, 0.1], dtype=np.float32)
        self.handoff.confirm("cam_1", self.identity, appearance)

        result = self.handoff.inherit("cam_2", appearance.copy())

        self.assertEqual(result["person_id"], "person_003")
        self.assertEqual(result["identity_source"], "handoff")
        self.assertEqual(result["handoff_from_camera"], "cam_1")

    def test_stranger_appearance_is_not_inherited(self):
        self.handoff.confirm(
            "cam_1", self.identity, np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )

        result = self.handoff.inherit(
            "cam_2", np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )

        self.assertIsNone(result)

    def test_ambiguous_multiple_people_are_not_inherited(self):
        appearance = np.array([0.8, 0.2, 0.1], dtype=np.float32)
        self.handoff.confirm("cam_1", self.identity, appearance)
        second = {
            "known": True,
            "person_id": "person_002",
            "person_number": 2,
            "name": "sxt",
        }
        self.handoff.confirm("cam_1", second, appearance)

        result = self.handoff.inherit("cam_2", appearance)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
