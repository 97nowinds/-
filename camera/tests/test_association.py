import unittest

import numpy as np

from association import AssociationObservation, AssociationTarget, BatchAssociator
from mtmc_config import load_mtmc_config
from reid_gallery import TrackFeatureGallery


def gallery(vector):
    result = TrackFeatureGallery(capacity=5, min_quality=0.1, top_k=3)
    result.add(np.asarray(vector), quality=0.9, camera_id="test", monotonic_time=1, unix_time=1)
    return result


class BatchAssociationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_mtmc_config()
        self.transitions = [
            {"from": "cam_1", "to": "cam_2", "max_gap_seconds": 8, "bidirectional": True},
            {"from": "cam_entrance", "to": "cam_1", "max_gap_seconds": 20},
        ]
        self.associator = BatchAssociator(self.config, self.transitions, "approximate")

    def observation(self, local_id, vector, camera="cam_2", now=2):
        return AssociationObservation(camera, local_id, now, gallery=gallery(vector))

    def target(self, global_id, vector, camera="cam_1", seen=1):
        return AssociationTarget(global_id, camera, seen, gallery=gallery(vector))

    def test_result_is_independent_of_input_order(self):
        observations = [self.observation(2, [0, 1]), self.observation(1, [1, 0])]
        targets = [self.target("person_2", [0, 1]), self.target("person_1", [1, 0])]
        first, _ = self.associator.associate(observations, targets)
        second, _ = self.associator.associate(list(reversed(observations)), list(reversed(targets)))
        mapping = lambda result: {(item["local_id"], item["global_id"]) for item in result}
        self.assertEqual(mapping(first), mapping(second))
        self.assertEqual(mapping(first), {(1, "person_1"), (2, "person_2")})

    def test_two_people_cross_one_to_one(self):
        decisions, _ = self.associator.associate(
            [self.observation(10, [1, 0]), self.observation(11, [0, 1])],
            [self.target("person_1", [1, 0]), self.target("person_2", [0, 1])],
        )
        assigned = [item["global_id"] for item in decisions if item["accepted"]]
        self.assertEqual(len(assigned), 2)
        self.assertEqual(len(set(assigned)), 2)

    def test_similar_clothing_stays_pending_when_margin_is_ambiguous(self):
        decisions, _ = self.associator.associate(
            [self.observation(10, [1, 0.01])],
            [self.target("person_1", [1, 0]), self.target("person_2", [1, 0.02])],
        )
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(decisions[0]["confidence"], "medium")

    def test_one_target_is_never_assigned_to_two_observations(self):
        decisions, _ = self.associator.associate(
            [self.observation(10, [1, 0]), self.observation(11, [1, 0])],
            [self.target("person_1", [1, 0])],
        )
        self.assertEqual(sum(item["accepted"] for item in decisions), 0)
        self.assertEqual(sum(item["confidence"] == "medium" for item in decisions), 1)

    def test_expired_candidate_is_rejected(self):
        decisions, evaluations = self.associator.associate(
            [self.observation(10, [1, 0], now=20)],
            [self.target("person_1", [1, 0], seen=1)],
        )
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(evaluations[0]["reason"], "candidate_expired")

    def test_reverse_transition_is_rejected(self):
        decisions, evaluations = self.associator.associate(
            [self.observation(10, [1, 0], camera="cam_entrance", now=2)],
            [self.target("person_1", [1, 0], camera="cam_1", seen=1)],
        )
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(evaluations[0]["reason"], "topology_rejected")

    def test_formal_geometry_rejects_impossible_distance(self):
        associator = BatchAssociator(self.config, self.transitions, "formal")
        observation = self.observation(10, [1, 0])
        observation.position = {"x": 9, "y": 5}
        target = self.target("person_1", [1, 0])
        target.position = {"x": 1, "y": 1}
        decisions, evaluations = associator.associate([observation], [target])
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(evaluations[0]["reason"], "geometry_distance_rejected")

    def test_approximate_simultaneous_pair_requires_manual_overlap_validation(self):
        observation = self.observation(10, [1, 0], camera="cam_2", now=2)
        target = self.target("person_1", [1, 0], camera="cam_1", seen=2)
        target.active_camera_ids = {"cam_1"}

        decisions, evaluations = self.associator.associate([observation], [target])

        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(
            evaluations[0]["reason"],
            "simultaneous_cross_camera_without_formal_overlap",
        )

    def test_manually_validated_overlap_allows_strong_reid_only_match(self):
        transitions = [
            {
                "from": "cam_1",
                "to": "cam_2",
                "max_gap_seconds": 8,
                "simultaneous_overlap_validated": True,
            }
        ]
        associator = BatchAssociator(self.config, transitions, "approximate")
        observation = self.observation(10, [1, 0], camera="cam_2", now=2)
        target = self.target("person_1", [1, 0], camera="cam_1", seen=2.1)
        target.active_camera_ids = {"cam_1"}

        decisions, evaluations = associator.associate([observation], [target])

        self.assertTrue(decisions[0]["accepted"])
        self.assertEqual(decisions[0]["global_id"], "person_1")
        self.assertTrue(evaluations[0]["simultaneous_overlap_validated"])

    def test_validated_overlap_still_rejects_medium_reid(self):
        transitions = [
            {
                "from": "cam_1",
                "to": "cam_2",
                "max_gap_seconds": 8,
                "simultaneous_overlap_validated": True,
            }
        ]
        associator = BatchAssociator(self.config, transitions, "approximate")
        observation = self.observation(10, [0.75, 0.6614], camera="cam_2", now=2)
        target = self.target("person_1", [1, 0], camera="cam_1", seen=2)
        target.active_camera_ids = {"cam_1"}

        decisions, evaluations = associator.associate([observation], [target])

        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(evaluations[0]["reason"], "validated_overlap_reid_below_high")


if __name__ == "__main__":
    unittest.main()
