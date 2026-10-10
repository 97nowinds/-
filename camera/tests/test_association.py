import unittest
from pathlib import Path

import numpy as np

from association import AssociationObservation, AssociationTarget, BatchAssociator
from floor_map import FloorMapProjector
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

    def test_camera_history_uses_actual_source_time_and_exit(self):
        observation = self.observation(10, [1, 0], now=11)
        target = self.target("person_1", [1, 0], camera="cam_entrance", seen=10.9)
        target.camera_observations = {
            "cam_1": {"last_seen": 10.5, "source_local_id": 3},
            "cam_entrance": {"last_seen": 10.9},
        }
        result = self.associator.evaluate(observation, target)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["source_camera"], "cam_1")
        self.assertEqual(result["source_local_id"], 3)
        self.assertAlmostEqual(result["time_gap_seconds"], 0.5)
        target.camera_observations["cam_1"]["last_seen"] = 1
        self.assertFalse(self.associator.evaluate(observation, target)["eligible"])

    def test_recorded_secondary_portals_accept_without_opening_whole_image(self):
        floor = FloorMapProjector.from_path(Path(__file__).resolve().parents[1] / "config/floor_map.json")
        associator = BatchAssociator(self.config, floor.config["camera_transitions"], "approximate")
        observation = self.observation(62, [1, 0], camera="cam_1", now=11)
        observation.image_anchor = {"x": (5 + 45 / 2) / 768, "y": (12 + 198) / 432}
        target = self.target("person_1", [1, 0], camera="cam_2", seen=10.9)
        target.image_anchor = {"x": 0.57, "y": 0.34}
        target.active_camera_ids = {"cam_2"}
        result = associator.evaluate(observation, target)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["handoff_route"], "secondary")
        self.assertTrue(result["simultaneous_overlap_validated"])
        target.image_anchor = {"x": 0.65, "y": 0.70}
        self.assertFalse(associator.evaluate(observation, target)["eligible"])
        target.image_anchor = {"x": 0.57, "y": 0.34}
        observation.image_anchor = {"x": 0.9, "y": 0.1}
        self.assertFalse(associator.evaluate(observation, target)["eligible"])

        # The secondary overlap permission must not authorize the main route.
        observation.image_anchor = {"x": 0.20, "y": 0.90}
        target.image_anchor = {"x": 0.65, "y": 0.70}
        result = associator.evaluate(observation, target)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "simultaneous_cross_camera_without_formal_overlap")

        observation = self.observation(21, [1, 0], camera="cam_2", now=11)
        observation.image_anchor = {"x": 0.57, "y": 0.34}
        target = self.target("person_1", [1, 0], camera="cam_entrance", seen=10.9)
        target.image_anchor = {"x": 0.19, "y": 0.42}
        self.assertTrue(associator.evaluate(observation, target)["eligible"])

        observation = self.observation(58, [1, 0], camera="cam_entrance", now=11)
        observation.image_anchor = {"x": 0.19, "y": 0.42}
        target = self.target("person_1", [1, 0], camera="cam_1", seen=10.9)
        target.image_anchor = {"x": 0.35, "y": 0.99}
        self.assertTrue(associator.evaluate(observation, target)["eligible"])

    def test_same_camera_cannot_bypass_pixel_motion_with_high_reid(self):
        observation = self.observation(10, [1, 0], camera="cam_1", now=2)
        observation.box = (0, 20, 40, 80)
        target = self.target("person_1", [1, 0], camera="cam_1", seen=1)
        target.box = (20, 20, 40, 80)
        target.pixel_velocity = (200, 0)
        result = self.associator.evaluate(observation, target)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "same_camera_pixel_motion_rejected")

    def test_two_people_cross_one_to_one(self):
        decisions, _ = self.associator.associate(
            [self.observation(10, [1, 0]), self.observation(11, [0, 1])],
            [self.target("person_1", [1, 0]), self.target("person_2", [0, 1])],
        )
        assigned = [item["global_id"] for item in decisions if item["accepted"]]
        self.assertEqual(len(assigned), 2)
        self.assertEqual(len(set(assigned)), 2)

    def test_bidirectional_transition_accepts_both_camera_orders(self):
        forward, _ = self.associator.associate(
            [self.observation(10, [1, 0], camera="cam_2", now=2)],
            [self.target("person_1", [1, 0], camera="cam_1", seen=1)],
        )
        reverse, _ = self.associator.associate(
            [self.observation(10, [1, 0], camera="cam_1", now=2)],
            [self.target("person_1", [1, 0], camera="cam_2", seen=1)],
        )

        self.assertTrue(forward[0]["accepted"])
        self.assertTrue(reverse[0]["accepted"])
        self.assertEqual(forward[0]["global_id"], reverse[0]["global_id"])

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

    def test_handoff_requires_source_exit_and_target_entry_gates(self):
        transitions = [{
            "from": "cam_1", "to": "cam_2", "max_gap_seconds": 8,
            "image_handoff_gates": {
                "cam_1": {"exit": {"points": [[0.7, 0.7], [1, 0.7], [1, 1], [0.7, 1]]}},
                "cam_2": {"entry": {"points": [[0, 0], [0.3, 0], [0.3, 0.3], [0, 0.3]]}},
            },
        }]
        associator = BatchAssociator(self.config, transitions, "approximate")
        observation = self.observation(10, [1, 0])
        target = self.target("person_1", [1, 0])
        observation.image_anchor = {"x": 0.15, "y": 0.15}
        target.image_anchor = {"x": 0.5, "y": 0.5}

        rejected, evaluations = associator.associate([observation], [target])
        self.assertFalse(rejected[0]["accepted"])
        self.assertEqual(evaluations[0]["reason"], "source_not_in_handoff_exit_gate")

        target.image_anchor = {"x": 0.85, "y": 0.85}
        accepted, evaluations = associator.associate([observation], [target])
        self.assertTrue(accepted[0]["accepted"])
        self.assertTrue(evaluations[0]["source_in_handoff_gate"])
        self.assertTrue(evaluations[0]["target_in_handoff_gate"])

    def test_recent_source_exit_gate_survives_short_source_track_loss(self):
        transition = {
            "from": "cam_1", "to": "cam_2", "max_gap_seconds": 2.0,
            "image_handoff_gates": {
                "cam_1": {"exit": {"points": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}},
                "cam_2": {"entry": {"points": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}},
            },
        }
        associator = BatchAssociator(self.config, [transition])
        observation = AssociationObservation(
            camera_id="cam_2", local_id=2, monotonic_time=11.0,
            appearance=np.array([1.0, 0.0]), image_anchor={"x": 0.8, "y": 0.5},
        )
        target = AssociationTarget(
            global_id="person_1", camera_id="cam_1", last_seen=10.9,
            appearance=np.array([1.0, 0.0]), image_anchor={"x": 0.8, "y": 0.5},
            recent_exit_gates={"cam_2": 10.7},
        )
        evaluation = associator.evaluate(observation, target)
        self.assertTrue(evaluation["eligible"])
        self.assertFalse(evaluation["source_in_handoff_gate"])
        self.assertTrue(evaluation["source_exit_gate_recent"])

    def test_handoff_exit_grace_is_limited_to_original_source_and_deadline(self):
        transition = {
            "from": "cam_2", "to": "cam_1", "max_gap_seconds": 8.0,
            "image_handoff_gates": {
                "cam_2": {"exit": {"points": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]]}},
                "cam_1": {"entry": {"points": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]]}},
            },
        }
        associator = BatchAssociator(self.config, [transition], "approximate")
        observation = AssociationObservation(
            camera_id="cam_1", local_id=1, monotonic_time=4.1,
            appearance=np.array([1.0, 0.0]), image_anchor={"x": 0.2, "y": 0.5},
            handoff_exit_grace={"person_1": 5.0},
        )
        original = AssociationTarget(
            global_id="person_1", camera_id="cam_2", last_seen=3.8,
            appearance=np.array([1.0, 0.0]), image_anchor={"x": 0.8, "y": 0.5},
            recent_exit_gates={"cam_1": 1.0},
        )
        other = AssociationTarget(
            global_id="person_2", camera_id="cam_2", last_seen=3.8,
            appearance=np.array([1.0, 0.0]), image_anchor={"x": 0.8, "y": 0.5},
            recent_exit_gates={"cam_1": 1.0},
        )
        self.assertTrue(associator.evaluate(observation, original)["eligible"])
        self.assertTrue(associator.evaluate(observation, original)["source_exit_gate_grace"])
        self.assertEqual(
            associator.evaluate(observation, other)["reason"],
            "source_not_in_handoff_exit_gate",
        )
        observation.monotonic_time = 5.1
        self.assertEqual(
            associator.evaluate(observation, original)["reason"],
            "source_not_in_handoff_exit_gate",
        )

    def test_recorded_cam1_ingress_is_eligible_even_without_map_position(self):
        floor = FloorMapProjector.from_path(
            Path(__file__).resolve().parents[1] / "config" / "floor_map.json"
        )
        transitions = floor.config["camera_transitions"]
        associator = BatchAssociator(self.config, transitions, "approximate")
        for source_camera, anchor in (
            ("cam_2", {"x": (33 + 99 / 2) / 768, "y": (62 + 366) / 432}),
            ("cam_entrance", {"x": (134 + 116 / 2) / 768, "y": (75 + 352) / 432}),
        ):
            with self.subTest(source_camera=source_camera):
                observation = self.observation(10, [1, 0], camera="cam_1", now=11)
                observation.image_anchor = anchor
                observation.position = None
                target = self.target("person_1", [1, 0], camera=source_camera, seen=10.9)
                target.recent_exit_gates = {"cam_1": 10.8}
                # Cam2 has already lost the white track; the entrance still
                # sees the black track during its physical overlap with Cam1.
                target.active_camera_ids = (
                    {source_camera} if source_camera == "cam_entrance" else set()
                )
                evaluation = associator.evaluate(observation, target)
                self.assertTrue(evaluation["eligible"])
                self.assertTrue(evaluation["target_in_handoff_gate"])
                self.assertTrue(evaluation["source_exit_gate_recent"])

                observation.image_anchor = {"x": 0.77, "y": 0.60}
                wrong_person = associator.evaluate(observation, target)
                self.assertEqual(wrong_person["reason"], "target_not_in_handoff_entry_gate")

                observation.image_anchor = {"x": anchor["x"], "y": 431 / 432}
                self.assertTrue(associator.evaluate(observation, target)["eligible"])

    def test_recorded_entrance_ingress_matches_cam2_before_cam1(self):
        floor = FloorMapProjector.from_path(
            Path(__file__).resolve().parents[1] / "config" / "floor_map.json"
        )
        associator = BatchAssociator(
            self.config, floor.config["camera_transitions"], "approximate"
        )
        observation = self.observation(24, [1, 0], camera="cam_entrance", now=11)
        observation.image_anchor = {
            "x": (19 + 111 / 2) / 960,
            "y": (98 + 438) / 540,
        }
        observation.position = None
        target = self.target("black_shirt", [1, 0], camera="cam_2", seen=10.9)
        target.recent_exit_gates = {"cam_entrance": 10.8}
        target.active_camera_ids = {"cam_2"}
        self.assertTrue(associator.evaluate(observation, target)["eligible"])

        observation.image_anchor = {"x": (255 + 88 / 2) / 960, "y": (1 + 185) / 540}
        self.assertEqual(
            associator.evaluate(observation, target)["reason"],
            "target_not_in_handoff_entry_gate",
        )

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

    def test_clear_dark_light_conflict_rejects_even_perfect_reid(self):
        associator = BatchAssociator(
            self.config,
            [{"from": "cam_2", "to": "cam_entrance", "max_gap_seconds": 8,
              "simultaneous_overlap_validated": True}],
            "approximate",
        )
        observation = self.observation(10, [1, 0], camera="cam_entrance", now=2)
        observation.clothing_tone = "light"
        target = self.target("person_1", [1, 0], camera="cam_2", seen=2)
        target.clothing_tone = "dark"
        target.active_camera_ids = {"cam_2"}

        decision, evaluations = associator.associate([observation], [target])

        self.assertFalse(decision[0]["accepted"])
        self.assertEqual(evaluations[0]["reason"], "clothing_tone_mismatch")
        observation.clothing_tone = "dark"
        self.assertTrue(associator.associate([observation], [target])[0][0]["accepted"])

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
        observation = self.observation(10, [0.60, 0.80], camera="cam_2", now=2)
        target = self.target("person_1", [1, 0], camera="cam_1", seen=2)
        target.active_camera_ids = {"cam_1"}

        decisions, evaluations = associator.associate([observation], [target])

        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(evaluations[0]["reason"], "validated_overlap_reid_below_high")

    def test_validated_overlap_can_require_stricter_simultaneous_reid(self):
        transitions = [
            {
                "from": "cam_entrance",
                "to": "cam_2",
                "max_gap_seconds": 20,
                "bidirectional": True,
                "simultaneous_overlap_validated": True,
                "simultaneous_reid_threshold": 0.95,
            }
        ]
        associator = BatchAssociator(self.config, transitions, "approximate")
        observation = self.observation(10, [0.9, 0.43589], camera="cam_entrance", now=2)
        target = self.target("person_1", [1, 0], camera="cam_2", seen=2)
        target.active_camera_ids = {"cam_2"}

        decisions, evaluations = associator.associate([observation], [target])

        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(evaluations[0]["reason"], "validated_overlap_reid_below_high")
        self.assertEqual(evaluations[0]["simultaneous_reid_threshold"], 0.95)


if __name__ == "__main__":
    unittest.main()
