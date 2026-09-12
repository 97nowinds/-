import unittest
from pathlib import Path

from floor_map import FloorMapProjector


def config():
    return {
        "name": "Test Lab",
        "width_m": 10,
        "height_m": 6,
        "calibrated": True,
        "zones": [
            {"name": "Main", "x": 0, "y": 0, "width": 10, "height": 6},
            {"name": "Overlap", "kind": "overlap", "x": 4, "y": 2, "width": 2, "height": 2},
        ],
        "fixtures": [],
        "cameras": {
            "cam_1": {
                "label": "Cam1",
                "position": [0, 0],
                "fov": [[0, 0], [10, 0], [10, 6]],
                "image_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
                "map_points": [[0, 0], [10, 0], [0, 6], [10, 6]],
            },
            "cam_2": {
                "label": "Cam2",
                "position": [10, 6],
                "fov": [[10, 6], [0, 6], [0, 0]],
                "image_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
                "map_points": [[0, 0], [10, 0], [0, 6], [10, 6]],
            },
        },
    }


class FloorMapProjectorTests(unittest.TestCase):
    def setUp(self):
        self.projector = FloorMapProjector(config())

    def test_projects_person_foot_point_to_floor_coordinate(self):
        position = self.projector.project("cam_1", (40, 20, 20, 50), (100, 100, 3))

        self.assertAlmostEqual(position["x"], 5.0, places=1)
        self.assertAlmostEqual(position["y"], 4.2, places=1)

    def test_calibration_status_exposes_geometry_trust(self):
        approximate_config = config()
        approximate_config["calibrated"] = False
        approximate_config["calibration"] = {
            "status": "approximate",
            "reason": "estimated control points",
        }
        approximate = FloorMapProjector(approximate_config)
        formal = FloorMapProjector(config())

        approximate_status = approximate.localization_status("cam_1")
        formal_status = formal.localization_status("cam_1")

        self.assertEqual(approximate_status["calibration_status"], "approximate")
        self.assertFalse(approximate_status["geometry_fusion_allowed"])
        self.assertIn("estimated control points", approximate_status["degraded_reason"])
        self.assertEqual(formal_status["calibration_status"], "formal")
        self.assertTrue(formal_status["geometry_fusion_allowed"])

    def test_overlap_zone_has_priority_over_large_room_zone(self):
        self.assertEqual(self.projector.zone_for(5, 3), "Overlap")
        self.assertEqual(self.projector.zone_for(2, 3), "Main")

    def test_solid_fixture_rejects_person_foot_point(self):
        floor_config = config()
        floor_config["fixtures"] = [
            {"type": "bench", "x": 2, "y": 2, "width": 3, "height": 1}
        ]
        projector = FloorMapProjector(floor_config)

        self.assertFalse(projector.is_walkable({"x": 3, "y": 2.5}))
        self.assertTrue(projector.is_walkable({"x": 7, "y": 4}))

    def test_unmapped_entrance_camera_does_not_invent_coordinates(self):
        floor_config = config()
        floor_config["cameras"]["cam_entrance"] = {
            "label": "Entrance",
            "position": [10, 5],
            "fov": [[10, 5], [8, 4], [10, 4]],
        }
        projector = FloorMapProjector(floor_config)

        self.assertIsNone(
            projector.project("cam_entrance", (20, 10, 30, 70), (100, 100, 3))
        )
        self.assertFalse(projector.has_projection("cam_entrance"))

    def test_entrance_tracking_anchor_marks_person_at_door(self):
        floor_config = config()
        floor_config["cameras"]["cam_entrance"] = {
            "label": "Entrance",
            "position": [10, 5],
            "fov": [[10, 5], [8, 4], [10, 4]],
            "tracking_anchor": [9.6, 4.8],
        }
        projector = FloorMapProjector(floor_config)

        position = projector.project(
            "cam_entrance", (20, 10, 30, 70), (100, 100, 3)
        )

        self.assertEqual(position, {"x": 9.6, "y": 4.8})

    def test_same_registered_person_from_two_cameras_is_merged(self):
        observations = [
            {
                "camera_id": "cam_1",
                "position": {"x": 4.8, "y": 3.0},
                "person_id": "person_003",
                "person_number": 3,
                "name": "yxq",
                "identity_source": "face",
                "observed_at": 100.0,
            },
            {
                "camera_id": "cam_2",
                "position": {"x": 5.2, "y": 3.2},
                "person_id": "person_003",
                "person_number": 3,
                "name": "yxq",
                "identity_source": "handoff",
                "observed_at": 100.1,
            },
        ]

        state = self.projector.state(observations, now=101.0)

        self.assertEqual(len(state["people"]), 1)
        person = state["people"][0]
        self.assertEqual(person["track_id"], "person_003")
        self.assertEqual(person["cameras"], ["cam_1", "cam_2"])
        self.assertEqual(person["zone"], "Overlap")
        self.assertTrue(person["identified"])
        self.assertEqual(person["identity_lock_status"], "locked")

    def test_unidentified_tracks_prefer_stable_observation_track_ids(self):
        observations = [
            {
                "camera_id": "cam_1",
                "track_id": "unknown-track-17",
                "position": {"x": 3, "y": 3},
                "observed_at": 10,
            },
            {
                "camera_id": "cam_1",
                "track_id": "unknown-track-18",
                "position": {"x": 7, "y": 3},
                "observed_at": 10,
            },
        ]

        people = self.projector.state(observations, now=11)["people"]

        self.assertEqual(len(people), 2)
        self.assertEqual(
            {person["track_id"] for person in people},
            {"unknown-track-17", "unknown-track-18"},
        )
        self.assertTrue(all(person["name"] == "未注册" for person in people))
        self.assertTrue(all(not person["identified"] for person in people))

    def test_observations_older_than_two_seconds_are_removed(self):
        observations = [
            {
                "camera_id": "cam_1",
                "track_id": "fresh",
                "position": {"x": 3, "y": 3},
                "observed_at": 98.0,
            },
            {
                "camera_id": "cam_2",
                "track_id": "stale",
                "position": {"x": 7, "y": 3},
                "observed_at": 97.99,
            },
        ]

        people = self.projector.state(observations, now=100.0)["people"]

        self.assertEqual([person["track_id"] for person in people], ["fresh"])

    def test_kalman_filter_smooths_map_jitter(self):
        first = [{
            "camera_id": "cam_1",
            "track_id": "moving",
            "position": {"x": 3.0, "y": 3.0},
            "observed_at": 10.0,
        }]
        second = [{
            "camera_id": "cam_1",
            "track_id": "moving",
            "position": {"x": 5.0, "y": 3.0},
            "observed_at": 10.2,
        }]

        self.projector.state(first, now=10.0)
        smoothed = self.projector.state(second, now=10.2)["people"][0]

        self.assertGreater(smoothed["x"], 3.0)
        self.assertLess(smoothed["x"], 5.0)

    def test_uncalibrated_dual_camera_target_uses_one_real_observation(self):
        floor_config = config()
        floor_config["calibrated"] = False
        projector = FloorMapProjector(floor_config)
        observations = [
            {"camera_id": "cam_1", "track_id": "person_1", "position": {"x": 1, "y": 1}, "observed_at": 10},
            {"camera_id": "cam_2", "track_id": "person_1", "position": {"x": 9, "y": 5}, "observed_at": 10},
        ]

        people = projector.state(observations, now=10)["people"]

        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["cameras"], ["cam_1"])
        self.assertEqual((people[0]["x"], people[0]["y"]), (1.0, 1.0))
        self.assertEqual(people[0]["zone"], "Main")

    def test_duplicate_identity_in_one_camera_is_split_into_distinct_people(self):
        observations = [
            {
                "camera_id": "cam_entrance",
                "local_id": local_id,
                "track_id": f"person_{local_id}",
                "global_track_id": f"person_{local_id}",
                "position": {"x": float(local_id), "y": 3.0},
                "person_id": "employee_1",
                "name": "worker",
                "identity_source": "face",
                "confidence": 0.9 - local_id * 0.01,
                "observed_at": 10.0,
            }
            for local_id in (1, 2, 3)
        ]

        people = self.projector.state(observations, now=10.0)["people"]

        self.assertEqual(len(people), 3)
        self.assertEqual(sum(person["identified"] for person in people), 1)
        self.assertEqual(len({person["track_id"] for person in people}), 3)

    def test_calibrated_but_inconsistent_camera_positions_are_not_averaged(self):
        observations = [
            {
                "camera_id": "cam_1",
                "track_id": "person_1",
                "person_id": "employee_1",
                "identity_source": "face",
                "position": {"x": 1.0, "y": 1.0},
                "observed_at": 10.0,
            },
            {
                "camera_id": "cam_2",
                "track_id": "person_1",
                "person_id": "employee_1",
                "identity_source": "handoff",
                "position": {"x": 9.0, "y": 5.0},
                "observed_at": 10.0,
            },
        ]

        person = self.projector.state(observations, now=10.0)["people"][0]

        self.assertEqual(person["cameras"], ["cam_1"])
        self.assertEqual((person["x"], person["y"]), (1.0, 1.0))

    def test_uncalibrated_camera_handoff_resets_position_filter(self):
        floor_config = config()
        floor_config["calibrated"] = False
        projector = FloorMapProjector(floor_config)
        projector.state(
            [{"camera_id": "cam_1", "track_id": "person_1", "position": {"x": 1, "y": 1}, "observed_at": 10}],
            now=10,
        )

        person = projector.state(
            [{"camera_id": "cam_2", "track_id": "person_1", "position": {"x": 9, "y": 5}, "observed_at": 10.2}],
            now=10.2,
        )["people"][0]

        self.assertEqual((person["x"], person["y"]), (9.0, 5.0))

    def test_actual_camera_positions_match_the_reversed_layout(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "floor_map.json"
        projector = FloorMapProjector.from_path(config_path)
        cam_1_x, cam_1_y = projector.public_config()["cameras"]["cam_1"]["position"]
        cam_2_x, cam_2_y = projector.public_config()["cameras"]["cam_2"]["position"]

        self.assertGreater(cam_1_x, projector.width_m / 2)
        self.assertLess(cam_1_y, projector.height_m / 2)
        self.assertLess(cam_2_x, projector.width_m / 2)
        self.assertGreater(cam_2_y, projector.height_m / 2)

    def test_actual_camera_mapping_can_distinguish_inner_and_main_aisles(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "floor_map.json"
        projector = FloorMapProjector.from_path(config_path)

        cam_2_inner = projector.project("cam_2", (0.10 * 1000, 0.80 * 1000, 0.02 * 1000, 0.06 * 1000), (1000, 1000, 3))
        cam_2_main = projector.project("cam_2", (0.60 * 1000, 0.80 * 1000, 0.02 * 1000, 0.06 * 1000), (1000, 1000, 3))

        self.assertEqual(projector.zone_for(**cam_2_inner), "副通道")
        self.assertEqual(projector.zone_for(**cam_2_main), "主通道")
        self.assertNotEqual(projector.zone_for(**cam_2_inner), projector.zone_for(**cam_2_main))


if __name__ == "__main__":
    unittest.main()
