import math
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

    def test_projection_rejects_foot_point_far_outside_four_point_domain(self):
        floor_config = config()
        floor_config["cameras"]["cam_1"]["image_points"] = [
            [0.4, 0.4], [0.6, 0.4], [0.4, 0.8], [0.6, 0.8]
        ]
        projector = FloorMapProjector(floor_config)

        position = projector.project("cam_1", (5, 10, 10, 30), (100, 100, 3))
        status = projector.localization_status("cam_1")

        self.assertIsNone(position)
        self.assertEqual(status["last_quality"], "low")
        self.assertIn("outside calibrated image domain", status["degraded_reason"])
        self.assertFalse(status["recent_projection"]["inside_domain"])

    def test_projection_margin_allows_small_detector_edge_jitter(self):
        floor_config = config()
        floor_config["cameras"]["cam_1"]["image_points"] = [
            [0.4, 0.4], [0.6, 0.4], [0.4, 0.8], [0.6, 0.8]
        ]
        projector = FloorMapProjector(floor_config)

        position = projector.project("cam_1", (33, 10, 10, 30), (100, 100, 3))
        status = projector.localization_status("cam_1")

        self.assertIsNotNone(position)
        self.assertEqual(status["last_quality"], "low")
        self.assertIn("near calibrated image boundary", status["degraded_reason"])

    def test_projection_domain_margin_is_validated(self):
        floor_config = config()
        floor_config["projection_domain_margin_normalized"] = 0.5

        with self.assertRaisesRegex(ValueError, "projection_domain_margin_normalized"):
            FloorMapProjector(floor_config)

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

    def test_nearest_walkable_position_moves_point_out_of_fixture(self):
        floor_config = config()
        floor_config["fixtures"] = [
            {"type": "bench", "x": 2, "y": 2, "width": 3, "height": 1}
        ]
        projector = FloorMapProjector(floor_config)

        position, adjusted = projector.nearest_walkable_position({"x": 3, "y": 2.5})

        self.assertTrue(adjusted)
        self.assertTrue(projector.is_walkable(position))
        self.assertLess(math.hypot(position["x"] - 3, position["y"] - 2.5), 0.7)

    def test_tracking_position_is_locked_to_configured_corridors(self):
        floor_config = config()
        floor_config["tracking_allowed_zone_ids"] = ["main_aisle", "side_aisle"]
        floor_config["zones"] = [
            {"id": "main_aisle", "name": "Main", "x": 0, "y": 4, "width": 8, "height": 1},
            {"id": "side_aisle", "name": "Side", "x": 8, "y": 1, "width": 1, "height": 4},
            {"id": "bench", "name": "Bench", "x": 1, "y": 1, "width": 6, "height": 2},
        ]
        projector = FloorMapProjector(floor_config)

        position, adjusted = projector.nearest_tracking_position({"x": 4, "y": 2})

        self.assertTrue(adjusted)
        self.assertEqual(position, {"x": 4.0, "y": 4.0})
        self.assertEqual(projector.zone_for(**position), "Main")

    def test_actual_map_only_reports_three_walkable_corridors(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "floor_map.json"
        projector = FloorMapProjector.from_path(config_path)
        allowed_names = {"副通道", "侧向通道", "主通道"}

        for point in ({"x": 5, "y": 0.8}, {"x": 5, "y": 3.2}, {"x": 5, "y": 6.2}, {"x": 11, "y": 3}):
            position, _ = projector.nearest_tracking_position(point)
            self.assertIn(projector.zone_for(**position), allowed_names)

    def test_smoothed_map_position_never_enters_solid_fixture(self):
        floor_config = config()
        floor_config["fixtures"] = [
            {"type": "bench", "x": 2, "y": 2, "width": 3, "height": 1}
        ]
        projector = FloorMapProjector(floor_config)

        person = projector.state(
            [{"camera_id": "cam_1", "track_id": "person_1", "position": {"x": 3, "y": 2.5}, "observed_at": 10}],
            now=10,
        )["people"][0]

        self.assertTrue(projector.is_walkable(person))
        self.assertTrue(person["walkability_adjusted"])

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

    def test_uncalibrated_camera_handoff_is_limited_to_walking_speed(self):
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

        self.assertNotEqual((person["x"], person["y"]), (9.0, 5.0))
        self.assertLessEqual(math.hypot(person["x"] - 1.0, person["y"] - 1.0), 0.45)
        self.assertTrue(person["motion_limited"])

    def test_kalman_prediction_cannot_overshoot_speed_gate(self):
        floor_config = config()
        floor_config["calibrated"] = False
        projector = FloorMapProjector(floor_config)
        positions = []
        for index, target_x in enumerate((1.0, 9.0, 9.0, 1.0, 9.0)):
            observed_at = 10.0 + index * 0.15
            person = projector.state(
                [
                    {
                        "camera_id": "cam_1",
                        "track_id": "person_1",
                        "position": {"x": target_x, "y": 3.0},
                        "observed_at": observed_at,
                    }
                ],
                now=observed_at,
                max_map_speed_mps=2.2,
            )["people"][0]
            positions.append((person["x"], person["y"]))

        for previous, current in zip(positions, positions[1:]):
            self.assertLessEqual(math.dist(previous, current), 2.2 * 0.15 + 0.005)

    def test_measurement_resume_does_not_spend_held_time_as_jump_credit(self):
        floor_config = config()
        floor_config["calibrated"] = False
        projector = FloorMapProjector(floor_config)
        first = projector.state(
            [{"camera_id": "cam_1", "track_id": "person_1", "position": {"x": 1, "y": 1}, "observed_at": 10}],
            now=10,
        )["people"][0]

        resumed = projector.state(
            [{"camera_id": "cam_2", "track_id": "person_1", "position": {"x": 9, "y": 5}, "observed_at": 15}],
            now=15,
            max_map_speed_mps=2.2,
            max_motion_gap_seconds=0.25,
            map_position_hold_seconds=8.0,
        )["people"][0]

        self.assertLessEqual(
            math.hypot(resumed["x"] - first["x"], resumed["y"] - first["y"]),
            2.2 * 0.25 + 0.005,
        )
        self.assertTrue(resumed["motion_limited"])

    def test_secondary_to_main_route_must_pass_through_side_corridor(self):
        floor_config = config()
        floor_config["zones"] = [
            {"id": "secondary", "name": "Secondary", "x": 0, "y": 1, "width": 8.5, "height": 1.2},
            {"id": "side", "name": "Side", "x": 8, "y": 1, "width": 1, "height": 4},
            {"id": "main", "name": "Main", "x": 0, "y": 4, "width": 9, "height": 1},
        ]
        floor_config["tracking_allowed_zone_ids"] = ["secondary", "side", "main"]
        projector = FloorMapProjector(floor_config)
        positions = []
        first = projector.state(
            [{"camera_id": "cam_1", "track_id": "person_1", "position": {"x": 7.5, "y": 1.8}, "observed_at": 10.0}],
            now=10.0,
        )["people"][0]
        positions.append(first)
        for index in range(1, 41):
            observed_at = 10.0 + index * 0.2
            person = projector.state(
                [{"camera_id": "cam_2", "track_id": "person_1", "position": {"x": 6.0, "y": 4.5}, "observed_at": observed_at}],
                now=observed_at,
                max_map_speed_mps=2.2,
            )["people"][0]
            positions.append(person)

        zones = [person["zone"] for person in positions]
        self.assertIn("Side", zones)
        self.assertIn("Main", zones)
        self.assertLess(zones.index("Side"), zones.index("Main"))
        self.assertTrue(any(person["corridor_routed"] for person in positions[1:]))
        for previous, current in zip(positions, positions[1:]):
            self.assertLessEqual(
                math.hypot(current["x"] - previous["x"], current["y"] - previous["y"]),
                2.2 * 0.2 + 0.015,
            )

    def test_active_track_holds_last_position_during_short_projection_gap(self):
        observation = {
            "camera_id": "cam_1",
            "track_id": "person_1",
            "position": {"x": 3.0, "y": 3.0},
            "observed_at": 10.0,
        }
        measured = self.projector.state(
            [observation],
            now=10.0,
            active_tracks=[{"track_id": "person_1", "cameras": ["cam_1"], "observed_at": 10.0}],
        )["people"][0]

        held = self.projector.state(
            [],
            now=12.5,
            active_tracks=[{"track_id": "person_1", "cameras": ["cam_1"], "observed_at": 12.5}],
            map_position_hold_seconds=4.0,
        )["people"][0]

        self.assertEqual((held["x"], held["y"]), (measured["x"], measured["y"]))
        self.assertTrue(held["position_estimated"])
        self.assertEqual(held["localization_state"], "held")
        self.assertEqual(held["position_age_seconds"], 2.5)

    def test_position_hold_requires_active_tracking_and_expires(self):
        observation = {
            "camera_id": "cam_1",
            "track_id": "person_1",
            "position": {"x": 3.0, "y": 3.0},
            "observed_at": 10.0,
        }
        self.projector.state(
            [observation],
            now=10.0,
            active_tracks=[{"track_id": "person_1", "cameras": ["cam_1"], "observed_at": 10.0}],
        )

        inactive = self.projector.state([], now=11.0, active_tracks=[])["people"]
        expired = self.projector.state(
            [],
            now=14.1,
            active_tracks=[{"track_id": "person_1", "cameras": ["cam_1"], "observed_at": 14.1}],
            map_position_hold_seconds=4.0,
        )["people"]

        self.assertEqual(inactive, [])
        self.assertEqual(expired, [])

    def test_uncalibrated_overlap_keeps_existing_position_source(self):
        floor_config = config()
        floor_config["calibrated"] = False
        projector = FloorMapProjector(floor_config)
        projector.state(
            [{"camera_id": "cam_1", "track_id": "person_1", "position": {"x": 2, "y": 2}, "observed_at": 10}],
            now=10,
        )

        person = projector.state(
            [
                {"camera_id": "cam_1", "track_id": "person_1", "position": {"x": 2.2, "y": 2}, "observed_at": 10.2},
                {"camera_id": "cam_2", "track_id": "person_1", "position": {"x": 9, "y": 5}, "observed_at": 10.2, "identity_source": "face"},
            ],
            now=10.2,
        )["people"][0]

        self.assertEqual(person["position_source_camera"], "cam_1")
        self.assertEqual(person["cameras"], ["cam_1"])

    def test_repeated_map_poll_does_not_reapply_same_measurement(self):
        projector = FloorMapProjector(config())
        observation = {"camera_id": "cam_1", "track_id": "person_1", "position": {"x": 4, "y": 3}, "observed_at": 10}
        first = projector.state([observation], now=10)["people"][0]
        second = projector.state([observation], now=10.2)["people"][0]

        self.assertEqual((first["x"], first["y"]), (second["x"], second["y"]))

    def test_actual_camera_positions_match_the_reversed_layout(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "floor_map.json"
        projector = FloorMapProjector.from_path(config_path)
        cam_1_x, cam_1_y = projector.public_config()["cameras"]["cam_1"]["position"]
        cam_2_x, cam_2_y = projector.public_config()["cameras"]["cam_2"]["position"]

        self.assertGreater(cam_1_x, projector.width_m / 2)
        self.assertLess(cam_1_y, projector.height_m / 2)
        self.assertLess(cam_2_x, projector.width_m / 2)
        self.assertGreater(cam_2_y, projector.height_m / 2)

    def test_actual_entrance_uses_saved_per_pixel_homography(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "floor_map.json"
        projector = FloorMapProjector.from_path(config_path)

        self.assertTrue(projector.has_projection("cam_entrance"))
        status = projector.localization_status("cam_entrance")
        self.assertEqual(status["mode"], "approximate_homography")
        first = projector.project("cam_entrance", (1400, 700, 180, 400), (1440, 2560, 3))
        second = projector.project("cam_entrance", (1700, 700, 180, 400), (1440, 2560, 3))
        self.assertNotEqual(first, second)

    def test_actual_cam_2_mapping_varies_within_annotated_domain(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "floor_map.json"
        projector = FloorMapProjector.from_path(config_path)

        first = projector.project(
            "cam_2", (0.68 * 1000, 0.80 * 1000, 0.02 * 1000, 0.06 * 1000), (1000, 1000, 3)
        )
        second = projector.project(
            "cam_2", (0.67 * 1000, 0.56 * 1000, 0.02 * 1000, 0.06 * 1000), (1000, 1000, 3)
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
