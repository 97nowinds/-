import unittest

import numpy as np

from yolo_person_tracker import CrossCameraTrackCoordinator, TrackTrailStore, YoloPersonTracker


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    id = FakeTensor([7.0, 3.0])
    xyxy = FakeTensor([[10.2, 20.4, 50.4, 120.6], [1, 2, 3, 4]])
    conf = FakeTensor([0.91, 0.76])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self, path):
        self.path = path
        self.kwargs = None
        self.predictor = object()

    def track(self, **kwargs):
        self.kwargs = kwargs
        return [FakeResult()]


class FakeFeatureExtractor:
    def extract(self, frame, box):
        x, y, width, height = box
        crop = frame[y : y + height, x : x + width]
        feature = crop.astype(np.float32).mean(axis=(0, 1))
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else None


class YoloPersonTrackerTests(unittest.TestCase):
    def test_tracks_person_class_with_persistent_bytetrack(self):
        tracker = YoloPersonTracker("model.pt", model_factory=FakeModel)
        frame = np.zeros((180, 320, 3), dtype=np.uint8)

        tracks = tracker.track(frame)

        self.assertEqual(tracks, [{"track_id": 7, "box": (10, 20, 40, 101), "confidence": 0.91}])
        self.assertTrue(tracker.model.kwargs["persist"])
        self.assertEqual(tracker.model.kwargs["tracker"], "bytetrack.yaml")
        self.assertEqual(tracker.model.kwargs["classes"], [0])

    def test_reset_discards_predictor_tracker_state(self):
        tracker = YoloPersonTracker("model.pt", model_factory=FakeModel)

        tracker.reset()

        self.assertIsNone(tracker.model.predictor)

    def test_deduplicates_highly_overlapping_tracker_boxes(self):
        tracks = [
            {"track_id": 7, "box": (10, 20, 40, 100), "confidence": 0.91},
            {"track_id": 8, "box": (11, 21, 40, 100), "confidence": 0.84},
            {"track_id": 9, "box": (120, 20, 40, 100), "confidence": 0.88},
        ]

        deduplicated = YoloPersonTracker.deduplicate_tracks(tracks)

        self.assertEqual([track["track_id"] for track in deduplicated], [7, 9])


class TrackTrailStoreTests(unittest.TestCase):
    def test_adds_only_meaningful_movement(self):
        trails = TrackTrailStore(minimum_step=2.0)
        trails.update("cam_1:7", (10, 10), now=1.0)
        trails.update("cam_1:7", (11, 10), now=2.0)
        trails.update("cam_1:7", (13, 10), now=3.0)

        self.assertEqual(trails.points("cam_1:7"), [(10, 10), (13, 10)])

    def test_prunes_stale_tracks(self):
        trails = TrackTrailStore(stale_seconds=3.0)
        trails.update("cam_1:7", (10, 10), now=1.0)

        trails.prune(now=4.1)

        self.assertEqual(trails.points("cam_1:7"), [])


class CrossCameraTrackCoordinatorTests(unittest.TestCase):
    class FloorMap:
        config = {
            "calibrated": True,
            "zones": [{"kind": "overlap", "x": 4, "y": 4, "width": 4, "height": 2}],
        }

    class UncalibratedFloorMap:
        config = {"calibrated": False, "zones": []}

    class TransitionFloorMap:
        config = {
            "calibrated": False,
            "zones": [],
            "camera_transitions": [
                {"from": "cam_entrance", "to": "cam_1", "max_gap_seconds": 20}
            ],
        }

    class ValidatedOverlapFloorMap:
        config = {
            "calibrated": False,
            "calibration": {"status": "approximate"},
            "zones": [],
            "camera_transitions": [
                {
                    "from": "cam_entrance",
                    "to": "cam_1",
                    "max_gap_seconds": 20,
                    "simultaneous_overlap_validated": True,
                }
            ],
        }

    class SpatialTransitionFloorMap:
        config = {
            "calibrated": True,
            "zones": [],
            "camera_transitions": [
                {
                    "from": "cam_entrance",
                    "to": "cam_1",
                    "max_gap_seconds": 20,
                    "spatial_handoff_max_age_seconds": 3,
                    "spatial_handoff_max_distance_m": 12.0,
                    "spatial_handoff_similarity_threshold": 0.35,
                }
            ],
        }

    def coordinator(self, floor_map, **kwargs):
        return CrossCameraTrackCoordinator(FakeFeatureExtractor(), floor_map, **kwargs)

    def test_batch_update_is_one_to_one_and_order_independent(self):
        floor_map = self.TransitionFloorMap()

        def run(reverse=False):
            coordinator = self.coordinator(floor_map)
            frame = np.zeros((100, 160, 3), dtype=np.uint8)
            frame[20:80, 10:50] = (20, 80, 180)
            frame[20:80, 100:140] = (180, 80, 20)
            source = [
                {"track_id": 1, "box": (10, 20, 40, 60), "position": None},
                {"track_id": 2, "box": (100, 20, 40, 60), "position": None},
            ]
            first = coordinator.update_batch(
                "cam_entrance",
                list(reversed(source)) if reverse else source,
                frame,
                monotonic_time=1.0,
                unix_time=100.0,
            )
            coordinator.update_batch(
                "cam_entrance",
                [],
                frame,
                active_local_ids=set(),
                monotonic_time=1.1,
                unix_time=100.1,
            )
            target = [
                {"track_id": 8, "box": (100, 20, 40, 60), "position": None},
                {"track_id": 7, "box": (10, 20, 40, 60), "position": None},
            ]
            second = coordinator.update_batch(
                "cam_1",
                list(reversed(target)) if reverse else target,
                frame,
                monotonic_time=1.2,
                unix_time=100.2,
            )
            return first, second

        normal = run(False)
        reversed_order = run(True)

        self.assertEqual(normal, reversed_order)
        self.assertEqual(normal[0][1], normal[1][7])
        self.assertEqual(normal[0][2], normal[1][8])
        self.assertEqual(len(set(normal[1].values())), 2)

    def test_existing_anonymous_track_merges_after_source_disappears(self):
        coordinator = self.coordinator(self.TransitionFloorMap())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = (20, 80, 180)
        track = [{"track_id": 1, "box": (30, 20, 40, 60), "position": None}]
        source = coordinator.update_batch(
            "cam_entrance", track, frame, monotonic_time=1.0, unix_time=100.0
        )[1]
        simultaneous = coordinator.update_batch(
            "cam_1", track, frame, monotonic_time=1.1, unix_time=100.1
        )[1]

        self.assertNotEqual(simultaneous, source)
        coordinator.update_batch(
            "cam_entrance", [], frame, active_local_ids=set(), monotonic_time=1.2, unix_time=100.2
        )
        merged = coordinator.update_batch(
            "cam_1", track, frame, monotonic_time=2.0, unix_time=101.0
        )[1]

        self.assertEqual(merged, source)
        self.assertEqual(coordinator.redirects[simultaneous], source)

    def test_validated_simultaneous_overlap_preserves_oldest_global_id(self):
        coordinator = self.coordinator(self.ValidatedOverlapFloorMap())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = (20, 80, 180)
        track = [{"track_id": 1, "box": (30, 20, 40, 60), "position": None}]
        oldest = coordinator.update_batch(
            "cam_entrance", track, frame, monotonic_time=1.0, unix_time=100.0
        )[1]

        simultaneous = coordinator.update_batch(
            "cam_1", track, frame, monotonic_time=1.1, unix_time=100.1
        )[1]

        self.assertEqual(simultaneous, oldest)
        self.assertNotIn(oldest, coordinator.redirects)

    def test_same_appearance_in_overlap_keeps_global_id(self):
        coordinator = self.coordinator(self.FloorMap(), similarity_threshold=0.70)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = (20, 80, 180)

        first = coordinator.update("cam_1", 1, frame, (30, 20, 40, 60), {"x": 5, "y": 5}, now=1)
        second = coordinator.update("cam_2", 7, frame, (30, 20, 40, 60), {"x": 6, "y": 5}, now=2)

        self.assertEqual(first, "person_1")
        self.assertEqual(second, first)

    def test_target_outside_overlap_does_not_steal_id(self):
        coordinator = self.coordinator(self.FloorMap(), similarity_threshold=0.70)
        red = np.zeros((100, 100, 3), dtype=np.uint8)
        red[20:80, 30:70] = (20, 80, 180)
        blue = np.zeros((100, 100, 3), dtype=np.uint8)
        blue[20:80, 30:70] = (180, 80, 20)

        first = coordinator.update("cam_1", 1, red, (30, 20, 40, 60), {"x": 5, "y": 5}, now=1)
        second = coordinator.update("cam_2", 7, blue, (30, 20, 40, 60), {"x": 1, "y": 1}, now=2)

        self.assertNotEqual(second, first)

    def test_existing_local_ids_merge_when_both_reach_overlap(self):
        coordinator = self.coordinator(self.FloorMap(), similarity_threshold=0.70)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = (20, 80, 180)

        first = coordinator.update("cam_1", 1, frame, (30, 20, 40, 60), {"x": 2, "y": 2}, now=1)
        second = coordinator.update("cam_2", 7, frame, (30, 20, 40, 60), {"x": 9, "y": 2}, now=1.1)
        coordinator.update("cam_1", 1, frame, (30, 20, 40, 60), {"x": 5, "y": 5}, now=2)
        merged = coordinator.update("cam_2", 7, frame, (30, 20, 40, 60), {"x": 5.5, "y": 5}, now=2.1)

        self.assertNotEqual(first, second)
        self.assertEqual(merged, "person_1")
        self.assertEqual(coordinator.local_bindings[("cam_1", 1)], merged)
        self.assertEqual(coordinator.local_bindings[("cam_2", 7)], merged)

    def test_same_camera_id_switch_reuses_recent_global_id(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = (20, 80, 180)

        first = coordinator.update(
            "cam_entrance", 1, frame, (30, 20, 40, 60), None,
            active_local_ids={1}, now=1,
        )
        second = coordinator.update(
            "cam_entrance", 2, frame, (31, 20, 40, 60), None,
            active_local_ids={2}, now=1.4,
        )

        self.assertEqual(second, first)
        self.assertEqual(coordinator.local_bindings[("cam_entrance", 2)], first)

    def test_same_camera_simultaneous_tracks_do_not_merge(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[20:80, 20:60] = (20, 80, 180)
        frame[20:80, 100:140] = (20, 80, 180)

        first = coordinator.update(
            "cam_entrance", 1, frame, (20, 20, 40, 60), None,
            active_local_ids={1, 2}, now=1,
        )
        second = coordinator.update(
            "cam_entrance", 2, frame, (100, 20, 40, 60), None,
            active_local_ids={1, 2}, now=1,
        )

        self.assertNotEqual(second, first)

    def test_overlap_rehydrates_identity_after_global_track_expires(self):
        coordinator = self.coordinator(self.FloorMap(), ttl_seconds=2.0, similarity_threshold=0.70)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = (20, 80, 180)

        first = coordinator.update(
            "cam_1", 1, frame, (30, 20, 40, 60), {"x": 5, "y": 5}, now=1
        )
        coordinator.set_identity(
            first,
            {"known": True, "person_id": "p1", "person_number": 1, "name": "yxq"},
            "cam_1",
            {"x": 5, "y": 5},
            now=1,
        )
        # The original local global track is stale, but its confirmed identity
        # remains in the longer handoff registry.
        second = coordinator.update(
            "cam_2", 7, frame, (30, 20, 40, 60), {"x": 5.5, "y": 5}, now=10
        )

        self.assertEqual(second, first)
        self.assertEqual(
            coordinator.identity_for(second, "cam_2", {"x": 5.5, "y": 5}, now=10)["name"],
            "yxq",
        )

    def test_uncalibrated_map_still_requires_reid_similarity(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        first_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        first_frame[20:80, 30:70] = (20, 80, 180)
        second_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        second_frame[20:80, 30:70] = (180, 80, 20)

        first = coordinator.update("cam_1", 1, first_frame, (30, 20, 40, 60), {"x": 2, "y": 2}, now=1)
        second = coordinator.update("cam_2", 7, second_frame, (30, 20, 40, 60), {"x": 9, "y": 2}, now=2)

        self.assertNotEqual(first, second)

    def test_frame_level_merge_rebinds_existing_local_tracks(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        first = coordinator._new_global(1, "cam_1", None, {"x": 2, "y": 2})
        second = coordinator._new_global(1, "cam_2", None, {"x": 9, "y": 2})
        coordinator.local_bindings[("cam_1", 3)] = first
        coordinator.local_bindings[("cam_2", 8)] = second

        merged = coordinator.merge_global_ids([first, second])

        self.assertEqual(merged, "person_1")
        self.assertEqual(coordinator.local_bindings[("cam_1", 3)], merged)
        self.assertEqual(coordinator.local_bindings[("cam_2", 8)], merged)

    def test_merge_rejects_two_active_owners_in_the_same_camera(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        first = coordinator._new_global(1, "cam_2", None, {"x": 2, "y": 5})
        second = coordinator._new_global(1, "cam_2", None, {"x": 8, "y": 5})
        coordinator.local_bindings[("cam_2", 11)] = first
        coordinator.local_bindings[("cam_2", 12)] = second
        coordinator.active_local_ids_by_camera["cam_2"] = {11, 12}

        merged = coordinator._merge(first, second)

        self.assertIsNone(merged)
        self.assertEqual(
            coordinator.last_merge_rejection_reason,
            "active_same_camera_owner_conflict",
        )
        self.assertEqual(coordinator.local_bindings[("cam_2", 11)], first)
        self.assertEqual(coordinator.local_bindings[("cam_2", 12)], second)
        self.assertNotIn(second, coordinator.redirects)

    def test_directed_transition_rehydrates_entrance_identity_without_map_position(self):
        coordinator = self.coordinator(
            self.TransitionFloorMap(), ttl_seconds=2.0, similarity_threshold=0.70
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = (20, 80, 180)
        entrance_id = coordinator.update(
            "cam_entrance", 1, frame, (30, 20, 40, 60), None, now=1
        )
        coordinator.set_identity(
            entrance_id,
            {"known": True, "person_id": "p1", "person_number": 1, "name": "yxq"},
            "cam_entrance",
            now=1,
        )

        indoor_id = coordinator.update(
            "cam_1", 7, frame, (30, 20, 40, 60), None, now=10
        )
        identity = coordinator.identity_for(indoor_id, "cam_1", None, now=10)

        self.assertEqual(indoor_id, entrance_id)
        self.assertEqual(identity["name"], "yxq")
        self.assertEqual(identity["identity_source"], "transition_handoff")

    def test_transition_rejects_reverse_direction_and_expired_identity(self):
        coordinator = self.coordinator(
            self.TransitionFloorMap(), ttl_seconds=2.0, similarity_threshold=0.70
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = (20, 80, 180)
        first = coordinator.update("cam_1", 1, frame, (30, 20, 40, 60), None, now=1)
        reverse = coordinator.update(
            "cam_entrance", 2, frame, (30, 20, 40, 60), None, now=2
        )
        entrance = coordinator.update(
            "cam_entrance", 3, frame, (30, 20, 40, 60), None, now=30
        )
        coordinator.set_identity(
            entrance,
            {"known": True, "person_id": "p1", "person_number": 1, "name": "yxq"},
            "cam_entrance",
            now=30,
        )
        expired = coordinator.update(
            "cam_1", 4, frame, (30, 20, 40, 60), None, now=51
        )

        self.assertNotEqual(reverse, first)
        self.assertNotEqual(expired, entrance)

    def test_locked_global_identity_cannot_be_overwritten(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        first = coordinator._new_global(1, "cam_entrance", None, None)

        accepted = coordinator.set_identity(
            first, {"known": True, "person_id": "p1", "name": "yxq"},
            "cam_entrance", now=1,
        )
        rejected = coordinator.set_identity(
            first, {"known": True, "person_id": "p2", "name": "other"},
            "cam_entrance", now=2,
        )

        self.assertEqual(accepted["identity_lock_status"], "locked")
        self.assertIsNone(rejected)
        self.assertEqual(
            coordinator.global_identities[first]["identity"]["person_id"], "p1"
        )

    def test_registered_person_cannot_own_two_active_global_tracks(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        first = coordinator._new_global(1, "cam_entrance", None, None)
        second = coordinator._new_global(1, "cam_1", None, None)
        identity = {"known": True, "person_id": "p1", "name": "yxq"}

        coordinator.set_identity(first, identity, "cam_entrance", now=1)
        rejected = coordinator.set_identity(second, identity, "cam_1", now=2)

        self.assertIsNone(rejected)
        self.assertEqual(coordinator.person_locks["p1"], first)
        self.assertNotIn(second, coordinator.global_identities)

    def test_entrance_face_conflict_does_not_merge_active_people(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        indoor = coordinator._new_global(1, "cam_1", None, {"x": 9.5, "y": 4.5})
        entrance = coordinator._new_global(
            2, "cam_entrance", None, {"x": 11.0, "y": 5.35}
        )
        identity = {"known": True, "person_id": "p1", "name": "yxq"}
        coordinator.set_identity(indoor, identity, "cam_1", now=1)

        confirmed = coordinator.set_identity(
            entrance,
            identity,
            "cam_entrance",
            {"x": 11.0, "y": 5.35},
            now=2,
        )

        self.assertIsNone(confirmed)
        self.assertEqual(coordinator.person_locks["p1"], indoor)
        self.assertIn(entrance, coordinator.global_tracks)
        self.assertNotIn(entrance, coordinator.global_identities)

    def test_three_entrance_tracks_cannot_share_one_registered_identity(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        tracks = [
            coordinator._new_global(1, "cam_entrance", None, None)
            for _ in range(3)
        ]
        identity = {"known": True, "person_id": "p1", "name": "worker"}

        results = [
            coordinator.set_identity(track, identity, "cam_entrance", now=2)
            for track in tracks
        ]

        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNone(results[2])
        self.assertEqual(coordinator.person_locks["p1"], tracks[0])
        self.assertEqual(set(coordinator.global_tracks), set(tracks))

    def test_different_registered_people_are_never_merged(self):
        coordinator = self.coordinator(self.UncalibratedFloorMap())
        first = coordinator._new_global(1, "cam_1", None, None)
        second = coordinator._new_global(1, "cam_2", None, None)
        coordinator.set_identity(
            first, {"known": True, "person_id": "p1", "name": "yxq"},
            "cam_1", now=1,
        )
        coordinator.set_identity(
            second, {"known": True, "person_id": "p2", "name": "other"},
            "cam_2", now=1,
        )

        merged = coordinator._merge(first, second)

        self.assertIsNone(merged)
        self.assertIn(first, coordinator.global_tracks)
        self.assertIn(second, coordinator.global_tracks)

    def test_existing_indoor_track_syncs_after_late_entrance_confirmation(self):
        coordinator = self.coordinator(
            self.SpatialTransitionFloorMap(), similarity_threshold=0.72
        )
        indoor = np.zeros((100, 100, 3), dtype=np.uint8)
        indoor[20:80, 30:70] = (50, 87, 0)
        entrance = np.zeros((100, 100, 3), dtype=np.uint8)
        entrance[20:80, 30:70] = (100, 0, 0)

        indoor_id = coordinator.update(
            "cam_1", 7, indoor, (30, 20, 40, 60), {"x": 3.0, "y": 2.0}, now=1
        )
        entrance_id = coordinator.update(
            "cam_entrance", 8, entrance, (30, 20, 40, 60),
            {"x": 11.0, "y": 5.35}, now=2,
        )
        coordinator.set_identity(
            entrance_id,
            {"known": True, "person_id": "p1", "name": "yxq"},
            "cam_entrance", {"x": 11.0, "y": 5.35}, now=2,
        )

        synced_id = coordinator.update(
            "cam_1", 7, indoor, (30, 20, 40, 60), {"x": 3.0, "y": 2.0}, now=2.2
        )
        identity = coordinator.identity_for(
            synced_id, "cam_1", {"x": 3.0, "y": 2.0}, now=2.2
        )

        self.assertEqual(synced_id, indoor_id)
        self.assertEqual(identity["person_id"], "p1")
        self.assertEqual(identity["handoff_match_source"], "directed_spatial_reid")
        self.assertGreaterEqual(identity["appearance_score"], 0.35)

    def test_late_identity_sync_rejects_ambiguous_indoor_crowd(self):
        coordinator = self.coordinator(
            self.SpatialTransitionFloorMap(), similarity_threshold=0.72
        )
        indoor = np.zeros((100, 100, 3), dtype=np.uint8)
        indoor[20:80, 30:70] = (50, 87, 0)
        entrance = np.zeros((100, 100, 3), dtype=np.uint8)
        entrance[20:80, 30:70] = (100, 0, 0)
        first = coordinator.update(
            "cam_1", 7, indoor, (30, 20, 40, 60), {"x": 9.7, "y": 4.3}, now=1
        )
        coordinator.update(
            "cam_1", 9, indoor, (30, 20, 40, 60), {"x": 9.5, "y": 4.4}, now=1
        )
        entrance_id = coordinator.update(
            "cam_entrance", 8, entrance, (30, 20, 40, 60),
            {"x": 11.0, "y": 5.35}, now=2,
        )
        coordinator.set_identity(
            entrance_id,
            {"known": True, "person_id": "p1", "name": "yxq"},
            "cam_entrance", {"x": 11.0, "y": 5.35}, now=2,
        )

        unchanged = coordinator.update(
            "cam_1", 7, indoor, (30, 20, 40, 60), {"x": 9.7, "y": 4.3}, now=2.2
        )

        self.assertEqual(unchanged, first)
        self.assertIsNone(
            coordinator.identity_for(
                unchanged, "cam_1", {"x": 9.7, "y": 4.3}, now=2.2
            )
        )


if __name__ == "__main__":
    unittest.main()
