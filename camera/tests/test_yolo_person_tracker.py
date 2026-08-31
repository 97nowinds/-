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

    def coordinator(self, floor_map, **kwargs):
        return CrossCameraTrackCoordinator(FakeFeatureExtractor(), floor_map, **kwargs)

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


if __name__ == "__main__":
    unittest.main()
