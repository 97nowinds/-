import unittest
from collections import defaultdict, deque

from app import CameraWorker


class FakeHandoff:
    def __init__(self):
        self.confirmations = []

    def confirm(self, camera_id, identity, appearance):
        self.confirmations.append((camera_id, identity["person_id"], appearance))


class CameraWorkerIdentityTests(unittest.TestCase):
    def worker(self):
        worker = CameraWorker.__new__(CameraWorker)
        worker.camera = {"id": "cam_entrance"}
        worker.handoff = FakeHandoff()
        worker.identity = None
        worker.identity_track_id = None
        worker.identity_votes = deque(maxlen=7)
        worker.track_identities = {}
        worker.identity_votes_by_track = defaultdict(lambda: deque(maxlen=7))
        worker.handoff_unknown_count = 0
        return worker

    @staticmethod
    def known(person_id):
        return {"known": True, "person_id": person_id, "name": person_id}

    def test_identity_votes_are_isolated_per_local_track(self):
        worker = self.worker()

        for _ in range(2):
            worker.accept_face_result(self.known("person_a"), "a", track_id=1)
        for _ in range(3):
            worker.accept_face_result(self.known("person_b"), "b", track_id=2)

        self.assertNotIn(1, worker.track_identities)
        self.assertEqual(worker.track_identities[2]["person_id"], "person_b")

        worker.accept_face_result(self.known("person_a"), "a", track_id=1)

        self.assertEqual(worker.track_identities[1]["person_id"], "person_a")
        self.assertEqual(worker.track_identities[2]["person_id"], "person_b")

    def test_single_known_vote_does_not_move_existing_identity(self):
        worker = self.worker()
        worker.track_identities[1] = self.known("person_a")
        worker.identity = worker.track_identities[1]
        worker.identity_track_id = 1

        worker.accept_face_result(self.known("person_b"), "b", track_id=2)

        self.assertEqual(worker.identity_track_id, 1)
        self.assertNotIn(2, worker.track_identities)

    def test_clearing_one_identity_preserves_other_tracks(self):
        worker = self.worker()
        worker.track_identities = {
            1: self.known("person_a"),
            2: self.known("person_b"),
        }
        worker.identity = worker.track_identities[1]
        worker.identity_track_id = 1

        worker.clear_identity(1)

        self.assertNotIn(1, worker.track_identities)
        self.assertEqual(worker.track_identities[2]["person_id"], "person_b")
        self.assertEqual(worker.identity_track_id, 2)

    def test_face_target_prefers_unidentified_track(self):
        worker = self.worker()
        worker.track_identities[1] = self.known("person_a")
        tracks = [
            {"track_id": 1, "box": (0, 0, 100, 200), "confidence": 0.95},
            {"track_id": 2, "box": (200, 0, 70, 160), "confidence": 0.80},
        ]

        selected = worker.select_face_target(tracks)

        self.assertEqual(selected["track_id"], 2)


if __name__ == "__main__":
    unittest.main()
