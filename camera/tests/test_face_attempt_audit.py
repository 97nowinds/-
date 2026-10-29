import unittest
from face_attempt_audit import FaceAttemptAudit, run_face_attempt, warm_face_models, scale_box
from types import SimpleNamespace


class FaceAuditTests(unittest.TestCase):
    def test_native_input_preserves_local_track_and_returns_preview_coordinates(self):
        import numpy as np
        small = np.zeros((540, 960, 3), dtype=np.uint8)
        native = np.zeros((1440, 2560, 3), dtype=np.uint8)
        calls = []
        result = {"known": True, "person_id": "registered"}
        def confirm(image, box, track):
            calls.append((image is native, box, track))
            return (320, 160, 128, 160), result
        worker = SimpleNamespace(confirm_face_identity=confirm)
        candidate = run_face_attempt(worker, small, (120, 60, 96, 240), 77, source_frame=native)
        self.assertEqual(calls, [(True, (320, 160, 256, 640), 77)])
        self.assertEqual(candidate[0], (120, 60, 48, 60))
        self.assertIs(candidate[1], result)

    def test_mapping_uses_actual_axes_and_clips_edges(self):
        self.assertEqual(scale_box((-1, -2, 11, 12), (100, 200), (200, 600)), (0, 0, 30, 20))
        self.assertEqual(scale_box((195, 95, 20, 20), (100, 200), (200, 600)), (585, 190, 15, 10))

    def test_fixed_face_shapes_warm_before_concurrent_work(self):
        seen = []
        def session():
            return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input")],
                                   run=lambda _, values: seen.append(values["input"].shape))
        engine = SimpleNamespace(available=True, det_size=(640, 640), analysis=SimpleNamespace(
            models={"detection": SimpleNamespace(session=session()),
                    "recognition": SimpleNamespace(session=session(), input_size=(112, 112))}))
        warm_face_models(engine)
        self.assertEqual(seen, [(1, 3, 640, 640), (1, 3, 112, 112)])

    def test_failed_attempt_returns_no_identity_and_next_attempt_can_succeed(self):
        def failed(*args):
            raise RuntimeError("simulated CUDA failure")
        worker = SimpleNamespace(camera={"id": "test"}, face_attempt_audit=FaceAttemptAudit(),
                                 confirm_face_identity=failed, face_recognition_error=None)
        with self.assertLogs(level="ERROR"):
            self.assertIsNone(run_face_attempt(worker, None, (0, 0, 10, 10), 7))
        self.assertEqual(worker.face_attempt_audit.payload()["reason_counts"], {"inference_error": 1})
        worker.confirm_face_identity = lambda *args: "actual candidate"
        self.assertEqual(run_face_attempt(worker, None, (0, 0, 10, 10), 7), "actual candidate")
        self.assertIsNone(worker.face_recognition_error)

    def test_bounded_metadata_keeps_failures_without_features(self):
        audit = FaceAttemptAudit(capacity=2)
        audit.record(track_id=1, detection_ms=10, total_ms=11)
        audit.record(track_id=2, detection_ms=12, total_ms=13,
                     result={"known": False, "reason": "face_too_small", "embedding": [123]})
        audit.record(track_id=2, detection_ms=14, total_ms=16,
                     result={"known": False, "distance": .6, "margin": .02}, face_box=(0, 0, 30, 40))
        data = audit.payload()
        self.assertEqual(len(data["recent"]), 2)
        self.assertEqual(data["reason_counts"]["no_face"], 1)
        self.assertNotIn("embedding", str(data))
        self.assertEqual(data["recent"][-1]["face_size_px"], [30, 40])
        self.assertAlmostEqual(data["recent"][-1]["similarity"], .4)
