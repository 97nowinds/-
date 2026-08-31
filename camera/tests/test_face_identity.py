import tempfile
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from face_identity import FaceCapturePolicy, FaceIdentityStore


class FaceIdentityResultTests(unittest.TestCase):
    def test_unknown_identity_uses_unregistered_label(self):
        result = FaceIdentityStore.unknown()

        self.assertFalse(result["known"])
        self.assertEqual(result["name"], "未注册")
        self.assertIsNone(result["person_id"])

    def test_required_arcface_never_falls_back_to_lbph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            people_path = root / "people.json"
            faces_dir = root / "faces"
            sample_dir = faces_dir / "person_001_session"
            sample_dir.mkdir(parents=True)
            people_path.write_text(
                json.dumps(
                    {
                        "person_001": {
                            "id": "person_001",
                            "name": "test",
                            "label": 1,
                            "sample_count": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            face = np.random.default_rng(4).integers(
                30, 220, (160, 160), dtype=np.uint8
            )
            cv2.imwrite(str(sample_dir / "0001.png"), face)
            store = FaceIdentityStore(
                people_path,
                faces_dir,
                modern_model_root=root / "missing-models",
                required_engine="arcface",
            )

            result = store.predict(face)

            self.assertFalse(result["known"])
            self.assertEqual(result["reason"], "arcface_unavailable")


class FaceCapturePolicyTests(unittest.TestCase):
    def test_policy_accepts_reasonable_face(self):
        face = np.full((120, 120), 128, dtype=np.uint8)
        face[20:100, 20:100] = 180
        face[40:80, 40:80] = 60
        face = np.clip(face + np.random.default_rng(1).integers(-8, 9, face.shape), 0, 255).astype(np.uint8)

        policy = FaceCapturePolicy(min_sharpness=5.0, min_contrast=1.0)
        result = policy.assess(face)

        self.assertTrue(result["accepted"])
        self.assertGreaterEqual(result["width"], 120)
        self.assertGreater(result["contrast"], 1.0)

    def test_policy_rejects_too_small_face(self):
        face = np.full((32, 32), 128, dtype=np.uint8)
        policy = FaceCapturePolicy()

        result = policy.assess(face)

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "face_too_small")


class FaceRegistrationSessionTests(unittest.TestCase):
    def test_registration_session_collects_and_finalizes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            class FakeStore:
                def __init__(self):
                    self.faces_dir = root / "faces"
                    self.faces_dir.mkdir(parents=True, exist_ok=True)
                    self.capture_policy = FaceCapturePolicy(min_sharpness=5.0, min_contrast=1.0)
                    self.registered = None
                    self.reloaded = False

                def save_face_sample(self, person_id, face_gray, *, directory=None, sample_index=None):
                    directory = Path(directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f"{sample_index:04d}.png"
                    path.write_bytes(b"sample")
                    return path

                def register_person(self, person_id, name, label, *, sample_count, created_at=None):
                    self.registered = {
                        "person_id": person_id,
                        "name": name,
                        "label": label,
                        "sample_count": sample_count,
                        "created_at": created_at,
                    }

                def reload(self):
                    self.reloaded = True

            store = FakeStore()
            session = FaceIdentityStore.start_registration(
                store,
                "person_010",
                "test_user",
                10,
                target_samples=2,
                min_samples=2,
                max_samples=3,
                session_id="session_a",
            )

            face = np.full((120, 120), 128, dtype=np.uint8)
            face[15:105, 15:105] = 175
            face[45:85, 45:85] = 55
            face = np.clip(face + np.random.default_rng(2).integers(-6, 7, face.shape), 0, 255).astype(np.uint8)

            first = session.add_sample(face)
            second = session.add_sample(face)
            summary = session.finalize()

            self.assertTrue(first["accepted"])
            self.assertTrue(second["accepted"])
            self.assertTrue(session.can_finish)
            self.assertEqual(summary["person_id"], "person_010")
            self.assertTrue(store.reloaded)
            self.assertEqual(store.registered["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
