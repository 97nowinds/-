import json
import tempfile
import unittest
from pathlib import Path

from mtmc_events import AssociationEventStore


class AssociationEventStoreTests(unittest.TestCase):
    def test_records_and_restores_history_without_live_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            store = AssociationEventStore(path, now_fn=lambda: 100.0)
            store.record("create", global_id="person_1", unix_time=99.0)
            restored = AssociationEventStore(path, now_fn=lambda: 100.0)

            self.assertEqual(restored.recent()[0]["global_id"], "person_1")
            self.assertFalse(restored.summary()["live_tracks_restored"])

    def test_biometric_vectors_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            store = AssociationEventStore(path)
            event = store.record(
                "associate",
                global_id="person_1",
                embedding=[0.1, 0.2],
                diagnostics={"feature": [1, 2], "score": 0.9},
            )
            raw = json.loads(path.read_text(encoding="utf-8"))

            self.assertNotIn("embedding", event)
            self.assertNotIn("feature", raw["diagnostics"])

    def test_retention_excludes_old_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(
                json.dumps({"sequence": 1, "event": "expire", "unix_time": 1.0}) + "\n",
                encoding="utf-8",
            )
            store = AssociationEventStore(path, retention_days=1, now_fn=lambda: 200000.0)
            self.assertEqual(store.recent(), [])
            self.assertEqual(path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
