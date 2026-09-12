import json
import tempfile
import unittest
from pathlib import Path

from mtmc_config import MTMCConfigError, load_mtmc_config


class MTMCConfigTests(unittest.TestCase):
    def test_loads_valid_defaults_and_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mtmc.json"
            path.write_text(json.dumps({"reid": {"gallery_capacity": 7}}), encoding="utf-8")
            config = load_mtmc_config(path)
        self.assertEqual(config["reid"]["gallery_capacity"], 7)
        self.assertEqual(config["algorithm_version"], "mtmc-2.0.0")

    def test_rejects_invalid_threshold_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mtmc.json"
            path.write_text(
                json.dumps({"reid": {"medium_similarity": 0.9, "high_similarity": 0.8}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MTMCConfigError, "medium_similarity"):
                load_mtmc_config(path)

    def test_rejects_event_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mtmc.json"
            path.write_text(json.dumps({"events": {"path": "../secret"}}), encoding="utf-8")
            with self.assertRaisesRegex(MTMCConfigError, "inside"):
                load_mtmc_config(path)


if __name__ == "__main__":
    unittest.main()
