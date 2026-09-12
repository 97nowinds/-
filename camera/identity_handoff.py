import threading
import time

import numpy as np


class IdentityHandoff:
    """Share confirmed identities between camera-local tracks by appearance."""

    def __init__(self, feature_extractor, ttl_seconds=12.0, similarity_threshold=0.72, match_margin=0.05):
        self.feature_extractor = feature_extractor
        self.ttl_seconds = float(ttl_seconds)
        self.similarity_threshold = float(similarity_threshold)
        self.match_margin = float(match_margin)
        self.lock = threading.RLock()
        self.observations = {}

    def appearance(self, frame, box):
        return self.feature_extractor.extract(frame, box)

    @staticmethod
    def similarity(first, second):
        if first is None or second is None:
            return -1.0
        first = np.asarray(first, dtype=np.float32).reshape(-1)
        second = np.asarray(second, dtype=np.float32).reshape(-1)
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.dot(first, second) / denominator) if denominator > 1e-8 else -1.0

    def confirm(self, camera_id, identity, appearance):
        if not identity.get("known"):
            return
        with self.lock:
            self.observations[identity["person_id"]] = {
                "identity": dict(identity),
                "camera_id": camera_id,
                "appearance": appearance.copy() if appearance is not None else None,
                "updated_at": time.monotonic(),
            }

    def inherit(self, camera_id, appearance):
        now = time.monotonic()
        with self.lock:
            candidates = []
            for person_id, observation in list(self.observations.items()):
                age = now - observation["updated_at"]
                if age > self.ttl_seconds:
                    self.observations.pop(person_id, None)
                    continue
                if observation["camera_id"] == camera_id:
                    continue
                score = self.similarity(observation["appearance"], appearance)
                if score >= self.similarity_threshold:
                    candidates.append((score, observation))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates:
                return None
            score, observation = candidates[0]
            if len(candidates) > 1 and score - candidates[1][0] < self.match_margin:
                return None
            result = dict(observation["identity"])
            result.update(
                {
                    "identity_source": "handoff",
                    "handoff_from_camera": observation["camera_id"],
                    "appearance_score": round(score, 3),
                }
            )
            return result
