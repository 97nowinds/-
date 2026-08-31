import threading
import time


class GlobalTargetCoordinator:
    """Own one global target id shared by all camera-local trackers."""

    def __init__(self):
        self.lock = threading.RLock()
        self.target_id = None
        self.source_camera = None
        self.histogram = None
        self.started_at = None
        self.observations = {}
        self.events = []

    def start(self, target_id, source_camera, histogram):
        with self.lock:
            self.target_id = str(target_id)
            self.source_camera = source_camera
            self.histogram = histogram.copy()
            self.started_at = time.time()
            self.observations = {}
            self.events = [
                {
                    "type": "target_started",
                    "camera_id": source_camera,
                    "at": self.started_at,
                }
            ]

    def stop(self):
        with self.lock:
            self.target_id = None
            self.source_camera = None
            self.histogram = None
            self.started_at = None
            self.observations = {}
            self.events = []

    def target(self):
        with self.lock:
            if self.target_id is None:
                return None
            return {
                "target_id": self.target_id,
                "source_camera": self.source_camera,
                "histogram": self.histogram.copy(),
            }

    def observe(self, camera_id, tracker_state, acquired=False):
        now = time.time()
        with self.lock:
            if self.target_id is None:
                return
            previous = self.observations.get(camera_id)
            self.observations[camera_id] = {
                "camera_id": camera_id,
                "status": tracker_state["status"],
                "box": tracker_state.get("box"),
                "confidence": tracker_state.get("confidence", 0.0),
                "updated_at": now,
            }
            if acquired and (previous is None or previous.get("status") != "tracking"):
                self.events.append(
                    {"type": "camera_acquired", "camera_id": camera_id, "at": now}
                )
                self.events = self.events[-20:]

    def snapshot(self):
        now = time.time()
        with self.lock:
            if self.target_id is None:
                return {
                    "active": False,
                    "target_id": None,
                    "state": "idle",
                    "active_cameras": [],
                    "observations": [],
                    "events": [],
                }
            current = [
                item
                for item in self.observations.values()
                if item["status"] == "tracking" and now - item["updated_at"] <= 2.0
            ]
            active_cameras = [item["camera_id"] for item in current]
            if len(active_cameras) >= 2:
                state = "overlap"
            elif active_cameras and active_cameras[0] != self.source_camera:
                state = "transferred"
            elif active_cameras:
                state = "tracking_source"
            else:
                state = "searching"
            return {
                "active": True,
                "target_id": self.target_id,
                "source_camera": self.source_camera,
                "state": state,
                "active_cameras": active_cameras,
                "observations": list(self.observations.values()),
                "events": list(self.events),
                "started_at": self.started_at,
            }
