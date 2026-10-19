"""ByteTrack with camera-owned IDs; preserve the installed matching algorithm."""
import threading
from ultralytics.trackers.byte_tracker import BYTETracker, STrack


class TrackIdAllocator:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def next(self):
        with self.lock:
            self.value += 1
            return self.value


class ScopedByteTracker(BYTETracker):
    def __init__(self, args, allocator=None):
        if not hasattr(BYTETracker, "track_class"):
            raise RuntimeError("Scoped IDs require the inspected ByteTrack track_class API")
        self.id_allocator = allocator or TrackIdAllocator()
        owner = self.id_allocator

        class CameraSTrack(STrack):
            def next_id(self):
                return owner.next()

        self.track_class = CameraSTrack
        super().__init__(args)

    def reset_id(self):
        # A camera reconnect clears trajectories, not another camera's IDs.
        # Its own allocator also survives reconnects to avoid local-ID reuse.
        pass
