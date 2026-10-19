"""Bounded exact snapshot/track lookup; never guess names from list order."""
from collections import OrderedDict
from copy import deepcopy
import threading


class SnapshotTrackIndex:
    def __init__(self, max_snapshots=300):
        self.max_snapshots = max_snapshots
        self.snapshots = OrderedDict()
        self.frames = {}
        self.lock = threading.Lock()

    def remember(self, event):
        camera = event['camera']
        timing=camera.get('timing') or {}
        time_domain=event.get('time_domain') or timing.get('event_time_domain') or timing.get('time_domain')
        key = (str(camera['camera_id']), str(event['snapshot_id']))
        tracks = {}
        for track in camera.get('tracks', []):
            track_id = str(track['track_id'])
            # Duplicate keys are ambiguous, not a reason to choose the last row.
            if track_id in tracks:
                tracks[track_id] = None
            else:
                tracks[track_id] = {field: deepcopy(track.get(field)) for field in
                    ('global_id', 'person_id', 'display_name', 'observed_at', 'bbox', 'bbox_format')}
                timing=camera.get('timing') or {}
                tracks[track_id]['time_domain']=(event.get('time_domain') or timing.get('event_time_domain')
                                                or timing.get('time_domain'))
        with self.lock:
            self.snapshots[key] = tracks
            self.frames[key] = {field: deepcopy(camera.get(field)) for field in
                ('image_url', 'image_sha256', 'frame_width', 'frame_height', 'source_frame_id')}
            self.frames[key].update(captured_at=event.get('captured_at'),time_domain=time_domain)
            self.snapshots.move_to_end(key)
            while len(self.snapshots) > self.max_snapshots:
                removed, _ = self.snapshots.popitem(last=False)
                self.frames.pop(removed, None)

    def evidence_frame(self, camera_id, snapshot_id):
        with self.lock:
            return deepcopy(self.frames.get((str(camera_id), str(snapshot_id))))

    def evidence_track(self, camera_id, snapshot_id, track_id):
        """Read original identity/observation time for strict evidence validation."""
        key = (str(camera_id), str(snapshot_id))
        with self.lock:
            return deepcopy(self.snapshots.get(key, {}).get(str(track_id)))

    def resolve(self, result):
        item = dict(result)
        required = ('camera_id', 'snapshot_id', 'track_id')
        match = None
        if all(result.get(key) is not None for key in required):
            match = self.evidence_track(*(result[key] for key in required))
        # Receiver predictions cannot assign an identity absent from the sent row.
        for field in ('global_id', 'person_id', 'display_name'):
            item[field] = match.get(field) if match else None
        # Local proxy annotation only, never a receiver-bound Schema field.
        item['handoff_match_status'] = 'matched' if match else 'unmatched'
        for field in ('person_name', 'name', 'global_track_id', 'person_number'):
            item.pop(field, None)
        return item
