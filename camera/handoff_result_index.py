"""Bounded exact snapshot/track lookup; never guess names from list order."""
from collections import OrderedDict
from copy import deepcopy
import threading


class SnapshotTrackIndex:
    def __init__(self, max_snapshots=300):
        self.max_snapshots = max_snapshots
        self.snapshots = OrderedDict()
        self.lock = threading.Lock()

    def remember(self, event):
        camera = event['camera']
        key = (str(camera['camera_id']), str(event['snapshot_id']))
        tracks = {}
        for track in camera.get('tracks', []):
            track_id = str(track['track_id'])
            # Duplicate keys are ambiguous, not a reason to choose the last row.
            if track_id in tracks:
                tracks[track_id] = None
            else:
                tracks[track_id] = {field: deepcopy(track.get(field)) for field in
                    ('global_id', 'person_id', 'display_name', 'observed_at')}
        with self.lock:
            self.snapshots[key] = tracks
            self.snapshots.move_to_end(key)
            while len(self.snapshots) > self.max_snapshots:
                self.snapshots.popitem(last=False)

    def resolve(self, result):
        item = dict(result)
        required = ('camera_id', 'snapshot_id', 'track_id')
        match = None
        if all(result.get(key) is not None for key in required):
            key = (str(result['camera_id']), str(result['snapshot_id']))
            with self.lock:
                match = deepcopy(self.snapshots.get(key, {}).get(str(result['track_id'])))
        # Receiver predictions cannot assign an identity absent from the sent row.
        for field in ('global_id', 'person_id', 'display_name'):
            item[field] = match.get(field) if match else None
        # Local proxy annotation only, never a receiver-bound Schema field.
        item['handoff_match_status'] = 'matched' if match else 'unmatched'
        for field in ('person_name', 'name', 'global_track_id', 'person_number'):
            item.pop(field, None)
        return item
