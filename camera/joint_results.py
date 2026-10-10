"""Local display projection and trace log; never invent model decisions."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import threading
import time


def timestamp(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('result timestamp lacks time zone')
    return stamp.timestamp()


class JointResultProjection:
    def __init__(self, history_path=None, wall=time.time, monotonic=time.monotonic):
        self.history_path = Path(history_path) if history_path else None
        self.wall, self.monotonic = wall, monotonic
        self.deadlines = {}
        self.newest = {}
        self.current = {}
        self.recorded = set()
        self.lock = threading.Lock()

    def project(self, payload):
        if payload.get('time_domain') != 'authority':
            raise ValueError('result time domain must be authority')
        result = deepcopy(payload)
        with self.lock:
            for item in result['analyses']:
                fingerprint = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if fingerprint not in self.recorded and self.history_path:
                    self.history_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.history_path.open('a', encoding='utf-8') as stream:
                        stream.write(fingerprint + '\n')
                    self.recorded.add(fingerprint)
                if item.get('status') == 'valid' and not self._expiry(item):
                    item.update(status='expired', is_interacting=None, reason_code='result_expired')
            rows = []
            for item in result['results']:
                if item.get('handoff_match_status') != 'matched':
                    continue
                if not self._expiry(item):
                    continue
                key = (item['camera_id'], item['track_id'])
                captured = timestamp(item['captured_at'])
                if captured < self.newest.get(key, float('-inf')):
                    previous = self.current.get(key)
                    if previous and self._expiry(previous):
                        rows.append(deepcopy(previous))
                    continue
                self.newest[key] = captured
                self.current[key] = deepcopy(item)
                rows.append(item)
            result['results'] = rows
        return result

    def _expiry(self, item):
        published, expires = timestamp(item['published_at']), timestamp(item['expires_at'])
        expires = min(expires, published + 5)
        key = (item['camera_id'], item['snapshot_id'], item['track_id'], item['instrument_id'],
               item['analysis_id'])
        # First observation fixes a local monotonic deadline. Queries cannot extend it.
        deadline = self.monotonic() + max(0, expires - self.wall())
        self.deadlines[key] = min(self.deadlines.get(key, deadline), deadline)
        return self.wall() < expires and self.monotonic() < self.deadlines[key]
