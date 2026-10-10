"""Small server-side transport primitives for the joint demo."""
from copy import deepcopy
import hashlib
import hmac
import threading
import time
import json
import uuid
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from pathlib import Path


def load_demo_config():
    return json.loads((Path(__file__).parent / 'config/joint_demo.json').read_text(encoding='utf-8'))


def iso_time(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec='milliseconds')


class AuthoritySync:
    """Periodic dedicated probes, fail closed until shared limits are confirmed."""
    def __init__(self, source_clock, config, token):
        self.source_clock, self.config, self.token = source_clock, config, token
        self.mapping = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        keys = ('sync_refresh_seconds', 'sync_valid_seconds',
                'sync_max_uncertainty_ms', 'sync_jump_limit_ms')
        if not self.token or any(self.config.get(key) is None or self.config[key] <= 0 for key in keys):
            return
        self.thread = threading.Thread(target=self._run, daemon=True, name='handoff-clock-sync')
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.probe()
            except (OSError, ValueError, KeyError, TypeError):
                with self.lock:
                    self.mapping = None
            self.stop_event.wait(self.config['sync_refresh_seconds'])

    def probe(self):
        probe_id = uuid.uuid4().hex
        t1, m1 = time.time(), time.monotonic()
        body = {'probe_id': probe_id, 'client_sent_at': iso_time(t1)}
        req = Request(self.config['image_base'] + '/api/handoff/time-sync',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.token}, method='POST')
        with urlopen(req, timeout=3) as response:
            result = json.load(response)
        t4, m4 = time.time(), time.monotonic()
        if result.get('probe_id') != probe_id or result.get('client_sent_at') != body['client_sent_at']:
            raise ValueError('sync echo mismatch')
        if abs((t4 - t1) - (m4 - m1)) * 1000 > self.config['sync_jump_limit_ms']:
            raise ValueError('local clock jumped')
        t2, t3 = [datetime.fromisoformat(result[key].replace('Z', '+00:00')).timestamp()
                  for key in ('authority_received_at', 'authority_sent_at')]
        sample = estimate_sync(t1, t2, t3, t4)
        sample['uncertainty_ms'] += 1  # Client echo is quantized to milliseconds.
        if sample['uncertainty_ms'] > self.config['sync_max_uncertainty_ms']:
            raise ValueError('sync uncertainty exceeds confirmed limit')
        with self.lock:
            old = self.mapping
            if old and (old['authority_clock'] != result['authority_clock'] or
                    abs(old['offset_ms'] - sample['offset_ms']) > self.config['sync_jump_limit_ms']):
                self.mapping = None
                raise ValueError('clock mapping changed; new probe required')
            self.mapping = dict(sample, sync_id=probe_id, authority_clock=result['authority_clock'],
                source_clock=self.source_clock, source_reference_at=t4,
                sync_valid_from=t1 + sample['offset_ms'] / 1000,
                sync_valid_until=t4 + sample['offset_ms'] / 1000 + self.config['sync_valid_seconds'],
                received_monotonic=m4, received_wall=t4,
                deadline=m4 + self.config['sync_valid_seconds'])

    def current(self):
        with self.lock:
            sample = deepcopy(self.mapping)
        if not sample or time.monotonic() >= sample['deadline']:
            return None
        jump = abs((time.time() - sample['received_wall']) -
                   (time.monotonic() - sample['received_monotonic'])) * 1000
        if jump > self.config['sync_jump_limit_ms']:
            with self.lock:
                self.mapping = None
            return None
        return sample


class ImmutableSnapshots:
    """Never evict live promises to make room, or reuse a snapshot identifier."""
    def __init__(self, retention_seconds=60, max_bytes=128 * 1024 * 1024,
                 monotonic=time.monotonic, wall=time.time):
        if retention_seconds < 30:
            raise ValueError('JPEG retention must be at least 30 seconds')
        self.retention = retention_seconds
        self.max_bytes = max_bytes
        self.monotonic, self.wall = monotonic, wall
        self.entries = {}
        self.retired = set()
        self.lock = threading.Lock()

    def _prune(self):
        now = self.monotonic()
        for key in list(self.entries):
            if now >= self.entries[key]['deadline']:
                del self.entries[key]
                self.retired.add(key)

    def put(self, snapshot_id, jpeg, metadata):
        jpeg = bytes(jpeg)
        with self.lock:
            self._prune()
            if snapshot_id in self.retired:
                raise ValueError('snapshot expired; identifier cannot be reused')
            old = self.entries.get(snapshot_id)
            if old:
                if old['jpeg'] != jpeg or old['original'] != metadata:
                    raise ValueError('snapshot content cannot change')
                return deepcopy(old['metadata'])
            if sum(len(item['jpeg']) for item in self.entries.values()) + len(jpeg) > self.max_bytes:
                raise BufferError('snapshot capacity exhausted; live JPEGs retained')
            available = self.wall()
            result = deepcopy(metadata)
            result.update(image_sha256=hashlib.sha256(jpeg).hexdigest(),
                          image_available_at=available,
                          image_expires_at=available + self.retention)
            self.entries[snapshot_id] = dict(jpeg=jpeg, original=deepcopy(metadata),
                metadata=result, deadline=self.monotonic() + self.retention)
            return deepcopy(result)

    def get(self, snapshot_id):
        with self.lock:
            self._prune()
            item = self.entries.get(snapshot_id)
            return item['jpeg'] if item else None


def bearer_authorized(headers, token):
    return bool(token) and hmac.compare_digest(
        headers.get('Authorization', ''), 'Bearer ' + token)


def estimate_sync(t1, t2, t3, t4):
    """Four timestamps in seconds; business receipt times are never inputs."""
    import math
    if not all(math.isfinite(value) for value in (t1, t2, t3, t4)):
        raise ValueError('non-finite sync timestamp')
    network = (t4 - t1) - (t3 - t2)
    if t4 < t1 or t3 < t2 or network < 0:
        raise ValueError('clock jump or invalid sync sample')
    return {'offset_ms': ((t2 - t1) + (t3 - t4)) * 500,
            'uncertainty_ms': network * 500}
