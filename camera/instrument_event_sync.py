"""Persist the receiver's existing sessions without reinterpreting model decisions."""
import ctypes
from ctypes import wintypes
import json
import math
import threading

from management_notifications import iso, stamp, route_accepts


def read_receiver_token(target):
    class Credential(ctypes.Structure):
        _fields_ = [('Flags', wintypes.DWORD), ('Type', wintypes.DWORD),
                    ('TargetName', wintypes.LPWSTR), ('Comment', wintypes.LPWSTR),
                    ('LastWritten', wintypes.FILETIME), ('CredentialBlobSize', wintypes.DWORD),
                    ('CredentialBlob', ctypes.c_void_p), ('Persist', wintypes.DWORD),
                    ('AttributeCount', wintypes.DWORD), ('Attributes', ctypes.c_void_p),
                    ('TargetAlias', wintypes.LPWSTR), ('UserName', wintypes.LPWSTR)]
    api = ctypes.WinDLL('advapi32', use_last_error=True)
    api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                             ctypes.POINTER(ctypes.POINTER(Credential))]
    api.CredReadW.restype = wintypes.BOOL
    api.CredFree.argtypes = [ctypes.c_void_p]
    pointer = ctypes.POINTER(Credential)()
    if not api.CredReadW(target, 1, 0, ctypes.byref(pointer)):
        raise RuntimeError('Receiver credential unavailable in current Windows account')
    try:
        credential = pointer.contents
        if not 1 <= credential.CredentialBlobSize <= 512:
            raise ValueError('Invalid receiver credential length')
        return ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize).decode('ascii')
    finally:
        api.CredFree(pointer)


class InstrumentEventSync:
    def __init__(self, service, client):
        self.service, self.client = service, client
        self.stop_event = threading.Event()
        self.status = {'state': 'starting', 'last_success_at': None, 'error': None}
        with service.lock, service.db:
            service.db.execute('CREATE TABLE IF NOT EXISTS instrument_sessions('
                               'id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated REAL NOT NULL)')
            service.db.execute("INSERT OR IGNORE INTO settings VALUES('instrument_notifications_since',?)",
                               (iso(service.wall()),))
            self.notify_since = stamp(service.db.execute(
                "SELECT value FROM settings WHERE key='instrument_notifications_since'").fetchone()[0])

    @staticmethod
    def validate_session(row):
        required = ('session_id', 'person_key', 'camera_id', 'instrument_id',
                    'instrument_name', 'person_label', 'started_at', 'last_observed_at', 'status')
        if any(not isinstance(row.get(k), str) or not 0 < len(row[k]) <= 256 for k in required):
            raise ValueError('Invalid receiver session')
        start, last = stamp(row['started_at']), stamp(row['last_observed_at'])
        if last < start or row['status'] not in {'open', 'ended', 'interrupted'}:
            raise ValueError('Invalid session time or state')
        end = row.get('ended_at')
        if row['status'] == 'ended':
            duration = row.get('duration_seconds')
            if (not end or type(duration) not in (int, float) or not math.isfinite(duration)
                    or stamp(end) < start or stamp(end) > last
                    or abs(duration - (stamp(end) - start)) > .002):
                raise ValueError('Invalid session duration')
        elif end is not None or row.get('duration_seconds') is not None:
            raise ValueError('Open session must not have an invented end')
        return {k: row.get(k) for k in (*required, 'ended_at', 'duration_seconds',
                                      'observed_duration_seconds', 'interruption_reason', 'last_valid_source_at', 'observation_window_seconds')}

    def ingest(self, body):
        if not isinstance(body, dict) or not isinstance(body.get('sessions'), list):
            raise ValueError('Receiver session history unavailable')
        people = {p['person_key']: p for p in body.get('people', []) if isinstance(p, dict) and p.get('person_key')}
        rows = [self.validate_session(r) for r in body['sessions']]
        service = self.service
        with service.lock, service.db:
            for row in rows:
                prior = service.db.execute('SELECT payload FROM instrument_sessions WHERE id=?',
                                           (row['session_id'],)).fetchone()
                if prior:
                    old = json.loads(prior[0])
                    if (any(old[k] != row[k] for k in ('started_at', 'person_key', 'camera_id', 'instrument_id'))
                            or (old['status'] == 'ended' and row != old)
                            or stamp(row['last_observed_at']) < stamp(old['last_observed_at'])):
                        continue
                service.db.execute('INSERT INTO instrument_sessions VALUES(?,?,?) '
                    'ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,updated=excluded.updated',
                    (row['session_id'], json.dumps(row, ensure_ascii=False), service.wall()))
                person = people.get(row['person_key'], {})
                for kind, at in [('started', row['started_at']), ('ended', row.get('ended_at'))]:
                    if not at:
                        continue
                    event = {'event_id': 'instrument:' + row['session_id'] + ':' + kind,
                        'event_type': 'instrument.' + kind, 'lab_id': service.config['lab_id'],
                        'person_id': person.get('person_id') or person.get('global_id') or 'track:' + row['person_key'],
                        'display_name': person.get('display_name') or '匿名轨迹 ' + row['person_label'],
                        'camera_id': row['camera_id'], 'instrument_id': row['instrument_id'],
                        'instrument_name': row['instrument_name'], 'session_id': row['session_id'],
                        'observed_at': at, 'started_at': row['started_at'], 'severity': 'L0',
                        'evidence_source': 'member_minicpm_persistent_session', 'time_domain': 'source_frame'}
                    if kind == 'ended':
                        event.update(ended_at=at, duration_seconds=str(row['duration_seconds']))
                    encoded = json.dumps(event, ensure_ascii=False, sort_keys=True)
                    inserted = service.db.execute('INSERT OR IGNORE INTO events(id,payload,received) VALUES(?,?,?)',
                        (event['event_id'], encoded, service.wall())).rowcount
                    if not inserted:
                        continue
                    service._audit('instrument-sync', 'event.record', event['event_id'])
                    # Backfill history into the backend; only new, recent transitions notify groups.
                    age = service.wall() - stamp(at)
                    if (event['event_type'] in service.config.get('silent_events', [])
                            or stamp(at) < self.notify_since or not -2 <= age <= 300):
                        continue
                    for route in service.config['routes']:
                        if route_accepts(route, event):
                            service.db.execute('INSERT OR IGNORE INTO outbox(event_id,route_id,state,due,priority) '
                                'VALUES(?,?,?,?,?)', (event['event_id'], route['id'], 'pending', service.wall(), 1))

    def poll_once(self):
        if self.client is None:
            raise RuntimeError('Receiver connection is not configured')
        response = self.client._request('GET', '/api/demo/events')
        if not response.ok:
            raise RuntimeError('Receiver history HTTP ' + str(response.status))
        self.ingest(response.body)
        self.status = {'state': 'connected', 'last_success_at': iso(self.service.wall()), 'error': None}

    def snapshot(self):
        with self.service.lock:
            rows = []
            for session, event in self.service.db.execute(
                    "SELECT s.payload,e.payload FROM instrument_sessions s "
                    "LEFT JOIN events e ON e.id='instrument:'||s.id||':started' "
                    "ORDER BY json_extract(s.payload,'$.started_at') DESC LIMIT 200"):
                row = json.loads(session)
                if event:
                    row['person_id'] = json.loads(event).get('person_id')
                rows.append(row)
        return {'sessions': rows, 'sync': dict(self.status), 'generated_at': iso(self.service.wall()),
                'time_domain': 'source_frame'}

    def start(self):
        def run():
            while not self.stop_event.is_set():
                try:
                    self.poll_once()
                except Exception as error:
                    self.status = {**self.status, 'state': 'unavailable', 'error': type(error).__name__}
                self.stop_event.wait(5)
        self.thread = threading.Thread(target=run, daemon=True, name='instrument-event-sync')
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
