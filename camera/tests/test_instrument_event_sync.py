import copy
import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instrument_event_sync import InstrumentEventSync
from management_notifications import ManagementNotifications, iso


class InstrumentSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = 1793020000.0
        config = json.loads((Path(__file__).parents[1] / 'config/management.json').read_text(encoding='utf-8'))
        self.sent = []
        self.service = ManagementNotifications(Path(self.temp.name) / 'test.sqlite3', config,
            environ={r['webhook_env']: 'https://example.invalid' for r in config['routes']},
            wall=lambda: self.now, sender=lambda url, body: self.sent.append(body))
        self.sync = InstrumentEventSync(self.service, None)

    def tearDown(self):
        self.service.db.close()
        self.temp.cleanup()

    def body(self, ended=False, age=0):
        start = self.now-age
        row = dict(session_id='test-session', person_key='test-person', person_label='track-test',
            camera_id='cam_1', instrument_id='instrument_006', instrument_name='高速离心机',
            started_at=iso(start), last_observed_at=iso(start+9 if ended else start),
            ended_at=iso(start+9) if ended else None, duration_seconds=9.0 if ended else None,
            observed_duration_seconds=9.0 if ended else 0.0, status='ended' if ended else 'open')
        return {'sessions': [row], 'people': []}

    def test_start_end_idempotent_across_restart_and_kept_silent(self):
        self.sync.ingest(self.body())
        self.sync.ingest(self.body())
        self.now += 9
        self.sync.ingest(self.body(True, age=9))
        InstrumentEventSync(self.service, None).ingest(self.body(True, age=9))
        self.assertEqual(self.service.db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 2)
        self.assertEqual(self.service.db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0], 0)
        self.assertFalse(self.service.deliver_once())
        self.assertEqual(self.sent, [])
        self.assertEqual(self.sync.snapshot()['sessions'][0]['duration_seconds'], 9)

    def test_backfill_does_not_notify_and_gap_does_not_end(self):
        self.sync.ingest(self.body(age=3600))
        self.now += 7200
        self.assertEqual(self.sync.snapshot()['sessions'][0]['status'], 'open')
        self.assertIsNone(self.sync.snapshot()['sessions'][0]['ended_at'])
        self.assertEqual(self.service.db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0], 0)

    def test_invalid_duration_rejected_atomically(self):
        body = self.body(True)
        body['sessions'][0]['duration_seconds'] = 999
        with self.assertRaises(ValueError): self.sync.ingest(body)
        self.assertEqual(self.service.db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 0)

    def test_closed_session_cannot_reopen(self):
        self.sync.ingest(self.body(True, age=9))
        self.sync.ingest(self.body(age=9))
        self.assertEqual(self.sync.snapshot()['sessions'][0]['status'], 'ended')


if __name__ == '__main__': unittest.main()
