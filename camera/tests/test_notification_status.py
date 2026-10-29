import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from management_notifications import ManagementNotifications, iso
from notification_status import notification_status
from snapshot_registry import SnapshotRegistry


class NotificationStatusTests(unittest.TestCase):
    def test_deployed_config_notifies_instrument_events_and_keeps_person_entries_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            config=json.loads((Path(__file__).resolve().parents[1]/'config/management.json').read_text(encoding='utf-8'))
            now=2000000000.0
            def forbidden_sender(*_args):
                raise AssertionError('This policy test must never send a real message')
            service=ManagementNotifications(Path(directory)/'events.db', config, environ={},
                wall=lambda:now, sender=forbidden_sender)
            base={'lab_id':config['lab_id'],'person_id':'synthetic-person',
                  'display_name':'合成测试人员','observed_at':iso(now)}
            try:
                for kind in ['person.entered','instrument.started','instrument.ended']:
                    event_id='synthetic-'+kind
                    service.record({**base,'event_id':event_id,'event_type':kind})
                    routes={r[0] for r in service.db.execute('SELECT route_id FROM outbox WHERE event_id=?', (event_id,))}
                    self.assertEqual(routes, set() if kind=='person.entered' else {'experimenters','managers'})
                for severity,audiences,expected in [
                    ('L1','experimenters',{'experimenters'}),
                    ('L2','experimenters,managers',{'experimenters','managers'})]:
                    event_id='synthetic-ppe-'+severity
                    service.record({**base,'event_id':event_id,'event_type':'safety.no_mask',
                        'severity':severity,'audiences':audiences,'incident_id':'synthetic-incident',
                        'evidence_id':'synthetic-evidence'})
                    routes={r[0] for r in service.db.execute('SELECT route_id FROM outbox WHERE event_id=?',(event_id,))}
                    self.assertEqual(routes,expected)
                service.record({'lab_id':config['lab_id'],'event_id':'synthetic-fire',
                    'event_type':'environment.flame','severity':'L3','device_id':'synthetic-sensor',
                    'observed_at':iso(now)})
                self.assertEqual({r[0] for r in service.db.execute(
                    "SELECT route_id FROM outbox WHERE event_id='synthetic-fire'")},
                    {'experimenters','managers','administrators'})
            finally:
                service.db.close()

    def test_reports_missing_inputs_without_exposing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'lab_id': 'lab', 'principals': [
                {'id': 'detector', 'role': 'producer', 'token_env': 'PRODUCER'}],
                'silent_events': ['person.entered', 'instrument.started'],
                'routes': [{'id': 'experimenters', 'audience': 'experimenters',
                    'labs': ['lab'], 'event_types': ['safety.no_mask'],
                    'webhook_env': 'HOOK', 'secret_env': 'SIGN'}]}
            service = ManagementNotifications(root/'events.db', config,
                environ={'HOOK': 'private-webhook-value', 'SIGN': 'private-signature-value'})
            service.evidence = object()
            runtime = SimpleNamespace(service=service,
                registry=SnapshotRegistry(root/'snapshots.db'),
                server=SimpleNamespace(member_client=None, live_publisher=None),
                last_poll_status='state_ok')
            try:
                with patch('notification_status.EVIDENCE_CREDENTIALS', root/'not-configured'):
                    result = notification_status(runtime)
                self.assertIn('producer_token_missing', result['readiness_blockers'])
                self.assertIn('no_fresh_snapshot_index', result['readiness_blockers'])
                self.assertTrue(result['groups'][0]['webhook_configured'])
                encoded = json.dumps(result)
                self.assertNotIn('private-webhook-value', encoded)
                self.assertNotIn('private-signature-value', encoded)
                self.assertEqual(result['silent_events'], config['silent_events'])
                self.assertFalse(result['member_polling_configured'])
            finally:
                service.db.close()


if __name__ == '__main__':
    unittest.main()
