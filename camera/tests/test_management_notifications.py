import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from management_notifications import ManagementNotifications, EvidenceAdapter, feishu_payload, iso
from handoff_result_index import SnapshotTrackIndex
from hazard_feed import forward_visual_prewarnings


class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = 2000000000.
        self.sent = []
        self.config = {'lab_id': 'lab', 'principals': [
            {'id': 'admin', 'role': 'admin', 'token_env': 'ADMIN'}],
            'routes': [{'id': 'managers', 'labs': ['lab'], 'instruments': ['i1'],
                        'event_types': ['person.entered', 'instrument.started', 'safety.no_mask'],
                        'webhook_env': 'HOOK', 'secret_env': 'SECRET'}]}
        self.env = {'ADMIN': 'admin-secret', 'MANAGER': 'manager-secret', 'YXQ': 'personal-secret',
                    'HOOK': 'https://open.feishu.cn/open-apis/bot/v2/hook/synthetic', 'SECRET': 'synthetic'}
        self.path = Path(self.temp.name) / 'events.db'
        self.service = self.make_service()
        self.adapter = EvidenceAdapter(self.service)
        self.index = SnapshotTrackIndex()
        self.index.remember({'snapshot_id': 's1', 'camera': {'camera_id': 'cam_2', 'tracks': [
            {'track_id': 't1', 'global_id': 'g1', 'person_id': 'registered1', 'display_name': 'yxq', 'observed_at': iso(self.now-1)},
            {'track_id': 't2', 'global_id': 'g2', 'person_id': 'registered2', 'display_name': 'other', 'observed_at': iso(self.now-1)},
            {'track_id': 'unknown', 'person_id': None}]}})

    def make_service(self):
        return ManagementNotifications(self.path, self.config, self.env, wall=lambda: self.now,
                                       sender=lambda url, payload: self.sent.append((url, payload)))

    def tearDown(self):
        self.service.db.close()
        self.temp.cleanup()

    def row(self, **updates):
        return dict({'camera_id': 'cam_2', 'snapshot_id': 's1', 'track_id': 't1', 'analysis_id': 'a1',
                     'person_id': 'forged', 'display_name': 'forged', 'instrument_id': 'i1',
                     'instrument_name': '超声波净化器', 'status': 'valid', 'is_interacting': True,
                     'time_domain': 'authority', 'captured_at': iso(self.now - 1),
                     'published_at': iso(self.now), 'expires_at': iso(self.now + 5)}, **updates)

    def test_exact_identity_and_only_true_interaction(self):
        self.assertTrue(self.adapter.analysis(self.row(), self.index))
        item = self.service.events({'role': 'admin'})[0]
        self.assertEqual((item['person_id'], item['display_name']), ('registered1', 'yxq'))
        self.assertFalse(self.adapter.analysis(self.row(is_interacting=False), self.index))
        self.assertFalse(self.adapter.analysis(self.row(status='pending'), self.index))
        self.assertFalse(self.adapter.analysis(self.row(snapshot_id='wrong'), self.index))
        self.assertFalse(self.adapter.analysis(self.row(track_id='unknown'), self.index))

    def test_crossed_people_are_not_matched_by_order(self):
        self.adapter.analysis(self.row(track_id='t2'), self.index)
        self.assertEqual(self.service.events({'role': 'admin'})[0]['display_name'], 'other')

    def test_stale_and_wrong_clock_rejected(self):
        self.assertFalse(self.adapter.analysis(self.row(expires_at=iso(self.now)), self.index))
        self.assertFalse(self.adapter.analysis(self.row(time_domain='gpu'), self.index))
        self.assertFalse(self.adapter.analysis(self.row(captured_at=iso(self.now-60)), self.index))
        self.assertFalse(self.adapter.analysis(self.row(captured_at=iso(self.now)), self.index))

    def test_mask_requires_positive_evidence(self):
        self.assertFalse(self.adapter.analysis(self.row(mask_state='unknown', confidence=.99), self.index, mask=True))
        self.assertFalse(self.adapter.analysis(self.row(mask_state='not_worn', confidence=.4), self.index, mask=True))
        self.assertFalse(self.adapter.analysis(self.row(mask_state='not_worn', confidence=float('nan')), self.index, mask=True))
        self.assertTrue(self.adapter.analysis(self.row(mask_state='not_worn', confidence=.95), self.index, mask=True))

    def test_persistence_cooldown_and_feishu_message(self):
        self.adapter.analysis(self.row(), self.index)
        self.assertFalse(self.adapter.analysis(self.row(analysis_id='another'), self.index))
        self.assertTrue(self.service.deliver_once())
        self.assertIn('yxq 正在操作超声波净化器', self.sent[0][1]['content']['text'])
        self.assertIn('sign', self.sent[0][1])
        self.service.db.close()
        self.service = self.make_service()
        self.assertEqual(self.service.events({'role': 'admin'})[0]['deliveries'][0]['state'], 'sent')
        self.assertFalse(EvidenceAdapter(self.service).analysis(self.row(), self.index))

    def test_live_visual_prewarnings_use_fire_routes_without_triggering_hardware_alarm(self):
        self.service.config['routes'][0]['event_types'].append('environment.flame')
        feed = self.path.with_suffix('.json')
        checked_at = self.now
        events = [
            {'kind': 'smoke', 'at': checked_at - 1, 'source_at': 1700000000, 'score': .72},
            {'kind': 'flame', 'at': checked_at - 1, 'source_at': 1700000001, 'score': .85},
            {'kind': 'spark', 'at': checked_at - 1, 'source_at': 1700000002},
            {'kind': 'smoke', 'at': checked_at - 10, 'source_at': 1699999990, 'score': .95},
        ]
        payload = {'observed_at_unix': checked_at, 'cameras': {
            'cam_entrance': {'online': True, 'frames': 4, 'recent': events},
            'cam_1': {'online': True, 'frames': 4, 'recent': [events[0]]},
        }}
        feed.write_text(json.dumps(payload))
        forward_visual_prewarnings(self.service, feed, checked_at - 5)
        for event in events[:2]:
            event['at'] += .1
        feed.write_text(json.dumps(payload))
        forward_visual_prewarnings(self.service, feed, checked_at - 5)
        self.assertEqual(self.service.db.execute('SELECT count(*) FROM events').fetchone()[0], 3)
        self.assertEqual(self.service.db.execute('SELECT count(*) FROM fire_snapshot_jobs').fetchone()[0], 0)
        self.now += 3
        for _ in range(3):
            self.assertTrue(self.service.deliver_once())
            self.now += 1
        texts = [message['content']['text'] for _, message in self.sent]
        self.assertEqual(sum('烟雾视觉预警 · 待复核' in text for text in texts), 2)
        self.assertEqual(sum('火焰视觉预警 · 待复核' in text for text in texts), 1)

    def test_failure_retry_does_not_leak_webhook(self):
        self.adapter.analysis(self.row(), self.index)
        def fail(url, payload):
            raise OSError('secret ' + url)
        self.service.sender = fail
        self.service.deliver_once()
        delivery = self.service.events({'role': 'admin'})[0]['deliveries'][0]
        self.assertEqual(delivery['error'], 'OSError')
        self.assertEqual(delivery['state'], 'pending')
        self.now += 3
        self.service.sender = lambda url, payload: self.sent.append(payload)
        self.service.deliver_once()
        self.assertEqual(len(self.sent), 1)

    def test_missing_credentials_preserves_pending_and_expiry(self):
        del self.env['HOOK']
        self.adapter.analysis(self.row(), self.index)
        self.assertFalse(self.service.deliver_once())
        self.now += 301
        self.assertTrue(self.service.deliver_once())
        self.assertEqual(self.service.events({'role': 'admin'})[0]['deliveries'][0]['state'], 'expired')


    def test_out_of_scope_instrument_not_routed(self):
        self.adapter.analysis(self.row(instrument_id='i2'), self.index)
        self.assertEqual(self.service.events({'role': 'admin'})[0]['deliveries'], [])
        self.assertEqual(self.service.events({'role': 'manager', 'labs': ['lab'], 'instruments': ['i1']}), [])

    def camera(self, entrance=False, **updates):
        return dict({'id': 'entrance' if entrance else 'cam_2', 'role': 'entrance_identity' if entrance else 'tracking',
            'status': 'running', 'replay': {'enabled': False}, 'detection_age_seconds': .1, 'capture_age_seconds': .1,
            'identities': [{'local_id': 1, 'global_track_id': 'g1', 'person_id': 'registered1', 'name': 'yxq', 'identity_lock_status': 'locked'}],
            'yolo_tracks': [{'local_id': 1, 'track_id': 'g1'}]}, **updates)

    def test_entrance_to_indoor_once_not_appearance_alone(self):
        self.adapter.cameras({'cameras': [self.camera(entrance=True)]})
        self.assertEqual(self.service.events({'role': 'admin'}), [])
        self.now += 1
        self.adapter.cameras({'cameras': [self.camera()]})
        self.assertEqual(len(self.service.events({'role': 'admin'})), 1)
        self.now += 125
        self.adapter.cameras({'cameras': [self.camera()]})
        self.assertEqual(len(self.service.events({'role': 'admin'})), 1)

    def test_replay_stale_and_ambiguous_entrance_not_accepted(self):
        for camera in [self.camera(True, replay={'enabled': True}),
                       self.camera(True, detection_age_seconds=5), self.camera(True, identities=[])]:
            self.adapter.cameras({'cameras': [camera]})
        self.now += 1
        self.adapter.cameras({'cameras': [self.camera()]})
        self.assertEqual(self.service.events({'role': 'admin'}), [])

    def test_signature_known_formula(self):
        import base64, hashlib, hmac
        payload = feishu_payload('test', 'secret', now=123)
        expected = base64.b64encode(hmac.new(b'123\nsecret', b'', hashlib.sha256).digest()).decode()
        self.assertEqual(payload['sign'], expected)

    def request(self, path, token='', body=None, method='GET'):
        from io import BytesIO
        from types import SimpleNamespace
        from management_api import handle_management
        encoded = json.dumps(body).encode() if body is not None else b''
        handler = SimpleNamespace(path=path, headers={'Authorization': 'Bearer '+token,
                    'Content-Length': str(len(encoded))}, rfile=BytesIO(encoded),
                    server=SimpleNamespace(management=SimpleNamespace(service=self.service,
                        evidence=self.adapter, index=lambda: self.index, last_poll_status='ok')))
        output = []
        handler._send_json = lambda payload, status=200: output.append((status, payload))
        self.assertTrue(handle_management(handler, method))
        return output[0]

    def test_http_permissions_and_secret_redaction(self):
        self.assertEqual(self.request('/api/management/events')[0], 401)
        self.assertEqual(self.request('/api/management/routes', 'manager-secret')[0], 401)
        self.assertEqual(self.request('/api/management/audit', 'personal-secret')[0], 401)
        result = self.request('/api/management/routes', 'admin-secret')
        self.assertEqual(result[0], 200)
        self.assertNotIn(self.env['HOOK'], json.dumps(result[1]))
        self.assertEqual(self.request('/api/management/ack', 'personal-secret', {'event_id':'anything'}, 'POST')[0], 401)

    def test_removed_account_endpoints_cannot_create_or_restore_users(self):
        for method in ['GET','POST']:
            status,_=self.request('/api/management/accounts','admin-secret',
                {'action':'create','account':{'id':'unused','role':'admin'}} if method=='POST' else None,method)
            self.assertEqual(status,404)
        self.assertFalse(hasattr(self.service,'accounts'))
        tables={row[0] for row in self.service.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn('accounts',tables)

    def test_maintenance_auth_and_group_routes_remain_persistent(self):
        self.adapter.analysis(self.row(),self.index)
        admin=self.service.authenticate('Bearer admin-secret')
        self.assertEqual(admin['role'],'admin')
        self.assertIsNone(self.service.authenticate('Bearer manager-secret'))
        event_id=self.service.events(admin)[0]['event_id']
        self.service.acknowledge(admin,event_id)
        self.service.set_routes(admin,[])
        self.service.db.close();self.service=self.make_service()
        self.assertEqual(self.service.config['routes'],[])
        self.assertEqual(self.service.events(admin)[0]['acknowledged_by'],'admin')
        self.service.deliver_once();self.assertEqual(self.sent,[])



    def test_policy_authorization_validation_and_persistence(self):
        policy={'ppe_rules':{'mask':{'enabled':True,'minimum_confidence':.95}},'ppe_confirmation':{
            'minimum_frames':3,'minimum_seconds':5,'maximum_gap_seconds':15,'escalation_seconds':60}}
        path='/api/management/policy'
        self.assertEqual(self.request(path,'manager-secret',policy,'POST')[0],401)
        self.assertEqual(self.request(path,'admin-secret',policy,'POST')[0],200)
        self.assertEqual(self.request(path,'admin-secret')[1],policy)
        bad={**policy,'ppe_confirmation':{**policy['ppe_confirmation'],'minimum_frames':1}}
        self.assertEqual(self.request(path,'admin-secret',bad,'POST')[0],400)
        restored=self.make_service()
        try:self.assertEqual(restored.config['ppe_confirmation'],policy['ppe_confirmation'])
        finally:restored.db.close()

    def test_unconfirmed_snapshot_clock_cannot_be_overridden_by_receiver(self):
        self.index.remember({'snapshot_id':'source-snapshot','camera':{'camera_id':'cam_2',
            'timing':{'event_time_domain':'source'},'tracks':[{'track_id':'t1','person_id':'registered1',
                'observed_at':iso(self.now-1)}]}})
        self.assertFalse(self.adapter.analysis(self.row(snapshot_id='source-snapshot'),self.index))
        self.service.config['require_authority_snapshot']=True
        self.assertFalse(self.adapter.analysis(self.row(),self.index))

    def test_evidence_disk_failure_does_not_consume_ppe_episode(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        self.service.config.update(evidence_required_events=['safety.no_mask'],ppe_confirmation={
            'minimum_frames':2,'minimum_seconds':1,'maximum_gap_seconds':5,'escalation_seconds':6})
        capture=Mock(side_effect=OSError('synthetic storage failure'))
        self.service.evidence=SimpleNamespace(capture=capture)
        for number in range(3):
            self.now+=1;snapshot='fresh'+str(number)
            self.index.remember({'snapshot_id':snapshot,'time_domain':'authority','camera':{'camera_id':'cam_2',
                'source_frame_id':snapshot,'tracks':[{'track_id':'t1','person_id':'registered1','observed_at':iso(self.now-1)}]}})
            if number==2:capture.side_effect=None;capture.return_value='a'*64
            accepted=self.adapter.analysis(self.row(snapshot_id=snapshot,mask_state='not_worn',confidence=.99),self.index,mask=True)
            self.assertEqual(accepted,number==2)
        self.assertEqual(capture.call_count,2)

    def test_producer_cannot_read_and_only_bound_evidence_accepted(self):
        self.service.config['principals'].append({'id': 'producer', 'role':'producer', 'labs':['lab'],
            'cameras':['cam_2'], 'token_env':'PRODUCER'})
        self.env['PRODUCER'] = 'producer-secret'
        self.assertEqual(self.request('/api/management/events', 'producer-secret')[0], 403)
        row = self.row(evidence_type='mask', mask_state='not_worn', confidence=.95)
        response = self.request('/api/management/evidence', 'producer-secret', row, 'POST')
        self.assertEqual(response[0], 200)
        self.assertTrue(response[1]['accepted'])
        row['camera_id'] = 'cam_1'
        self.assertEqual(self.request('/api/management/evidence', 'producer-secret', row, 'POST')[0], 403)
        self.assertEqual(self.request('/api/management/evidence', 'admin-secret', row, 'POST')[0], 403)

    def test_real_http_management_page_and_bearer_auth(self):
        from types import SimpleNamespace
        from http.server import ThreadingHTTPServer
        import threading
        from urllib.request import urlopen, Request
        from urllib.error import HTTPError
        from frontend_server import FrontendHandler
        server = ThreadingHTTPServer(('127.0.0.1', 0), FrontendHandler)
        server.management = SimpleNamespace(service=self.service, evidence=self.adapter,
                                            index=lambda: self.index, last_poll_status='ok')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:' + str(server.server_port)
        try:
            with urlopen(base+'/management', timeout=3) as response:
                self.assertIn('实验室通知与告警管理', response.read().decode())
            with self.assertRaises(HTTPError) as error:
                urlopen(base+'/api/management/events', timeout=3)
            self.assertEqual(error.exception.code, 401)
            with urlopen(Request(base+'/api/management/events',
                                 headers={'Authorization':'Bearer admin-secret'}), timeout=3) as response:
                self.assertEqual(json.load(response)['role'], 'admin')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
