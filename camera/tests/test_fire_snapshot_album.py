import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from management_notifications import ManagementNotifications, iso
from fire_snapshot_album import FireSnapshotAlbums, CAMERAS, scene_time
from hazard_live_worker import FlameAlertWindow, face_overlap
from hazard_feed import forward_visual_prewarnings
from qwen_flame import QwenFlameWindow


class FireAlbumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = 2000000000.
        config = {'lab_id': 'synthetic-lab', 'routes': [
            {'id': name, 'labs': ['synthetic-lab'], 'event_types': ['environment.flame', 'environment.flame_recovered'],
             'webhook_env': name} for name in ['experimenters', 'managers', 'administrators']]}
        self.sent = []
        self.service = ManagementNotifications(Path(self.tmp.name)/'test.db', config,
            environ={k:k for k in ['experimenters','managers','administrators']}, wall=lambda:self.now,
            sender=lambda url, body:self.sent.append((url,body)))
        from PIL import Image
        from io import BytesIO
        stream=BytesIO(); Image.new('RGB',(2560,1440),'white').save(stream,format='JPEG')
        jpeg=stream.getvalue()
        self.client = type('Client', (), {'credentials': {'app_id': 'synthetic'},
            'upload_image': lambda _, data, **kwargs:'synthetic-image'})()
        self.service.evidence = type('Store', (), {'read': lambda _, eid:jpeg})()
        self.service.image_client = self.client
        self.calls = []
        def capture(event, camera):
            self.calls.append(camera)
            return {'status':'available', 'camera_id':camera, 'evidence_id':camera,
                'snapshot_id':'synthetic-'+camera, 'source_frame_id':'frame-'+camera,
                'captured_at':iso(self.now), 'alarm_delta_seconds':0}
        self.worker = FireSnapshotAlbums(self.service, capture)

    def tearDown(self):
        self.service.db.close(); self.tmp.cleanup()

    def event(self, source='local_serial', event_id='synthetic-fire', kind='environment.flame'):
        event = {'event_id':event_id, 'event_type':kind, 'lab_id':'synthetic-lab', 'device_id':'synthetic-device',
            'incident_id':event_id, 'severity':'L3', 'observed_at':iso(self.now), 'evidence_source':source}
        self.service.record(event)
        return event

    def deliver_texts(self):
        for _ in range(3):
            self.now += 1.1; self.assertTrue(self.service.deliver_once())

    def deliver_albums(self):
        from feishu_app import EVIDENCE_CREDENTIALS
        # Use only mtime; never read the machine's real protected credentials.
        with patch('fire_snapshot_album.FrozenSceneCapture'), patch.object(type(EVIDENCE_CREDENTIALS), 'stat') as st:
            st.return_value.st_mtime_ns=123
            self.service.image_credentials_revision=123
            for _ in range(3): self.assertTrue(self.worker.deliver_once())

    def test_hardware_three_images_to_three_groups_after_text(self):
        event=self.event(); self.worker.capture_once()
        self.assertCountEqual(self.calls, CAMERAS)
        self.assertFalse(self.worker.deliver_once())  # Must wait for emergency text.
        self.deliver_texts(); self.deliver_albums()
        self.assertEqual(len(self.sent),6)
        self.assertTrue(all(p['msg_type']=='text' for _,p in self.sent[:3]))
        for _,p in self.sent[3:]:
            rows=p['content']['post']['zh_cn']['content']
            self.assertEqual(sum(x['tag']=='img' for row in rows for x in row),3)
        self.assertFalse(self.worker.deliver_once())
        self.assertFalse(self.service.record(event))
        self.assertEqual(self.service.db.execute('SELECT count(*) FROM fire_snapshot_jobs').fetchone()[0],1)

    def test_visual_alarm_uses_same_three_camera_path(self):
        self.event('fire_signal_fusion'); self.worker.capture_once()
        self.assertCountEqual(self.calls,CAMERAS)

    def test_visual_prewarning_sends_only_the_triggering_camera_frame(self):
        event = {'event_id': 'visual-fire', 'event_type': 'environment.visual_flame',
                 'lab_id': 'synthetic-lab', 'device_id': 'cam_2', 'camera_id': 'cam_2',
                 'observed_at': iso(self.now), 'evidence_source': 'cluster_visual',
                 'visual_confidence': '76.6%', 'evidence_id': 'trigger-frame'}
        self.assertTrue(self.service.record(event))
        self.assertFalse(self.worker.capture_once())
        self.assertFalse(self.worker.deliver_once())
        self.deliver_texts(); self.deliver_albums()
        self.assertEqual(len(self.sent), 6)
        for _, payload in self.sent[3:]:
            rows = payload['content']['post']['zh_cn']['content']
            self.assertEqual(sum(tag['tag'] == 'img' for row in rows for tag in row), 1)
            self.assertIn('触发视觉预警的模型输入画面', rows[0][0]['text'])

    def test_smoke_evidence_and_text_are_sent_together_once(self):
        event = {'event_id': 'visual-smoke', 'event_type': 'environment.visual_smoke',
                 'lab_id': 'synthetic-lab', 'device_id': 'cam_entrance', 'camera_id': 'cam_entrance',
                 'observed_at': iso(self.now), 'evidence_source': 'cluster_visual',
                 'visual_confidence': '62.7%'}
        self.assertTrue(self.service.record(event))
        self.assertTrue(self.service.deliver_once())
        self.assertEqual(self.sent, [])
        self.assertFalse(self.service.record({**event, 'evidence_id': 'smoke-frame'}))
        self.now += 3
        with patch('notification_delivery.evidence_image_key', return_value='smoke-image'):
            self.deliver_texts()
        self.assertFalse(self.worker.capture_once())
        self.assertFalse(self.worker.deliver_once())
        self.assertEqual(len(self.sent), 3)

    def test_qwen_fire_feed_delivers_one_message_with_evidence(self):
        self.service.config['routes'] = [self.service.config['routes'][0]]
        path = Path(self.tmp.name) / 'hazard.json'
        payload = {'observed_at_unix': self.now, 'cameras': {'cam_2': {
            'online': True, 'frames': 10, 'recent': [{
                'kind': 'flame', 'stage': 'visual_initial', 'model': 'qwen',
                'source_at': self.now, 'at': self.now, 'score': None,
                'evidence_id': 'a' * 64}]}}}
        path.write_text(json.dumps(payload))
        forward_visual_prewarnings(self.service, path, self.now - 1)
        forward_visual_prewarnings(self.service, path, self.now - 1)
        with patch('notification_delivery.evidence_image_key', return_value='qwen-image'):
            self.assertTrue(self.service.deliver_once())
            self.assertFalse(self.service.deliver_once())
        self.assertEqual(len(self.sent), 1)
        _, message = self.sent[0]
        post = message['content']['post']['zh_cn']
        self.assertIn('Qwen', post['title'])
        self.assertEqual(sum(x['tag'] == 'img' for row in post['content'] for x in row), 1)
        self.assertEqual(self.service.db.execute('SELECT count(*) FROM fire_snapshot_jobs').fetchone()[0], 0)

    def test_qwen_flame_requires_repeat_at_the_same_location(self):
        window = QwenFlameWindow()
        tray, door = [5, 5, 30, 30], [200, 100, 30, 30]
        self.assertIsNone(window.observe([tray], 0.0))
        self.assertIsNone(window.observe([], 0.5))
        self.assertIsNone(window.observe([door], 1.0))
        self.assertIsNone(window.observe([tray], 4.0))
        self.assertEqual(window.observe([tray], 5.0), ('visual_initial', tray))
        for t in range(6, 15):
            self.assertIsNone(window.observe([tray], t))
        self.assertEqual(window.observe([tray], 15), ('visual_ongoing', tray))
        for _, payload in self.sent:
            post = payload['content']['post']['zh_cn']
            self.assertIn('烟雾视觉预警', post['title'])
            self.assertEqual(sum(tag['tag'] == 'img' for row in post['content'] for tag in row), 1)

    def test_visual_flame_repeats_while_present_and_restarts_after_clear(self):
        window = FlameAlertWindow()
        results = [window.observe(True, i / 4) for i in range(45)]
        self.assertEqual([result for result in results if result], ['visual_initial', 'visual_ongoing'])
        self.assertTrue(all(window.observe(False, 12 + i / 4) is None for i in range(5)))
        results = [window.observe(True, 14 + i / 4) for i in range(3)]
        self.assertEqual(results, [None, None, 'visual_initial'])
        for index, stage in enumerate(('visual_initial', 'visual_ongoing', 'visual_initial')):
            self.now += 10
            ident = f'flame-{index}'
            self.assertTrue(self.service.record({'event_id': ident, 'event_type': 'environment.visual_flame',
                'lab_id': 'synthetic-lab', 'camera_id': 'cam_2', 'device_id': 'cam_2', 'observed_at': iso(self.now),
                'visual_confidence': '76.6%', 'evidence_source': 'cluster_visual', 'fire_stage': stage}))
        self.assertEqual(self.service.db.execute('SELECT count(*) FROM outbox').fetchone()[0], 9)

    def test_visual_flame_expires_stale_samples_and_ignores_one_flash(self):
        window = FlameAlertWindow()
        self.assertIsNone(window.observe(True, 0.0))
        self.assertIsNone(window.observe(True, 0.25))
        self.assertIsNone(window.observe(True, 10.0))
        self.assertIsNone(window.observe(False, 10.25))
        self.assertIsNone(window.observe(True, 12.0))
        self.assertIsNone(window.observe(True, 12.25))
        self.assertEqual(window.observe(True, 12.50), 'visual_initial')

    def test_flame_face_filter_requires_high_mutual_overlap(self):
        face = (1320, 899, 141, 177)
        self.assertTrue(face_overlap((1301, 862, 156, 204), face))
        self.assertFalse(face_overlap((1420, 900, 32, 54), face))
        self.assertFalse(face_overlap((1470, 869, 28, 44), face))

    def test_one_offline_camera_retries_then_sends_two_images_with_missing_label(self):
        capture=self.worker.capture
        self.worker.capture=lambda e,c: (_ for _ in ()).throw(OSError()) if c=='cam_1' else capture(e,c)
        self.event(); self.worker.capture_once()
        self.assertEqual(self.service.db.execute('SELECT state FROM fire_snapshot_jobs').fetchone()[0],'pending')
        self.deliver_texts(); self.now+=21; self.worker.capture_once(); self.deliver_albums()
        p=self.sent[-1][1]; rows=p['content']['post']['zh_cn']['content']
        self.assertEqual(sum(x['tag']=='img' for r in rows for x in r),2)
        self.assertIn('未取得有效快照',json.dumps(p,ensure_ascii=False))

    def test_expired_job_after_restart_never_captures_new_scene(self):
        self.event(); self.now+=21
        self.worker.capture_once(); self.assertEqual(self.calls,[])
        self.assertEqual(self.service.db.execute('SELECT state FROM fire_snapshot_jobs').fetchone()[0],'captured')

    def test_recovery_does_not_generate_new_alarm_album(self):
        self.event(kind='environment.flame_recovered')
        self.assertFalse(self.worker.capture_once())

    def test_upload_failure_does_not_block_emergency_text_and_remains_retryable(self):
        self.event(); self.worker.capture_once(); self.deliver_texts()
        from feishu_app import EVIDENCE_CREDENTIALS
        with patch.object(type(EVIDENCE_CREDENTIALS),'stat',side_effect=OSError()):
            self.assertTrue(self.worker.deliver_once())
        self.assertEqual(len(self.sent),3)
        r=self.service.db.execute("SELECT state,error FROM fire_snapshot_deliveries WHERE attempts=1").fetchone()
        self.assertEqual(tuple(r),('pending','OSError'))

    def test_clock_mapping_and_actual_capture_time_required(self):
        frozen={'source_clock':'gpu','timestamp_source':'host_receive', 'source_captured_at':self.now-10,
            'source_occurred_at':self.now-9.5, 'sync_mapping': {'source_clock':'gpu','authority_clock':'host',
                'offset_ms':10000, 'uncertainty_ms':10, 'sync_valid_from':self.now-20,'sync_valid_until':self.now+20}}
        self.assertEqual(scene_time(frozen,'host',self.now,self.now),self.now)
        with self.assertRaises(ValueError):scene_time(frozen,'old-host',self.now,self.now)
        frozen['timestamp_source']='recording_replay'
        with self.assertRaises(ValueError):scene_time(frozen,'host',self.now,self.now)
        frozen['timestamp_source']='host_receive'; frozen['source_captured_at']-=10
        with self.assertRaises(ValueError):scene_time(frozen,'host',self.now,self.now)


if __name__=='__main__': unittest.main()
