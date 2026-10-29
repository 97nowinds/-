import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from management_notifications import ManagementNotifications, iso
from fire_snapshot_album import FireSnapshotAlbums, CAMERAS, scene_time


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
