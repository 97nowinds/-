import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from management_notifications import ManagementNotifications, EvidenceAdapter
from ppe_live_ingress import prepare_scene, ingest


class LivePPEIngressTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.temp = tempfile.TemporaryDirectory()
        config = json.loads((Path(__file__).parents[1]/'config/management.json').read_text(encoding='utf-8'))
        config['ppe_rules']['lab_coat']['enabled'] = True
        self.service = ManagementNotifications(Path(self.temp.name)/'test.db', config,
            wall=lambda: self.now, environ={}, sender=Mock())
        self.service.evidence = Mock()
        self.service.evidence.capture.return_value = 'a'*64
        self.runtime = SimpleNamespace(service=self.service, registry=Mock(),
            evidence=EvidenceAdapter(self.service), ppe_live_status={})

    def tearDown(self):
        self.service.db.close()
        self.temp.cleanup()

    def body(self):
        snapshot = 'snapshot_'+str(int(self.now))
        return {'frozen': {'camera_id': 'cam_1', 'snapshot_id': snapshot,
            'source_frame_id': snapshot, 'source_clock': 'gpu', 'timestamp_source': 'host_receive',
            'source_captured_at': self.now, 'source_occurred_at': self.now,
            'image_transform': 'identity', 'full_frame': True, 'frame_width': 640, 'frame_height': 480,
            'image_sha256': 'a'*64, 'camera': {'identities': [], 'yolo_tracks': [
                {'track_id': 't1', 'local_id': 1, 'box': [10, 20, 200, 400], 'detection_fresh': True}]}},
            'sync_mapping': {'source_clock': 'gpu', 'authority_clock': 'host', 'offset_ms': 0,
                'uncertainty_ms': 10, 'sync_valid_from': 90, 'sync_valid_until': 200},
            'tracks': [{'track_id': 't1', 'bbox': [10, 20, 210, 420],
                'mask': {'state': 'not_worn', 'confidence': .98},
                'lab_coat': 'not_worn', 'coat_confidence': .96}]}

    def test_both_ppe_events_after_distinct_frames_and_history_only_instruments(self):
        for self.now in (100., 103., 106.):
            result = ingest(self.runtime, self.body(), 'host')
        self.assertEqual(result['accepted'], 2)
        events = self.service.events({'role': 'admin'})
        self.assertEqual({e['event_type'] for e in events}, {'safety.no_mask', 'safety.no_lab_coat'})
        self.assertTrue(all(e['person_id']=='track:cam_1:t1' for e in events))
        self.assertEqual(self.service.db.execute('SELECT COUNT(*) FROM outbox').fetchone()[0], 2)
        self.service.sender.assert_not_called()

    def test_unknown_and_low_confidence_never_trigger(self):
        for self.now in (100., 103., 106., 109.):
            body = self.body()
            body['tracks'][0]['mask']['state'] = 'unknown'
            body['tracks'][0]['coat_confidence'] = None
            ingest(self.runtime, body, 'host')
        self.assertEqual(self.service.events({'role': 'admin'}), [])

    def test_duplicate_frame_does_not_count_twice(self):
        body = self.body()
        ingest(self.runtime, body, 'host')
        self.now += 6
        ingest(self.runtime, body, 'host')
        self.assertEqual(self.service.events({'role': 'admin'}), [])

    def test_wrong_crop_rejected_before_state_changes(self):
        body = self.body()
        body['tracks'][0]['bbox'][0] += 5
        with self.assertRaises(ValueError): ingest(self.runtime, body, 'host')
        self.runtime.registry.publish.assert_not_called()
        self.assertEqual(self.service.db.execute('SELECT COUNT(*) FROM ppe_states').fetchone()[0], 0)

    def test_clock_replay_and_stale_frames_rejected(self):
        body = self.body()
        for change in ({'timestamp_source': 'recording_replay'}, {'source_captured_at': 70}):
            with self.assertRaises(ValueError):
                prepare_scene({**body, 'frozen': {**body['frozen'], **change}}, 'host', self.now)
        with self.assertRaises(ValueError): prepare_scene(body, 'different-host', self.now)
        body['sync_mapping']['uncertainty_ms'] = 101
        with self.assertRaises(ValueError): prepare_scene(body, 'host', self.now)


if __name__ == '__main__': unittest.main()
