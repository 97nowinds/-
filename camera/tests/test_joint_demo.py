import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path
from io import BytesIO
from email.message import Message
from types import SimpleNamespace
from unittest.mock import Mock, patch
from copy import deepcopy

from handoff_transport import ImmutableSnapshots, estimate_sync, iso_time
from joint_demo import build_joint_event
from joint_results import JointResultProjection
from map_handoff import MapHandler
from frontend_server import FrontendHandler, LiveSnapshotPublisher, HANDOFF_TEST_JPEG
from member_integration import MemberServiceClient


class TransportTests(unittest.TestCase):
    def test_retention_retry_and_capacity(self):
        now = [100.0]
        store = ImmutableSnapshots(max_bytes=3, wall=lambda: now[0], monotonic=lambda: now[0])
        first = store.put('a', b'123', {'track': 'g1'})
        now[0] = 110
        self.assertEqual(first, store.put('a', b'123', {'track': 'g1'}))
        with self.assertRaises(ValueError):
            store.put('a', b'124', {'track': 'g1'})
        with self.assertRaises(BufferError):
            store.put('b', b'4', {})
        now[0] = 159.9
        self.assertEqual(store.get('a'), b'123')
        now[0] = 160
        self.assertIsNone(store.get('a'))
        with self.assertRaises(ValueError):
            store.put('a', b'123', {'track': 'g1'})

    def test_four_timestamp_offset_and_uncertainty(self):
        sample = estimate_sync(100, 102.1, 102.2, 100.3)
        self.assertAlmostEqual(sample['offset_ms'], 2000)
        self.assertAlmostEqual(sample['uncertainty_ms'], 100)
        with self.assertRaises(ValueError):
            estimate_sync(100, 102.2, 102.1, 100.3)

    def test_sync_and_image_auth_routes(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), MapHandler)
        server.sync_token, server.image_token = 'sync-only', 'image-only'
        server.authority_clock = 'test-main'
        server.snapshots = ImmutableSnapshots()
        server.snapshots.put('a', b'jpeg-test', {})
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            body = json.dumps({'probe_id': 'p1', 'client_sent_at': iso_time(100)}).encode()
            with self.assertRaises(HTTPError) as failure:
                urlopen(Request(base + '/api/handoff/time-sync', data=body,
                    headers={'Authorization': 'Bearer image-only'}))
            self.assertEqual(failure.exception.code, 403)
            with urlopen(Request(base + '/api/handoff/time-sync', data=body,
                    headers={'Authorization': 'Bearer sync-only'})) as response:
                result = json.load(response)
            self.assertEqual(result['probe_id'], 'p1')
            self.assertEqual(result['authority_clock'], 'test-main')
            self.assertLessEqual(result['authority_received_at'], result['authority_sent_at'])
            with urlopen(Request(base + '/api/handoff/snapshots/a.jpg',
                    headers={'Authorization': 'Bearer image-only'})) as response:
                self.assertEqual(response.read(), b'jpeg-test')
                self.assertEqual(response.headers.get_content_type(), 'image/jpeg')
            with self.assertRaises(HTTPError):
                urlopen(base + '/api/handoff/snapshots/a.jpg')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def fixtures():
    config = dict(image_base='http://192.168.31.127:5175', automatic_enabled=True, alignment_verified=True, zone_revision='z1',
        receiver_schema_file='schema.json', sync_refresh_seconds=1, sync_valid_seconds=10,
        sync_max_uncertainty_ms=50, sync_jump_limit_ms=20, result_publish_max_age_seconds=2,
        location_max_delta_seconds=1, inference_start_max_age_seconds=2,
        alignment={'cam_1': dict(alignment_revision='a1', transform='full_frame_resize',
            jpeg_width=1280, jpeg_height=720, source_dimensions=[2560, 1440], hd_width=2560, hd_height=1440)})
    frozen = dict(snapshot_id='s1', source_frame_id='f1', camera_id='cam_1',
        source_captured_at=100, source_occurred_at=100.2, timestamp_source='host_receive',
        frame_width=1280, frame_height=720, source_dimensions=[2560, 1440],
        map_id='m1', map_revision='mr1', image_url='http://192.168.31.127:5175/api/handoff/snapshots/s1.jpg',
        image_sha256='a' * 64, image_available_at=102.3, image_expires_at=162.3,
        sync_mapping=dict(offset_ms=2000, uncertainty_ms=10, source_clock='src',
            authority_clock='main', sync_id='sync1', source_reference_at=99,
            sync_valid_from=101, sync_valid_until=111))
    tracks = [dict(track_id='g1', global_id='g1', local_track_id=2, person_id=None,
        display_name=None, observed_at=iso_time(100), bbox=[10, 20, 30, 40],
        bbox_format='xywh_pixels', confidence=0.9,
        location=dict(x_m=1, y_m=2, observed_at=iso_time(99.9)))]
    return config, frozen, tracks


class EventTests(unittest.TestCase):
    def test_convert_once_and_preserve_originals(self):
        config, frozen, tracks = fixtures()
        event = build_joint_event(frozen, tracks, config)
        self.assertEqual(event['captured_at'], iso_time(102))
        location = event['camera']['tracks'][0]['location']
        self.assertEqual(location['observed_at'], iso_time(101.9))
        self.assertEqual(location['source_observed_at'], iso_time(99.9))
        self.assertIsNone(event['camera']['tracks'][0]['display_name'])
        self.assertEqual(tracks[0]['observed_at'], iso_time(100))

    def test_missing_sync_or_alignment_never_sends(self):
        config, frozen, tracks = fixtures()
        frozen['sync_mapping'] = None
        with self.assertRaises(ValueError):
            build_joint_event(frozen, tracks, config)
        config, frozen, tracks = fixtures()
        frozen['source_dimensions'] = [1920, 1080]
        with self.assertRaises(ValueError):
            build_joint_event(frozen, tracks, config)

    def test_duplicate_ids_rejected(self):
        config, frozen, tracks = fixtures()
        with self.assertRaises(ValueError):
            build_joint_event(frozen, tracks * 2, config)


class ResultTests(unittest.TestCase):
    def row(self, snapshot='s1', captured=99):
        return dict(analysis_id=snapshot, camera_id='cam_1', snapshot_id=snapshot,
            track_id='g1', instrument_id='i1', captured_at=iso_time(captured),
            published_at=iso_time(100), expires_at=iso_time(105),
            status='valid', is_interacting=True, handoff_match_status='matched')

    def test_queries_cannot_extend_ttl_and_history_survives(self):
        now = [101.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.jsonl'
            project = JointResultProjection(path, wall=lambda: now[0], monotonic=lambda: now[0])
            row = self.row()
            payload = dict(time_domain='authority', results=[row], analyses=[row], errors=[])
            self.assertEqual(len(project.project(payload)['results']), 1)
            now[0] = 105
            result = project.project(payload)
            self.assertEqual(result['results'], [])
            self.assertEqual(result['analyses'][0]['status'], 'expired')
            self.assertIsNone(result['analyses'][0]['is_interacting'])
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertTrue(json.loads(path.read_text())['is_interacting'])

    def test_late_results_never_override_newer_result(self):
        project = JointResultProjection(wall=lambda: 101, monotonic=lambda: 101)
        newer, older = self.row('s2', 100), self.row('s1', 99)
        project.project(dict(time_domain='authority', results=[newer], analyses=[], errors=[]))
        result = project.project(dict(time_domain='authority', results=[older], analyses=[older], errors=[]))
        self.assertEqual(result['results'][0]['snapshot_id'], 's2')
        self.assertEqual(len(result['analyses']), 1)


class PublisherTests(unittest.TestCase):
    def test_client_explicit_contract_is_forwarded(self):
        client = MemberServiceClient('http://127.0.0.1:8765', 'test-token')
        client._request = Mock()
        client.latest_interactions('cam_1', contract='lab-interaction-results-1.2')
        path = client._request.call_args.args[1]
        self.assertIn('camera_id=cam_1', path)
        self.assertIn('contract=lab-interaction-results-1.2', path)

    def test_legacy_response_cannot_masquerade_as_new_contract(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), FrontendHandler)
        server.member_client = Mock()
        server.member_client.latest_interactions.return_value = SimpleNamespace(
            status=200, body={'results': [], 'errors': []})
        server.live_publisher = None
        server.joint_result_projection = JointResultProjection()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}/api/interaction-results/latest'
        try:
            with urlopen(base) as response:
                self.assertEqual(response.status, 200)
            with self.assertRaises(HTTPError) as failure:
                urlopen(base + '?contract=lab-interaction-results-1.2')
            self.assertEqual(failure.exception.code, 502)
            server.member_client.latest_interactions.return_value.body = {
                'result_contract': 'lab-interaction-results-1.2', 'time_domain': 'authority',
                'results': [], 'errors': [],
                'analyses': [{'status': 'unavailable', 'is_interacting': False}]}
            with self.assertRaises(HTTPError) as failure:
                urlopen(base + '?contract=lab-interaction-results-1.2')
            self.assertEqual(failure.exception.code, 502)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_ack_loss_retries_same_event_and_indexes_before_ack(self):
        config, frozen, tracks = fixtures()
        frozen.update(frame_width=64, frame_height=36,
            map_observations=[dict(track_id='g1', position={'x': 1, 'y': 2}, observed_at=100)],
            camera={'yolo_tracks': [dict(track_id='g1', local_id=2, box=[1, 2, 3, 4])], 'identities': []})
        config['alignment']['cam_1'].update(jpeg_width=64, jpeg_height=36)
        import hashlib
        frozen['image_sha256'] = hashlib.sha256(HANDOFF_TEST_JPEG).hexdigest()
        replies = []
        server = SimpleNamespace(member_client=Mock())
        with tempfile.TemporaryDirectory() as directory:
            schema = Path(directory) / 'test-schema.json'
            # This is only a transport test; the real adapter uses the peer's Schema.
            schema.write_text(json.dumps({'type': 'object'}), encoding='utf-8')
            config['receiver_schema_file'] = str(schema)
            with patch('frontend_server.load_demo_config', return_value=config):
                publisher = LiveSnapshotPublisher(server, camera_id='cam_1')

            def send(event):
                self.assertEqual(publisher.result_index.resolve(dict(camera_id='cam_1',
                    snapshot_id='s1', track_id='g1'))['handoff_match_status'], 'matched')
                replies.append(deepcopy(event))
                return SimpleNamespace(status=500 if len(replies) == 1 else 200,
                    body={} if len(replies) == 1 else {'receipt_id': 'receipt1'})

            server.member_client.send_handoff.side_effect = send
            metadata_response = BytesIO(json.dumps(frozen).encode())
            jpeg_response = BytesIO(HANDOFF_TEST_JPEG)
            jpeg_response.headers = Message()
            jpeg_response.headers['Content-Type'] = 'image/jpeg'
            with patch('frontend_server.load_demo_config', return_value=config), \
                 patch.dict('os.environ', {'LAB_HANDOFF_IMAGE_READ_TOKEN': 'test-read'}), \
                 patch('frontend_server.urlopen', side_effect=[metadata_response, jpeg_response]) as fetch, \
                 patch('frontend_server.time.time', return_value=102.4), \
                 patch('frontend_server.time.monotonic', return_value=10):
                with self.assertRaises(RuntimeError):
                    publisher._publish_once()
                publisher._publish_once()
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(replies[0], replies[1])
                self.assertIsNone(publisher._pending_event)


if __name__ == '__main__':
    unittest.main()
