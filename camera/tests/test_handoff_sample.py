import unittest
import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from copy import deepcopy
from handoff_transport import ImmutableSnapshots
from map_handoff import MapHandler
from handoff_sample import build_sample_event
from joint_demo import build_joint_event
from test_joint_demo import fixtures


class SampleTests(unittest.TestCase):
    def sample(self):
        _, frozen, _ = fixtures()
        frozen.update(frame_width=2560, frame_height=1440, source_dimensions=[2560, 1440],
            source_clock='capture-real', full_frame=True, image_transform='identity',
            camera=dict(yolo_tracks=[dict(global_id='g1', track_id='g1', local_id=2,
                confidence=.9, box=[100, 200, 300, 400])], identities=[]),
            map_observations=[dict(track_id='g1', position=dict(x=1, y=2),
                observed_at=99.9, quality='approximate')])
        return frozen

    def test_preserve_ids_and_real_source_times_without_claiming_sync(self):
        frozen = self.sample()
        before = deepcopy(frozen)
        event = build_sample_event(frozen)
        self.assertEqual(event['schema_version'], 'lab-handoff-1.2')
        self.assertEqual(event['camera']['tracks'][0]['track_id'], 'g1')
        self.assertEqual(event['camera']['tracks'][0]['bbox'], [100, 200, 300, 400])
        self.assertEqual(event['camera']['tracks'][0]['location']['observed_at'], '1970-01-01T00:01:39.900+00:00')
        self.assertEqual(event['camera']['timing']['event_time_domain'], 'source')
        self.assertFalse(event['camera']['timing']['conversion_applied'])
        self.assertEqual(frozen, before)

    def test_no_person_no_location_or_nonoriginal_image_is_rejected(self):
        for key, value in [('map_observations', []), ('full_frame', False), ('timestamp_source', 'recording_replay')]:
            frozen = self.sample()
            frozen[key] = value
            with self.assertRaises(ValueError):
                build_sample_event(frozen)
        frozen = self.sample()
        frozen['camera']['yolo_tracks'] = []
        with self.assertRaises(ValueError):
            build_sample_event(frozen)

    def test_automatic_original_path_keeps_acceptance_gates(self):
        config, frozen, tracks = fixtures()
        frozen.update(frame_width=2560, frame_height=1440, source_dimensions=[2560, 1440],
                      image_transform='identity', full_frame=True)
        config['alignment']['cam_1'].update(transform='identity', jpeg_width=2560, jpeg_height=1440)
        self.assertEqual(build_joint_event(frozen, tracks, config)['camera']['geometry']['transform'], 'identity')
        frozen['full_frame'] = False
        with self.assertRaises(ValueError):
            build_joint_event(frozen, tracks, config)

    def test_selected_archive_keeps_auth_and_expiry_after_live_cache_expires(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'selected-1'
            directory.mkdir()
            jpeg = b'selected-original-jpeg'
            (directory/'snapshot.jpg').write_bytes(jpeg)
            manifest = dict(image_sha256=hashlib.sha256(jpeg).hexdigest(), expires_at=time.time()+60)
            (directory/'manifest.json').write_text(json.dumps(manifest))
            server = ThreadingHTTPServer(('127.0.0.1', 0), MapHandler)
            server.image_token = 'image-only'
            server.sample_archive = Path(temporary)
            server.snapshots = ImmutableSnapshots()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}/api/handoff/snapshots/selected-1.jpg'
            try:
                with self.assertRaises(HTTPError) as error:
                    urlopen(url)
                self.assertEqual(error.exception.code, 403)
                with urlopen(Request(url, headers={'Authorization':'Bearer image-only'})) as response:
                    self.assertEqual(response.read(), jpeg)
                manifest['expires_at'] = time.time()-1
                (directory/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url, headers={'Authorization':'Bearer image-only'}))
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
