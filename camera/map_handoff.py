"""Read-only map and live position contract for the instrument teammate."""
import argparse
import hashlib
import json
import base64
import os
import re
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
from handoff_transport import ImmutableSnapshots, bearer_authorized, load_demo_config


def map_snapshot(state):
    floor = state.get('floor_map') or {}
    layout = {key: floor.get(key) for key in (
        'name', 'width_m', 'height_m', 'fixtures', 'zones', 'cameras',
        'calibrated', 'calibration_status', 'calibration_reason')}
    calibration = floor.get('calibration') or {}
    layout['calibration_status'] = floor.get('calibration_status') or calibration.get('status')
    layout['calibration_reason'] = floor.get('calibration_reason') or calibration.get('reason')
    layout['cameras'] = {
        key: {field: camera[field] for field in ('label', 'position', 'fov') if field in camera}
        for key, camera in (floor.get('cameras') or {}).items()
    }
    revision = hashlib.sha256(json.dumps(layout, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    people = []
    for person in floor.get('people') or []:
        item = dict(person)
        item['x_m'] = item.pop('x', None)
        item['y_m'] = item.pop('y', None)
        people.append(item)
    tracks = []
    for camera in state.get('cameras') or []:
        for track in camera.get('yolo_tracks') or []:
            tracks.append({key: track.get(key) for key in (
                'track_id', 'local_id', 'count_status', 'counted', 'count_duplicate_of',
                'localization_quality', 'localization_reason', 'detection_fresh')} | {'camera_id': camera['id']})
    return {
        'schema_version': 'lab-map-handoff-1.0',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        'map_id': 'lab-floor', 'map_revision': revision,
        'coordinate_system': {'unit': 'm', 'origin': 'top_left', 'x_positive': 'right', 'y_positive': 'down'},
        'layout': layout, 'people': people, 'camera_tracks': tracks,
        'recommended_poll_interval_ms': 1000,
        'display_interval_ms': 5000,
    }


def read_snapshot(api_base):
    try:
        with urlopen(api_base.rstrip('/') + '/api/dashboard', timeout=5) as response:
            result = map_snapshot(json.load(response))
        result['backend_status'] = 'online'
        return result
    except (OSError, ValueError):
        config = json.loads((Path(__file__).parent / 'config/floor_map.json').read_text(encoding='utf-8'))
        result = map_snapshot({'floor_map': config})
        result['backend_status'] = 'unavailable'
        result['error'] = 'camera backend unavailable; positions are not available'
        return result


class MapHandler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        received_at = time.time()
        if self.path != '/api/handoff/time-sync':
            self._json({'error': 'not found'}, 404)
            return
        if not bearer_authorized(self.headers, self.server.sync_token):
            self._json({'error': 'sync read permission required'}, 403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError()
            data = json.loads(self.rfile.read(length))
            if (set(data) != {'probe_id', 'client_sent_at'}
                    or not isinstance(data['probe_id'], str)
                    or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', data['probe_id'])
                    or not isinstance(data['client_sent_at'], str)):
                raise ValueError()
            stamp = datetime.fromisoformat(data['client_sent_at'].replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            self._json({'error': 'invalid sync request'}, 400)
            return
        self._json({**data, 'authority_clock': self.server.authority_clock,
            'authority_received_at': datetime.fromtimestamp(received_at, timezone.utc).isoformat(),
            'authority_sent_at': datetime.fromtimestamp(time.time(), timezone.utc).isoformat()})

    def do_GET(self):
        route = self.path.split('?')[0]
        image = re.fullmatch(r'/api/handoff/snapshots/([A-Za-z0-9_-]{1,128})\.jpg', route)
        frozen = re.fullmatch(r'/api/handoff/frozen/(cam_1|cam_2|cam_entrance)', route)
        if image or frozen:
            if not bearer_authorized(self.headers, self.server.image_token):
                self._json({'error': 'snapshot read permission required'}, 403)
                return
            if image:
                jpeg = self.server.snapshots.get(image.group(1))
                if jpeg is None:
                    self._json({'error': 'snapshot not found or expired'}, 404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(jpeg)
                return
            try:
                request = Request(self.server.api_base.rstrip('/') + route,
                    headers={'Authorization': 'Bearer ' + self.server.image_token})
                with urlopen(request, timeout=5) as response:
                    payload = json.load(response)
                jpeg = base64.b64decode(payload.pop('jpeg_base64'), validate=True)
                metadata = self.server.snapshots.put(payload['snapshot_id'], jpeg, payload)
                metadata['image_url'] = ('http://192.168.31.127:5175/api/handoff/snapshots/'
                                         + payload['snapshot_id'] + '.jpg')
                self._json(metadata)
            except (OSError, ValueError, KeyError, BufferError):
                self._json({'error': 'frozen detection snapshot unavailable'}, 503)
            return
        if self.path.split('?')[0] not in ('/api/handoff/map', '/api/handoff/positions', '/health'):
            self.send_error(404)
            return
        try:
            payload = read_snapshot(self.server.api_base)
            if self.path == '/health':
                payload = {'status': payload['backend_status'], 'service': 'lab-map-handoff',
                           'service_revision': 'joint-demo-minimal-r1'}
            elif self.path.split('?')[0] == '/api/handoff/positions':
                payload.pop('layout')
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(503 if payload.get('backend_status') == 'unavailable' and self.path != '/api/handoff/map' else 200)
        except Exception:
            body = json.dumps({'status': 'unavailable', 'error': 'camera backend unavailable', 'people': []}).encode()
            self.send_response(503)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', default='http://127.0.0.1:15001')
    parser.add_argument('--port', type=int, default=5175)
    parser.add_argument('--credentials-file', type=Path)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('0.0.0.0', args.port), MapHandler)
    server.api_base = args.api
    credentials = (json.loads(args.credentials_file.read_text(encoding='utf-8'))
                   if args.credentials_file else {})
    server.image_token = os.environ.get('LAB_HANDOFF_IMAGE_READ_TOKEN', '') or credentials.get('image_read_token', '')
    server.sync_token = os.environ.get('LAB_HANDOFF_SYNC_READ_TOKEN', '') or credentials.get('sync_read_token', '')
    server.authority_clock = 'main-' + uuid.uuid4().hex
    server.snapshots = ImmutableSnapshots(retention_seconds=load_demo_config()['jpeg_retention_seconds'])
    server.serve_forever()
