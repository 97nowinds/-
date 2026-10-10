"""Read-only transport smoke check. Never prints credential values."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:5175')
    parser.add_argument('--credentials-file', type=Path, required=True)
    args = parser.parse_args()
    credentials = json.loads(args.credentials_file.read_text(encoding='utf-8'))

    def request(path, token=None, body=None):
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        try:
            with urlopen(Request(args.base + path, headers=headers,
                    data=json.dumps(body).encode() if body else None), timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, None

    health_code, health = request('/health')
    assert health_code == 200 and health['service_revision'] == 'joint-demo-minimal-r1'
    probe = {'probe_id': 'transport-smoke-check',
             'client_sent_at': datetime.now(timezone.utc).isoformat()}
    sync_code, sync = request('/api/handoff/time-sync', credentials['sync_read_token'], probe)
    assert sync_code == 200 and sync['probe_id'] == probe['probe_id']
    denied_code, _ = request('/api/handoff/time-sync', credentials['image_read_token'], probe)
    assert denied_code == 403
    image_code, _ = request('/api/handoff/snapshots/transport-check-missing.jpg', credentials['image_read_token'])
    assert image_code == 404
    frozen_code, _ = request('/api/handoff/frozen/cam_2', credentials['image_read_token'])
    print(json.dumps({'service_revision': health['service_revision'], 'health': health_code,
        'time_sync': sync_code, 'wrong_permission': denied_code,
        'missing_image': image_code, 'frozen_evidence': frozen_code,
        'real_jpeg_verified': False}))


if __name__ == '__main__':
    main()
