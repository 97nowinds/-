"""Explicit cam_2 live transport; does not enable automatic model analysis."""
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen, install_opener, build_opener, ProxyHandler

from frontend_server import _jpeg_dimensions
from handoff_sample import build_sample_event
from member_integration import MemberServiceClient, extract_receipt_id
from handoff_transport import load_demo_config


def prepare_live_event(frozen, jpeg):
    if frozen['camera_id'] != 'cam_2':
        raise ValueError('this stage only sends cam_2')
    if not frozen.get('source_frame_id'):
        raise ValueError('source frame identifier required')
    if (frozen['map_id'], frozen['map_revision']) != ('lab-floor', 'd9ea86e4bd80d086'):
        raise ValueError('map differs from agreed map')
    if (frozen['frame_width'], frozen['frame_height']) != (2560, 1440):
        raise ValueError('original 2560x1440 JPEG required')
    if len(jpeg) > 8 * 1024 * 1024:
        raise ValueError('JPEG exceeds 8 MiB')
    if hashlib.sha256(jpeg).hexdigest() != frozen['image_sha256']:
        raise ValueError('JPEG digest mismatch')
    if _jpeg_dimensions(jpeg) != (frozen['frame_width'], frozen['frame_height']):
        raise ValueError('JPEG dimensions mismatch')
    event = build_sample_event(frozen, require_person_location=False)
    for track in event['camera']['tracks']:
        x, y, width, height = track['bbox']
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x+width > 2560 or y+height > 1440:
            raise ValueError('person box outside original JPEG')
    return event


def main():
    # This service only contacts the explicitly configured LAN/Tailscale peers.
    # A desktop Internet proxy must not forward these private endpoints.
    install_opener(build_opener(ProxyHandler({})))
    config = load_demo_config()
    if not config.get('continuous_send_enabled', False):
        raise SystemExit('Continuous business sending awaits explicit joint confirmation; not started.')
    root = Path(__file__).resolve().parent
    credentials = json.loads((root/'runtime/handoff-credentials.json').read_text(encoding='utf-8-sig'))
    headers = {'Authorization': 'Bearer ' + credentials['image_read_token']}
    client = MemberServiceClient(config['sender_receiver_url'], os.environ.get('LAB_MEMBER_SERVICE_TOKEN', ''), direct=True)
    interval = 1 / config['sender_target_fps']
    status_path = root/'runtime/continuous-cam2-status.json'
    status = dict(camera_id='cam_2', target_fps=3, transport_enabled=True,
        automatic_analysis_enabled=False, time_domain='source', sync_status='unavailable',
        receiver_url=client.base_url, pid=os.getpid(), sent_frames=0, failed_frames=0,
        duplicate_skips=0, backlog=0, started_at=time.time(), frames=[])
    last_id = None
    recent_source_ids = set()
    source_id_order = []
    deadline = time.monotonic()
    while True:
        started = time.monotonic()
        try:
            with urlopen(Request('http://127.0.0.1:5175/api/handoff/frozen/cam_2', headers=headers), timeout=3) as response:
                frozen = json.load(response)
            public_base = config['sender_image_public_base'].rstrip('/')
            if public_base:
                frozen['image_url'] = public_base + '/api/handoff/snapshots/' + frozen['snapshot_id'] + '.jpg'
            if frozen['snapshot_id'] == last_id or frozen['source_frame_id'] in recent_source_ids:
                status['duplicate_skips'] += 1
                status['state'] = 'waiting_new_source_frame'
            else:
                last_id = frozen['snapshot_id']
                recent_source_ids.add(frozen['source_frame_id'])
                source_id_order.append(frozen['source_frame_id'])
                if len(source_id_order) > 300:
                    recent_source_ids.remove(source_id_order.pop(0))
                fetch_started = time.monotonic()
                with urlopen(Request(frozen['image_url'], headers=headers), timeout=3) as response:
                    if response.headers.get_content_type() != 'image/jpeg':
                        raise ValueError('image route did not return JPEG')
                    jpeg = response.read(8 * 1024 * 1024 + 1)
                fetch_seconds = time.monotonic()-fetch_started
                event = prepare_live_event(frozen, jpeg)
                if time.time() >= frozen['image_expires_at']:
                    raise ValueError('image expired before handoff')
                # No retry queue: move to the newest frame after errors. Every
                # event has immutable evidence; no archived sample is loaded.
                sent = time.monotonic()
                response = client.send_handoff(event)
                receipt = extract_receipt_id(response.body)
                if not response.ok or not receipt:
                    code = response.body.get('error', response.body.get('detail')) if isinstance(response.body, dict) else None
                    raise RuntimeError(f'receiver HTTP {response.status}; receipt={bool(receipt)}; detail={str(code)[:200]}')
                row = dict(snapshot_id=last_id, source_frame_id=frozen['source_frame_id'],
                    captured_at=event['captured_at'], sent_at=time.time(), jpeg_bytes=len(jpeg),
                    fetch_seconds=fetch_seconds, handoff_seconds=time.monotonic()-sent,
                    width=frozen['frame_width'], height=frozen['frame_height'],
                    track_ids=[track['track_id'] for track in event['camera']['tracks']],
                    tracks_with_location=sum('location' in track for track in event['camera']['tracks']),
                    receipt_id=receipt, http_status=response.status)
                status['sent_frames'] += 1
                status['frames'] = (status['frames']+[row])[-90:]
                status.update(state='sending', last_error=None, last_frame=row)
                (root/'runtime/continuous-cam2-last-event.json').write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as error:
            status['failed_frames'] += 1
            status.update(state='unavailable', last_error=f'{type(error).__name__}: {error}'[:400])
        status['updated_at'] = time.time()
        status['last_cycle_seconds'] = time.monotonic()-started
        rows = status['frames']
        status['recent_fps'] = (len(rows)-1)/(rows[-1]['sent_at']-rows[0]['sent_at']) if len(rows)>1 else 0
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(status_path)
        # Keep a fixed cadence; never catch up with a burst of old frames.
        deadline += interval
        if deadline < time.monotonic():
            deadline = time.monotonic()
        time.sleep(max(0, deadline-time.monotonic()))


if __name__ == '__main__':
    main()
