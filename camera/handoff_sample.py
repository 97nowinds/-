"""Manual source-clock sample export; never enables or sends model tasks."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from handoff_transport import iso_time, load_demo_config
from frontend_server import LiveSnapshotPublisher, _jpeg_dimensions


def build_sample_event(frozen, require_person_location=True):
    width, height = frozen['frame_width'], frozen['frame_height']
    if (frozen.get('image_transform') != 'identity' or not frozen.get('full_frame')
            or frozen['source_dimensions'] != [width, height]
            or frozen.get('timestamp_source') != 'host_receive'):
        raise ValueError('live original full-frame evidence required')
    captured = iso_time(frozen['source_captured_at'])
    people = [{**obs, 'x': obs['position']['x'], 'y': obs['position']['y'],
               'position_observed_at': obs['observed_at']}
              for obs in frozen['map_observations']
              if not obs.get('position_estimated') and obs.get('position')]
    tracks = LiveSnapshotPublisher._tracks(frozen['camera'], captured, people,
        max_location_delta_seconds=load_demo_config()['location_max_delta_seconds'])
    if require_person_location and (not tracks or not any(track.get('location') for track in tracks)):
        raise ValueError('sample requires a real person with a fresh map observation')
    if len({track['track_id'] for track in tracks}) != len(tracks):
        raise ValueError('ambiguous track identity')
    for track in tracks:
        # Names are personal data. IDs and real observation times are retained.
        track['display_name'] = None
        if track.get('location'):
            track['location']['valid'] = True
    return dict(schema_version='lab-handoff-1.2', event_type='snapshot.frame',
        event_id='snapshot.frame:' + frozen['snapshot_id'], snapshot_id=frozen['snapshot_id'],
        captured_at=captured, occurred_at=iso_time(frozen['source_occurred_at']),
        producer='lab-camera-host', priority='normal', status='ready',
        data_classification='anonymized_monitoring',
        map_id=frozen['map_id'], map_revision=frozen['map_revision'],
        zone_revision=load_demo_config().get('zone_revision'),
        camera=dict(camera_id=frozen['camera_id'], source_frame_id=frozen['source_frame_id'],
            frame_width=width, frame_height=height, image_url=frozen['image_url'],
            image_sha256=frozen['image_sha256'],
            image_available_at=iso_time(frozen['image_available_at']),
            image_expires_at=iso_time(frozen['image_expires_at']),
            geometry=dict(binding='detection_frame', transform='identity',
                detection_width=width, detection_height=height, hd_width=width, hd_height=height,
                alignment_revision=None, hd_alignment_status='pending'),
            timing=dict(source_clock=frozen['source_clock'], event_time_domain='source',
                conversion_applied=False, capture_time_kind='decoder_received_at',
                source_captured_at=captured,
                source_occurred_at=iso_time(frozen['source_occurred_at']),
                sync_status='unavailable'), tracks=tracks))


def export_sample(base, camera_id, token, output, archive=None, public_base=None,
                  require_person_location=True):
    headers = {'Authorization': 'Bearer ' + token}
    with urlopen(Request(base.rstrip('/') + '/api/handoff/frozen/' + camera_id,
                         headers=headers), timeout=5) as response:
        frozen = json.load(response)
    if public_base:
        frozen['image_url'] = public_base.rstrip('/') + '/api/handoff/snapshots/' + frozen['snapshot_id'] + '.jpg'
    event = build_sample_event(frozen, require_person_location=require_person_location)
    started = time.perf_counter()
    with urlopen(Request(frozen['image_url'], headers=headers), timeout=3) as response:
        if response.headers.get_content_type() != 'image/jpeg':
            raise ValueError('image route must return JPEG')
        jpeg = response.read(8 * 1024 * 1024 + 1)
    elapsed = time.perf_counter() - started
    if len(jpeg) > 8 * 1024 * 1024:
        raise ValueError('JPEG exceeds receiver 8 MiB limit')
    if hashlib.sha256(jpeg).hexdigest() != frozen['image_sha256']:
        raise ValueError('JPEG digest mismatch')
    if _jpeg_dimensions(jpeg) != (frozen['frame_width'], frozen['frame_height']):
        raise ValueError('JPEG dimensions mismatch')
    if time.time() >= frozen['image_expires_at']:
        raise ValueError('sample image expired')
    if archive is not None:
        directory = Path(archive) / frozen['snapshot_id']
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['image_sha256'] != frozen['image_sha256']:
                raise ValueError('archived sample content cannot change')
        else:
            manifest = dict(image_sha256=frozen['image_sha256'], expires_at=time.time() + 86400)
            with (directory / 'snapshot.jpg').open('xb') as stream:
                stream.write(jpeg)
            with manifest_path.open('x', encoding='utf-8') as stream:
                json.dump(manifest, stream)
        event['camera']['image_expires_at'] = iso_time(manifest['expires_at'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'snapshot.frame.json').write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'snapshot.jpg').write_bytes(jpeg)
    report = dict(snapshot_id=frozen['snapshot_id'], image_url=frozen['image_url'],
        jpeg_bytes=len(jpeg), fetch_seconds=elapsed, frame_width=frozen['frame_width'],
        frame_height=frozen['frame_height'], service_revision=frozen['service_revision'],
        expires_at=event['camera']['image_expires_at'], automatic_enabled=False)
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=load_demo_config()['image_base'])
    parser.add_argument('--camera', default='cam_2', choices=['cam_1', 'cam_2'])
    parser.add_argument('--credentials-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    credentials = json.loads(args.credentials_file.read_text(encoding='utf-8-sig'))
    print(json.dumps(export_sample(args.base, args.camera, credentials['image_read_token'], args.output, args.archive), indent=2))
