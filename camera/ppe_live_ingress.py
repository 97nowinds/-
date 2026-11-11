"""Bind authenticated GPU PPE results to the exact frozen frame and host clock."""
import math
import re

from handoff_result_index import SnapshotTrackIndex
from management_notifications import iso


def prepare_scene(body, authority_clock, now):
    frozen = body['frozen']
    sync = body['sync_mapping']
    camera = frozen['camera_id']
    snapshot = frozen['snapshot_id']
    if (camera not in {'cam_1', 'cam_2', 'cam_entrance'}
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', snapshot)
            or frozen.get('timestamp_source') != 'host_receive'
            or frozen.get('image_transform') != 'identity' or not frozen.get('full_frame')
            or sync.get('source_clock') != frozen.get('source_clock')
            or sync.get('authority_clock') != authority_clock):
        raise ValueError('PPE frame/clock binding invalid')
    for key in ('offset_ms', 'uncertainty_ms', 'sync_valid_from', 'sync_valid_until'):
        if type(sync.get(key)) not in (int, float) or not math.isfinite(sync[key]):
            raise ValueError('PPE clock sample invalid')
    captured = frozen['source_captured_at'] + sync['offset_ms'] / 1000
    occurred = frozen['source_occurred_at'] + sync['offset_ms'] / 1000
    if (not 0 <= sync['uncertainty_ms'] <= 100
            or not sync['sync_valid_from'] <= captured <= occurred <= sync['sync_valid_until']
            or now >= sync['sync_valid_until'] or occurred > now + 2
            or not -2 <= now - captured <= 15):
        raise ValueError('PPE frame stale or clock proof expired')
    width, height = frozen['frame_width'], frozen['frame_height']
    tracks = []
    for raw in frozen['camera'].get('yolo_tracks', []):
        tid = raw.get('global_id') or raw.get('track_id')
        box = raw.get('box')
        if not tid or raw.get('detection_fresh') is False:
            continue
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
            raise ValueError('PPE box invalid')
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > width+1 or y+h > height+1:
            raise ValueError('PPE box outside original frame')
        identities = [i for i in frozen['camera'].get('identities', [])
                      if i.get('local_id') == raw.get('local_id')
                      and i.get('global_track_id') == tid
                      and i.get('identity_lock_status') == 'locked' and i.get('person_id')]
        identity = identities[0] if len(identities) == 1 else {}
        tracks.append({'track_id': tid, 'global_id': tid,
                       'person_id': identity.get('person_id') or 'track:' + camera + ':' + tid,
                       'display_name': identity.get('name') or '未识别人员（' + tid + '）',
                       'observed_at': iso(captured), 'bbox': box, 'bbox_format': 'xywh_pixels'})
    if len({t['track_id'] for t in tracks}) != len(tracks):
        raise ValueError('PPE track ambiguous')
    return {'snapshot_id': 'ppe:' + snapshot, 'captured_at': iso(captured),
            'occurred_at': iso(occurred), 'time_domain': 'authority', 'camera': {
                'camera_id': camera, 'source_frame_id': frozen['source_frame_id'],
                'frame_width': width, 'frame_height': height,
                'image_sha256': frozen['image_sha256'],
                'image_url': 'http://127.0.0.1:5177/api/handoff/snapshots/' + snapshot + '.jpg',
                'tracks': tracks}}


def ingest(runtime, body, authority_clock):
    service = runtime.service
    if not service.config.get('ppe_live', {}).get('enabled'):
        raise ValueError('live PPE disabled')
    scene = prepare_scene(body, authority_clock, service.wall())
    tracks = {t['track_id']: t for t in scene['camera']['tracks']}
    rows = body['tracks']
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError('PPE result count invalid')
    # Validate all bindings before persisting/advancing any confirmation state.
    seen = set()
    for row in rows:
        tid = row['track_id']
        if tid in seen or tid not in tracks:
            raise ValueError('PPE result track not in original frame')
        seen.add(tid)
        x, y, w, h = tracks[tid]['bbox']
        if row.get('bbox') != [x, y, x+w, y+h]:
            raise ValueError('PPE result crop differs from original track')
    runtime.registry.publish(scene)
    index = SnapshotTrackIndex()
    index.remember(scene)
    accepted = 0
    for row in rows:
        track = tracks[row['track_id']]
        if (scene['camera']['camera_id'] in service.config.get('ppe_preparation', {}).get('camera_ids', [])
                and not track['person_id'].startswith('track:')):
            from management_notifications import stamp
            service.preparation.see(track['person_id'], stamp(scene['captured_at']), entrance_face=True)
        predictions = {'mask': row.get('mask', {}), 'lab_coat': {
            'state': row.get('lab_coat'), 'confidence': row.get('coat_confidence')}}
        for item, prediction in predictions.items():
            result = {'evidence_type': 'ppe', 'ppe_type': item, 'status': 'valid',
                      'analysis_id': scene['snapshot_id'] + ':' + row['track_id'] + ':' + item,
                      'camera_id': scene['camera']['camera_id'], 'snapshot_id': scene['snapshot_id'],
                      'track_id': row['track_id'], 'time_domain': 'authority',
                      'captured_at': scene['captured_at'], 'published_at': iso(service.wall()),
                      'expires_at': iso(service.wall()+5), 'ppe_state': prediction.get('state'),
                      'confidence': prediction.get('confidence'),
                      'visibility': 'valid' if prediction.get('state') in {'worn', 'not_worn'} else 'unknown'}
            accepted += bool(runtime.evidence.analysis(result, index, ppe_type=item))
    runtime.ppe_live_status = {'state': 'receiving', 'last_received_at': iso(service.wall()),
                               'camera_id': scene['camera']['camera_id'],
                               'snapshot_id': scene['snapshot_id'], 'tracks': len(rows),
                               'last_new_events': accepted,
                               'received_frames': runtime.ppe_live_status.get('received_frames', 0)+1}
    return {'accepted': accepted, 'processed_tracks': len(rows), 'mode': 'submit'}
