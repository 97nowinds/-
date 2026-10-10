"""Sender adapter for the agreed minimal demo; no receiver/model logic."""
from copy import deepcopy
from handoff_transport import iso_time


def build_joint_event(frozen, tracks, config):
    if not config.get('automatic_enabled') or not config.get('alignment_verified'):
        raise ValueError('automatic demo awaits alignment acceptance')
    if not config.get('zone_revision') or not config.get('receiver_schema_file'):
        raise ValueError('confirmed zone revision and receiver Schema required')
    for key in ('sync_refresh_seconds', 'sync_valid_seconds', 'sync_max_uncertainty_ms',
                'sync_jump_limit_ms', 'result_publish_max_age_seconds'):
        if config.get(key) is None:
            raise ValueError('shared timing parameters not confirmed')
    sync = frozen.get('sync_mapping')
    if not sync or frozen.get('timestamp_source') != 'host_receive':
        raise ValueError('valid live source sync mapping required')
    if sync['uncertainty_ms'] > config['sync_max_uncertainty_ms']:
        raise ValueError('source sync uncertainty exceeds limit')
    offset = sync['offset_ms'] / 1000
    captured = frozen['source_captured_at'] + offset
    occurred = frozen['source_occurred_at'] + offset
    if not sync['sync_valid_from'] <= captured <= occurred <= sync['sync_valid_until']:
        raise ValueError('source timestamps outside sync validity')
    alignment = config['alignment'].get(frozen['camera_id'])
    if (not alignment or not alignment.get('alignment_revision')
            or alignment.get('transform') != 'full_frame_resize'):
        raise ValueError('camera alignment unavailable')
    width, height = frozen['frame_width'], frozen['frame_height']
    if (alignment.get('jpeg_width'), alignment.get('jpeg_height')) != (width, height):
        raise ValueError('JPEG dimensions differ from accepted alignment')
    if alignment.get('source_dimensions') != frozen['source_dimensions']:
        raise ValueError('source dimensions differ from accepted alignment')
    tracks = deepcopy(tracks)
    for track in tracks:
        track['observed_at'] = iso_time(captured)
        location = track.get('location')
        if location:
            from datetime import datetime
            raw = datetime.fromisoformat(location['observed_at']).timestamp()
            translated = raw + offset
            if (abs(translated - captured) > config['location_max_delta_seconds']
                    or not sync['sync_valid_from'] <= translated <= sync['sync_valid_until']):
                del track['location']
            else:
                location.update(source_observed_at=iso_time(raw),
                                observed_at=iso_time(translated), valid=True)
    identifiers = [track['track_id'] for track in tracks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('ambiguous duplicate track identity')
    event = {
        'schema_version': 'lab-handoff-1.2', 'event_type': 'snapshot.frame',
        'event_id': 'snapshot.frame:' + frozen['snapshot_id'],
        'snapshot_id': frozen['snapshot_id'], 'captured_at': iso_time(captured),
        'occurred_at': iso_time(occurred), 'producer': 'lab-camera-host',
        'priority': 'normal', 'status': 'ready', 'data_classification': 'anonymized_monitoring',
        'map_id': frozen['map_id'], 'map_revision': frozen['map_revision'],
        'zone_revision': config['zone_revision'],
        'camera': {
            'camera_id': frozen['camera_id'], 'source_frame_id': frozen['source_frame_id'],
            'frame_width': width, 'frame_height': height,
            'image_url': frozen['image_url'], 'image_sha256': frozen['image_sha256'],
            'image_available_at': iso_time(frozen['image_available_at']),
            'image_expires_at': iso_time(frozen['image_expires_at']),
            'geometry': {'binding': 'detection_frame', 'transform': 'full_frame_resize',
                'detection_width': width, 'detection_height': height,
                'hd_width': alignment['hd_width'], 'hd_height': alignment['hd_height'],
                'alignment_revision': alignment['alignment_revision'], 'hd_alignment_status': 'verified'},
            'timing': {
                'source_clock': sync['source_clock'], 'authority_clock': sync['authority_clock'],
                'event_time_domain': 'authority', 'conversion_applied': True,
                'capture_time_kind': 'decoder_received_at',
                'source_captured_at': iso_time(frozen['source_captured_at']),
                'source_occurred_at': iso_time(frozen['source_occurred_at']),
                'sync_id': sync['sync_id'], 'sync_status': 'valid',
                'sync_valid_from': iso_time(sync['sync_valid_from']),
                'sync_valid_until': iso_time(sync['sync_valid_until']),
                'source_reference_at': iso_time(sync['source_reference_at']),
                'offset_at_reference_ms': sync['offset_ms'], 'drift_ppm': 0,
                'uncertainty_ms': sync['uncertainty_ms']},
            'tracks': tracks}}
    return event
