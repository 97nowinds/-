"""Generate local synthetic examples only. Never sends them to the receiver."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from member_integration import build_monitoring_snapshot_frame_event

tracks = [
    {'track_id': 'synthetic-global-1', 'global_id': 'synthetic-global-1',
     'local_track_id': 7, 'person_id': 'synthetic-person-1', 'display_name': 'synthetic-A',
     'bbox': [5, 4, 8, 20], 'bbox_format': 'xywh_pixels', 'confidence': 0.9,
     'observed_at': '2026-10-08T05:00:00.100+00:00',
     'location': {'x_m': 4.2, 'y_m': 1.8, 'quality': 'medium',
                  'trajectory_status': 'stationary', 'observed_at': '2026-10-08T05:00:00.000+00:00'}},
    {'track_id': 'synthetic-global-2', 'global_id': 'synthetic-global-2',
     'local_track_id': 8, 'person_id': None, 'display_name': None,
     'bbox': [25, 4, 8, 20], 'bbox_format': 'xywh_pixels', 'confidence': 0.85,
     'observed_at': '2026-10-08T05:00:00.100+00:00'}
]
arguments = dict(event_id='snapshot.frame:synthetic-alignment-001',
    snapshot_id='synthetic-alignment-001', camera_id='cam_2',
    captured_at='2026-10-08T05:00:00.100+00:00',
    image_url='http://192.168.31.127:5174/api/handoff/test.jpg',
    frame_width=64, frame_height=36, tracks=tracks)
output = root / 'contracts'
output.mkdir(exist_ok=True)
legacy = build_monitoring_snapshot_frame_event(**arguments,
    map_id='synthetic-map', map_revision='synthetic-revision')
(output / 'snapshot-frame.synthetic.legacy.json').write_text(
    json.dumps(legacy, ensure_ascii=False, indent=2), encoding='utf-8')
# Proposed semantics, not enabled in the legacy publisher or sent anywhere.
for track in tracks:
    track['track_id'] = str(track['local_track_id'])
proposed = build_monitoring_snapshot_frame_event(**arguments,
    map_id='synthetic-map', map_revision='synthetic-revision', include_map_reference=True)
(output / 'snapshot-frame.synthetic.proposed.json').write_text(
    json.dumps(proposed, ensure_ascii=False, indent=2), encoding='utf-8')
print('Two local synthetic files generated; no network requests made.')
