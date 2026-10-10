"""Replay saved detections against real frames; synthetic identity seeding is opt-in."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor_map import FloorMapProjector
from mtmc_events import AssociationEventStore
from reid_embedder import ReIDEmbedder
from yolo_person_tracker import CrossCameraTrackCoordinator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reid-model', type=Path, default=Path('models/osnet_x1_0_msmt17.pth'))
    parser.add_argument('--floor-map', type=Path, default=Path('config/floor_map.json'))
    parser.add_argument('--synthetic-seed-camera', help='Testing only: seed a synthetic identity on the first single-person frame')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.session / 'info.json').read_text(encoding='utf-8-sig'))
    capture_start = min(datetime.fromisoformat(stream['first_frame_at']).timestamp() for stream in info['streams'].values())
    configurations = {item['id']: item for item in json.loads(Path('config/cameras.json').read_text(encoding='utf-8-sig'))}
    coordinator = CrossCameraTrackCoordinator(
        ReIDEmbedder(args.reid_model), FloorMapProjector.from_path(args.floor_map),
        event_store=AssociationEventStore(args.output.with_suffix('.events.jsonl')),
    )
    captures = {cid: cv2.VideoCapture(str(args.session / stream['video_file'])) for cid, stream in info['streams'].items()}
    indices = {cid: -1 for cid in captures}
    last_ids, retained, last_seen = {}, {}, {}
    seeded = False
    with args.output.open('w', encoding='utf-8') as output:
        for number, line in enumerate(args.detections.read_text(encoding='utf-8-sig').splitlines(), 1):
            row = json.loads(line)
            cid, index = row['camera_id'], row['frame_index']
            while indices[cid] < index:
                ok, frame = captures[cid].read()
                if not ok:
                    raise RuntimeError(f'{cid}: missing frame {index}')
                indices[cid] += 1
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            now = row['seconds'] + 1.0
            tracks = row['tracks']
            current = {int(item['track_id']) for item in tracks}
            for local in current:
                last_seen[(cid, local)] = now
            still_retained = set()
            for local in retained.get(cid, set()) - current:
                if now - last_seen[(cid, local)] <= configurations[cid].get('track_hold_seconds', 1.5):
                    still_retained.add(local)
                else:
                    coordinator.forget_local(cid, local)
                    last_seen.pop((cid, local), None)
            retained[cid] = current | still_retained
            if row.get('held'):
                ids = last_ids[cid]
            else:
                ids = coordinator.update_batch(cid, tracks, frame, active_local_ids=current, monotonic_time=now, unix_time=capture_start + row['seconds'])
            last_ids[cid] = ids
            if args.synthetic_seed_camera == cid and not seeded and len(tracks) == 1:
                seeded = True
                coordinator.set_identity(ids[tracks[0]['track_id']], {
                    'known': True, 'person_id': 'synthetic_identity', 'name': 'SYNTHETIC_IDENTITY',
                }, cid, now=now)
            resolved = []
            for track in tracks:
                global_id = ids[track['track_id']]
                identity = coordinator.identity_for(global_id, cid, track.get('position'), now=now)
                if identity:
                    identity = coordinator.set_identity(global_id, identity, cid, track.get('position'), now=now)
                resolved.append({**track, 'global_id': global_id, 'identity': identity})
            output.write(json.dumps({**row, 'tracks': resolved, 'identity_seed_is_synthetic': bool(args.synthetic_seed_camera)}, ensure_ascii=False) + '\n')
            if number % 500 == 0:
                output.flush()
                print(f'processed={number} seconds={row["seconds"]:.1f}', flush=True)
    for capture in captures.values():
        capture.release()
    print(f'complete output={args.output}', flush=True)


if __name__ == '__main__':
    main()
