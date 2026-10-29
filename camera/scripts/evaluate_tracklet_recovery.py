"""Replay fixed recorded detections on original video crops; no identity ground truth."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--enabled', action='store_true')
    parser.add_argument('--limit-seconds', type=float, default=180)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import cv2
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    from floor_map import FloorMapProjector
    from mtmc_config import load_mtmc_config
    from reid_embedder import ReIDEmbedder
    from yolo_person_tracker import CrossCameraTrackCoordinator

    class Events:
        def __init__(self):
            self.counts = Counter()
        def record(self, event, **fields):
            self.counts[(event, fields.get('reason', ''))] += 1

    config = load_mtmc_config(args.source / 'config/mtmc.json')
    if args.enabled:
        config['reid']['tracklet_recovery_enabled'] = True
    else:
        config['reid']['tracklet_recovery_enabled'] = False
    events = Events()
    reid = ReIDEmbedder(args.source / 'models/osnet_x1_0_msmt17.pth', device='cuda:0')
    coordinator = CrossCameraTrackCoordinator(
        reid, FloorMapProjector.from_path(args.source / 'config/floor_map.json'),
        mtmc_config=config, event_store=events,
        handoff_policy_path=args.source / 'config/handoff_policy.json')
    info = json.loads((args.session / 'info.json').read_text(encoding='utf-8-sig'))
    captures = {cid: cv2.VideoCapture(str(args.session / s['video_file'])) for cid, s in info['streams'].items()}
    indices = {cid: -1 for cid in captures}
    frames, retained, last_seen, local_to_global = {}, defaultdict(set), {}, {}
    counts = Counter()
    changes, timing = [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    opener = gzip.open if args.detections.suffix == '.gz' else open
    with opener(args.detections, 'rt', encoding='utf-8-sig') as inputs, args.output.open('w', encoding='utf-8') as output:
        for line in inputs:
            row = json.loads(line)
            if row['seconds'] > args.limit_seconds:
                break
            if row.get('held'):
                continue  # Cached detector output supplies no new observations.
            cid, index = row['camera_id'], int(row['frame_index'])
            while indices[cid] < index:
                ok, frame = captures[cid].read()
                if not ok:
                    raise RuntimeError(f'{cid}: missing original frame {index}')
                frames[cid] = frame
                indices[cid] += 1
            frame = frames[cid]
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            now = float(row['seconds']) + 1
            tracks = [{k: v for k, v in t.items() if k not in ('global_id', 'identity')} for t in row['tracks']]
            current = {int(t['track_id']) for t in tracks}
            for local in current:
                last_seen[(cid, local)] = now
            for local in retained[cid] - current:
                if now - last_seen[(cid, local)] > 1.5:
                    coordinator.forget_local(cid, local)
            retained[cid] = current | {local for local in retained[cid]
                                        if now - last_seen[(cid, local)] <= 1.5}
            tick = time.monotonic()
            result = coordinator.update_batch(cid, tracks, frame, active_local_ids=current,
                                                monotonic_time=now, unix_time=now)
            timing.append((time.monotonic() - tick) * 1000)
            for local, global_id in result.items():
                key = (cid, local)
                previous = local_to_global.get(key)
                if previous and previous != global_id:
                    changes.append({'camera_id': cid, 'local_id': local, 'seconds': row['seconds'],
                                    'before': previous, 'after': global_id})
                local_to_global[key] = global_id
            counts[cid] += 1
            output.write(json.dumps({'camera_id': cid, 'frame_index': index, 'seconds': row['seconds'],
                                     'ids': result,
                                     'tracklets': {str(local): evidence.snapshot() for (camera, local), evidence
                                                   in getattr(coordinator, 'tracklet_evidence', {}).items() if camera == cid},
                                     'decisions': list(coordinator.recent_associations)[-len(tracks):] if tracks else []}) + '\n')
            if sum(counts.values()) % 500 == 0:
                output.flush()
                print('processed', sum(counts.values()), 'seconds', row['seconds'], flush=True)
    for capture in captures.values():
        capture.release()
    import numpy as np
    report = {'enabled': args.enabled, 'session': str(args.session), 'fresh_batches': dict(counts),
              'elapsed_seconds': round(time.monotonic() - started, 2), 'allocated_global_ids': coordinator.next_id - 1,
              'local_binding_changes': changes,
              'events': {f'{event}:{reason}': n for (event, reason), n in events.counts.items()},
              'association_ms_p50': float(np.median(timing)), 'association_ms_p95': float(np.percentile(timing, 95)),
              'identity_ground_truth': False, 'face_recognition_run': False,
              'limit': 'Fixed saved detections; original video crops; no claim of IDF1, real name correctness, or detector improvement.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('complete', str(args.output), flush=True)


if __name__ == '__main__':
    main()
