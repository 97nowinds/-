"""Replay detector/ReID association in capture timestamp order without live writes."""
import argparse
import csv
import json
import inspect
import hashlib
import time
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor_map import FloorMapProjector
from reid_embedder import ReIDEmbedder
from yolo_person_tracker import CrossCameraTrackCoordinator, YoloPersonTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--floor-map', type=Path, default=Path('config/floor_map.json'))
    parser.add_argument('--max-seconds', type=float, default=100000)
    parser.add_argument('--reid-model', type=Path, default=Path('models/osnet_x1_0_msmt17.pth'))
    parser.add_argument('--audit-relink', action='store_true', help='Read-only candidate evidence, including rejected candidates')
    parser.add_argument('--detector', type=Path, default=Path('models/yolov8n.pt'))
    parser.add_argument('--guarded-lost-low', action='store_true', help='Evaluate guarded low-score Lost recovery on Cam2 only')
    args = parser.parse_args()
    info = json.loads((args.session / 'info.json').read_text(encoding='utf-8'))
    configs = {c['id']: c for c in json.loads(Path('config/cameras.json').read_text(encoding='utf-8'))}
    floor = FloorMapProjector.from_path(args.floor_map)
    embedder = ReIDEmbedder(args.reid_model)
    coordinator = CrossCameraTrackCoordinator(embedder, floor_map=floor)
    relink_audit = []
    if args.audit_relink:
        original_relink = coordinator._same_camera_relink_assignments
        def audited_relink(camera_id, prepared, active_local_ids, now, unix_time):
            relink_audit.clear()
            for item in prepared:
                key = (camera_id, item['local_id'])
                if key in coordinator.local_bindings and key not in coordinator.relink_evidence:
                    continue
                candidates = []
                for gid, global_target in coordinator.global_tracks.items():
                    target = coordinator._camera_observation(global_target, camera_id)
                    if target is None or target.get('box') is None:
                        continue
                    age = now-target['last_seen']
                    if age > 12 or age < 0:
                        continue
                    box, previous = item['box'], target['box']
                    scale = max(*box[2:], *previous[2:], 1)
                    gallery = coordinator.global_galleries.get(gid)
                    comparison = item['gallery'].compare(gallery) if gallery is not None else {}
                    candidates.append({'global_id':gid, 'age':age,
                        'active_owner': any(c==camera_id and lid in active_local_ids and bound==gid for (c,lid),bound in coordinator.local_bindings.items()),
                        'gallery_similarity':comparison.get('score'),
                        'vector_similarity':coordinator.similarity(item.get('appearance'), target.get('appearance')),
                        'current_has_appearance':item.get('appearance') is not None,
                        'iou':YoloPersonTracker.box_iou(box,previous),
                        'center_ratio':coordinator._box_center_distance(box,previous)/scale,
                        'current_tone':item.get('clothing_tone'),'previous_tone':target.get('clothing_tone')})
                relink_audit.append({'local_id':item['local_id'],'candidates':candidates})
            return original_relink(camera_id, prepared, active_local_ids, now, unix_time)
        coordinator._same_camera_relink_assignments = audited_relink
    captures, trackers, schedule = {}, {}, []
    for cid, stream in info['streams'].items():
        captures[cid] = cv2.VideoCapture(str(args.session / stream['video_file']))
        cfg = configs[cid]
        options = dict(confidence=cfg.get('yolo_confidence', .45), image_size=cfg.get('yolo_image_size', 640))
        if 'tracker_config' in inspect.signature(YoloPersonTracker).parameters:
            options['tracker_config'] = cfg.get('yolo_tracker_config', 'bytetrack.yaml')
        if args.guarded_lost_low and cid == 'cam_2':
            options['tracker_config'] = 'config/bytetrack_guarded_lab.yaml'
            options['reid_embedder'] = embedder
        trackers[cid] = YoloPersonTracker(args.detector, **options)
        with (args.session / stream['timestamps_file']).open(newline='') as source:
            for row in csv.DictReader(source):
                schedule.append((float(row['unix_time']), cid, int(row['frame_index'])))
    schedule.sort()
    start = schedule[0][0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.meta.json').write_text(json.dumps({
        'session_id': info['session_id'], 'replay_only': True,
        'detector_model': args.detector.name,
        'guarded_lost_low': args.guarded_lost_low,
        'guarded_implementation_sha256': hashlib.sha256(Path('guarded_byte_tracker.py').read_bytes()).hexdigest() if args.guarded_lost_low else None,
        'detector_sha256': hashlib.sha256(args.detector.read_bytes()).hexdigest(),
        'reid_model': args.reid_model.name,
        'mtmc_config_sha256': hashlib.sha256(Path('config/mtmc.json').read_bytes()).hexdigest(),
        'floor_map_sha256': hashlib.sha256(args.floor_map.read_bytes()).hexdigest(),
        'face_recognition_run': False, 'identity_ground_truth': False,
        'limit': 'Capture timestamp ordered replay; no live queue/drop/transport/face workload. Global IDs are replay IDs, not registered names.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = 0
    active = {}
    last_tracks = {}
    last_detection = {}
    last_ids = {}
    last_seen = {}
    with args.output.open('w', encoding='utf-8') as output:
        for stamp, cid, index in schedule:
            if stamp-start > args.max_seconds:
                break
            ok, frame = captures[cid].read()
            if not ok:
                raise RuntimeError(f'{cid}: missing frame {index}')
            if index % configs[cid].get('yolo_frame_stride', 1):
                continue
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            track_started = time.perf_counter()
            tracks = trackers[cid].track(frame, observed_at=stamp)
            tracker_ms = (time.perf_counter()-track_started)*1000
            held = not tracks and stamp-last_detection.get(cid, -1e20) <= configs[cid].get('track_hold_seconds', 1.5)
            if held:
                tracks = [dict(t) for t in last_tracks[cid]]
            elif tracks:
                last_tracks[cid] = [dict(t) for t in tracks]
                last_detection[cid] = stamp
            current = {t['track_id'] for t in tracks}
            for local_id in current:
                last_seen[(cid, local_id)] = stamp
            retained = set()
            for missing in active.get(cid, set()) - current:
                if stamp-last_seen[(cid, missing)] <= configs[cid].get('track_hold_seconds', 1.5):
                    retained.add(missing)
                else:
                    coordinator.forget_local(cid, missing)
            active[cid] = current | retained
            for track in tracks:
                track['raw_position'] = floor.project(cid, track['box'], frame.shape)
                location = floor.project_track(cid, track['box'], frame.shape)
                track['position'] = location['position']
                track['position_estimated'] = bool(location.get('estimated'))
                x, y, width, height = track['box']
                track['image_anchor'] = {
                    'x': (x + width * .5) / frame.shape[1],
                    'y': min(y + height, frame.shape[0] - 1) / frame.shape[0],
                }
                if track.get('partial_body'):
                    track['position'] = None
                    track['position_estimated'] = False
                    track['image_anchor'] = None
                    track['localization_quality'] = 'unavailable'
            associate_started = time.perf_counter()
            ids = last_ids[cid] if held else coordinator.update_batch(cid, tracks=tracks, frame=frame, active_local_ids=current, monotonic_time=stamp-start+1, unix_time=stamp)
            association_ms = (time.perf_counter()-associate_started)*1000
            last_ids[cid] = ids
            report = {'camera_id': cid, 'frame_index': index, 'seconds': round(stamp-start, 4), 'held': held, 'tracks': [{**t, 'global_id': ids[t['track_id']]} for t in tracks],
                'tracker_diagnostics': trackers[cid].diagnostics(),
                'association_events': list(coordinator.recent_associations)[-8:],
                'identity_ground_truth': False,
                'face_recognition_run': False,
                'tracker_ms':tracker_ms,'association_ms':association_ms}
            if args.audit_relink and not held:
                report['relink_candidate_audit'] = list(relink_audit)
            internal_trackers = getattr(trackers[cid].model.predictor, 'trackers', [])
            if internal_trackers:
                internal = internal_trackers[0]
                report['local_tracker_states'] = {
                    state: [int(t.track_id) for t in getattr(internal, state, [])]
                    for state in ('tracked_stracks', 'lost_stracks', 'removed_stracks')
                }
            if len(set(ids.values())) < len(ids):
                report['duplicate_diagnostics'] = {
                    'bindings': {str(k): v for k, v in coordinator.local_bindings.items()},
                    'active': {k: sorted(v) for k,v in coordinator.active_local_ids_by_camera.items()},
                    'events': list(coordinator.recent_associations)[-15:],
                }
            output.write(json.dumps(report, ensure_ascii=False) + '\n')
            rows += 1
            if rows % 200 == 0:
                output.flush()
                print(f'processed={rows} seconds={stamp-start:.1f}', flush=True)
    for cap in captures.values():
        cap.release()
    print(f'complete rows={rows} output={args.output}', flush=True)


if __name__ == '__main__':
    main()
