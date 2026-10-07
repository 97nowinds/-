"""Replay detector/ReID association in capture timestamp order without live writes."""
import argparse
import csv
import json
import inspect
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
    args = parser.parse_args()
    info = json.loads((args.session / 'info.json').read_text(encoding='utf-8'))
    configs = {c['id']: c for c in json.loads(Path('config/cameras.json').read_text(encoding='utf-8'))}
    floor = FloorMapProjector.from_path(args.floor_map)
    embedder = ReIDEmbedder('../person_track/weights/reID/719rank1.pth')
    coordinator = CrossCameraTrackCoordinator(embedder, floor_map=floor)
    captures, trackers, schedule = {}, {}, []
    for cid, stream in info['streams'].items():
        captures[cid] = cv2.VideoCapture(str(args.session / stream['video_file']))
        cfg = configs[cid]
        options = dict(confidence=cfg.get('yolo_confidence', .45), image_size=cfg.get('yolo_image_size', 640))
        if 'tracker_config' in inspect.signature(YoloPersonTracker).parameters:
            options['tracker_config'] = cfg.get('yolo_tracker_config', 'bytetrack.yaml')
        trackers[cid] = YoloPersonTracker('models/yolov8n.pt', **options)
        with (args.session / stream['timestamps_file']).open(newline='') as source:
            for row in csv.DictReader(source):
                schedule.append((float(row['unix_time']), cid, int(row['frame_index'])))
    schedule.sort()
    start = schedule[0][0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    active = {}
    last_tracks = {}
    last_detection = {}
    last_ids = {}
    with args.output.open('w', encoding='utf-8') as output:
        for stamp, cid, index in schedule:
            if stamp-start > args.max_seconds:
                break
            ok, frame = captures[cid].read()
            if not ok:
                raise RuntimeError(f'{cid}: missing frame {index}')
            if index % configs[cid].get('yolo_frame_stride', 1):
                continue
            tracks = trackers[cid].track(frame)
            held = not tracks and stamp-last_detection.get(cid, -1e20) <= configs[cid].get('track_hold_seconds', 1.5)
            if held:
                tracks = [dict(t) for t in last_tracks[cid]]
            elif tracks:
                last_tracks[cid] = [dict(t) for t in tracks]
                last_detection[cid] = stamp
            current = {t['track_id'] for t in tracks}
            for missing in active.get(cid, set()) - current:
                coordinator.forget_local(cid, missing)
            active[cid] = current
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
            ids = last_ids[cid] if held else coordinator.update_batch(cid, tracks=tracks, frame=frame, monotonic_time=stamp-start+1, unix_time=stamp)
            last_ids[cid] = ids
            report = {'camera_id': cid, 'frame_index': index, 'seconds': round(stamp-start, 4), 'held': held, 'tracks': [{**t, 'global_id': ids[t['track_id']]} for t in tracks]}
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
