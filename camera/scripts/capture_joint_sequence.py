"""Capture a short local evidence sequence; never POST events or resume senders."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen, install_opener, build_opener, ProxyHandler

from frontend_server import _jpeg_dimensions
from handoff_sample import build_sample_event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=50)
    parser.add_argument('--keep-unavailable-location', action='store_true')
    parser.add_argument('--keep-all-tracks', action='store_true',
                        help='Retain empty/multiple-track frames for continuous geometry review')
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    install_opener(build_opener(ProxyHandler({})))
    credentials = json.loads((root/'runtime/handoff-credentials.json').read_text(encoding='utf-8-sig'))
    headers = {'Authorization': 'Bearer ' + credentials['image_read_token']}
    args.output.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    empty, invalid = 0, 0
    started = time.monotonic()
    next_tick = started
    print('Local capture ready; continuous business sending remains paused.', flush=True)
    while time.monotonic()-started < args.seconds:
        if args.stop_file and args.stop_file.exists():
            break
        try:
            cycle_started = time.perf_counter()
            with urlopen(Request('http://127.0.0.1:5175/api/handoff/frozen/cam_2', headers=headers), timeout=3) as response:
                frozen = json.load(response)
            metadata_elapsed = time.perf_counter()-cycle_started
            source_id = frozen['source_frame_id']
            if source_id in seen:
                continue
            seen.add(source_id)
            if not args.keep_all_tracks and len(frozen['camera']['yolo_tracks']) != 1:
                empty += 1
                continue
            if (frozen['map_id'], frozen['map_revision']) != ('lab-floor', 'd9ea86e4bd80d086'):
                raise ValueError('map differs from agreed map')
            if (frozen['frame_width'], frozen['frame_height']) != (2560, 1440):
                raise ValueError('original 2560x1440 frame required')
            event = build_sample_event(frozen, require_person_location=not args.keep_unavailable_location)
            for track in event['camera']['tracks']:
                x, y, width, height = track['bbox']
                if x < 0 or y < 0 or width <= 0 or height <= 0 or x+width > 2560 or y+height > 1440:
                    raise ValueError('restored box outside original frame')
            image_started = time.perf_counter()
            image_url = 'http://127.0.0.1:5175/api/handoff/snapshots/'+frozen['snapshot_id']+'.jpg'
            with urlopen(Request(image_url, headers=headers), timeout=3) as response:
                jpeg = response.read(8*1024*1024+1)
            image_elapsed = time.perf_counter()-image_started
            if (len(jpeg)>8*1024*1024 or hashlib.sha256(jpeg).hexdigest()!=frozen['image_sha256']
                    or _jpeg_dimensions(jpeg)!=(2560,1440)):
                raise ValueError('image validation failed')
            event['camera']['image_url'] = 'http://100.106.153.101:5175/api/handoff/snapshots/'+frozen['snapshot_id']+'.jpg'
            index=len(rows)+1
            stem=f'frame-{index:03d}'
            (args.output/(stem+'.jpg')).write_bytes(jpeg)
            (args.output/(stem+'.json')).write_text(json.dumps(event,ensure_ascii=False,indent=2),encoding='utf-8')
            rows.append(dict(index=index, stem=stem, elapsed_seconds=round(time.monotonic()-started,3),
                snapshot_id=frozen['snapshot_id'], source_frame_id=source_id,
                captured_at=event['captured_at'],
                track_id=event['camera']['tracks'][0]['track_id'] if len(event['camera']['tracks'])==1 else None,
                tracks=event['camera']['tracks'], source_clock=frozen['source_clock'],
                source_captured_at=frozen['source_captured_at'], source_occurred_at=frozen['source_occurred_at'],
                source_detection_age_ms=(frozen['source_occurred_at']-frozen['source_captured_at'])*1000,
                metadata_fetch_ms=metadata_elapsed*1000,image_fetch_ms=image_elapsed*1000,
                cycle_ms=(time.perf_counter()-cycle_started)*1000,
                jpeg_bytes=len(jpeg)))
            if len(rows)%15==0:
                print(json.dumps({'captured':len(rows),'elapsed_seconds':rows[-1]['elapsed_seconds'],'track_id':rows[-1]['track_id']}),flush=True)
        except Exception as error:
            invalid += 1
            print(type(error).__name__+': '+str(error),flush=True)
        finally:
            next_tick += 1/3
            if next_tick < time.monotonic(): next_tick=time.monotonic()
            time.sleep(max(0,next_tick-time.monotonic()))
    result=dict(duration_seconds=time.monotonic()-started,frames=rows,skipped_empty_or_multiple=empty,
        rejected_frames=invalid,continuous_business_sending=False)
    (args.output/'sequence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'captured_frames':len(rows),'output':str(args.output),
        'track_ids':sorted({track['track_id'] for r in rows for track in r['tracks']})}),flush=True)


if __name__=='__main__':
    main()
