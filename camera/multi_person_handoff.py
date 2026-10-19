"""Explicitly started two-camera joint test. Stop via runtime/multi-person-stop.request."""
import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import psutil
from bounded_live_handoff import session, encode, body, submit, saved_confirmation
from continuous_handoff import prepare_live_event


def main():
    root = Path(__file__).resolve().parent
    stop_path = root/'runtime/multi-person-stop.request'
    status_path = root/'runtime/multi-person-status.json'
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding='utf-8'))
        if old.get('state') == 'running' and psutil.pid_exists(old.get('pid', -1)):
            raise SystemExit('joint sender already running')
    stop_path.unlink(missing_ok=True)
    original = json.loads((root/'runtime/continuous-cam2-status.json').read_text(encoding='utf-8-sig'))
    process = psutil.Process(original['pid'])
    if not any('continuous_handoff.py' in a for a in process.cmdline()):
        raise SystemExit('unexpected credential process')
    token = process.environ()['LAB_MEMBER_SERVICE_TOKEN']
    credentials = json.loads((root/'runtime/handoff-credentials.json').read_text(encoding='utf-8-sig'))
    secrets = (token, credentials['image_read_token'], credentials['sync_read_token'])
    receiver = 'http://192.168.31.189:8765'
    images = 'http://192.168.31.127:5175'
    output = root.parent/'output'/('multi-person-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True)
    lock = threading.RLock()
    stop = threading.Event()
    started = time.perf_counter()
    counts = {c: dict(captured=0, accepted=0, saved=0, failed=0, duplicates=0,
                     with_tracks=0, with_locations=0, with_estimated_locations=0,
                     with_identity=0, queue_peak=0, backpressure_waits=0) for c in ('cam_1', 'cam_2')}
    queues = {c: queue.Queue(maxsize=30) for c in counts}
    capture_done = {c: threading.Event() for c in counts}
    pending = {}
    source_ids = set()
    local_bytes = [0]
    model_count = [0]
    model_positive = [0]
    last_error = [None]

    def save(path, data):
        text = json.dumps(data, ensure_ascii=False, indent=2)
        if any(v and v in text for v in secrets):
            raise RuntimeError('credential detected; output suppressed')
        with lock:
            target = Path(path)
            temporary = target.with_suffix(target.suffix+'.tmp')
            temporary.write_text(text, encoding='utf-8')
            temporary.replace(target)

    def log(name, data):
        text = json.dumps(data, ensure_ascii=False)
        if any(v and v in text for v in secrets):
            raise RuntimeError('credential detected; output suppressed')
        with lock:
            with (output/name).open('a', encoding='utf-8') as stream:
                stream.write(text+'\n')

    client = session(token)
    response = client.get(receiver+'/api/v1.2/capabilities', timeout=(5, 10))
    capabilities = body(response)
    save(output/'capabilities.json', dict(http_status=response.status_code, body=capabilities))
    policy = capabilities.get('map_candidate_policy', {})
    if (response.status_code != 200 or capabilities.get('service_version') != '1.4.8'
            or policy.get('estimated_positions') != 'accepted_for_candidate_selection'
            or not capabilities.get('multi_person_analysis', {}).get('supported')
            or not capabilities.get('auto_analysis_enabled')):
        raise SystemExit('receiver policy differs; no business input sent')

    def capture(camera):
        s = session(credentials['image_read_token'])
        previous = None
        tick = 0
        while not stop.is_set():
            # Bound input age without dropping a captured original or stopping
            # the other camera. Actual capture frequency is reported honestly.
            if queues[camera].qsize() >= 6:
                with lock:
                    counts[camera]['backpressure_waits'] += 1
                stop.wait(.1)
                continue
            due = started+tick/3
            if stop.wait(max(0, due-time.perf_counter())):
                break
            try:
                response = s.get(images+'/api/handoff/frozen/'+camera, timeout=(2, 5))
                response.raise_for_status()
                frozen = response.json()
                if frozen['source_frame_id'] == previous:
                    with lock:
                        counts[camera]['duplicates'] += 1
                    tick += 1
                    continue
                response = s.get(frozen['image_url'], timeout=(2, 5))
                response.raise_for_status()
                jpeg = response.content
                event = prepare_live_event(frozen, jpeg, camera_id=camera, include_estimated_locations=True)
                expected = images+'/api/handoff/snapshots/'+frozen['snapshot_id']+'.jpg'
                if event['camera']['image_url'] != expected:
                    raise ValueError('image URL differs from selected LAN endpoint')
                for track in event['camera']['tracks']:
                    identities = [i for i in frozen['camera'].get('identities', [])
                                  if i.get('local_id') == track['local_track_id']
                                  and i.get('global_track_id') == track['global_id']
                                  and i.get('identity_lock_status') == 'locked'
                                  and track.get('person_id') and i.get('person_id') == track['person_id']]
                    track['display_name'] = identities[0].get('name') if len(identities) == 1 else None
                with lock:
                    index = counts[camera]['captured']+1
                    if local_bytes[0]+len(jpeg) > 1024**3 or shutil.disk_usage(output).free < 512*1024**2:
                        last_error[0] = 'local unconfirmed image capacity reached; capture stopped, originals retained'
                        stop.set(); break
                stem = camera+'-'+str(index).zfill(7)
                image_path = output/(stem+'.jpg')
                with image_path.open('xb') as stream:
                    stream.write(jpeg); stream.flush(); os.fsync(stream.fileno())
                save(output/(stem+'.request.json'), event)
                save(output/(stem+'.frozen.json'), frozen)
                with lock:
                    previous = frozen['source_frame_id']
                    source_ids.add(previous)
                    local_bytes[0] += len(jpeg)
                    count = counts[camera]
                    count['captured'] += 1
                    tracks = event['camera']['tracks']
                    count['with_tracks'] += bool(tracks)
                    count['with_locations'] += any(t.get('location') for t in tracks)
                    count['with_estimated_locations'] += any(t.get('location', {}).get('position_estimated') is True for t in tracks)
                    count['with_identity'] += any(t.get('person_id') for t in tracks)
                item = dict(camera=camera, index=index, stem=stem, event=event,
                            jpeg_bytes=len(jpeg), image_path=image_path)
                log('capture.jsonl', dict(camera=camera, index=index, event_id=event['event_id'],
                    source_frame_id=previous, captured_at=event['captured_at'], tracks=tracks,
                    jpeg_bytes=len(jpeg), elapsed_seconds=time.perf_counter()-started))
                try:
                    queues[camera].put_nowait(item)
                except queue.Full:
                    last_error[0] = camera+' submit queue reached 30; test stopped, original input retained'
                    stop.set(); break
                with lock:
                    count['queue_peak'] = max(count['queue_peak'], queues[camera].qsize())
            except Exception as exc:
                with lock:
                    last_error[0] = camera+' capture '+type(exc).__name__
                log('errors.jsonl', dict(camera=camera, stage='capture', error=type(exc).__name__))
                stop.wait(.3)
            tick = max(tick+1, int((time.perf_counter()-started)*3)+1)
        capture_done[camera].set()

    def sending(camera):
        s = session(token)
        while not capture_done[camera].is_set() or not queues[camera].empty():
            try:
                item = queues[camera].get(timeout=.2)
            except queue.Empty:
                continue
            event = item['event']
            def record(data):
                log('post-attempts.jsonl', dict(camera=camera, index=item['index'],
                    event_id=event['event_id'], **data))
            expires = datetime.fromisoformat(event['camera']['image_expires_at']).timestamp()
            result = submit(s, receiver+'/api/handoffs', encode(event), expires,
                            time.monotonic()+45, record)
            save(output/(item['stem']+'.submission.json'), result)
            with lock:
                if result.get('receipt_id'):
                    counts[camera]['accepted'] += 1
                    pending[result['receipt_id']] = item
                else:
                    counts[camera]['failed'] += 1
                    last_error[0] = camera+' '+result['state']
            queues[camera].task_done()

    def receipts():
        s = session(token)
        while (not all(v.is_set() for v in capture_done.values())
               or any(q.unfinished_tasks for q in queues.values()) or pending):
            with lock:
                batch = list(pending.items())[:30]
            for rid, item in batch:
                try:
                    response = s.get(receiver+'/api/handoffs/receipts/'+quote(rid, safe=''), timeout=(5, 10))
                    data = body(response)
                    log('receipt-checks.jsonl', dict(camera=item['camera'], index=item['index'],
                        receipt_id=rid, http_status=response.status_code, body=data))
                    if response.status_code == 200 and saved_confirmation(data):
                        save(output/(item['stem']+'.saved.json'), data)
                        with lock:
                            counts[item['camera']]['saved'] += 1
                            pending.pop(rid, None)
                        # Keep three representative originals per camera. All other
                        # originals are removed ONLY after a persisted save proof.
                        if item['index'] > 3:
                            item['image_path'].unlink(missing_ok=True)
                            with lock:
                                local_bytes[0] -= item['jpeg_bytes']
                    elif data.get('input_state') in ('failed', 'recovery_held', 'submission_unknown'):
                        save(output/(item['stem']+'.failed-receipt.json'), data)
                        with lock:
                            counts[item['camera']]['failed'] += 1
                            pending.pop(rid, None)
                            last_error[0] = item['camera']+' receipt '+str(data.get('error'))
                except Exception as exc:
                    log('errors.jsonl', dict(stage='receipt', error=type(exc).__name__))
            time.sleep(.3)

    def history():
        s = session(token)
        cursor = 0
        while not stop.is_set():
            try:
                response = s.get(receiver+f'/api/interaction-results/history?after={cursor}&limit=100', timeout=(5, 10))
                data = body(response)
                if response.status_code == 200 and isinstance(data.get('results'), list):
                    for result in data['results']:
                        log('results.jsonl', result)
                        with lock:
                            matches = any(source in json.dumps(result) for source in source_ids)
                        if matches:
                            log('matched-results.jsonl', result)
                            model_count[0] += 1
                            model_positive[0] += result.get('is_interacting') is True
                    # Save actual result records before advancing the durable cursor.
                    following = data.get('next_cursor')
                    if following is not None:
                        cursor = following
                        save(output/'cursor.json', dict(next_cursor=cursor))
            except Exception as exc:
                log('errors.jsonl', dict(stage='history', error=type(exc).__name__))
            stop.wait(3)

    workers = []
    for camera in counts:
        for fn in (capture, sending):
            t = threading.Thread(target=fn, args=(camera,), daemon=True)
            t.start(); workers.append(t)
    for fn in (receipts, history):
        t = threading.Thread(target=fn, daemon=True)
        t.start(); workers.append(t)
    print('DUAL_STARTED '+str(output), flush=True)
    def status(state):
        with lock:
            return dict(pid=os.getpid(), state=state, output=str(output),
                started_at=datetime.fromtimestamp(time.time()-(time.perf_counter()-started), timezone.utc).isoformat(),
                elapsed_seconds=time.perf_counter()-started, target_fps_per_camera=3,
                cameras={c: dict(**v, queue=queues[c].qsize(),
                    actual_capture_fps=v['captured']/max(.001,time.perf_counter()-started))
                    for c,v in counts.items()},
                pending_save_confirmations=len(pending), retained_image_bytes=local_bytes[0],
                model_results=model_count[0], model_positive_results=model_positive[0],
                last_error=last_error[0], stop_file=str(stop_path),
                updated_at=datetime.now(timezone.utc).isoformat())
    while not stop.is_set():
        save(status_path, status('running'))
        if stop_path.exists():
            stop.set(); break
        time.sleep(1)
    save(status_path, status('draining'))
    deadline = time.monotonic()+65
    while any(t.is_alive() for t in workers) and time.monotonic()<deadline:
        for t in workers:
            t.join(timeout=.1)
    final = status('stopped')
    save(status_path, final); save(output/'summary.json', final)


if __name__ == '__main__':
    main()
