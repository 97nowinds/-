"""Finite real-camera transport check; never changes resident services or gates."""
import argparse
import json
import queue
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import psutil
import requests

from continuous_handoff import prepare_live_event
from member_integration import extract_receipt_id


def session(token):
    s = requests.Session()
    s.trust_env = False
    s.headers.update(Authorization='Bearer ' + token)
    return s


def encode(event):
    return json.dumps(event, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def body(response):
    try:
        return response.json()
    except ValueError:
        return {'non_json_body': response.text}


def retry_delay(response, attempt):
    try:
        return min(10., max(1., float(response.headers.get('Retry-After', 1))))
    except (ValueError, TypeError):
        return min(4., 2 ** (attempt - 1))


def saved_confirmation(data):
    """Only an explicit boolean is a deletion permission; status is insufficient."""
    if not isinstance(data, dict):
        return False
    if data.get('source_may_delete') is True:
        return True
    return isinstance(data.get('receipt'), dict) and data['receipt'].get('source_may_delete') is True


def submit(s, url, raw, expires, deadline, record, sleep=time.sleep):
    """Retry identical bytes in place, preserving order and original image expiry."""
    for attempt in range(1, 5):
        if time.time() >= expires or time.monotonic() >= deadline:
            return {'state': 'unsent_or_unknown_expired', 'attempts': attempt - 1}
        start = time.monotonic()
        try:
            r = s.post(url, data=raw, headers={'Content-Type': 'application/json; charset=utf-8'}, timeout=(5, 10))
            data = body(r)
            record({'attempt': attempt, 'http_status': r.status_code, 'body': data,
                    'elapsed_ms': (time.monotonic() - start) * 1000,
                    'remaining_image_seconds': expires - time.time()})
            receipt = extract_receipt_id(data)
            if 200 <= r.status_code < 300 and receipt:
                return {'state': 'accepted', 'receipt_id': receipt, 'attempts': attempt}
            if r.status_code not in (429, 500, 502, 503, 504):
                return {'state': 'rejected_or_missing_receipt', 'http_status': r.status_code, 'attempts': attempt}
            delay = retry_delay(r, attempt)
        except requests.RequestException as error:
            # No Authorization, URL credentials or raw exception text in evidence.
            record({'attempt': attempt, 'http_status': None, 'error': type(error).__name__,
                    'receipt_unknown': True, 'elapsed_ms': (time.monotonic() - start) * 1000})
            delay = min(4., 2 ** (attempt - 1))
        if attempt < 4:
            sleep(min(delay, max(0., expires - time.time()), max(0., deadline - time.monotonic())))
    return {'state': 'retry_limit_unknown_or_busy', 'attempts': 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--fps', type=float, default=3)
    ap.add_argument('--settle-seconds', type=float, default=65)
    ap.add_argument('--receiver', default='http://192.168.31.189:8765')
    ap.add_argument('--image-base', default='http://192.168.31.127:5175')
    ap.add_argument('--camera', choices=('cam_1', 'cam_2'), default='cam_2')
    args = ap.parse_args()
    if not (0 < args.seconds <= 30 and 0 < args.fps <= 3 and 0 <= args.settle_seconds <= 90):
        raise SystemExit('finite check only: <=30 seconds, <=3 fps, <=90 seconds drain')
    root = Path(__file__).resolve().parent
    state = json.loads((root/'runtime/continuous-cam2-status.json').read_text(encoding='utf-8-sig'))
    proc = psutil.Process(state['pid'])
    if not any('continuous_handoff.py' in a for a in proc.cmdline()):
        raise SystemExit('unexpected credential process')
    token = proc.environ()['LAB_MEMBER_SERVICE_TOKEN']
    creds = json.loads((root/'runtime/handoff-credentials.json').read_text(encoding='utf-8-sig'))
    secrets = (token, creds['image_read_token'], creds['sync_read_token'])
    receiver = args.receiver.rstrip('/')
    local = args.image_base.rstrip('/')
    public = local
    out = root.parent/'output'/('live-input-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True)
    lock = threading.RLock()

    def save(path, data):
        text = json.dumps(data, ensure_ascii=False, indent=2)
        if any(v and v in text for v in secrets):
            raise RuntimeError('credential detected; evidence suppressed')
        with lock:
            (out/path).write_text(text, encoding='utf-8')

    def log(name, data):
        text = json.dumps(data, ensure_ascii=False)
        if any(v and v in text for v in secrets):
            raise RuntimeError('credential detected; evidence suppressed')
        with lock:
            with (out/name).open('a', encoding='utf-8') as f:
                f.write(text+'\n')

    health_client = requests.Session()
    health_client.trust_env = False
    r = health_client.get(receiver+'/health', timeout=(5, 10))
    save('health.json', {'http_status': r.status_code, 'body': body(r)})
    if r.status_code != 200:
        raise SystemExit('receiver health failed; no frames submitted')
    check = session(token)
    r = check.get(receiver+'/api/v1.2/capabilities', timeout=(5, 10))
    cap = body(r); save('capabilities.json', {'http_status': r.status_code, 'body': cap})
    if r.status_code != 200 or cap.get('service_version') != '1.4.6' or not cap.get('durable_image_cache'):
        raise SystemExit('receiver prerequisite differs; no frames submitted')
    q = queue.Queue()
    rows = []; accepted = {}; confirmations = {}; finished = threading.Event()
    start = time.monotonic(); capture_end = start + args.seconds
    deadline = capture_end + args.settle_seconds
    counts = {'attempts': 0, 'max_submit_queue': 0, 'submit_in_flight': 0}

    def sending():
        s = session(token)
        while True:
            item = q.get()
            if item is None:
                q.task_done(); return
            i, event, raw = item
            with lock:
                counts['submit_in_flight'] = 1
            def record(data):
                with lock:
                    counts['attempts'] += 1
                log('post-attempts.jsonl', {'index': i, 'event_id': event['event_id'], **data})
            result = submit(s, receiver+'/api/handoffs', raw,
                            datetime.fromisoformat(event['camera']['image_expires_at']).timestamp(), deadline, record)
            with lock:
                if result.get('receipt_id'):
                    accepted[i] = result['receipt_id']
                counts['submit_in_flight'] = 0
            save(f'frame-{i:03d}.submission.json', result)
            q.task_done()

    def polling():
        s = session(token)
        while time.monotonic() < deadline:
            with lock:
                pending = [(i, rid) for i, rid in accepted.items() if i not in confirmations]
            for i, rid in pending:
                if time.monotonic() >= deadline:
                    break
                try:
                    r = s.get(receiver+'/api/handoffs/receipts/'+quote(rid, safe=''), timeout=(5, 10))
                    data = body(r)
                    log('receipt-checks.jsonl', {'index': i, 'http_status': r.status_code, 'body': data,
                                               'received_at': datetime.now(timezone.utc).isoformat()})
                    if r.status_code == 200 and saved_confirmation(data):
                        with lock:
                            confirmations[i] = data
                        save(f'frame-{i:03d}.saved.json', data)
                except requests.RequestException as e:
                    log('receipt-checks.jsonl', {'index': i, 'http_status': None, 'error': type(e).__name__})
            if finished.is_set() and q.unfinished_tasks == 0:
                with lock:
                    if len(confirmations) == len(accepted):
                        return
            time.sleep(.5)

    sender = threading.Thread(target=sending, daemon=True)
    poller = threading.Thread(target=polling, daemon=True)
    sender.start(); poller.start()
    images = session(creds['image_read_token'])
    seen = set(); tick = 0
    print('START '+str(out)+' '+datetime.now(timezone.utc).isoformat(), flush=True)
    try:
        while time.monotonic() < capture_end:
            target = start + tick/args.fps
            if time.monotonic() < target:
                time.sleep(min(target-time.monotonic(), max(0., capture_end-time.monotonic())))
            if time.monotonic() >= capture_end:
                break
            began = time.monotonic()
            try:
                r = images.get(local+'/api/handoff/frozen/'+args.camera, timeout=(2, 5)); r.raise_for_status()
                frozen = r.json()
                if frozen['source_frame_id'] in seen:
                    log('capture-errors.jsonl', {'tick': tick, 'error': 'duplicate_source_not_sent'})
                else:
                    r = images.get(local+'/api/handoff/snapshots/'+frozen['snapshot_id']+'.jpg', timeout=(2, 5)); r.raise_for_status()
                    jpeg = r.content; event = prepare_live_event(frozen, jpeg, camera_id=args.camera)
                    expected_url = public+'/api/handoff/snapshots/'+frozen['snapshot_id']+'.jpg'
                    if frozen['image_url'] != expected_url:
                        raise ValueError('frozen image URL differs from selected LAN endpoint')
                    event['camera']['image_url'] = frozen['image_url']
                    for t in event['camera']['tracks']:
                        ids = [v for v in frozen['camera'].get('identities', []) if v.get('local_id') == t['local_track_id'] and v.get('global_track_id') == t['global_id'] and v.get('identity_lock_status') == 'locked' and t.get('person_id') and v.get('person_id') == t['person_id']]
                        t['display_name'] = ids[0].get('name') if len(ids) == 1 else None
                    i = len(rows)+1; raw = encode(event)
                    # Persist complete original and same-frame metadata before queueing.
                    with (out/f'frame-{i:03d}.jpg').open('xb') as f:
                        f.write(jpeg); f.flush()
                        import os
                        os.fsync(f.fileno())
                    save(f'frame-{i:03d}.request.json', event)
                    save(f'frame-{i:03d}.frozen.json', frozen)
                    row = {'index': i, 'source_frame_id': frozen['source_frame_id'], 'source_captured_at': frozen['source_captured_at'], 'captured_at': event['captured_at'], 'jpeg_bytes': len(jpeg), 'fetch_ms': (time.monotonic()-began)*1000, 'queued_elapsed_s': time.monotonic()-start, 'source_age_at_queue_s': time.time()-frozen['source_captured_at'], 'tracks': event['camera']['tracks']}
                    rows.append(row); seen.add(frozen['source_frame_id']); log('capture.jsonl', row)
                    q.put((i, event, raw))
                    counts['max_submit_queue'] = max(counts['max_submit_queue'], q.qsize())
            except Exception as e:
                log('capture-errors.jsonl', {'tick': tick, 'error': type(e).__name__})
            tick = max(tick+1, int((time.monotonic()-start)*args.fps)+1)
    finally:
        capture_elapsed = time.monotonic()-start
        save('capture-end.json', {'capture_elapsed_seconds': capture_elapsed, 'captured': len(rows), 'submit_queue': q.qsize(), 'accepted': len(accepted), 'saved_confirmed': len(confirmations)})
        q.put(None); finished.set()
    print('CAPTURE_ENDED '+json.dumps({'captured': len(rows), 'accepted': len(accepted), 'saved': len(confirmations), 'queued': q.qsize()-1}), flush=True)
    while (sender.is_alive() or poller.is_alive()) and time.monotonic() < deadline+16:
        sender.join(timeout=.5); poller.join(timeout=.5)
    intervals = [rows[i]['source_captured_at']-rows[i-1]['source_captured_at'] for i in range(1, len(rows))]
    summary = {'target_seconds': args.seconds, 'target_fps': args.fps, 'capture_elapsed_seconds': capture_elapsed, 'captured': len(rows), 'captured_per_wall_second': len(rows)/capture_elapsed, 'unique_source_frames': len(seen), 'post_attempts': counts['attempts'], 'accepted': len(accepted), 'source_may_delete_confirmed': len(confirmations), 'unconfirmed_originals_retained': len(rows)-len(confirmations), 'all_originals_retained': True, 'max_submit_queue': counts['max_submit_queue'], 'pending_submit_queue': max(0,q.qsize()), 'source_intervals_seconds': intervals, 'source_interval_min': min(intervals) if intervals else None, 'source_interval_median': statistics.median(intervals) if intervals else None, 'source_interval_max': max(intervals) if intervals else None, 'image_bytes_total': sum(v['jpeg_bytes'] for v in rows), 'source_span_seconds': rows[-1]['source_captured_at']-rows[0]['source_captured_at'] if len(rows)>1 else None, 'resident_services_changed': False, 'output': str(out)}
    save('summary.json', summary)
    print('COMPLETE '+json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
