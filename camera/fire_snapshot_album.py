"""Durable three-camera scene supplements, independent of emergency text delivery."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
import math
import threading
import time
from urllib.request import Request, build_opener, ProxyHandler

CAMERAS = ('cam_1', 'cam_2', 'cam_entrance')


def group_preview(jpeg):
    """Bound network cost; original encrypted 2K evidence remains unchanged."""
    from io import BytesIO
    from PIL import Image
    with Image.open(BytesIO(jpeg)) as image:
        image = image.convert('RGB')
        image.thumbnail((1280, 720))
        output = BytesIO(); image.save(output, format='JPEG', quality=75, optimize=True)
        return output.getvalue()


def initialize(service):
    service.db.executescript('''
        CREATE TABLE IF NOT EXISTS fire_snapshot_jobs(
            event_id TEXT PRIMARY KEY, deadline REAL, items TEXT, state TEXT);
        CREATE TABLE IF NOT EXISTS fire_snapshot_deliveries(
            event_id TEXT, route_id TEXT, state TEXT, attempts INTEGER DEFAULT 0,
            due REAL, error TEXT, PRIMARY KEY(event_id,route_id));
    ''')


def enqueue(service, event):
    if event['event_type'] != 'environment.flame':
        return
    service.db.execute('INSERT OR IGNORE INTO fire_snapshot_jobs VALUES(?,?,?,?)',
                       (event['event_id'], service.wall()+20, '{}', 'pending'))


def scene_time(frozen, authority, now, alarm):
    """Convert actual GPU receive time through its verified, current mapping."""
    sync = frozen.get('sync_mapping') or {}
    if (frozen.get('timestamp_source') != 'host_receive'
            or sync.get('source_clock') != frozen.get('source_clock')
            or sync.get('authority_clock') != authority):
        raise ValueError('clock_unverified')
    keys = ('offset_ms', 'uncertainty_ms', 'sync_valid_from', 'sync_valid_until')
    if any(type(sync.get(k)) not in (float, int) or not math.isfinite(sync[k]) for k in keys):
        raise ValueError('clock_unverified')
    captured = frozen['source_captured_at'] + sync['offset_ms']/1000
    occurred = frozen['source_occurred_at'] + sync['offset_ms']/1000
    if (not 0 <= sync['uncertainty_ms'] <= 100
            or not sync['sync_valid_from'] <= captured <= occurred <= sync['sync_valid_until']
            or now >= sync['sync_valid_until'] or occurred > now+2):
        raise ValueError('clock_unverified')
    if not -2 <= now-captured <= 5 or not -5 <= captured-alarm <= 20:
        raise ValueError('frame_stale')
    return captured


class FrozenSceneCapture:
    """Read existing protected frozen frames; no inference or RTSP credentials."""
    def __init__(self, service, credentials_path, base='http://127.0.0.1:5175'):
        self.service, self.credentials_path, self.base = service, credentials_path, base

    def __call__(self, event, camera):
        from alert_evidence import NoRedirect
        from handoff_result_index import SnapshotTrackIndex
        from management_notifications import iso, stamp
        creds = json.loads(self.credentials_path.read_text(encoding='utf-8-sig'))
        opener = build_opener(ProxyHandler({}), NoRedirect())
        began = time.monotonic()
        with opener.open(Request(self.base+'/api/handoff/frozen/'+camera,
                         headers={'Authorization': 'Bearer '+creds['image_read_token']}), timeout=3) as response:
            frozen = json.loads(response.read(4*1024*1024))
        if frozen.get('camera_id') != camera:
            raise ValueError('camera_mismatch')
        received = self.service.wall()
        if time.monotonic()-began > 5 or not -2 <= received-stamp(event['observed_at']) <= 25:
            raise ValueError('request_stale')
        if frozen.get('timestamp_source') != 'host_receive':
            raise ValueError('replay_not_allowed')
        source_time = frozen.get('source_captured_at')
        if type(source_time) not in (int,float) or not math.isfinite(source_time) or not frozen.get('source_clock'):
            raise ValueError('source_time_missing')
        captured, domain = source_time, 'source'
        # The trusted /frozen endpoint enforces frame age against its own GPU
        # clock. Without an authority mapping, preserve that clock explicitly
        # and report main-host receipt separately; never invent an offset.
        if frozen.get('sync_mapping'):
            probe = {'probe_id': 'fire_album_'+str(time.time_ns()), 'client_sent_at': iso(self.service.wall())}
            request = Request(self.base+'/api/handoff/time-sync', data=json.dumps(probe).encode(),
                headers={'Authorization': 'Bearer '+creds['sync_read_token'], 'Content-Type': 'application/json'})
            with opener.open(request, timeout=3) as response:
                authority = json.load(response)['authority_clock']
            captured = scene_time(frozen, authority, self.service.wall(), stamp(event['observed_at']))
            domain = 'authority'
        # Never follow a URL supplied by an upstream result. The image origin is fixed locally.
        import re
        snapshot = frozen['snapshot_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', snapshot):
            raise ValueError('snapshot_invalid')
        snapshot_event = {'snapshot_id': snapshot, 'captured_at': iso(captured),
            'time_domain': domain, 'camera': {
                'camera_id': camera, 'source_frame_id': frozen['source_frame_id'],
                'image_url': self.base+'/api/handoff/snapshots/'+snapshot+'.jpg',
                'image_sha256': frozen['image_sha256'], 'frame_width': frozen['frame_width'],
                'frame_height': frozen['frame_height'], 'tracks': []}}
        index = SnapshotTrackIndex(); index.remember(snapshot_event)
        row = {'camera_id': camera, 'snapshot_id': snapshot, 'captured_at': iso(captured)}
        evidence = self.service.evidence.capture_scene(row, index)
        return {'status': 'available', 'evidence_id': evidence, 'camera_id': camera,
                'snapshot_id': snapshot, 'captured_at': iso(captured),
                'source_captured_at': iso(source_time), 'source_clock': frozen['source_clock'],
                'time_domain': domain, 'host_received_at': iso(received),
                'source_frame_id': frozen['source_frame_id'],
                'alarm_delta_seconds': round(captured-stamp(event['observed_at']), 3) if domain=='authority' else None}


def album_payload(event, items, keys, secret, now):
    from management_notifications import feishu_payload, stamp
    local = lambda value: datetime.fromtimestamp(stamp(value), timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    rows = [[{'tag': 'text', 'text': '火焰报警现场快照（不代表三路均发现火焰）\n'
        '群内图片为压缩预览，原始快照加密保留后台。\n报警时间：'+local(event['observed_at'])+' 北京时间\n'
        '来源：'+event.get('sources', event.get('evidence_source', 'unknown'))+
        '\n事件：'+event['event_id']}]]
    for camera in CAMERAS:
        item = items[camera]
        if item['status'] == 'available':
            timing = ('主机时间域画面接收：'+local(item['captured_at'])+'｜距报警 '+str(item['alarm_delta_seconds'])+' 秒'
                if item.get('time_domain','authority')=='authority' else
                'GPU 源画面接收：'+local(item['source_captured_at'])+'（源时钟，未换算）\n主机获取：'+local(item['host_received_at']))
            rows.append([{'tag': 'text', 'text': camera+'｜'+timing}])
            rows.append([{'tag': 'img', 'image_key': keys[camera]}])
        else:
            rows.append([{'tag': 'text', 'text': camera+'：未取得有效快照（断线、画面过期或时间校验未通过）'}])
    payload = {'msg_type': 'post', 'content': {'post': {'zh_cn': {
        'title': '实验室火焰报警｜三路现场快照', 'content': rows}}}}
    signed = feishu_payload('', secret, now)
    for key in ('timestamp', 'sign'):
        if key in signed: payload[key] = signed[key]
    return payload


class FireSnapshotAlbums:
    def __init__(self, service, capture):
        self.service, self.capture = service, capture
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name='fire-three-camera-evidence')
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=8)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.capture_once()
                self.deliver_once()
            except Exception:
                # Do not log URLs, protected credentials, or exception bodies.
                pass
            self.stop_event.wait(.5)

    def capture_once(self):
        s = self.service
        with s.lock:
            job = s.db.execute("SELECT j.*,e.payload FROM fire_snapshot_jobs j JOIN events e ON e.id=j.event_id "
                               "WHERE j.state='pending' ORDER BY e.received LIMIT 1").fetchone()
        if not job: return False
        event, items = json.loads(job['payload']), json.loads(job['items'])
        missing = [c for c in CAMERAS if items.get(c, {}).get('status') != 'available']
        def capture(camera):
            try:
                item = self.capture(event, camera)
                if item.get('camera_id') != camera or item.get('status') != 'available':
                    raise ValueError('invalid_capture')
                return camera, item
            except Exception:
                return camera, {'camera_id': camera, 'status': 'unavailable'}
        # Deadline prevents taking a new, unrelated picture after a long outage/restart.
        if s.wall() <= job['deadline']:
            with ThreadPoolExecutor(max_workers=3) as pool:
                items.update(dict(pool.map(capture, missing)))
        else:
            for camera in missing: items[camera] = {'camera_id': camera, 'status': 'unavailable'}
        ready = all(items.get(c, {}).get('status') == 'available' for c in CAMERAS) or s.wall() >= job['deadline']
        with s.lock, s.db:
            s.db.execute('UPDATE fire_snapshot_jobs SET items=?,state=? WHERE event_id=?',
                         (json.dumps(items), 'captured' if ready else 'pending', event['event_id']))
            if ready:
                for route in s.db.execute('SELECT route_id FROM outbox WHERE event_id=?', (event['event_id'],)).fetchall():
                    s.db.execute('INSERT OR IGNORE INTO fire_snapshot_deliveries VALUES(?,?,?,?,?,?)',
                                 (event['event_id'], route['route_id'], 'pending', 0, s.wall(), None))
        return True

    def deliver_once(self):
        from feishu_app import EVIDENCE_CREDENTIALS, FeishuAppClient, load_credentials
        from management_notifications import scoped
        s = self.service
        with s.lock, s.db:
            s.db.execute("UPDATE fire_snapshot_deliveries SET state='pending' WHERE state='sending' AND due<=?", (s.wall(),))
            row = s.db.execute("SELECT d.*,j.items,e.payload,e.received FROM fire_snapshot_deliveries d "
                'JOIN fire_snapshot_jobs j ON j.event_id=d.event_id JOIN events e ON e.id=d.event_id '
                'JOIN outbox o ON o.event_id=d.event_id AND o.route_id=d.route_id '
                "WHERE d.state='pending' AND d.due<=? AND o.state='sent' ORDER BY e.received LIMIT 1", (s.wall(),)).fetchone()
            if not row: return False
            event, items = json.loads(row['payload']), json.loads(row['items'])
            route = next((r for r in s.config['routes'] if r['id'] == row['route_id']), None)
            if (not route or event['event_type'] not in route['event_types'] or not scoped(
                    dict(role='route', labs=route['labs'], instruments=route.get('instruments')), event)):
                state = 'cancelled'
            elif s.wall()-row['received'] > 300:
                state = 'expired'
            else: state = None
            if state:
                s.db.execute('UPDATE fire_snapshot_deliveries SET state=? WHERE event_id=? AND route_id=?',
                             (state, row['event_id'], row['route_id']))
                return True
            if not s.env.get(route['webhook_env']): return False
            s.db.execute("UPDATE fire_snapshot_deliveries SET state='sending',due=? WHERE event_id=? AND route_id=?",
                         (s.wall()+120, row['event_id'], row['route_id']))
        try:
            keys = {}
            if any(i['status'] == 'available' for i in items.values()):
                with s.image_lock:
                    revision = EVIDENCE_CREDENTIALS.stat().st_mtime_ns
                    if s.image_client is None or s.image_credentials_revision != revision:
                        s.image_client = FeishuAppClient(load_credentials(EVIDENCE_CREDENTIALS), wall=s.wall)
                        s.image_credentials_revision = revision
                    client = s.image_client; app_id = client.credentials['app_id']
                    for camera in CAMERAS:
                        item = items[camera]
                        if item['status'] != 'available': continue
                        with s.lock:
                            saved = s.db.execute('SELECT image_key FROM evidence_uploads WHERE evidence_id=? AND app_id=?',
                                                 (item['evidence_id'], app_id)).fetchone()
                        key = saved[0] if saved else client.upload_image(
                            group_preview(s.evidence.read(item['evidence_id'])), timeout=20)
                        if not saved:
                            with s.lock, s.db:
                                s.db.execute('INSERT OR REPLACE INTO evidence_uploads VALUES(?,?,?)', (item['evidence_id'], app_id, key))
                        keys[camera] = key
            payload = album_payload(event, items, keys, s.env.get(route.get('secret_env', ''), ''), s.wall())
            s.sender(s.env[route['webhook_env']], payload)
            state, error = 'sent', None
        except Exception as exc:
            state, error = ('failed' if row['attempts'] >= 4 else 'pending'), type(exc).__name__
        with s.lock, s.db:
            s.db.execute('UPDATE fire_snapshot_deliveries SET state=?,attempts=attempts+1,due=?,error=? WHERE event_id=? AND route_id=?',
                         (state, s.wall()+min(60, 2**(row['attempts']+1)), error, row['event_id'], row['route_id']))
        return True
