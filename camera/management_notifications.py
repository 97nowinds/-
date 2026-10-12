"""Evidence-based laboratory events and a durable, scoped Feishu outbox.

No detector or identity assignments live here. Secrets are environment-only.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

KINDS = {'person.entered', 'instrument.started', 'safety.no_mask', 'safety.no_lab_coat', 'safety.no_gloves'}
ROLES = {'viewer', 'experimenter', 'manager', 'admin', 'producer'}


def stamp(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timestamp requires timezone')
    return parsed.timestamp()


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec='milliseconds')


def scoped(principal, event):
    if principal['role'] == 'admin':
        return True
    if event['lab_id'] not in principal.get('labs', []):
        return False
    if principal['role'] == 'experimenter' and event['person_id'] not in principal.get('person_ids', []):
        return False
    instruments = principal.get('instruments')
    return not event.get('instrument_id') or instruments is None or event['instrument_id'] in instruments


def feishu_payload(text, secret='', now=None):
    payload = {'msg_type': 'text', 'content': {'text': text}}
    if secret:
        timestamp = str(int(time.time() if now is None else now))
        key = (timestamp + '\n' + secret).encode()
        payload.update(timestamp=timestamp, sign=base64.b64encode(
            hmac.new(key, b'', hashlib.sha256).digest()).decode())
    return payload


def send_feishu(url, payload):
    from feishu_app import FeishuError
    if url.startswith('cli_'):
        raise FeishuError('第一栏填的是自建应用 App ID。请改为群自定义机器人的完整 Webhook 地址。')
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname != 'open.feishu.cn'
            or parsed.port not in (None, 443) or parsed.username or parsed.password
            or parsed.query or parsed.fragment
            or not parsed.path.startswith('/open-apis/bot/v2/hook/')):
        raise FeishuError('Webhook格式不正确：应为 https://open.feishu.cn/open-apis/bot/v2/hook/ 开头的完整地址。')
    req = Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
                  headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(req, timeout=5) as response:
        body = json.loads(response.read(65536))
    if body.get('code', body.get('StatusCode', -1)) != 0:
        code = body.get('code', body.get('StatusCode', -1))
        hints = {19001:'Webhook无效，请重新复制群自定义机器人的地址。',
                 19021:'签名校验失败，请填写该群机器人的签名密钥；App Secret不能代替它。',
                 19022:'机器人IP白名单不允许当前主机的出口地址。',
                 19024:'消息不符合机器人关键词限制；本次测试文字包含“实验室”。',
                 11232:'飞书限流，请稍后重试。'}
        safe_code = str(code) if isinstance(code, int) else 'unknown'
        raise FeishuError('飞书返回错误码 '+safe_code+'：'+hints.get(code,'请求被拒绝，请检查机器人设置。'))


class ManagementNotifications:
    def __init__(self, path, config, environ=None, wall=time.time, sender=send_feishu):
        self.config = json.loads(json.dumps(config))
        self.env = os.environ if environ is None else environ
        self.wall, self.sender = wall, sender
        self.lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                received REAL NOT NULL, acknowledged_by TEXT, acknowledged_at REAL);
            CREATE TABLE IF NOT EXISTS outbox(event_id TEXT, route_id TEXT, state TEXT,
                attempts INTEGER DEFAULT 0, due REAL, error TEXT,
                PRIMARY KEY(event_id, route_id));
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, at REAL, actor TEXT,
                action TEXT, target TEXT);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS evidence_uploads(evidence_id TEXT,app_id TEXT,image_key TEXT,
                PRIMARY KEY(evidence_id,app_id));
        ''')
        saved = self.db.execute("SELECT value FROM settings WHERE key='routes'").fetchone()
        if saved:
            self.config['routes'] = json.loads(saved['value'])
        self._validate_config(self.config)
        self.stop_event = threading.Event()
        self.thread = None
        self.last_send = 0.0
        self.app_client = None
        self.app_credentials_revision = None
        self.evidence = None
        self.image_client = None
        self.image_credentials_revision = None

    @staticmethod
    def _validate_config(config):
        if not config.get('lab_id'):
            raise ValueError('lab_id required')
        ids = set()
        for principal in config.get('principals', []):
            if (principal.get('role') not in ROLES or not principal.get('id')
                    or not principal.get('token_env') or principal['id'] in ids):
                raise ValueError('invalid principal')
            ids.add(principal['id'])
        ids = set()
        for route in config.get('routes', []):
            if (not route.get('id') or route['id'] in ids
                    or not set(route.get('event_types', [])) <= KINDS
                    or not route.get('labs')
                    or route.get('transport', 'webhook') not in {'webhook', 'app'}
                    or (route.get('transport', 'webhook') == 'webhook' and not route.get('webhook_env'))
                    or (route.get('transport') == 'app' and route.get('recipient_key') != 'primary')
                    or set(route) - {'id', 'labs', 'instruments', 'event_types', 'webhook_env', 'secret_env',
                                     'transport', 'recipient_key'}):
                raise ValueError('invalid notification route')
            ids.add(route['id'])

    def authenticate(self, header):
        if not header.startswith('Bearer '):
            return None
        supplied = header[7:]
        if not supplied:
            return None
        matches = [p for p in self.config.get('principals', [])
                   if self.env.get(p['token_env']) and hmac.compare_digest(
                       supplied.encode(), self.env[p['token_env']].encode())]
        return dict(matches[0]) if len(matches) == 1 else None

    def record(self, event):
        # Explicit allowlist prevents accidental transmission of biometrics/secrets.
        fields = {'event_id', 'event_type', 'lab_id', 'person_id', 'display_name',
                  'camera_id', 'snapshot_id', 'track_id', 'global_id',
                  'instrument_id', 'instrument_name', 'observed_at', 'evidence_source', 'evidence_id'}
        item = {k: v for k, v in event.items() if k in fields}
        if (item.get('event_type') not in KINDS or item.get('lab_id') != self.config['lab_id']
                or not item.get('event_id') or not item.get('person_id')):
            raise ValueError('known identity and valid event required')
        for value in item.values():
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError('invalid event field')
        age = self.wall() - stamp(item['observed_at'])
        if age < -2 or age > self.config.get('event_max_age_seconds', 15):
            raise ValueError('event timestamp stale or future')
        if item['event_type'] in self.config.get('evidence_required_events', []) and not item.get('evidence_id'):
            raise ValueError('exact-frame evidence required')
        with self.lock, self.db:
            encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
            previous = self.db.execute('SELECT payload FROM events WHERE id=?', (item['event_id'],)).fetchone()
            if previous:
                if previous['payload'] != encoded:
                    raise ValueError('event_id conflict')
                return False
            # Cooldown is persistent and shared across cameras; short ID churn does not spam.
            semantic = (item['event_type'], item['person_id'], item.get('instrument_id'))
            for row in self.db.execute('SELECT payload FROM events WHERE received>?',
                                       (self.wall() - self.config.get('cooldown_seconds', 120),)):
                old = json.loads(row['payload'])
                if (old['event_type'], old['person_id'], old.get('instrument_id')) == semantic:
                    return False
            self.db.execute('INSERT INTO events(id,payload,received) VALUES(?,?,?)',
                            (item['event_id'], encoded, self.wall()))
            for route in self.config.get('routes', []):
                principal = dict(role='manager', labs=route['labs'], instruments=route.get('instruments'))
                if (item['event_type'] not in self.config.get('silent_events', [])
                        and item['event_type'] in route['event_types'] and scoped(principal, item)):
                    self.db.execute('INSERT INTO outbox(event_id,route_id,state,due) VALUES(?,?,?,?)',
                                    (item['event_id'], route['id'], 'pending', self.wall()))
            self._audit('system', 'event.record', item['event_id'])
        return True

    def _audit(self, actor, action, target):
        self.db.execute('INSERT INTO audit(at,actor,action,target) VALUES(?,?,?,?)',
                        (self.wall(), actor, action, target))

    def cooling_down(self,event):
        semantic=(event['event_type'],event['person_id'],event.get('instrument_id'))
        with self.lock:
            for row in self.db.execute('SELECT payload FROM events WHERE received>?',
                (self.wall()-self.config.get('cooldown_seconds',120),)):
                previous=json.loads(row['payload'])
                if (previous['event_type'],previous['person_id'],previous.get('instrument_id'))==semantic:
                    return True
        return False

    def events(self, principal):
        with self.lock:
            output = []
            for row in self.db.execute('SELECT * FROM events ORDER BY received DESC LIMIT 1000'):
                item = json.loads(row['payload'])
                if scoped(principal, item):
                    item.update(acknowledged_by=row['acknowledged_by'], acknowledged_at=row['acknowledged_at'],
                                deliveries=[dict(r) for r in self.db.execute(
                                    'SELECT route_id,state,attempts,error FROM outbox WHERE event_id=?', (row['id'],))])
                    output.append(item)
                if len(output) >= 100:
                    break
            return output

    def acknowledge(self, principal, event_id):
        if principal['role'] not in {'manager', 'admin'}:
            raise PermissionError('manager role required')
        with self.lock, self.db:
            row = self.db.execute('SELECT payload FROM events WHERE id=?', (event_id,)).fetchone()
            if not row or not scoped(principal, json.loads(row['payload'])):
                raise PermissionError('event access denied')
            self.db.execute('UPDATE events SET acknowledged_by=?,acknowledged_at=? WHERE id=?',
                            (principal['id'], self.wall(), event_id))
            self._audit(principal['id'], 'event.ack', event_id)

    def set_routes(self, principal, routes):
        if principal['role'] != 'admin':
            raise PermissionError('admin role required')
        with self.lock, self.db:
            candidate = {**self.config, 'routes': routes}
            self._validate_config(candidate)
            self.config = candidate
            self.db.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',
                            ('routes', json.dumps(routes, ensure_ascii=False)))
            self._audit(principal['id'], 'routes.update', 'persisted')

    @staticmethod
    def message(event):
        name = event.get('display_name') or '身份已确认人员'
        if event['event_type'] == 'person.entered':
            action = '进入实验室（入口与室内连续身份观测）'
        elif event['event_type'] == 'instrument.started':
            action = '正在操作' + event.get('instrument_name', event['instrument_id'])
        else:
            action = {'safety.no_mask':'未戴口罩','safety.no_lab_coat':'未穿实验服',
                      'safety.no_gloves':'未戴手套'}[event['event_type']]+'（防护检测结果）'
        camera='\n摄像头：'+event['camera_id'] if event.get('camera_id') else ''
        snapshot='\n快照：'+event['snapshot_id'] if event.get('snapshot_id') else ''
        return f"实验室通知\n{name} {action}\n实验室：{event['lab_id']}{camera}\n观测时间：{event['observed_at']}{snapshot}\n事件：{event['event_id']}"

    def deliver_once(self):
        with self.lock:
            row = self.db.execute('SELECT o.*, e.payload FROM outbox o JOIN events e ON e.id=o.event_id '
                                  "WHERE o.state='pending' AND o.due<=? ORDER BY o.due LIMIT 1", (self.wall(),)).fetchone()
            if not row or self.wall() - self.last_send < 1:
                return False
            route = next((r for r in self.config['routes'] if r['id'] == row['route_id']), None)
            event = json.loads(row['payload'])
            if self.wall() - stamp(event['observed_at']) > self.config.get('delivery_max_age_seconds', 300):
                self.db.execute("UPDATE outbox SET state='expired' WHERE event_id=? AND route_id=?",
                                (row['event_id'], row['route_id']))
                self.db.commit()
                return True
            # Re-evaluate route scope after an administrator revokes it.
            allowed = route and event['event_type'] in route['event_types'] and scoped(
                dict(role='manager', labs=route['labs'], instruments=route.get('instruments')), event)
            if not allowed:
                self.db.execute("UPDATE outbox SET state='cancelled' WHERE event_id=? AND route_id=?",
                                (row['event_id'], row['route_id']))
                self.db.commit()
                return True
            application = route.get('transport') == 'app'
            from feishu_app import DEFAULT_CREDENTIALS, load_credentials, FeishuAppClient
            url = self.env.get(route.get('webhook_env', ''), '')
            configured = DEFAULT_CREDENTIALS.exists() if application else bool(url)
            if not configured:
                self.db.execute('UPDATE outbox SET due=? WHERE event_id=? AND route_id=?',
                                (self.wall() + 30, row['event_id'], row['route_id']))
                self.db.commit()
                return False  # Leave durable pending until configured.
            self.last_send = self.wall()
            # Network I/O is on a dedicated worker, never on the inference path.
            try:
                if application:
                    revision = DEFAULT_CREDENTIALS.stat().st_mtime_ns
                    if self.app_client is None or revision != self.app_credentials_revision:
                        self.app_client = FeishuAppClient(load_credentials(), wall=self.wall)
                        self.app_credentials_revision = revision
                    recipient = self.app_client.credentials['primary_open_id']
                    self.app_client.send_text(recipient, self.message(event), event['event_id'])
                else:
                    if event.get('evidence_id'):
                        from feishu_app import EVIDENCE_CREDENTIALS
                        from alert_evidence import group_evidence_payload
                        if self.evidence is None or not EVIDENCE_CREDENTIALS.exists():
                            raise ValueError('image transport not configured')
                        revision=EVIDENCE_CREDENTIALS.stat().st_mtime_ns
                        if self.image_client is None or self.image_credentials_revision!=revision:
                            self.image_client=FeishuAppClient(load_credentials(EVIDENCE_CREDENTIALS),wall=self.wall)
                            self.image_credentials_revision=revision
                        app_id=self.image_client.credentials['app_id']
                        old=self.db.execute('SELECT image_key FROM evidence_uploads WHERE evidence_id=? AND app_id=?',
                                            (event['evidence_id'],app_id)).fetchone()
                        image_key=old['image_key'] if old else self.image_client.upload_image(self.evidence.read(event['evidence_id']))
                        if not old:
                            with self.db:
                                self.db.execute('INSERT INTO evidence_uploads VALUES(?,?,?)',(event['evidence_id'],app_id,image_key))
                        payload=group_evidence_payload(self.message(event),image_key,
                            self.env.get(route.get('secret_env',''),''),self.wall())
                    else:
                        payload=feishu_payload(self.message(event),self.env.get(route.get('secret_env',''),''),self.wall())
                    self.sender(url,payload)
                state, error = 'sent', None
            except Exception as exc:
                # Never log exception text: it can contain a credential-bearing URL.
                state, error = ('failed' if row['attempts'] >= 4 else 'pending'), type(exc).__name__
            with self.db:
                self.db.execute('UPDATE outbox SET state=?, attempts=attempts+1,due=?,error=? '
                                'WHERE event_id=? AND route_id=?',
                                (state, self.wall() + min(300, 2 ** (row['attempts'] + 1)), error,
                                 row['event_id'], row['route_id']))
            return True

    def start(self):
        def run():
            while not self.stop_event.wait(.5):
                try:
                    self.deliver_once()
                except Exception:
                    pass  # Isolation: notification failure never stops camera services.
        self.thread = threading.Thread(target=run, daemon=True, name='lab-feishu-outbox')
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)


class EvidenceAdapter:
    def __init__(self, service):
        self.service = service
        self.entrances = {}
        self.indoor_seen = {}
        self.lock = threading.RLock()

    def base(self, kind, row, observed_at, source):
        key = '|'.join(str(row.get(k, '')) for k in
                       ('camera_id', 'snapshot_id', 'track_id', 'person_id', 'instrument_id'))
        digest = hashlib.sha256((kind + key + str(observed_at)).encode()).hexdigest()[:32]
        event = {k: str(row[k]) for k in ('camera_id', 'snapshot_id', 'track_id', 'global_id',
                 'person_id', 'display_name', 'instrument_id', 'instrument_name') if row.get(k)}
        return {**event, 'event_id': kind + ':' + digest, 'event_type': kind,
                'lab_id': self.service.config['lab_id'], 'observed_at': observed_at, 'evidence_source': source}

    def analysis(self, row, index, mask=False, ppe_type=None):
        # Ignore receiver names. Resolve only the original full snapshot key.
        matched = index.resolve(row)
        if matched.get('handoff_match_status') != 'matched' or not matched.get('person_id'):
            return False
        original = index.evidence_track(row['camera_id'], row['snapshot_id'], row['track_id'])
        if (not original or not original.get('observed_at')
                or abs(stamp(original['observed_at']) - stamp(row['captured_at'])) > .005):
            return False  # An old frame cannot be made current with a new timestamp.
        now = self.service.wall()
        if (row.get('status') != 'valid' or row.get('time_domain') != 'authority'
                or not row.get('analysis_id')
                or min(stamp(row['expires_at']), stamp(row['published_at']) + 5) <= now
                or stamp(row['published_at']) > now + 2):
            return False
        if mask or ppe_type:
            ppe_type=ppe_type or 'mask'
            rule=self.service.config.get('ppe_rules',{}).get(ppe_type)
            if (ppe_type not in {'mask','lab_coat','gloves'} or
                    (rule is not None and not rule.get('enabled')) or (not mask and rule is None)):
                return False
            if row.get('evidence_type')=='ppe' and row.get('visibility')!='valid':
                return False
            confidence = row.get('confidence', 0)
            state=row.get('ppe_state') if row.get('evidence_type')=='ppe' else row.get('mask_state')
            minimum=rule.get('minimum_confidence',.9) if rule else .9
            if (state != 'not_worn' or type(confidence) not in (int, float)
                    or not math.isfinite(confidence) or not minimum <= confidence <= 1):
                return False
            kind = 'safety.no_'+ppe_type
        else:
            if row.get('is_interacting') is not True or not row.get('instrument_id'):
                return False
            kind = 'instrument.started'
        event=self.base(kind,matched,row['captured_at'],'ppe_detector' if mask or ppe_type else 'member_minicpm')
        if self.service.cooling_down(event):
            return False
        if kind in self.service.config.get('evidence_required_events', []):
            if self.service.evidence is None:
                return False
            try:
                event['evidence_id']=self.service.evidence.capture(matched,index)
            except ValueError:
                with self.service.lock,self.service.db:
                    self.service._audit('system','evidence.unavailable',str(row.get('snapshot_id','unknown'))[:128])
                return False
        return self.service.record(event)

    def cameras(self, state):
        # Relative entry sequence uses host receipt clock, never compares GPU wall clocks.
        now = self.service.wall()
        with self.lock:
            self.entrances = {k: v for k, v in self.entrances.items() if now - v < 20}
            self.indoor_seen = {k: v for k, v in self.indoor_seen.items() if now - v < 60}
            indoor = []
            for camera in state.get('cameras', []):
                if (camera.get('status') != 'running' or camera.get('replay', {}).get('enabled')
                        or camera.get('detection_age_seconds') is None
                        or camera['detection_age_seconds'] > 1.5
                        or camera.get('capture_age_seconds') is None or camera['capture_age_seconds'] > 2):
                    continue
                for identity in camera.get('identities', []):
                    matching = [t for t in camera.get('yolo_tracks', [])
                                if t.get('local_id') == identity.get('local_id')
                                and t.get('track_id') == identity.get('global_track_id')
                                and not t.get('position_estimated')]
                    if (len(matching) != 1 or not identity.get('person_id')
                            or identity.get('identity_lock_status') != 'locked'):
                        continue
                    person = identity['person_id']
                    if camera.get('role') == 'entrance_identity':
                        if person not in self.indoor_seen:
                            self.entrances.setdefault(person, now)
                    else:
                        indoor.append((camera, identity))
            for camera, identity in indoor:
                person = identity['person_id']
                if person in self.entrances and self.entrances[person] < now:
                    row = dict(camera_id=camera['id'], person_id=person,
                               display_name=identity.get('name'), track_id=identity['global_track_id'])
                    self.service.record(self.base('person.entered', row, iso(now),
                                        'host_received_entrance_then_indoor'))
                    del self.entrances[person]
                self.indoor_seen[person] = now
