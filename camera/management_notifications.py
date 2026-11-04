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

KINDS = {'person.entered', 'instrument.started', 'instrument.ended', 'safety.no_mask', 'safety.no_lab_coat', 'safety.no_gloves',
         'environment.flame','environment.flame_recovered'}
ROLES = {'admin', 'producer'}


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
    instruments = principal.get('instruments')
    cameras = principal.get('cameras')
    return ((not event.get('instrument_id') or instruments is None or event['instrument_id'] in instruments)
            and (not event.get('camera_id') or cameras is None or event['camera_id'] in cameras))


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
        saved_policy=self.db.execute("SELECT value FROM settings WHERE key='ppe_policy'").fetchone()
        if saved_policy:self.config.update(json.loads(saved_policy['value']))
        self._validate_config(self.config)
        self.stop_event = threading.Event()
        self.thread = None
        self.last_send = 0.0
        self.evidence = None
        self.image_client = None
        self.image_credentials_revision = None
        from safety_incidents import SafetyIncidents
        self.incidents=SafetyIncidents(self)
        from ppe_preparation import PPEPreparation
        self.preparation=PPEPreparation(self)
        from ppe_policy import PPEPolicy
        self.ppe_policy=PPEPolicy(self)
        from fire_fusion import FireFusion
        self.fire_fusion=FireFusion(self)
        self.image_lock=threading.RLock()
        from fire_snapshot_album import initialize as initialize_fire_snapshots
        initialize_fire_snapshots(self)
        with self.db:
            columns={row[1] for row in self.db.execute('PRAGMA table_info(outbox)')}
            for name,declaration in [('priority','INTEGER DEFAULT 0'),('lease_until','REAL DEFAULT 0'),('claim','TEXT')]:
                if name not in columns:self.db.execute('ALTER TABLE outbox ADD COLUMN '+name+' '+declaration)

    @staticmethod
    def _validate_config(config):
        if not config.get('lab_id'):
            raise ValueError('lab_id required')
        from management_policy import validate_policy
        from fire_fusion import validate_config as validate_fire_config
        validate_fire_config(config)
        validate_policy({k:config[k] for k in ('ppe_rules','ppe_confirmation','ppe_preparation') if k in config})
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
                    or route.get('transport', 'webhook') != 'webhook'
                    or not route.get('webhook_env')
                    or set(route) - {'id', 'labs', 'instruments', 'event_types', 'webhook_env', 'secret_env',
                                     'transport','audience'}):
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
                  'instrument_id', 'instrument_name', 'observed_at', 'evidence_source', 'evidence_id',
                  'severity','incident_id','device_id','audiences','fire_stage','zone_id','sources','image_observed_at',
                  'hardware_observed_at','visual_observed_at'}
        item = {k: v for k, v in event.items() if k in fields}
        item.setdefault('severity','L0' if item.get('event_type') in {'person.entered','instrument.started'} else 'L1')
        if (item.get('event_type') not in KINDS or item.get('lab_id') != self.config['lab_id']
                or not item.get('event_id') or (not item.get('person_id') and
                    not (item['event_type'].startswith('environment.') and item.get('device_id')))):
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
            semantic = (item['event_type'], item.get('person_id'), item.get('instrument_id'))
            for row in self.db.execute('SELECT payload FROM events WHERE received>?',
                                       (self.wall() - self.config.get('cooldown_seconds', 120),)):
                old = json.loads(row['payload'])
                if (not item.get('incident_id') and item.get('severity')!='L3' and
                        (old['event_type'], old.get('person_id'), old.get('instrument_id')) == semantic):
                    return False
            self.db.execute('INSERT INTO events(id,payload,received) VALUES(?,?,?)',
                            (item['event_id'], encoded, self.wall()))
            if item.get('incident_id') and item['event_type'].startswith('safety.'):
                self.incidents.note_ppe(item)
            for route in self.config.get('routes', []):
                principal = dict(role='route', labs=route['labs'], instruments=route.get('instruments'))
                if (item['event_type'] not in self.config.get('silent_events', [])
                        and item['event_type'] in route['event_types'] and scoped(principal, item)):
                    if item.get('audiences') and route.get('audience',route['id']) not in item['audiences'].split(','):
                        continue
                    self.db.execute('INSERT INTO outbox(event_id,route_id,state,due,priority) VALUES(?,?,?,?,?)',
                                    (item['event_id'], route['id'], 'pending', self.wall(),3 if item.get('severity')=='L3' else 1))
            from fire_snapshot_album import enqueue as enqueue_fire_snapshots
            enqueue_fire_snapshots(self, item)
            self._audit('system', 'event.record', item['event_id'])
        return True

    def _audit(self, actor, action, target):
        self.db.execute('INSERT INTO audit(at,actor,action,target) VALUES(?,?,?,?)',
                        (self.wall(), actor, action, target))

    def cooling_down(self,event):
        if event.get('incident_id'):return False
        semantic=(event['event_type'],event.get('person_id'),event.get('instrument_id'))
        with self.lock:
            for row in self.db.execute('SELECT payload FROM events WHERE received>?',
                (self.wall()-self.config.get('cooldown_seconds',120),)):
                previous=json.loads(row['payload'])
                if (previous['event_type'],previous.get('person_id'),previous.get('instrument_id'))==semantic:
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
                    album = self.db.execute('SELECT items,state FROM fire_snapshot_jobs WHERE event_id=?', (row['id'],)).fetchone()
                    if album:
                        item['fire_snapshots'] = {'state': album['state'], 'cameras': json.loads(album['items']),
                            'deliveries': [dict(r) for r in self.db.execute(
                                'SELECT route_id,state,attempts,error FROM fire_snapshot_deliveries WHERE event_id=?', (row['id'],))]}
                    if item.get('session_id'):
                        table=self.db.execute("SELECT 1 FROM sqlite_master WHERE name='instrument_sessions'").fetchone()
                        current=self.db.execute('SELECT payload FROM instrument_sessions WHERE id=?',(item['session_id'],)).fetchone() if table else None
                        if current:
                            observation=json.loads(current[0])
                            item.update(lifecycle_status=observation['status'],
                                        interruption_reason=observation.get('interruption_reason'),
                                        last_valid_source_at=observation.get('last_valid_source_at'))
                    output.append(item)
                if len(output) >= 100:
                    break
            return output

    def acknowledge(self, principal, event_id):
        if principal['role'] != 'admin':
            raise PermissionError('maintenance administrator required')
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
        if event['event_type'] in {'instrument.started', 'instrument.ended'} and event.get('session_id'):
            from datetime import timedelta
            local = lambda at: datetime.fromtimestamp(stamp(at), timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
            ending = ('\n结束：' + local(event['ended_at']) + '\n时长：' + event['duration_seconds'] + ' 秒'
                      if event.get('ended_at') else '\n结束：尚未取得有效结束判断')
            return (f"实验室仪器记录｜{'开始使用' if event['event_type'].endswith('started') else '结束使用'}"
                    f"\n人员：{event['display_name']}\n仪器：{event['instrument_name']}"
                    f"\n摄像头：{event['camera_id']}\n开始：{local(event['started_at'])}{ending}"
                    f"\n时间依据：源画面（北京时间）\n会话：{event['session_id']}\n事件：{event['event_id']}")
        if event['event_type'].startswith('environment.'):
            stage=event.get('fire_stage')
            title=('【L3 紧急】视觉与硬件双源火焰确认，需现场核查' if stage=='dual_confirmed' else
                   '【L3 紧急】火焰报警（单源），需现场核查') if event['event_type']=='environment.flame' else '火焰信号恢复，等待现场核查关闭'
            return (f"实验室通知\n{title}\n实验室：{event['lab_id']}\n设备：{event['device_id']}"
                    f"\n区域：{event.get('zone_id','未配置')}\n来源：{event.get('sources',event.get('evidence_source','unknown'))}"
                    f"\n视觉快照时间：{event.get('image_observed_at','无视觉图片')}"
                    f"\n硬件观测：{event.get('hardware_observed_at','无融合观测')}\n视觉观测：{event.get('visual_observed_at','无融合观测')}"
                    f"\n原始观测时间：{event['observed_at']}\n告警：{event['incident_id']}\n事件：{event['event_id']}")
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
        return f"实验室通知 {event.get('severity','')}\n{name} {action}\n实验室：{event['lab_id']}{camera}\n观测时间：{event['observed_at']}{snapshot}\n事件：{event['event_id']}"

    def deliver_once(self):
        from notification_delivery import deliver
        return deliver(self)

    def start(self):
        def run():
            while not self.stop_event.wait(.5):
                try:
                    self.deliver_once()
                except Exception:
                    pass  # Isolation: notification failure never stops camera services.
        self.threads=[threading.Thread(target=run,daemon=True,name='lab-feishu-outbox-'+str(i)) for i in range(3)]
        for worker in self.threads:worker.start()

    def stop(self):
        self.stop_event.set()
        for worker in getattr(self,"threads",[]):worker.join(timeout=6)


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
                or original.get('time_domain') not in (None,'authority')
                or (self.service.config.get('require_authority_snapshot',False) and original.get('time_domain')!='authority')
                or abs(stamp(original['observed_at']) - stamp(row['captured_at'])) > .005):
            return False  # An old frame cannot be made current with a new timestamp.
        now = self.service.wall()
        if (row.get('status') != 'valid' or row.get('time_domain') != 'authority'
                or not row.get('analysis_id')
                or min(stamp(row['expires_at']), stamp(row['published_at']) + 5) <= now
                or stamp(row['published_at']) > now + 2):
            return False
        if not -2<=now-stamp(row['captured_at'])<=self.service.config.get('event_max_age_seconds',15):return False
        if mask or ppe_type:
            ppe_type=ppe_type or 'mask'
            rule=self.service.config.get('ppe_rules',{}).get(ppe_type)
            if (ppe_type not in {'mask','lab_coat','gloves'} or
                    (rule is not None and not rule.get('enabled')) or (not mask and rule is None)):
                return False
            if row.get('evidence_type')=='ppe' and row.get('visibility')!='valid':
                self.service.ppe_policy.observe(matched['person_id'],'safety.no_'+ppe_type,
                    row.get('instrument_id'),stamp(row['captured_at']),row['snapshot_id'],'unknown')
                return False
            confidence = row.get('confidence', 0)
            state=row.get('ppe_state') if row.get('evidence_type')=='ppe' else row.get('mask_state')
            minimum=rule.get('minimum_confidence',.9) if rule else .9
            kind = 'safety.no_'+ppe_type
            valid_confidence=(type(confidence) in (int,float) and math.isfinite(confidence) and minimum<=confidence<=1)
            frame=index.evidence_frame(row['camera_id'],row['snapshot_id']) or {}
            policy_result=self.service.ppe_policy.observe(matched['person_id'],kind,row.get('instrument_id'),
                stamp(row['captured_at']),row['camera_id']+':'+str(frame.get('source_frame_id') or row['snapshot_id']),
                state if valid_confidence else 'unknown')
            if state!='not_worn' or not valid_confidence or policy_result is None:return False
        else:
            if row.get('is_interacting') is not True or not row.get('instrument_id'):
                return False
            preparation=self.service.config.get('ppe_preparation',{})
            if preparation.get('enabled') and preparation.get('end_on_confirmed_interaction',True):
                self.service.preparation.finish(matched['person_id'],stamp(row['captured_at']),
                                                'confirmed_instrument_interaction')
            kind = 'instrument.started'
        event=self.base(kind,matched,row['captured_at'],'ppe_detector' if mask or ppe_type else 'member_minicpm')
        if mask or ppe_type:
            event.update(policy_result)
            if event.get('incident_id'):event['event_id']=event['incident_id']+':'+event['severity']
        if self.service.cooling_down(event):
            return False
        if kind in self.service.config.get('evidence_required_events', []):
            if self.service.evidence is None:
                if event.get('incident_id'):self.service.ppe_policy.retry(event['incident_id'],event['severity'])
                return False
            try:
                event['evidence_id']=self.service.evidence.capture(matched,index)
            except Exception:
                if event.get('incident_id'):self.service.ppe_policy.retry(event['incident_id'],event['severity'])
                with self.service.lock,self.service.db:
                    self.service._audit('system','evidence.unavailable',str(row.get('snapshot_id','unknown'))[:128])
                return False
        try:
            return self.service.record(event)
        except Exception:
            if event.get('incident_id'):self.service.ppe_policy.retry(event['incident_id'],event['severity'])
            raise

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
                    preparation=self.service.config.get('ppe_preparation',{})
                    self.service.preparation.see(person,now,entrance_face=(
                        camera['id'] in preparation.get('camera_ids',[])
                        and identity.get('identity_source')=='face'))
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
