"""HTTP and read-only polling adapters for the notification subsystem."""
import json
import threading
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from management_notifications import ManagementNotifications, EvidenceAdapter
from handoff_result_index import SnapshotTrackIndex


class ManagementRuntime:
    def __init__(self, server, config_path):
        self.server = server
        config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        self.service = ManagementNotifications(Path(config_path).parent.parent /
                                              'runtime' / 'management.sqlite3', config)
        from feishu_app import DEFAULT_CREDENTIALS, GROUP_CREDENTIALS, EVIDENCE_CREDENTIALS, load_credentials
        credential_path = (EVIDENCE_CREDENTIALS if EVIDENCE_CREDENTIALS.exists() else
                           GROUP_CREDENTIALS if GROUP_CREDENTIALS.exists() else DEFAULT_CREDENTIALS)
        if credential_path.exists():
            credentials = load_credentials(credential_path)
            self.service.env = dict(self.service.env)
            self.service.env.setdefault('LAB_MANAGEMENT_ADMIN_TOKEN', credentials['management_admin_token'])
            if credential_path == GROUP_CREDENTIALS:
                self.service.env['LAB_FEISHU_GROUP_WEBHOOK'] = credentials['webhook']
                self.service.env['LAB_FEISHU_GROUP_SECRET'] = credentials.get('secret', '')
            if credential_path == EVIDENCE_CREDENTIALS:
                for audience, bot in credentials['groups'].items():
                    self.service.env['LAB_FEISHU_'+audience.upper()+'_WEBHOOK']=bot['webhook']
                    self.service.env['LAB_FEISHU_'+audience.upper()+'_SECRET']=bot.get('secret','')
        from alert_evidence import AlertEvidence
        import os
        image_token=os.environ.get('LAB_HANDOFF_IMAGE_READ_TOKEN','')
        handoff_credentials=Path(config_path).parent.parent/'runtime'/'handoff-credentials.json'
        if not image_token and handoff_credentials.exists():
            image_token=json.loads(handoff_credentials.read_text(encoding='utf-8-sig')).get('image_read_token','')
        origins=set(config.get('evidence_image_origins',['http://127.0.0.1:5175']))|{server.api_base}
        self.service.evidence=AlertEvidence(Path(config_path).parent.parent/'runtime'/'alert-evidence',origins,image_token)
        self.evidence = EvidenceAdapter(self.service)
        self.stop_event = threading.Event()
        self.last_poll_status = 'starting'
        self.thread = None

    def index(self):
        publisher = self.server.live_publisher
        return publisher.result_index if publisher else SnapshotTrackIndex()

    def start(self):
        self.service.start()
        self.thread = threading.Thread(target=self._poll, daemon=True, name='lab-management-evidence')
        self.thread.start()

    def _poll(self):
        while not self.stop_event.is_set():
            try:
                with urlopen(self.server.api_base + '/api/state', timeout=3) as response:
                    state = json.loads(response.read(4 * 1024 * 1024))
                self.evidence.cameras(state)
                self.last_poll_status = 'state_ok'
            except Exception:
                self.last_poll_status = 'state_unavailable'
            try:
                client = self.server.member_client
                if client:
                    response = client.latest_interactions(contract='lab-interaction-results-1.2')
                    body = response.body
                    if (response.ok and isinstance(body, dict) and body.get('time_domain') == 'authority'
                            and body.get('result_contract') == 'lab-interaction-results-1.2'):
                        for row in body.get('analyses', []):
                            if isinstance(row, dict):
                                try:
                                    self.evidence.analysis({**row, 'time_domain': 'authority'}, self.index())
                                except (ValueError, KeyError, TypeError):
                                    pass
            except Exception:
                self.last_poll_status = 'member_unavailable'
            self.stop_event.wait(1)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=7)
        self.service.stop()


def handle_management(handler, method):
    """Return True if handled. Every data operation requires an explicit role."""
    path = urlsplit(handler.path).path
    if path == '/management' and method == 'GET':
        handler._send_file(Path(__file__).parent / 'templates' / 'management.html', 'text/html; charset=utf-8')
        return True
    if not path.startswith('/api/management/'):
        return False
    runtime = getattr(handler.server, 'management', None)
    if runtime is None:
        handler._send_json({'error': 'management not configured'}, status=503)
        return True
    service = runtime.service
    principal = service.authenticate(handler.headers.get('Authorization', ''))
    if principal is None:
        handler._send_json({'error': 'authentication required'}, status=401)
        return True
    try:
        if principal['role'] == 'producer' and path != '/api/management/evidence':
            raise PermissionError('producer can only submit evidence')
        if method == 'GET' and path.startswith('/api/management/evidence/'):
            import re
            from management_notifications import scoped
            match=re.fullmatch(r'/api/management/evidence/([a-f0-9]{64})\.jpg',path)
            authorized=False
            if match:
                with service.lock:
                    for row in service.db.execute('SELECT payload FROM events'):
                        event=json.loads(row['payload'])
                        if event.get('evidence_id')==match[1] and scoped(principal,event):
                            authorized=True;break
            if not authorized or service.evidence is None:
                handler._send_json({'error':'evidence not available'},status=404)
            else:
                handler._send_bytes(service.evidence.read(match[1]),'image/jpeg')
        elif method == 'GET' and path == '/api/management/events':
            handler._send_json({'events': service.events(principal), 'role': principal['role'],
                                'poll_status': runtime.last_poll_status})
        elif method == 'GET' and path == '/api/management/routes':
            if principal['role'] != 'admin':
                raise PermissionError('admin required')
            handler._send_json({'routes': service.config['routes']})
        elif method == 'GET' and path == '/api/management/audit':
            if principal['role'] != 'admin':
                raise PermissionError('admin required')
            with service.lock:
                rows = [dict(row) for row in service.db.execute('SELECT * FROM audit ORDER BY id DESC LIMIT 100')]
            handler._send_json({'audit': rows})
        elif method == 'POST' and path in {'/api/management/ack', '/api/management/routes', '/api/management/evidence'}:
            length = int(handler.headers.get('Content-Length', '0'))
            if length < 1 or length > 16384:
                raise ValueError('body must be 1..16384 bytes')
            body = json.loads(handler.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('object required')
            if path.endswith('/ack'):
                service.acknowledge(principal, body['event_id'])
                handler._send_json({'ok': True})
            elif path.endswith('/routes'):
                service.set_routes(principal, body['routes'])
                handler._send_json({'ok': True})
            else:
                if (principal['role'] != 'producer' or service.config['lab_id'] not in principal.get('labs', [])
                        or body.get('camera_id') not in principal.get('cameras', [])):
                    raise PermissionError('producer camera scope required')
                if body.get('evidence_type') not in {'mask', 'ppe', 'interaction'}:
                    raise ValueError('mask, ppe or interaction evidence required')
                if body['evidence_type']=='ppe' and body.get('ppe_type') not in {'mask','lab_coat','gloves'}:
                    raise ValueError('unsupported PPE type')
                recorded = runtime.evidence.analysis(body, runtime.index(), mask=body['evidence_type'] == 'mask',
                    ppe_type=body.get('ppe_type') if body['evidence_type']=='ppe' else None)
                with service.lock, service.db:
                    service._audit(principal['id'], 'evidence.submit', str(body.get('analysis_id', 'unknown'))[:128])
                handler._send_json({'accepted': recorded,
                                    'meaning': 'event recorded' if recorded else 'no new eligible event'})
        else:
            handler._send_json({'error': 'route not found'}, status=404)
    except PermissionError:
        handler._send_json({'error': 'access denied'}, status=403)
    except (ValueError, KeyError, TypeError, OverflowError):
        handler._send_json({'error': 'invalid evidence or request'}, status=400)
    return True
