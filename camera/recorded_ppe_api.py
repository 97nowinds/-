"""Run the offline PPE reviewer through the existing cluster connection."""
import json
from pathlib import Path
import subprocess
import threading

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()


def handle_recorded_ppe(handler, action):
    if not LOCK.acquire(blocking=False):
        handler._send_json({'error': '正在处理另一批录像，请稍后重试'}, status=503)
        return
    try:
        handler.connection.settimeout(20)
        length = int(handler.headers.get('Content-Length', '0'))
        if not 0 < length <= 12 * 1024 * 1024:
            raise ValueError('截图批次过大')
        body = json.loads(handler.rfile.read(length))
        if action == 'scan' and not 1 <= len(body['frames']) <= 12:
            raise ValueError('每次最多截取 12 张')
        config = json.loads((ROOT/'config/resident_services.json').read_text(encoding='utf-8-sig'))
        key = str(Path(config['ssh_identity']).expanduser())
        proxy = f'ssh -i "{key}" -o BatchMode=yes -o ConnectTimeout=8 -W %h:%p {config["ssh_jump"]}'
        command = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-i', key,
                   '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ProxyCommand='+proxy,
                   config['ssh_target'], 'env', 'PYTHONPATH=/cluster/dataX/xqyi/lab-camera-python',
                   '/cluster/home/opt/anaconda3/envs/omat25/bin/python',
                   '/cluster/dataX/xqyi/labsuite-20261110/review_recorded_ppe.py']
        result = subprocess.run(command, input=json.dumps({'action': action, 'body': body}),
            encoding='utf-8', capture_output=True, timeout=100,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result.check_returncode()
        line = next(line for line in result.stdout.splitlines() if line.startswith('PPE_RESULT '))
        payload = json.loads(line.removeprefix('PPE_RESULT '))
        handler._send_json(payload, status=400 if 'error' in payload else 200)
    except (ValueError, TypeError, KeyError):
        handler._send_json({'error': '截图或标注数据无效'}, status=400)
    except (OSError, subprocess.SubprocessError, StopIteration):
        handler._send_json({'error': '集群录像审核服务暂不可用'}, status=503)
    finally:
        LOCK.release()


def serve_recording(handler, action):
    import re
    from urllib.error import HTTPError
    from urllib.parse import parse_qs, urlsplit
    from urllib.request import build_opener, ProxyHandler

    if action == 'ppe-prepared':
        path = ROOT / 'runtime/ppe-prepared-recordings.json'
        handler._send_json(json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else [])
        return
    if action == 'ppe-prepared-all':
        query = parse_qs(urlsplit(handler.path).query)
        recording = query.get('recording', [''])[0]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}/[A-Za-z0-9_-]{1,80}', recording):
            handler._send_json({'error': '无效的录像会话'}, status=400)
            return
        subject, session = recording.split('/')
        directory = ROOT / 'runtime/ppe-prepared-all' / subject
        source = query.get('source', [''])[0]
        if source:
            if not re.fullmatch(r'(cam_1|cam_2|cam_entrance)-[0-9]{1,9}', source):
                handler._send_json({'error': '无效的候选照片'}, status=400)
                return
            path = directory / session / (source + '.json')
            if not path.is_file():
                handler._send_json({'error': '候选原图不存在'}, status=404)
                return
        else:
            path = directory / (session + '.json')
        if path.is_file():
            handler._send_file(path, 'application/json; charset=utf-8')
        else:
            handler._send_json([])
        return
    listing = action == 'ppe-recordings'
    if listing:
        path = '/api/recordings'
    else:
        match = re.fullmatch(r'ppe-recording-frame/([\w-]{1,80})/([\w-]{1,80})/(cam_1|cam_2|cam_entrance)', action, re.ASCII)
        index = parse_qs(urlsplit(handler.path).query).get('index', [''])[0]
        if not match or not re.fullmatch(r'[0-9]{1,9}', index):
            handler._send_json({'error': '无效的录像或帧编号'}, status=400)
            return
        subject, session, camera = match.groups()
        path = f'/api/recordings/{subject}/{session}/frame/{camera}?index={index}'
    try:
        with build_opener(ProxyHandler({})).open(handler.server.api_base.rstrip('/')+path, timeout=12) as response:
            limit = 8 * 1024 * 1024
            data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError('recording response too large')
            if listing:
                handler._send_json(json.loads(data))
            elif data.startswith(b'\xff\xd8'):
                handler._send_bytes(data, 'image/jpeg')
            else:
                raise ValueError('invalid JPEG')
    except HTTPError as error:
        handler._send_json({'error': '该录像帧无法读取'}, status=error.code)
    except (OSError, ValueError):
        handler._send_json({'error': '录像服务暂不可用'}, status=502)
