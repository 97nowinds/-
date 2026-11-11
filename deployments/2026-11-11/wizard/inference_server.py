"""Loopback-only setup wizard and serialized fresh DINOv2/SAM job API.

Old preview masks/preview.json are not served. Every POST creates a new run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import traceback
from urllib.parse import unquote, urlsplit
import uuid
from json_io import read_json, write_json
from capture_store import CaptureStore, CAPTURE_SCENE, public_scene, scene_digest
from scene_set_store import SceneSetStore
from camera_capture import CameraService, CameraError
from review_store import ReviewStore, ReviewError, ReviewConflict

ROOT = Path(__file__).resolve().parent
FRONTEND = {'index.html', 'styles.css', 'app.js', 'favicon.svg',
    'viewer_motion.js', 'viewer_motion.css', 'wizard_runtime.css'}
RUNTIME_PYTHON = ROOT.parents[1] / 'runtime/python/python.exe'
JOB_PATTERN = re.compile(r'[a-f0-9]{32}')
CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'none'"


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def public_failure_message(message):
    if any(token in message for token in ('[WinError 5]', '[WinError 32]', '[WinError 33]')):
        return '任务文件暂时无法访问，本次识别已停止。请重新识别；详细原因已保存在任务日志中。'
    return message[:500]


class Jobs:
    """One active process; no hidden parallel GPU inference or result cache."""
    def __init__(self, root, config):
        self.root = root; self.config = config; self.lock = threading.Lock(); self.active = None
        # Terminal error delivery only; never contains masks or successful
        # inference results. Prevent a dead worker from looking "running" if
        # a permanent disk error prevents its final status from being saved.
        self.failed_statuses = {}
        (self.root / 'runs').mkdir(exist_ok=True)

    def create(self, scene_id, input_scene=None):
        with self.lock:
            if self.active is not None:
                return None, dict(error='已有推理任务正在运行，请等待当前任务完成。',
                    activeJobId=self.active['id'], sceneId=self.active['sceneId'])
            if input_scene is None:
                input_scene = next(s for s in self.config['scenes'] if s['id'] == scene_id)
            job_id = uuid.uuid4().hex; run = self.root / 'runs' / job_id
            run.mkdir()
            status = dict(id=job_id, sceneId=scene_id, status='queued', stage='queued',
                message='新任务已创建，准备提交到集群', createdAt=datetime.now(timezone.utc).isoformat())
            write_json(run / 'status.json', status)
            write_json(run / 'request.json', dict(id=job_id, sceneId=scene_id,
                freshInferenceRequested=True, input=input_scene, inputSha256=scene_digest(input_scene)))
            self.active = status.copy()
            threading.Thread(target=self.work, args=(job_id, scene_id), daemon=True).start()
            return status, None

    def work(self, job_id, scene_id):
        run = self.root / 'runs' / job_id
        try:
            from cluster_inference import run_cluster
            status = read_json(run / 'status.json')
            status.update(status='running', stage='cluster_inference', message='正在集群识别，结果返回后可复核')
            write_json(run / 'status.json', status)
            with (run / 'worker.log').open('wb') as log:
                code = run_cluster(self.root, job_id, scene_id, log)
            status = read_json(run / 'status.json')
            if code != 0 and status.get('status') != 'failed':
                raise RuntimeError('推理进程异常退出，请查看本次任务日志并重试。')
            if code == 0 and (status.get('status') != 'succeeded' or not (run / 'result.json').is_file()):
                raise RuntimeError('推理进程未产生完整结果。')
        except Exception as error:
            trace = traceback.format_exc()
            try:
                (run / 'server-error.log').write_text(trace, encoding='utf-8')
            except OSError:
                pass
            try:
                status = read_json(run / 'status.json')
            except (OSError, ValueError):
                status = dict(id=job_id, sceneId=scene_id)
            status.update(status='failed', stage='failed', message=public_failure_message(str(error)),
                          updatedAt=datetime.now(timezone.utc).isoformat())
            with self.lock:
                self.failed_statuses[job_id] = status
                if len(self.failed_statuses) > 128:
                    self.failed_statuses.pop(next(iter(self.failed_statuses)))
            try:
                write_json(run / 'status.json', status)
            except OSError:
                # get() can still return the terminal failure above.
                pass
        finally:
            with self.lock:
                self.active = None

    def get(self, job_id):
        if not JOB_PATTERN.fullmatch(job_id):
            return None
        with self.lock:
            if job_id in self.failed_statuses:
                return self.failed_statuses[job_id].copy()
        run = self.root / 'runs' / job_id
        try:
            status = read_json(run / 'status.json')
            if status.get('status') == 'failed':
                status['message'] = public_failure_message(str(status.get('message', '本次识别未完成，可以重试。')))
            if status.get('status') == 'succeeded':
                result = read_json(run / 'result.json')
                if result.get('provenance', {}).get('freshInference') is not True:
                    return dict(id=job_id, status='failed', stage='verification_failed', message='结果缺少新推理来源记录。')
                request_path = run / 'request.json'
                if request_path.is_file():
                    request = read_json(request_path)
                    if 'input' in request:
                        frozen = request['input']
                        if (request.get('id') != job_id or result.get('id') != job_id
                                or result.get('provenance', {}).get('jobId') != job_id
                                or request.get('sceneId') != frozen.get('id')
                                or result.get('sceneId') != frozen.get('id')
                                or request.get('inputSha256') != scene_digest(frozen)
                                or result.get('provenance', {}).get('inputSceneSha256') != request['inputSha256']
                                or public_scene(result.get('scene', {})) != public_scene(frozen)
                                or result.get('provenance', {}).get('capture') != frozen.get('capture')):
                            return dict(id=job_id, status='failed', stage='verification_failed', message='识别结果与本次输入画面不一致。')
                return dict(status, **result)
            return status
        except (OSError, ValueError):
            return None
        except (KeyError, TypeError):
            return dict(id=job_id, status='failed', stage='verification_failed', message='任务来源或结果记录不完整，请重新识别。')


def make_handler(root=ROOT, cameras=None):
    root = Path(root).resolve()
    config = read_json(root / 'data/inference_config.json')
    jobs = Jobs(root, config)
    captures = CaptureStore(root)
    scene_sets = SceneSetStore(root, config)
    reviews = ReviewStore(root)
    cameras = cameras if cameras is not None else CameraService(root)
    assets = {}
    for scene in config['scenes']:
        for field, digest in [('image', 'sha256'), ('thumbnail', 'thumbnailSha256')]:
            relative = scene[field].lstrip('/')
            if not relative.startswith('assets/') or not (root / relative).resolve().is_relative_to(root / 'assets'):
                raise ValueError('Invalid raw scene whitelist')
            assets[relative] = scene[digest]

    class Handler(BaseHTTPRequestHandler):
        server_version = 'InstrumentSetupFresh/2'
        protocol_version = 'HTTP/1.1'

        def log_message(self, format, *args):
            pass

        def local_request(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            return self.client_address[0] in ('127.0.0.1', '::1') and self.headers.get('Host') in allowed

        def send_payload(self, payload, content_type, status=200, head=False):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', CSP)
            self.end_headers()
            if not head:
                self.wfile.write(payload)

        def json(self, value, status=200, head=False):
            self.send_payload(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                'application/json; charset=utf-8', status, head)

        def decoded_path(self):
            try:
                decoded = unquote(urlsplit(self.path).path, errors='strict')
            except (ValueError, UnicodeError):
                return None
            if any(c in decoded for c in ('\\', '\x00', ':')) or any(p in ('.', '..') for p in decoded.split('/')):
                return None
            return decoded

        def do_GET(self):
            self.serve(False)

        def do_HEAD(self):
            self.serve(True)

        def serve(self, head):
            if not self.local_request():
                self.json(dict(error='Loopback requests only'), 403, head); return
            decoded = self.decoded_path()
            if decoded is None:
                self.json(dict(error='Not found'), 404, head); return
            if decoded == '/api/scenes':
                scenes = [{k: s[k] for k in ('id', 'title', 'description', 'image', 'thumbnail', 'width', 'height')}
                          for s in config['scenes']]
                self.json(dict(mode='fresh_inference', defaultSceneId='query_cam2', scenes=scenes,
                    notice='点击生成后将在本机实际重新运行 DINOv2 和 SAM 2.1；所有候选需人工确认。'), head=head); return
            if decoded == '/api/cameras':
                self.json(dict(cameras=cameras.list_cameras()), head=head); return
            if decoded == '/api/scene-sets':
                try:
                    self.json(dict(sceneSets=scene_sets.list_sets()), head=head)
                except (OSError, ValueError):
                    self.json(dict(error='已保存画面暂时无法读取，请重试。'), 503, head)
                return
            match = re.fullmatch(r'/api/cameras/(cam_[12])/frame', decoded)
            if match:
                if head:
                    self.json(dict(error='请使用 GET 获取实时帧'), 405, head); return
                try:
                    frame = cameras.latest(match.group(1), timeout=8)
                    if not 0 <= time.monotonic() - frame.received_monotonic <= 2:
                        raise ValueError('实时画面已断开，请重新连接摄像头。')
                    self.send_payload(frame.preview_jpeg, 'image/jpeg')
                except (CameraError, ValueError) as error:
                    self.json(dict(error=str(error)), 503)
                return
            match = re.fullmatch(r'/captures/([a-f0-9]{32})/frame\.jpg', decoded)
            if match:
                try:
                    snapshot = captures.load('capture_' + match.group(1))
                    payload = (root / snapshot['image'].lstrip('/')).read_bytes()
                    if sha_bytes(payload) != snapshot['sha256']:
                        raise ValueError('Capture changed')
                    self.send_payload(payload, 'image/jpeg', head=head)
                except (OSError, ValueError, KeyError, TypeError):
                    self.json(dict(error='截取画面不存在或校验失败'), 404, head)
                return
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})', decoded)
            if match:
                result = jobs.get(match.group(1))
                self.json(result or dict(error='任务不存在'), 200 if result else 404, head); return
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/review', decoded)
            if match:
                try:
                    self.json(reviews.read_state(match.group(1)), head=head)
                except ReviewError as error:
                    self.json(dict(error=str(error)), 409, head)
                except (OSError, ValueError, KeyError, TypeError):
                    self.json(dict(error='这次任务的确认配置暂时无法读取，请检查识别结果。'), 404, head)
                return
            relative = 'index.html' if decoded in ('', '/') else decoded.lstrip('/')
            expected = assets.get(relative)
            if relative not in FRONTEND and expected is None:
                match = re.fullmatch(r'runs/([a-f0-9]{32})/assets/([a-zA-Z0-9_-]+\.(?:png|jpg))', relative)
                if not match:
                    self.json(dict(error='Not found'), 404, head); return
                run = root / 'runs' / match.group(1)
                try:
                    status = jobs.get(match.group(1))
                    if not status or status.get('status') != 'succeeded':
                        raise ValueError('Job not complete')
                    manifest = read_json(run / 'asset_manifest.json')
                    row = next(r for r in manifest['files'] if r['path'] == 'assets/'+match.group(2))
                    expected = row['sha256']
                except (OSError, ValueError, StopIteration, KeyError):
                    self.json(dict(error='Not found'), 404, head); return
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                self.json(dict(error='Not found'), 404, head); return
            try:
                payload = path.read_bytes()
            except OSError:
                self.json(dict(error='Not found'), 404, head); return
            if expected is not None and sha_bytes(payload) != expected:
                self.json(dict(error='Asset verification failed'), 409, head); return
            content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            if path.suffix in {'.html', '.css', '.js', '.json', '.svg'}:
                content_type += '; charset=utf-8'
            self.send_payload(payload, content_type, head=head)

        def do_POST(self):
            if not self.local_request():
                self.json(dict(error='Loopback requests only'), 403); return
            endpoint = self.decoded_path()
            review_match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/review', endpoint or '')
            if not review_match and endpoint not in {'/api/jobs', '/api/captures', '/api/scene-sets', '/api/scene-sets/capture', '/api/cameras/connect', '/api/cameras/retry'}:
                self.json(dict(error='Not found'), 404); return
            origin = self.headers.get('Origin')
            if origin is not None and origin not in (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
                self.json(dict(error='Cross-origin requests denied'), 403); return
            if self.headers.get('Transfer-Encoding') is not None:
                self.close_connection = True
                self.json(dict(error='Unsupported request framing'), 400); return
            if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                self.json(dict(error='Content-Type must be application/json'), 415); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                limit = 65536 if review_match else 1024
                if not 1 <= length <= limit:
                    self.close_connection = True
                    self.json(dict(error=f'Request body limit is {limit} bytes'), 413); return
                self.connection.settimeout(5)
                body = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(body, dict):
                    raise ValueError('Invalid JSON object')
            except (ValueError, UnicodeError, OSError, TypeError):
                self.json(dict(error='Invalid JSON request'), 400); return
            if review_match:
                try:
                    self.json(reviews.save(review_match.group(1), body), 201)
                except ReviewConflict as error:
                    self.json(dict(error=str(error)), 409)
                except ReviewError as error:
                    self.json(dict(error=str(error)), 400)
                except (ValueError, KeyError, TypeError):
                    self.json(dict(error='识别来源或所选区域不完整，请重新检查本次结果。'), 400)
                except OSError:
                    self.json(dict(error='确认配置暂时无法保存，请稍后重试。'), 503)
                return
            if endpoint == '/api/cameras/connect':
                if set(body) != {'password'} or not isinstance(body['password'], str) or not 1 <= len(body['password']) <= 256:
                    self.json(dict(error='请输入摄像头连接密码'), 400); return
                try:
                    cameras.set_password(body['password'])
                    self.json(dict(cameras=cameras.list_cameras()))
                except (CameraError, ValueError) as error:
                    self.json(dict(error=str(error)), 400)
                return
            if endpoint == '/api/cameras/retry':
                if set(body) != {'cameraId'} or not isinstance(body['cameraId'], str) or body['cameraId'] not in {'cam_1', 'cam_2'}:
                    self.json(dict(error='请选择有效的摄像头'), 400); return
                try:
                    cameras.retry(body['cameraId'])
                    self.json(dict(cameras=cameras.list_cameras()))
                except (CameraError, ValueError) as error:
                    self.json(dict(error=str(error)), 503)
                return
            if endpoint == '/api/captures':
                if set(body) != {'cameraId'} or not isinstance(body['cameraId'], str) or body['cameraId'] not in {'cam_1', 'cam_2'}:
                    self.json(dict(error='请选择有效的摄像头'), 400); return
                try:
                    requested = time.monotonic()
                    sequence = cameras.current_sequence(body['cameraId'])
                    frame = cameras.latest(body['cameraId'], timeout=8, after_sequence=sequence)
                    info = next(s for s in cameras.list_cameras() if s['id'] == body['cameraId'])
                    snapshot = captures.create(frame, requested_monotonic=requested, title=info['title'])
                    self.json(dict(scene=public_scene(snapshot)), 201)
                except (CameraError, ValueError) as error:
                    self.json(dict(error=str(error)), 503)
                except OSError:
                    self.json(dict(error='截取画面未能保存，请重新截取。'), 503)
                return
            if endpoint == '/api/scene-sets/capture':
                if (set(body) != {'name'} or not isinstance(body['name'], str)
                        or not 1 <= len(body['name'].strip()) <= 60):
                    self.json(dict(error='请提供 1 至 60 个字符的画面套名称。'), 400); return
                # One software trigger for both cameras. This does not claim
                # hardware-synchronized exposures. Each camera's first fresh
                # decoded frame is frozen by its own worker immediately, so
                # waiting for the other camera cannot make that frame stale.
                requested = time.monotonic()
                try:
                    camera_ids = ('cam_1', 'cam_2')
                    sequences = {camera_id: cameras.current_sequence(camera_id)
                                 for camera_id in camera_ids}
                    camera_info = {item['id']: item for item in cameras.list_cameras()}

                    def freeze_camera(camera_id):
                        frame = cameras.latest(camera_id, timeout=8,
                                               after_sequence=sequences[camera_id])
                        if (frame.camera_id != camera_id
                                or type(frame.sequence) is not int
                                or frame.sequence <= sequences[camera_id]):
                            raise ValueError('摄像头帧来源或序列不一致')
                        return captures.create(frame, requested_monotonic=requested,
                                               title=camera_info[camera_id]['title'])

                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='paired-capture') as pool:
                        pending = [pool.submit(freeze_camera, camera_id) for camera_id in camera_ids]
                        pair = [future.result() for future in pending]
                    group = scene_sets.create(body['name'], [item['id'] for item in pair])
                    self.json(dict(sceneSet=group), 201)
                except Exception:
                    # Device/decoder errors may contain connection details.
                    # Never publish them or silently fill a missing view with
                    # a previous snapshot. No incomplete set is committed.
                    self.json(dict(error='两个视角未能完整截取保存，请检查摄像头连接后重试。'), 503)
                return
            if endpoint == '/api/scene-sets':
                if set(body) != {'name', 'sceneIds'}:
                    self.json(dict(error='请提供画面套名称和两个视角的画面'), 400); return
                try:
                    self.json(dict(sceneSet=scene_sets.create(body['name'], body['sceneIds'])), 201)
                except ValueError as error:
                    self.json(dict(error=str(error)), 400)
                except OSError:
                    self.json(dict(error='这套画面未能保存，请重试。'), 503)
                return
            if set(body) != {'sceneId'} or not isinstance(body['sceneId'], str):
                self.json(dict(error='请选择有效的输入画面'), 400); return
            source = next((s for s in config['scenes'] if s['id'] == body['sceneId']), None)
            if source is None and CAPTURE_SCENE.fullmatch(body['sceneId']):
                try:
                    source = captures.load(body['sceneId'])
                except (OSError, ValueError, KeyError, TypeError):
                    self.json(dict(error='截取画面不存在或已改变，请重新截取。'), 400); return
            if source is None:
                self.json(dict(error='请选择有效的输入画面'), 400); return
            status, busy = jobs.create(body['sceneId'], source)
            self.json(busy if busy else status, 409 if busy else 202)

        def deny_write(self):
            self.send_response(405)
            self.send_header('Allow', 'GET, HEAD, POST')
            self.send_header('Content-Length', '0')
            self.end_headers()

        do_PUT = do_PATCH = do_DELETE = do_OPTIONS = deny_write

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8888)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    cameras = CameraService(ROOT)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(cameras=cameras))
    server.daemon_threads = True
    print(f'Setup wizard with fresh inference: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cameras.close()


if __name__ == '__main__':
    main()
