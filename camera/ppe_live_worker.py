"""Cluster GPU live PPE observer. No notification permissions or fabricated confidence."""
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parent


def main():
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from ppe_detector import MaskDetector
    torch.set_num_threads(2)
    credentials = json.loads((ROOT / 'runtime/handoff-credentials.json').read_text())
    model_config = json.loads((ROOT / 'config/ppe_observer.json').read_text())
    mask = MaskDetector(ROOT / 'runtime/vendor_aizoo/FaceMaskDetection-master')
    model_path = model_config['coat_model']
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    coat = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
        device_map='cuda:0', local_files_only=True, attn_implementation='sdpa').eval()
    opener = build_opener(ProxyHandler({}))
    headers = {'Authorization': 'Bearer ' + credentials['image_read_token']}
    base = 'http://127.0.0.1:25177'
    lock = threading.Lock()
    state = {'mode': 'observe', 'started_at': time.time(), 'frames': 0,
             'observations': 0, 'cameras': {}, 'notification_eligible': False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/health', '/api/ppe/status'):
                self.send_error(404)
                return
            with lock:
                body = json.dumps(state, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 15004), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seen = {}
    try:
        while True:
            for camera in ('cam_1', 'cam_2', 'cam_entrance'):
                try:
                    with opener.open(Request(base + '/api/handoff/frozen/' + camera,
                                             headers=headers), timeout=5) as response:
                        frozen = json.load(response)
                    if frozen['source_frame_id'] == seen.get(camera):
                        continue
                    snapshot = frozen['snapshot_id']
                    if not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', snapshot):
                        raise ValueError('snapshot key invalid')
                    with opener.open(Request(base + '/api/handoff/snapshots/' + snapshot + '.jpg',
                                             headers=headers), timeout=5) as response:
                        jpeg = response.read(10 * 1024 * 1024 + 1)
                    if hashlib.sha256(jpeg).hexdigest() != frozen['image_sha256']:
                        raise ValueError('snapshot hash mismatch')
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None or list(frame.shape[1::-1]) != [frozen['frame_width'], frozen['frame_height']]:
                        raise ValueError('snapshot shape mismatch')
                    if time.time() - frozen['source_captured_at'] > 15:
                        raise ValueError('stale source frame')
                    if frozen.get('image_transform') != 'identity' or not frozen.get('full_frame'):
                        raise ValueError('original full-frame snapshot required')
                    rows = []
                    for raw_track in frozen['camera'].get('yolo_tracks', []):
                        box = raw_track.get('box', [])
                        if len(box) != 4:
                            continue
                        x, y, width, height = map(float, box)
                        if x < 0 or y < 0 or width <= 0 or height <= 0 or x+width > frame.shape[1] or y+height > frame.shape[0]:
                            raise ValueError('source-frame box invalid')
                        tid = raw_track.get('global_id') or raw_track.get('track_id')
                        if not tid:
                            continue
                        identities = [item for item in frozen['camera'].get('identities', [])
                            if item.get('local_id') == raw_track.get('local_id')
                            and item.get('global_track_id') == tid and item.get('identity_lock_status') == 'locked']
                        track = {'track_id': tid, 'person_id': identities[0].get('person_id') if len(identities)==1 else None,
                                 'bbox': [x, y, x+width, y+height]}
                        x1, y1, x2, y2 = map(int, track['bbox'])
                        crop = frame[max(0, y1):min(frame.shape[0], y2),
                                     max(0, x1):min(frame.shape[1], x2)]
                        if crop.size == 0:
                            continue
                        mask_result = mask.observe(crop)
                        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        inputs = processor.apply_chat_template([{'role': 'user', 'content': [
                            {'type': 'image', 'image': image}, {'type': 'text', 'text':
                             '判断此人是否穿着长款实验室白大褂。普通白色衣服不算。看不清回答未知。只回答：实验服=已穿戴或未穿戴或未知。'}]}],
                            tokenize=True, add_generation_prompt=True, return_dict=True,
                            return_tensors='pt', processor_kwargs={'downsample_mode': '4x',
                            'max_slice_nums': 1, 'use_image_id': False}).to(coat.device)
                        with torch.inference_mode():
                            output = coat.generate(**inputs, downsample_mode='4x',
                                                   max_new_tokens=32, do_sample=False)
                        raw = processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                                     skip_special_tokens=True)[0].strip()
                        match = re.fullmatch(r'实验服\s*[=：:]\s*(已穿戴|未穿戴|未知)[。.]?', raw)
                        rows.append({'track_id': track['track_id'], 'person_id': track.get('person_id'),
                            'bbox': track['bbox'], 'mask': mask_result,
                            'lab_coat': {'已穿戴': 'worn', '未穿戴': 'not_worn', '未知': 'unknown'}.get(match[1] if match else '', 'unknown'),
                            'coat_confidence': None, 'coat_raw': raw})
                    if rows:
                        audit = ROOT / 'runtime/ppe-observation-evidence'
                        audit.mkdir(exist_ok=True)
                        slot = state['observations'] % 60
                        name = camera + '-' + str(slot)
                        (audit / (name + '.jpg')).write_bytes(jpeg)
                        (audit / (name + '.json')).write_text(json.dumps({
                            'snapshot_id': snapshot, 'source_frame_id': frozen['source_frame_id'],
                            'source_captured_at': frozen['source_captured_at'],
                            'image_sha256': frozen['image_sha256'], 'tracks': rows}, ensure_ascii=False))
                    seen[camera] = frozen['source_frame_id']
                    with lock:
                        state['frames'] += 1
                        state['observations'] += len(rows)
                        state['updated_at'] = time.time()
                        state['cameras'][camera] = {'snapshot_id': snapshot,
                            'source_frame_id': frozen['source_frame_id'],
                            'source_captured_at': frozen['source_captured_at'],
                            'observed_at': time.time(), 'time_domain': 'source_unverified',
                            'status': 'observed', 'tracks': rows}
                except Exception as exc:
                    with lock:
                        state['cameras'][camera] = {'status': 'unavailable',
                            'error_type': type(exc).__name__, 'observed_at': time.time(), 'tracks': []}
            with lock:
                serialized = json.dumps(state, ensure_ascii=False, indent=2)
            path = ROOT / 'runtime/ppe-observer-status.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(serialized)
            temporary.replace(path)
            time.sleep(1)
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
