"""Offline PPE candidate extraction and operator-confirmed sample storage."""
import base64
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/cluster/home/xqyi/lab-safety-host/camera')
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data/ppe_samples'


def decode(value):
    data = base64.b64decode(value, validate=True)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or max(image.shape[:2]) > 3000:
        raise ValueError('无效的截图')
    return data, image


def jpeg(image):
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('截图编码失败')
    return base64.b64encode(encoded).decode()


def fingerprint(image):
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (17, 16))
    return small[:, 1:] > small[:, :-1]


def sharpness(image):
    height, width = image.shape[:2]
    scaled = cv2.resize(image, (min(width, 160), min(height, 160)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


def scan(body):
    from ultralytics import YOLO
    from ppe_detector import MaskDetector
    import torch

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    detector = YOLO(str(ROOT / 'models/yolov8n.pt'))
    classifier = YOLO(str(ROOT / 'runtime/coat-yolov8-cls-20261109/run-fp32/weights/best.pt'))
    mask = MaskDetector(ROOT / 'runtime/vendor_aizoo/FaceMaskDetection-master', device) if device != 'cpu' else None
    automatic = body.get('automatic') is True
    candidates, fingerprints = [], []
    for frame in body['frames']:
        _, image = decode(frame['jpeg'])
        boxes = detector.predict(image, classes=[0], conf=.4, device=device, verbose=False)[0].boxes.xyxy.tolist()
        if len(boxes) != 1:
            continue
        x, y, right, bottom = map(int, boxes[0])
        crop = image[y:bottom, x:right]
        if min(crop.shape[:2]) < 48:
            continue
        sig = fingerprint(crop)
        similar = max((float(np.mean(sig == prior)) for prior in fingerprints), default=0)
        fingerprints.append(sig)
        head = crop[:max(1, round(crop.shape[0] * .55))]
        p = classifier.predict(crop, imgsz=320, device=device, verbose=False)[0]
        observation = mask.observe(crop) if mask else {'state': 'unknown'}
        mask_hint = observation['state']
        face_quality = None
        if mask_hint == 'worn':
            a,b,c,d = map(int, observation['bbox'])
            face = crop[b:d,a:c]
            if min(face.shape[:2]) >= 36:
                face_quality = sharpness(face)
        torso = crop[round(crop.shape[0]*.18):round(crop.shape[0]*.7)]
        coat_quality = sharpness(torso)
        coat_hint = p.names[p.probs.top1]
        coat_score = float(p.probs.top1conf)
        matches = []
        if face_quality is not None and face_quality >= 65:
            matches.append('mask')
        if min(torso.shape[:2]) >= 80 and coat_quality >= 55 and coat_hint == 'worn' and coat_score >= .55:
            matches.append('lab_coat')
        if automatic and not matches:
            continue
        candidates.append(dict(second=frame['second'], source=frame['jpeg'], person=jpeg(crop),
            head=jpeg(head), bbox=[x,y,right-x,bottom-y], similarity=round(similar,3),
            mask=mask_hint, lab_coat=coat_hint, coat_score=coat_score, matches=matches,
            face_quality=face_quality, coat_quality=coat_quality,
            fingerprint=np.packbits(sig).tobytes().hex()))
    return {'candidates': candidates}


def save(body):
    camera = body['camera']
    if camera not in {'cam_1','cam_2','cam_entrance'}:
        raise ValueError('请选择录像对应摄像头')
    labels = body['labels']
    if not labels or set(labels) - {'mask','lab_coat'} or any(
            value not in {'worn','not_worn','unknown'} for value in labels.values()):
        raise ValueError('请选择人工标签')
    source, image = decode(body['source'])
    x,y,w,h = body['bbox']
    if any(type(v) is not int for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<1 or x+w>image.shape[1] or y+h>image.shape[0]:
        raise ValueError('人员框超出截图')
    digest = hashlib.sha256(source).hexdigest()
    sample = hashlib.sha256(f'{camera}:{digest}:{x}:{y}:{w}:{h}'.encode()).hexdigest()[:32]
    directory = DATA / 'images'
    directory.mkdir(exist_ok=True)
    paths = [directory / f'{sample}_{suffix}.jpg' for suffix in ('source','person','head_region')]
    crop = image[y:y+h,x:x+w]
    now = time.time()
    provenance = json.dumps(dict(source_kind='manual_video_review', source_group=body['recording'][:150],
                                 video_second=body['second'], capture_clock_available=False), ensure_ascii=False)
    with sqlite3.connect(DATA/'samples.sqlite3',timeout=15) as db:
        db.execute('BEGIN IMMEDIATE')
        if not db.execute('SELECT 1 FROM samples WHERE sample_id=?',(sample,)).fetchone():
            paths[0].write_bytes(source)
            for path,img in zip(paths[1:],(crop,crop[:max(1,round(h*.55))])):
                path.write_bytes(base64.b64decode(jpeg(img)))
            db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (sample,camera,'replay:'+digest,'','replay:single',digest,
                 *(str(p.relative_to(DATA)) for p in paths),json.dumps(body['bbox']),'{}',provenance,'unassigned',now))
        for item,state in labels.items():
            db.execute('INSERT INTO labels VALUES(?,?,?,?,?,?,?) ON CONFLICT(sample_id,item) DO UPDATE SET '
                       'state=excluded.state,visibility=excluded.visibility,basis=excluded.basis,'
                       'annotation_source=excluded.annotation_source,annotated_at=excluded.annotated_at',
                       (sample,item,state,'not_reliably_visible' if state=='unknown' else 'visible',
                        'operator_reviewed_video_crop','user_visual_review',now))
    return {'sample_id':sample}


if __name__=='__main__':
    message=json.load(sys.stdin)
    try:
        result={'scan':scan,'save':save}[message['action']](message['body'])
    except (ValueError, KeyError, TypeError) as error:
        result={'error':str(error)}
    print('PPE_RESULT '+json.dumps(result,ensure_ascii=True),flush=True)
