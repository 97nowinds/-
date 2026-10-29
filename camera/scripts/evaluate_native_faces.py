"""Paired read-only GPU face audit on fixed recorded body detections.

Never imports the application, sends events, writes galleries or exports vectors.
The recorded native frame may already be smaller than the original camera stream.
"""
import argparse
import ast
from collections import Counter, defaultdict, deque
import gzip
import json
from pathlib import Path
import sys
import threading
import time
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--session', type=Path)
    p.add_argument('--detections', type=Path)
    p.add_argument('--limit-seconds', type=float, default=300)
    args = p.parse_args()
    if bool(args.session) != bool(args.detections):
        p.error('--session and --detections must be provided together')
    sys.path.insert(0, str(args.source))
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    from face_identity import FaceIdentityStore, FaceCapturePolicy, ModernFaceEngine
    from face_attempt_audit import scale_box, warm_face_models
    engine = ModernFaceEngine(args.source / 'models/insightface')
    assert engine.available, engine.error
    assert 'CUDAExecutionProvider' in engine.providers
    warm_face_models(engine)
    people = json.loads((args.source / 'data/people.json').read_text())
    refs = {}
    for pid in people:
        values = [np.load(f, allow_pickle=False)
                  for d in (args.source/'data/known_faces').glob(pid+'_*') for f in d.glob('*.npy')]
        if len(values) >= 5:
            mean = np.mean(values, axis=0)
            refs[pid] = mean / np.linalg.norm(mean)
    assert refs, 'No valid registered gallery'
    def store():
        s = FaceIdentityStore.__new__(FaceIdentityStore)
        s.lock = threading.RLock(); s.capture_policy = FaceCapturePolicy()
        s.required_engine = 'arcface'; s.modern = engine; s.sface = None
        s.people = people; s.models = {k: None for k in people}; s.modern_features = refs
        s.query_features = defaultdict(lambda: deque(maxlen=5))
        return s
    tree = ast.parse((args.source/'app.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CameraWorker')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'detect_face_in_person_box')
    ns = {'cv2': cv2}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_face_roi', 'exec'), ns)
    args.output.mkdir(parents=True, exist_ok=True)
    sessions = [('base01','20260921_172359',60), ('base04','20261009_142701',90),
                ('kadun','20261009_141659',90), ('yxq_side_20261009','20261009_134334',100),
                ('kaoqing','20261010_142755',60), ('multi_person_20261016','20261016_183228',180)]
    if args.session:
        sessions = [(args.session.parent.name, args.session.name, args.limit_seconds)]
    summaries = []
    for group, stamp, limit in sessions:
        directory = args.session or args.source/'data/recordings'/group/stamp
        info = json.loads((directory/'info.json').read_text(encoding='utf-8-sig'))
        cache = args.detections or args.source/'runtime/hub-candidate4-20261019'/f'candidate-{group}-{stamp}.jsonl.gz'
        stores = {mode: store() for mode in ('preview960','recorded_native')}
        captures = {cid: cv2.VideoCapture(str(directory/s['video_file'])) for cid,s in info['streams'].items()}
        indices = {cid: -1 for cid in captures}; frames = {}
        last, checked, seen = defaultdict(lambda: -1e9), {}, {}
        votes = defaultdict(lambda: deque(maxlen=7)); first = {}; attempts = []
        workers = {mode: types.SimpleNamespace(face_store=s) for mode,s in stores.items()}
        with gzip.open(cache, 'rt', encoding='utf-8-sig') as lines:
            for line in lines:
                row = json.loads(line); cid = row['camera_id']; sec = row['seconds']
                if sec > limit: break
                if row.get('held'): continue
                tracks = row['tracks']; active = {t['track_id'] for t in tracks}
                for key in list(seen):
                    if key[0] == cid and key[1] not in active and sec-seen[key] > 1.5:
                        for mode,s in stores.items():
                            s.forget_track(key); votes.pop((mode,*key), None)
                        seen.pop(key)
                for tid in active: seen[(cid,tid)] = sec
                if not tracks or sec-last[cid] < .8: continue
                last[cid] = sec
                # Fixed paired target selection: both resolutions see exactly the same person/frame.
                target = min(tracks, key=lambda t:(checked.get((cid,t['track_id']),-1e9),
                             -t['confidence']*t['box'][2]*t['box'][3]))
                tid = target['track_id']; checked[(cid,tid)] = sec
                idx = int(row['frame_index'])
                while indices[cid] < idx:
                    ok, image = captures[cid].read()
                    if not ok: raise RuntimeError(f'Missing frame {cid} {idx}')
                    indices[cid] += 1; frames[cid] = image
                native = frames[cid]
                preview = native if native.shape[1] <= 960 else cv2.resize(native,(960,int(native.shape[0]*960/native.shape[1])))
                for mode,image in [('preview960',preview),('recorded_native',native)]:
                    box = scale_box(target['box'], preview.shape, image.shape)
                    start = time.perf_counter()
                    det = ns['detect_face_in_person_box'](workers[mode], image, box)
                    out = {'camera_id':cid,'local_id':tid,'seconds':sec,'frame_index':idx,
                           'mode':mode,'dimensions':[image.shape[1],image.shape[0]],'reason':'no_face'}
                    if det is not None:
                        fb, face, aligned, emb = det
                        result = stores[mode].predict(face,track_key=(cid,tid),query_embedding=emb)
                        reason = 'known_candidate' if result['known'] else result.get('reason') or 'score_or_margin_rejected'
                        out.update(reason=reason,face_size_px=list(fb[2:]),person_id=result.get('person_id'),
                                   similarity=result.get('similarity'),margin=result.get('margin'))
                        key = (mode,cid,tid); votes[key].append(result.get('person_id') if result['known'] else None)
                        if result['known']:
                            winner,n = Counter(votes[key]).most_common(1)[0]
                            if winner == result['person_id'] and n >= 3:
                                first.setdefault((mode,cid,tid,result['person_id']),sec)
                    out['total_ms'] = round((time.perf_counter()-start)*1000,3)
                    attempts.append(out)
        for cap in captures.values(): cap.release()
        cameras = {}
        for cid in captures:
            cameras[cid] = {}
            for mode in stores:
                subset = [a for a in attempts if a['camera_id']==cid and a['mode']==mode]
                times = [a['total_ms'] for a in subset]
                confirmations = [{'local_id':t,'person_id':pid,'seconds':sec}
                                 for (m,c,t,pid),sec in first.items() if m==mode and c==cid]
                cameras[cid][mode] = {'attempts':len(subset),'reasons':dict(Counter(a['reason'] for a in subset)),
                    'confirmed_track_person_pairs':len(confirmations),'first_confirmations':confirmations,
                    'median_ms':float(np.median(times)) if times else None,
                    'p95_ms':float(np.percentile(times,95)) if times else None}
        summary = {'session':group,'limit_seconds':limit,'cameras':cameras,
                   'recorded_dimensions':{c:[s['width'],s['height']] for c,s in info['streams'].items()}}
        (args.output/f'{group}.json').write_text(json.dumps({'summary':summary,'attempts':attempts},indent=2))
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'gallery_people':len(refs),'sessions':summaries,
        'limitations':['Fixed saved body detections; no detector or ReID rerun.',
            'Paired fresh-frame round-robin at 0.8s, not an exact online scheduler.',
            'Known candidates and three-vote confirmations are before global identity locks.',
            'No identity ground truth: these are not accuracy or continuity measurements.',
            'Recorded native resolution is not the 2560x1440 live source.']},indent=2))


if __name__ == '__main__': main()
