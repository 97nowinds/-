"""Read-only face-stage audit using actual recorded tracks and registered gallery.
No app import, no gallery writes, no identity assignment or notifications.
"""
import argparse,ast,json,time,threading,collections,types,sys
from pathlib import Path
import cv2
import torch  # Match the live app import order and preload its CUDA/cuDNN libraries.
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from face_identity import FaceIdentityStore,FaceCapturePolicy,ModernFaceEngine

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--tracks',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--interval',type=float,default=.8);p.add_argument('--face-max-width',type=int,default=960);args=p.parse_args()
 root=Path.cwd();model=ModernFaceEngine(root/'models/insightface');assert model.available,model.error
 store=FaceIdentityStore.__new__(FaceIdentityStore);store.lock=threading.RLock();store.capture_policy=FaceCapturePolicy();store.required_engine='arcface';store.modern=model;store.sface=None;store.people=json.loads((args.data/'people.json').read_text());store.models={k:None for k in store.people};store.modern_features={};store.query_features=collections.defaultdict(lambda:collections.deque(maxlen=5))
 for pid in store.people:
  values=[np.load(f,allow_pickle=False) for d in (args.data/'known_faces').glob(pid+'_*') for f in d.glob('*.npy')]
  if len(values)>=5:
   v=np.mean(values,axis=0);store.modern_features[pid]=v/np.linalg.norm(v)
 tree=ast.parse((root/'app.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CameraWorker');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='detect_face_in_person_box');ns={'cv2':cv2};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_camera_face_roi','exec'),ns)
 worker=types.SimpleNamespace(face_store=store);detect=types.MethodType(ns['detect_face_in_person_box'],worker)
 cap=cv2.VideoCapture(str(args.session/'cam_entrance.mp4'));last=-1e10;last_frame=-1;attempts=[];votes=collections.defaultdict(lambda:collections.deque(maxlen=7));last_seen={};known_at={};last_checked={};native=[]
 rows=[json.loads(s) for s in args.tracks.read_text().splitlines() if s.strip()];rows=[r for r in rows if r['camera_id']=='cam_entrance']
 for row in rows:
  tracks=row['tracks'];sec=row['seconds'];active={t['track_id'] for t in tracks}
  for tid in list(last_seen):
   if tid not in active and sec-last_seen[tid]>1.5:
    store.forget_track(('cam_entrance',tid));votes.pop(tid,None);known_at.pop(tid,None);last_seen.pop(tid,None)
  for tid in active:last_seen[tid]=sec
  if not tracks or sec-last<args.interval:continue
  last=sec;unknown=[t for t in tracks if t['track_id'] not in known_at];target=min(unknown or tracks,key=lambda t:(last_checked.get(t['track_id'],0),-t['confidence']*t['box'][2]*t['box'][3]));tid=target['track_id'];last_checked[tid]=sec
  idx=row['frame_index']
  if idx!=last_frame+1:cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
  ok,original=cap.read();last_frame=idx
  if not ok:raise RuntimeError('missing recorded frame')
  frame=original
  if args.face_max_width and frame.shape[1]>args.face_max_width:frame=cv2.resize(frame,(args.face_max_width,int(frame.shape[0]*args.face_max_width/frame.shape[1])))
  source_width=min(960,original.shape[1]);scale=frame.shape[1]/source_width
  box=tuple(int(round(v*scale)) for v in target['box']);start=time.perf_counter();detection=detect(frame,box);detect_ms=(time.perf_counter()-start)*1000
  out={'seconds':sec,'frame_index':idx,'local_id':tid,'global_id':target['global_id'],'held_track':row['held'],'detect_embed_ms':round(detect_ms,3),'result':'no_face'}
  if detection is not None:
   face_box,face,aligned,emb=detection;st=time.perf_counter();result=store.predict(face,track_key=('cam_entrance',tid),query_embedding=emb);out.update(result='known_prelock' if result['known'] else result.get('reason') or 'score_or_margin_rejected',face_box=face_box,quality=result.get('quality'),similarity=result.get('similarity') if result.get('similarity') is not None else (1-result['distance'] if result.get('distance') is not None else None),margin=result.get('margin'),predict_ms=round((time.perf_counter()-st)*1000,3),person_id=result.get('person_id'),feature_frames=result.get('feature_frames'))
   if result['known']:
    votes[tid].append(result['person_id']);winner,n=collections.Counter(votes[tid]).most_common(1)[0]
    if n>=3 and winner==result['person_id']:known_at.setdefault(tid,sec);out['three_vote_at']=known_at[tid]
   else:votes[tid].append(None)
  attempts.append(out)
  if len(native)<30:
   sx=original.shape[1]/frame.shape[1];sy=original.shape[0]/frame.shape[0];nb=tuple(int(round(v*(sx if j%2==0 else sy))) for j,v in enumerate(box));st=time.perf_counter();nd=detect(original,nb);rec={'seconds':sec,'processed_result':out['result'],'native_video_dimensions':[original.shape[1],original.shape[0]],'detect_ms':round((time.perf_counter()-st)*1000,3),'face':nd is not None}
   if nd:rec.update(face_box=nd[0],quality=store.capture_policy.assess(nd[1]))
   native.append(rec)
 cap.release();times=[a['detect_embed_ms'] for a in attempts[1:]]
 report={'replay_only':True,'interval_seconds':args.interval,'face_max_width':args.face_max_width,'limit':'Recorded entrance video, sampled using recorded local tracks and 0.8s schedule; not exact online scheduling, no identity-lock or cross-camera proof; native video is not original full-resolution RTSP.','providers':{k:v.session.get_providers() for k,v in model.analysis.models.items()},'gallery_people':len(store.modern_features),'attempts':attempts,'reason_counts':dict(collections.Counter(a['result'] for a in attempts)),'detect_embed_ms_excluding_first':{'median':float(np.median(times)) if times else None,'p95':float(np.percentile(times,95)) if times else None},'native_video_comparison':native}
 args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('attempts','native_video_comparison')},ensure_ascii=True))
if __name__=='__main__':main()
