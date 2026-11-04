"""Read-only paired gallery replay on saved single-person body tracks."""
from pathlib import Path
import os,sys,json,gzip,ast,types,sqlite3,threading,time,hashlib,collections,contextlib,io
import cv2,numpy as np,torch
ROOT=Path('/cluster/home/xqyi/lab-safety-host/camera');sys.path.insert(0,str(ROOT));OUT=ROOT/'runtime/face-gallery-ab-20261104';OUT.mkdir(exist_ok=True);torch.set_num_threads(2);cv2.setNumThreads(2)
from face_identity import FaceIdentityStore,FaceCapturePolicy,ModernFaceEngine
from face_attempt_audit import scale_box,warm_face_models
from identity_revalidation import face_owned_by_body
recording='normal_single_2k_20261027/20261027_141441';folder=ROOT/'data/recordings'/recording;people=json.loads((ROOT/'data/people.json').read_text());base={};hashes={}
for pid in people:
 vals=[]
 for d in sorted((ROOT/'data/known_faces').glob(pid+'_*')):
  for p in sorted(d.glob('*.npy')):
   data=p.read_bytes();hashes[str(p)]=hashlib.sha256(data).hexdigest();v=np.load(p,allow_pickle=False).reshape(-1);assert v.size==512 and np.isfinite(v).all();vals.append(v)
 if vals:base[pid]=vals
c=sqlite3.connect('file:'+str(ROOT/'data/face_adaptation/samples.sqlite3')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;add=collections.defaultdict(list);add_manifest=[];excluded=[]
for row in c.execute("select * from samples where status='approved' order by person,captured"):
 d=dict(row);quality=json.loads(d['quality'])
 if quality.get('recording')==recording or recording in d['source_frame']:
  excluded.append({'id':d['id'],'person':d['person'],'reason':'same_recording_as_test'});continue
 if float(quality.get('sharpness',0))<800:excluded.append({'id':d['id'],'reason':'below_current_capture_quality'});continue
 p=ROOT/'data/face_adaptation'/(d['id']+'.npy');v=np.load(p,allow_pickle=False).reshape(-1);assert v.size==512 and np.isfinite(v).all();add[d['person']].append(v);add_manifest.append({'id':d['id'],'person':d['person'],'camera':d['camera'],'source_frame':d['source_frame'],'feature_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
refs={}
for mode in ['current','target_added','all_added']:
 refs[mode]={}
 for pid,values in base.items():
  vs=values+(add[pid] if mode=='all_added' or (mode=='target_added' and pid=='person001') else []);m=np.mean(vs,axis=0);refs[mode][pid]=m/np.linalg.norm(m)
engine=ModernFaceEngine(ROOT/'models/insightface');assert engine.available and all('CUDAExecutionProvider' in m.session.get_providers() for m in engine.analysis.models.values());warm_face_models(engine)
def store(ref):
 s=FaceIdentityStore.__new__(FaceIdentityStore);s.lock=threading.RLock();s.capture_policy=FaceCapturePolicy();s.required_engine='arcface';s.modern=engine;s.sface=None;s.people=people;s.models={k:None for k in people};s.modern_features=ref;s.query_features=collections.defaultdict(lambda:collections.deque(maxlen=5));return s
stores={mode:store(ref) for mode,ref in refs.items()};source=(ROOT/'app.py').read_text();tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CameraWorker');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='detect_face_in_person_box');ns={'cv2':cv2};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_roi','exec'),ns);worker=types.SimpleNamespace(face_store=types.SimpleNamespace(modern=engine,align_face=lambda *a:None));rows=[json.loads(l) for l in gzip.open(ROOT/'runtime/native-face-20261027/normal-single/detections.jsonl.gz','rt')]
info=json.loads((folder/'info.json').read_text());caps={cid:cv2.VideoCapture(str(folder/s['video_file'])) for cid,s in info['streams'].items()};indices={cid:-1 for cid in caps};images={};last=collections.defaultdict(lambda:-1e9);checked={};seen={};votes=collections.defaultdict(list);vote_start={};bindings={mode:{} for mode in stores};confirmations={mode:[] for mode in stores};attempts=[];coverage=[];started=time.time()
for ri,row in enumerate(rows):
 cid,sec,idx=row['camera_id'],row['seconds'],int(row['frame_index'])
 if row.get('held'):continue
 tracks=row['tracks'];active={t['track_id'] for t in tracks}
 for key in list(seen):
  if key[0]==cid and key[1] not in active and sec-seen[key]>1.5:
   for mode,s in stores.items():s.forget_track(key);bindings[mode].pop(key,None);votes.pop((mode,*key),None);vote_start.pop((mode,*key),None)
   seen.pop(key)
 for tid in active:seen[(cid,tid)]=sec
 if tracks and sec-last[cid]>=.8:
  last[cid]=sec;target=min(tracks,key=lambda t:(checked.get((cid,t['track_id']),-1e9),-t['confidence']*t['box'][2]*t['box'][3]));tid=target['track_id'];key=(cid,tid);checked[key]=sec
  while indices[cid]<idx:
   ok,im=caps[cid].read()
   if not ok:raise RuntimeError('Missing source frame')
   indices[cid]+=1;images[cid]=im
  frame=images[cid];box=scale_box(target['box'],(540,960,3),frame.shape);peers=[scale_box(t['box'],(540,960,3),frame.shape) for t in tracks if t['track_id']!=tid];det=ns['detect_face_in_person_box'](worker,frame,box);reason='no_face';quality=None
  if det and not face_owned_by_body(det[0],box,peers):det=None;reason='ambiguous_face_owner'
  record={'camera_id':cid,'local_id':tid,'seconds':sec,'frame_index':idx,'models':{}}
  for mode,s in stores.items():
   vk=(mode,*key)
   if not det:res={'known':False,'reason':reason}
   else:res=s.predict(det[1],track_key=key,query_embedding=det[3])
   pid=res.get('person_id') if res['known'] else None
   if not votes[vk] or sec-vote_start.get(vk,sec)>10 or votes[vk][-1]!=pid:votes[vk]=[];vote_start[vk]=sec
   if pid:votes[vk].append(pid)
   else:votes[vk]=[]
   if pid and len(votes[vk])>=3:
    if bindings[mode].get(key)!=pid:
     bindings[mode][key]=pid;confirmations[mode].append({'camera_id':cid,'local_id':tid,'person_id':pid,'seconds':sec,'frame_index':idx})
   record['models'][mode]={'known':res['known'],'person_id':pid,'similarity':res.get('similarity'),'margin':res.get('margin'),'reason':'known_candidate' if pid else res.get('reason') or 'score_or_margin_rejected','feature_frames':res.get('feature_frames'),'consecutive_votes':len(votes[vk]),'confirmed_identity':bindings[mode].get(key)}
  attempts.append(record)
 coverage.append({'camera_id':cid,'seconds':sec,'frame_index':idx,'tracks':len(tracks),'correct_bound_tracks':{mode:sum(bindings[mode].get((cid,t['track_id']))=='person001' for t in tracks) for mode in stores},'wrong_bound_tracks':{mode:sum(bindings[mode].get((cid,t['track_id'])) not in [None,'person001'] for t in tracks) for mode in stores}})
 if ri%200==0:(OUT/'progress.json').write_text(json.dumps({'rows':ri,'total_rows':len(rows),'attempts':len(attempts),'elapsed_seconds':round(time.time()-started,1)}));print('PROGRESS',ri,len(rows),len(attempts),round(time.time()-started,1),flush=True)
for cap in caps.values():cap.release()
summary={}
for cid in caps:
 aa=[a for a in attempts if a['camera_id']==cid];cc=[a for a in coverage if a['camera_id']==cid];summary[cid]={}
 for mode in stores:
  mm=[a['models'][mode] for a in aa];den=sum(a['tracks'] for a in cc);correct=sum(a['correct_bound_tracks'][mode] for a in cc);wrong=sum(a['wrong_bound_tracks'][mode] for a in cc);conf=[a for a in confirmations[mode] if a['camera_id']==cid]
  summary[cid][mode]={'attempts':len(aa),'correct_candidate_passes':sum(a['known'] and a['person_id']=='person001' for a in mm),'other_identity_passes':sum(a['known'] and a['person_id']!='person001' for a in mm),'candidate_pass_over_all_attempts':sum(a['known'] and a['person_id']=='person001' for a in mm)/len(mm) if mm else None,'reason_counts':dict(collections.Counter(a['reason'] for a in mm)),'three_consecutive_confirmations':conf,'detected_track_samples':den,'correct_confirmed_track_samples':correct,'wrong_confirmed_track_samples':wrong,'offline_confirmed_track_sample_coverage':correct/den if den else None}
for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'Live gallery changed during audit'
report={'recording':recording,'recording_seconds':info['elapsed_seconds'],'expected_identity':'person001','expected_identity_basis':'previous single-person recording audit and user route context, not a new exhaustive frame-level annotation','gallery_status':'approved samples not enrolled; temporary in-memory comparison only','gallery_base_counts':{pid:len(vs) for pid,vs in base.items()},'approved_added_counts':{pid:len(vs) for pid,vs in add.items()},'included_samples':add_manifest,'excluded_samples':excluded,'same_recording_samples_excluded_from_gallery':True,'thresholds':{'similarity':.55,'margin':.12,'query_history':5,'consecutive_votes':3,'vote_window_seconds':10},'summary':summary,'attempts':attempts,'coverage':coverage,'confirmations':confirmations,'elapsed_seconds':round(time.time()-started,2),'providers':engine.providers,'gallery_unchanged':True,'limitations':['Saved body tracks are fixed; detector and ReID were not retrained or changed.','Coverage is offline three-consecutive-vote coverage of saved body-track samples before global identity locks, not exact online name display or frame-annotated identity accuracy.','This single-person positive recording cannot measure multi-person false identity rate.','Four approved reference images originating from this test recording were excluded from all expanded galleries.','The temporary gallery is not enrolled or reloaded into production.']};(OUT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('DONE',json.dumps({'summary':summary,'added_counts':report['approved_added_counts'],'excluded':len(excluded)},ensure_ascii=True),flush=True)
