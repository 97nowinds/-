import os,sys,pathlib,json,gzip,ast,types,contextlib,io
os.environ['CUDA_VISIBLE_DEVICES']='0'
root=pathlib.Path(os.environ.get('LAB_CAMERA_ROOT','/cluster/home/xqyi/lab-safety-host/camera'));sys.path.insert(0,str(root))
import torch,cv2,numpy as np
from face_identity import ModernFaceEngine
from face_attempt_audit import scale_box
cv2.setNumThreads(2)
with contextlib.redirect_stdout(io.StringIO()):engine=ModernFaceEngine(root/'models/insightface')
assert engine.available and all('CUDAExecutionProvider' in m.session.get_providers() for m in engine.analysis.models.values())
refs={};samples={}
for d in (root/'data/known_faces').iterdir():
 if not d.is_dir():continue
 pid=d.name.split('_')[0];vs=[np.load(f,allow_pickle=False).reshape(-1) for f in sorted(d.glob('*.npy'))]
 if vs:samples.setdefault(pid,[]).extend(vs)
for pid,vs in samples.items():
 m=np.mean(vs,axis=0);refs[pid]=m/np.linalg.norm(m)
tree=ast.parse((root/'app.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CameraWorker');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='detect_face_in_person_box');ns={'cv2':cv2};exec(compile(ast.Module(body=[fn],type_ignores=[]),'roi','exec'),ns)
w=types.SimpleNamespace(face_store=types.SimpleNamespace(modern=engine,align_face=lambda *a:None))
rows=[json.loads(l) for l in gzip.open(root/'runtime/native-face-20261027/normal-single/detections.jsonl.gz','rt')]
folder=root/'data/recordings/normal_single_2k_20261027/20261027_141441'

from face_identity import FaceCapturePolicy
policy=FaceCapturePolicy(min_side=64,min_sharpness=60)
train=[];training_ids=[];cap=cv2.VideoCapture(str(folder/'cam_1.mp4'))
for row in rows:
 if row['camera_id']!='cam_1' or row['seconds']>=50 or not row['tracks']:continue
 idx=row['frame_index'];t=max(row['tracks'],key=lambda t:t['box'][2]*t['box'][3]);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read()
 if not ok:continue
 found=ns['detect_face_in_person_box'](w,im,scale_box(t['box'],(540,960,3),im.shape))
 if not found or not policy.assess(found[1])['accepted']:continue
 q=found[3];q=q/np.linalg.norm(q);rank=sorted([(float(q@v),pid) for pid,v in refs.items()],reverse=True)
 if rank[0][1]!='person001' or rank[0][0]<.30 or rank[0][0]-rank[1][0]<.08:continue
 if any(float(q@v)>.995 for v in train):continue
 train.append(q);training_ids.append(idx)
 if len(train)==20:break
cap.release();assert train
combined=np.mean(np.array(samples['person001']+train),axis=0);combined/=np.linalg.norm(combined)
adapt=np.mean(train,axis=0);adapt/=np.linalg.norm(adapt)
paired=json.loads((root/'runtime/native-face-20261027/normal-single/paired/normal_single_2k_20261027.json').read_text())
evalrows=[a for a in paired['attempts'] if a['mode']=='recorded_native' and (a['camera_id']!='cam_1' or a['seconds']>=60)]
lookup={(r['camera_id'],r['frame_index']):r for r in rows};records=[]
for a in evalrows:
 cid,idx=a['camera_id'],a['frame_index'];assert not (cid=='cam_1' and idx in training_ids)
 row=lookup[(cid,idx)];t=next(t for t in row['tracks'] if t['track_id']==a['local_id']);cap=cv2.VideoCapture(str(folder/(cid+'.mp4')));cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read();cap.release()
 if not ok:continue
 found=ns['detect_face_in_person_box'](w,im,scale_box(t['box'],(540,960,3),im.shape))
 if not found or not policy.assess(found[1])['accepted']:continue
 q=found[3];q=q/np.linalg.norm(q);base=float(q@refs['person001']);other=max(float(q@v) for pid,v in refs.items() if pid!='person001');new=float(q@combined);site=float(q@adapt)
 records.append({'camera':cid,'frame':idx,'base':base,'append20':new,'site_centroid_observe_only':site,'other_best':other,'base_pass':base>=.55 and base-other>=.12,'append_pass':new>=.55 and new-other>=.12})
summary={}
for cid in ['cam_1','cam_2','cam_entrance']:
 rr=[x for x in records if x['camera']==cid];summary[cid]={'quality_accepted_heldout':len(rr),'base_pass':sum(x['base_pass'] for x in rr),'append_pass':sum(x['append_pass'] for x in rr),'mean_base':float(np.mean([x['base'] for x in rr])) if rr else None,'mean_append':float(np.mean([x['append20'] for x in rr])) if rr else None}
result={'training_count':len(train),'training_camera':'cam_1','training_before_seconds':50,'training_frame_ids':training_ids,'summary':summary,'heldout':records,'limitations':['Model-suggested person001 pseudo-labels, not independently verified identities.','Training one passage; heldout cam1 >=60sec and other cameras are disjoint.','Only offline gallery simulation; live gallery unchanged.','Candidate pass counts are not accuracy.']}
out=root/'runtime/site-gallery-simulation-20261029.json';out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='heldout'},indent=2))
