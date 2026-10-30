import os,sys,pathlib,json,gzip,ast,types,collections,threading,contextlib,io
os.environ['CUDA_VISIBLE_DEVICES']='0'
r=pathlib.Path(os.environ.get('LAB_CAMERA_ROOT','/cluster/home/xqyi/lab-safety-host/camera'));sys.path.insert(0,str(r))
import torch,cv2,numpy as np
torch.set_num_threads(2);cv2.setNumThreads(2)
from face_identity import ModernFaceEngine,FaceCapturePolicy
from face_attempt_audit import scale_box
with contextlib.redirect_stdout(io.StringIO()):engine=ModernFaceEngine(r/'models/insightface')
assert all('CUDAExecutionProvider' in m.session.get_providers() for m in engine.analysis.models.values())
refs={}
for d in (r/'data/known_faces').iterdir():
 if not d.is_dir():continue
 vs=[np.load(f,allow_pickle=False).reshape(-1) for f in d.glob('*.npy')]
 if vs:
  v=np.mean(vs,axis=0);refs[d.name.split('_')[0]]=v/np.linalg.norm(v)
pids=list(refs);mat=np.array([refs[p] for p in pids]);policy=FaceCapturePolicy()
tree=ast.parse((r/'app.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CameraWorker');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='detect_face_in_person_box');ns={'cv2':cv2};exec(compile(ast.Module(body=[fn],type_ignores=[]),'roi','exec'),ns)
w=types.SimpleNamespace(face_store=types.SimpleNamespace(modern=engine,align_face=lambda *a:None))
paired=json.loads((r/'runtime/native-face-20261027/normal-single/paired/normal_single_2k_20261027.json').read_text())
attempts=[a for a in paired['attempts'] if a['mode']=='recorded_native']
rows={(x['camera_id'],x['frame_index']):x for x in [json.loads(l) for l in gzip.open(r/'runtime/native-face-20261027/normal-single/detections.jsonl.gz','rt')]}
folder=r/'data/recordings/normal_single_2k_20261027/20261027_141441'
caps={c:cv2.VideoCapture(str(folder/(c+'.mp4'))) for c in ['cam_1','cam_2','cam_entrance']};indices={c:-1 for c in caps}
hist5=collections.defaultdict(lambda:collections.deque(maxlen=5));hist3=collections.defaultdict(lambda:collections.deque(maxlen=3));last={};votes=collections.defaultdict(lambda:collections.deque(maxlen=7));records=[]
for a in attempts:
 cid,idx,tid,sec=a['camera_id'],a['frame_index'],a['local_id'],a['seconds'];key=(cid,tid);row=rows[(cid,idx)];t=next(x for x in row['tracks'] if x['track_id']==tid)
 while indices[cid]<idx:
  ok,im=caps[cid].read();assert ok;indices[cid]+=1
 box=scale_box(t['box'],(540,960,3),im.shape);found=ns['detect_face_in_person_box'](w,im,box)
 rec={'camera':cid,'frame':idx,'seconds':sec,'track':tid,'modes':{}}
 if key in last and sec-last[key]>3:hist3[key].clear()
 last[key]=sec
 if not found:rec['reason']='no_face';records.append(rec);continue
 fb,face,_,q=found;quality=policy.assess(face)
 if not quality['accepted']:rec['reason']=quality['reason'];records.append(rec);continue
 q=q/np.linalg.norm(q);h=hist5[key]
 if h:
  cent=np.mean(h,axis=0);cent/=np.linalg.norm(cent)
  if float(q@cent)<.65:h.clear()
 h.append(q);hist3[key].append((idx,q));single=mat@q;v=np.mean(h,axis=0);v/=np.linalg.norm(v)
 scores={'single':single,'current_feature5':mat@v}
 if len(hist3[key])==3:scores['three_score_mean']=np.mean([mat@v for _,v in hist3[key]],axis=0)
 for mode,ss in scores.items():
  ids=np.argsort(ss)[::-1];best=float(ss[ids[0]]);margin=best-float(ss[ids[1]]);passed=best>=.55 and margin>=.12;pid=pids[ids[0]];vk=(mode,*key);votes[vk].append(pid if passed else None)
  confirmed=passed and collections.Counter(votes[vk])[pid]>=3
  rec['modes'][mode]={'score':round(best,5),'margin':round(margin,5),'candidate':pid,'passed':passed,'confirmed':confirmed}
 records.append(rec)
for c in caps.values():c.release()
summary={}
for cid in caps:
 summary[cid]={}
 for mode in ['single','current_feature5','three_score_mean']:
  subset=[x for x in records if x['camera']==cid];scored=[x for x in subset if mode in x['modes']];passed=[x for x in scored if x['modes'][mode]['passed']];confirmed={(x['track'],x['modes'][mode]['candidate']) for x in scored if x['modes'][mode]['confirmed']}
  summary[cid][mode]={'attempts':len(subset),'scored_windows':len(scored),'passed':len(passed),'confirmed_track_person_pairs':len(confirmed),'max_score':max([x['modes'][mode]['score'] for x in scored],default=None)}
out=r/'runtime/face-three-score-20261029';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps({'summary':summary,'records':records,'limitations':['No manual identity ground truth; candidate counts are not accuracy.','Three-score means require three distinct accepted frames and max3s gap.','Current feature5 reproduced selected fixed targets; not complete online scheduler or global lock.']},indent=2));print(json.dumps(summary,indent=2))
