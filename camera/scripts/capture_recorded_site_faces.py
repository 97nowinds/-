"""GPU-only replay capture using the live site sampler's ROI and quality rules.
Writes reviewed candidates only; never enrolls or assigns a definitive identity.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys,json,gzip,collections,time,uuid,shutil,datetime
from pathlib import Path
ROOT=Path(os.environ.get('LAB_CAMERA_ROOT','/cluster/home/xqyi/lab-safety-host/camera'))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,"/cluster/dataX/xqyi/lab-camera-python")
import torch,cv2,numpy as np
from face_identity import ModernFaceEngine,FaceCapturePolicy
from face_adaptation import SiteSamples
from face_attempt_audit import scale_box

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    engine=ModernFaceEngine(ROOT/'models/insightface')
    assert engine.available and all('CUDAExecutionProvider' in m.session.get_providers() for m in engine.analysis.models.values()), 'CUDA required'
    policy=FaceCapturePolicy(min_side=64,min_sharpness=800)
    people=json.loads((ROOT/'data/people.json').read_text());refs={}
    for pid in people:
        vs=[np.load(p,allow_pickle=False).reshape(-1) for d in (ROOT/'data/known_faces').glob(pid+'_*') for p in d.glob('*.npy')]
        if vs:
            v=np.mean(vs,axis=0);refs[pid]=v/np.linalg.norm(v)
    recording='normal_single_2k_20261027/20261027_141441'
    folder=ROOT/'data/recordings'/recording
    rows=[json.loads(l) for l in gzip.open(ROOT/'runtime/native-face-20261027/normal-single/detections.jsonl.gz','rt')]
    out=ROOT/'runtime/recorded-site-faces-20261030';out.mkdir(exist_ok=True)
    clock=[time.time()];stage=SiteSamples(out/'candidates',wall=lambda:clock[0]);sid=stage.start()
    caps={cid:cv2.VideoCapture(str(folder/(cid+'.mp4'))) for cid in ['cam_1','cam_2','cam_entrance']}
    indices={cid:-1 for cid in caps};counts=collections.Counter();accepted=[]
    try:
        for n,row in enumerate(rows):
            cid=row['camera_id'];idx=row['frame_index'];cap=caps[cid]
            if row.get('held'):counts['held_skipped']+=1;continue
            while indices[cid]<idx:
                ok,im=cap.read();indices[cid]+=1
                if not ok:break
            if not ok:counts['decode_failed']+=1;continue
            counts['frames']+=1
            for t in row.get('tracks',[])[:6]:
                counts['person_rois']+=1
                x,y,w,h=map(int,scale_box(t['box'],(540,960,3),im.shape))
                left=max(0,x-int(w*.2));top=max(0,y-int(h*.04));right=min(im.shape[1],x+w+int(w*.2));bottom=min(im.shape[0],y+int(h*.64))
                roi=im[top:bottom,left:right]
                if roi.size==0:continue
                faces=engine.analysis.get(roi)
                if len(faces)!=1:counts['not_one_face']+=1;continue
                f=faces[0];a,b,c,d=map(int,f.bbox);a=max(0,a);b=max(0,b);c=min(roi.shape[1],c);d=min(roi.shape[0],d)
                crop=roi[b:d,a:c];quality=policy.assess(crop)
                if not quality['accepted'] or float(f.det_score)<.65:counts[quality.get('reason') or 'low_detection']+=1;continue
                q=np.asarray(f.normed_embedding,dtype=np.float32);ranking=sorted([(float(q@v),pid) for pid,v in refs.items()],reverse=True)
                best,pid=ranking[0] if ranking else (0,None)
                suggested=pid if best>=.35 and (len(ranking)<2 or best-ranking[1][0]>=.08) else None
                k=f.kps;eye=max(float(np.linalg.norm(k[1]-k[0])),1);vertical=float((k[2][1]-(k[0][1]+k[1][1])/2)/eye);roll=float(abs(k[1][1]-k[0][1])/eye)
                pose=('鼻眼比例低' if vertical<.35 else '鼻眼比例中' if vertical<.65 else '鼻眼比例高')+(' / 倾斜' if roll>.2 else '')
                quality.update(source_kind='recording',recording=recording,video_seconds=round(row['seconds'],3),frame_index=idx,timestamp_basis='recording start label + video offset; not verified capture clock')
                if max(crop.shape[:2])>512:crop=cv2.resize(crop,(int(crop.shape[1]*512/max(crop.shape[:2])),int(crop.shape[0]*512/max(crop.shape[:2]))))
                jpg=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tobytes()
                accepted.append((cid,idx,t['track_id'],suggested,best,pose,row['seconds'],quality,jpg,q))
            if n%200==0:(out/'progress.json').write_text(json.dumps({'processed':n,'rows':len(rows),'counts':dict(counts),'quality_candidates':len(accepted)}))
        # Keep high quality frames while enforcing source-video spacing and visual diversity.
        chosen=collections.defaultdict(list)
        for item in sorted(accepted,key=lambda a:a[7]['sharpness'],reverse=True):
            cid,idx,tid,suggested,best,pose,sec,quality,jpg,q=item
            bucket=('suggested:'+suggested) if suggested else cid+':track:'+str(tid)
            if any(abs(sec-s)<60 for s in chosen[(bucket,cid)]):counts['source_interval']+=1;continue
            clock[0]+=61 # monotonic staging clock; source-video spacing was checked above
            result=stage.add(sid,cid,'recording:'+recording+':'+str(idx),tid,suggested,best,pose,clock[0],quality,jpg,q)
            counts['stage_'+result]+=1
            if result=='saved':chosen[(bucket,cid)].append(sec)
        live=SiteSamples(ROOT/'data/face_adaptation');imported=[]
        with live.lock,live.db:
            live.db.execute('INSERT INTO sessions VALUES(?,?,?,0)',(sid,time.time(),time.time()))
            for r in stage.db.execute('SELECT * FROM samples'):
                if live.db.execute('SELECT count(*) FROM samples').fetchone()[0]>=1000:break
                if live.db.execute('SELECT 1 FROM samples WHERE camera=? AND source_frame=? AND track=?',(r['camera'],r['source_frame'],r['track'])).fetchone():counts['already_imported']+=1;continue
                for ext in ('.jpg','.npy'):shutil.copy2(stage.root/(r['id']+ext),live.root/(r['id']+ext))
                values=dict(r);quality=json.loads(values['quality'])
                start=datetime.datetime(2026,10,27,14,14,41,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
                values['captured']=start+quality['video_seconds']
                live.db.execute('INSERT INTO samples VALUES('+','.join('?' for _ in values)+')',tuple(values.values()));imported.append(r['id'])
        result={'recording':recording,'counts':dict(counts),'imported':len(imported),'ids':imported,'quality_min':800,'identity':'manual review required','live_session_unchanged':True,'capture_time':'recording folder label + video offset, not verified hardware capture time'}
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        for c in caps.values():c.release()
if __name__=='__main__':main()
