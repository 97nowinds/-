"""Loopback GPU0 sampling sidecar; bounded images, human enrollment approval."""
import base64
import hmac
import json
import os
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parent


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    import cv2
    import numpy as np
    from flask import Flask,request,jsonify,send_file
    from face_identity import ModernFaceEngine,FaceCapturePolicy
    from face_adaptation import SiteSamples
    torch.set_num_threads(2);cv2.setNumThreads(2)
    engine=None
    policy=FaceCapturePolicy(min_side=64,min_sharpness=800)
    store=SiteSamples(ROOT/'data/face_adaptation')
    token=(ROOT/'runtime/face-adaptation.token').read_text().strip()
    image_token=os.environ['LAB_HANDOFF_IMAGE_READ_TOKEN']
    def people():return json.loads((ROOT/'data/people.json').read_text())
    def refs():
        result={}
        for pid in people():
            vs=[np.load(p,allow_pickle=False).reshape(-1) for d in (ROOT/'data/known_faces').glob(pid+'_*') for p in d.glob('*.npy')]
            if vs:
                q=np.mean(vs,axis=0);result[pid]=q/np.linalg.norm(q)
        return result
    references=refs()
    def poll():
        nonlocal engine
        while True:
            session=store.active()
            if session:
                if engine is None:
                    try:
                        engine=ModernFaceEngine(ROOT/'models/insightface')
                        if not engine.available or not all('CUDAExecutionProvider' in m.session.get_providers() for m in engine.analysis.models.values()):
                            raise RuntimeError('GPU provider required')
                    except Exception as exc:
                        engine=None;store.error=type(exc).__name__;time.sleep(15);continue
                for camera in ['cam_1','cam_2','cam_entrance']:
                    try:
                        req=Request('http://127.0.0.1:5000/api/handoff/frozen/'+camera,headers={'Authorization':'Bearer '+image_token})
                        with urlopen(req,timeout=3) as response:frozen=json.load(response)
                        if frozen.get('timestamp_source')!='host_receive' or not 0<=time.time()-frozen['source_captured_at']<=5:continue
                        im=cv2.imdecode(np.frombuffer(base64.b64decode(frozen['jpeg_base64']),np.uint8),cv2.IMREAD_COLOR)
                        if im is None:continue
                        tracks=frozen['camera'].get('yolo_tracks',[])[:6]
                        for t in tracks:
                            x,y,w,h=map(int,t['box']);left=max(0,x-int(w*.2));top=max(0,y-int(h*.04))
                            right=min(im.shape[1],x+w+int(w*.2));bottom=min(im.shape[0],y+int(h*.64))
                            roi=im[top:bottom,left:right]
                            if roi.size==0:continue
                            faces=engine.analysis.get(roi)
                            if len(faces)!=1:continue
                            f=faces[0];a,b,c,d=map(int,f.bbox);a=max(0,a);b=max(0,b);c=min(roi.shape[1],c);d=min(roi.shape[0],d)
                            crop=roi[b:d,a:c];quality=policy.assess(crop)
                            if not quality['accepted'] or float(f.det_score)<.65:continue
                            q=np.asarray(f.normed_embedding,dtype=np.float32);ranking=sorted([(float(q@v),pid) for pid,v in references.items()],reverse=True)
                            best,pid=ranking[0] if ranking else (0,None)
                            suggested=pid if best>=.35 and (len(ranking)<2 or best-ranking[1][0]>=.08) else None
                            k=f.kps;eye=max(float(np.linalg.norm(k[1]-k[0])),1);vertical=float((k[2][1]-(k[0][1]+k[1][1])/2)/eye)
                            roll=float(abs(k[1][1]-k[0][1])/eye)
                            pose=('鼻眼比例低' if vertical<.35 else '鼻眼比例中' if vertical<.65 else '鼻眼比例高')+(' / 倾斜' if roll>.2 else '')
                            if max(crop.shape[:2])>512:crop=cv2.resize(crop,(int(crop.shape[1]*512/max(crop.shape[:2])),int(crop.shape[0]*512/max(crop.shape[:2]))))
                            jpeg=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tobytes()
                            store.add(session['id'],camera,frozen['source_frame_id'],t.get('local_id',t.get('track_id')),suggested,best,pose,frozen['source_captured_at'],quality,jpeg,q)
                        store.error=None
                    except Exception as exc:store.error=type(exc).__name__
            elif engine is not None:
                engine=None
                import gc
                gc.collect();torch.cuda.empty_cache()
            time.sleep(15)
    threading.Thread(target=poll,daemon=True).start()
    app=Flask(__name__)
    @app.before_request
    def auth():
        if not hmac.compare_digest(request.headers.get('Authorization',''),'Bearer '+token):return jsonify(error='unauthorized'),401
    @app.get('/status')
    def status():return jsonify(store.status(people()))
    @app.get('/images/<sid>.jpg')
    def image(sid):
        import re
        if not re.fullmatch('[a-f0-9]{32}',sid):return jsonify(error='not found'),404
        return send_file(store.root/(sid+'.jpg'),mimetype='image/jpeg',max_age=0)
    @app.post('/start')
    def start():return jsonify(session_id=store.start())
    @app.post('/stop')
    def stop():store.stop();return jsonify(ok=True)
    @app.post('/review')
    def review():
        data=request.get_json();store.review(data.get('ids'),data.get('person_id'),people(),data.get('reject') is True)
        return jsonify(ok=True)
    @app.post('/enroll')
    def enroll():
        data=request.get_json()
        if data.get('confirmed') is not True:return jsonify(error='explicit confirmation required'),400
        result=store.enroll(data.get('person_id'),ROOT/'data/known_faces',people())
        # A protected reload request changes only gallery memory, not the worker/model.
        req=Request('http://127.0.0.1:5000/api/face-adaptation/reload',data=b'{}',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        try:
            with urlopen(req,timeout=15) as response:json.load(response)
            result['reload_required']=False
        except Exception:result['reload_required']=True
        references.clear();references.update(refs())
        return jsonify(result)
    @app.post('/unreview')
    def unreview():
        store.unreview(request.get_json().get('ids'));return jsonify(ok=True)
    @app.post('/reload')
    def reload_gallery():
        req=Request('http://127.0.0.1:5000/api/face-adaptation/reload',data=b'{}',
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with urlopen(req,timeout=25) as response:json.load(response)
        references.clear();references.update(refs())
        return jsonify(ok=True)
    @app.errorhandler(ValueError)
    def bad_request(error):return jsonify(error=str(error)),400
    app.run(host='127.0.0.1',port=15003,threaded=True)


if __name__=='__main__':main()
