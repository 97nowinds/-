"""Independent latest-frame fire worker. Observe mode never sends notifications."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.request import Request,build_opener,ProxyHandler

import cv2
import numpy as np
from alert_evidence import NoRedirect
from fire_detector import FireDetector
from handoff_transport import iso_time
from snapshot_registry import SnapshotRegistry

ROOT=Path(__file__).resolve().parent


def authority_scene(frozen,authority_clock,now):
    """Use only a current mapping to this running main host clock service."""
    sync=frozen.get('sync_mapping') or {}
    if (not sync or sync.get('source_clock')!=frozen.get('source_clock') or
        sync.get('authority_clock')!=authority_clock or frozen.get('timestamp_source')!='host_receive'):
        raise ValueError('verified authority clock mapping unavailable')
    keys=('offset_ms','uncertainty_ms','sync_valid_from','sync_valid_until')
    if any(type(sync.get(k)) not in (int,float) or not np.isfinite(sync[k]) for k in keys):
        raise ValueError('invalid clock mapping')
    if not 0<=sync['uncertainty_ms']<=100:raise ValueError('clock uncertainty exceeds fire limit')
    captured=frozen['source_captured_at']+sync['offset_ms']/1000
    occurred=frozen['source_occurred_at']+sync['offset_ms']/1000
    if (not sync['sync_valid_from']<=captured<=occurred<=sync['sync_valid_until'] or
            occurred>now+2 or not now<sync['sync_valid_until']):
        raise ValueError('clock mapping expired or timestamps outside bounds')
    if not -2<=now-captured<=15:raise ValueError('fire snapshot stale')
    return {'snapshot_id':'fire:'+frozen['snapshot_id'],'captured_at':iso_time(captured),
        'occurred_at':iso_time(occurred),'time_domain':'authority','camera':{
            'camera_id':frozen['camera_id'],'source_frame_id':frozen['source_frame_id'],
            'frame_width':frozen['frame_width'],'frame_height':frozen['frame_height'],
            'image_sha256':frozen['image_sha256'],'image_url':frozen['image_url'],'tracks':[]}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--interval',type=float,default=2);p.add_argument('--cameras',nargs='+',default=['cam_1','cam_2','cam_entrance'])
    p.add_argument('--submit',action='store_true',help='Submit only with proven authority time and producer credentials')
    args=p.parse_args()
    if args.interval<1 or args.seconds<=0:p.error('interval >=1 and positive seconds required')
    token=os.environ.get('LAB_MANAGEMENT_PRODUCER_TOKEN','')
    if args.submit and not token:p.error('producer token must be supplied through local process environment')
    credentials=json.loads((ROOT/'runtime/handoff-credentials.json').read_text(encoding='utf-8-sig'))
    provenance=json.loads((ROOT/'config/fire_model.json').read_text(encoding='utf-8'))
    alarm_minimum=json.loads((ROOT/'config/management.json').read_text(encoding='utf-8')).get('fire_fusion',{}).get('visual_minimum_confidence',.9)
    if args.submit and provenance.get('state')!='validated':p.error('fire model remains experimental; observation only')
    detector=FireDetector(ROOT/provenance['path'],provenance['sha256'],threads=provenance['cpu_threads'])
    cv2.setNumThreads(1);opener=build_opener(ProxyHandler({}),NoRedirect());base='http://127.0.0.1:5175'
    image_headers={'Authorization':'Bearer '+credentials['image_read_token']}
    registry=SnapshotRegistry(ROOT/'runtime/snapshot-registry.sqlite3') if args.submit else None
    status={'pid':os.getpid(),'mode':'submit' if args.submit else 'observe','model_sha256':provenance['sha256'],
        'frames':0,'duplicate_skips':0,'eligible_fire_frames':0,'submitted':0,'errors':0,'cameras':{},'started_at':iso_time(time.time())}
    status_path=ROOT/'runtime/fire-worker-status.json'
    # One model worker owns this lock; a second launch must not double CPU work.
    lock=(ROOT/'runtime/fire-worker.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    deadline=time.monotonic()+args.seconds;seen={}
    try:
        while time.monotonic()<deadline:
            began=time.monotonic()
            for camera in args.cameras:
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',camera):raise ValueError('invalid camera id')
                try:
                    with opener.open(Request(base+'/api/handoff/frozen/'+camera,headers=image_headers),timeout=3) as r:
                        frozen=json.loads(r.read(4*1024*1024))
                    source=frozen['source_frame_id'];snapshot=frozen['snapshot_id']
                    if source==seen.get(camera):status['duplicate_skips']+=1;continue
                    if not re.fullmatch(r'[A-Za-z0-9_-]{1,180}',snapshot):raise ValueError('invalid snapshot id')
                    url=base+'/api/handoff/snapshots/'+snapshot+'.jpg'
                    with opener.open(Request(url,headers=image_headers),timeout=3) as r:
                        if r.headers.get_content_type()!='image/jpeg':raise ValueError('snapshot MIME mismatch')
                        jpeg=r.read(10*1024*1024+1)
                    if len(jpeg)>10*1024*1024 or hashlib.sha256(jpeg).hexdigest()!=frozen['image_sha256']:
                        raise ValueError('snapshot checksum mismatch')
                    image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                    if image is None or list(image.shape[1::-1])!=[frozen['frame_width'],frozen['frame_height']]:
                        raise ValueError('snapshot dimensions mismatch')
                    seen[camera]=source;prediction=detector.predict(image,provenance['candidate_confidence'])
                    verdict=detector.verdict(prediction,alarm_minimum)
                    status['frames']+=1;status['eligible_fire_frames']+=verdict['fire_state']=='detected'
                    row={'source_frame_id':source,'source_captured_at':frozen['source_captured_at'],
                        'inference_ms':prediction['inference_ms'],'fire_state':verdict['fire_state'],
                        'detections':prediction['detections'],'notification_gate':'observe_only'}
                    if args.submit:
                        try:
                            probe={'probe_id':'fire_'+str(time.time_ns()),'client_sent_at':iso_time(time.time())}
                            with opener.open(Request(base+'/api/handoff/time-sync',data=json.dumps(probe).encode(),
                                headers={'Authorization':'Bearer '+credentials['sync_read_token'],'Content-Type':'application/json'}),timeout=3) as r:
                                authority=json.load(r)['authority_clock']
                            frozen['image_url']=url
                            event=authority_scene(frozen,authority,time.time());registry.publish(event)
                            now=iso_time(time.time());body={'evidence_type':'fire','analysis_id':'fire-model:'+snapshot,
                                'camera_id':camera,'snapshot_id':event['snapshot_id'],'time_domain':'authority',
                                'captured_at':event['captured_at'],'published_at':now,'expires_at':iso_time(time.time()+5),
                                'status':'valid',**verdict}
                            with opener.open(Request('http://127.0.0.1:5173/api/management/evidence',data=json.dumps(body).encode(),
                                headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}),timeout=5) as r:reply=json.load(r)
                            row['notification_gate']='accepted' if reply.get('accepted') else 'no_new_event';status['submitted']+=1
                        except ValueError:row['notification_gate']='authority_mapping_unavailable'
                    status['cameras'][camera]=row
                except Exception as exc:
                    status['errors']+=1;status['cameras'][camera]={'status':'unavailable','error_type':type(exc).__name__}
            status['updated_at']=iso_time(time.time());temporary=status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(status_path)
            time.sleep(max(0,min(args.interval-(time.monotonic()-began),deadline-time.monotonic())))
    finally:
        status['stopped_at']=iso_time(time.time());status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        lock.close()
    print(json.dumps({k:v for k,v in status.items() if k!='cameras'}))


if __name__=='__main__':main()
