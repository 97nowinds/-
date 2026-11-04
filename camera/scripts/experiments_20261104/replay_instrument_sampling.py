from pathlib import Path
import os,sys,json,time,hashlib,collections
import cv2
R=Path('/cluster/home/xqyi/lab-safety-host/camera');out=R/'runtime/instrument-sampling-fix-20261104';out.mkdir(exist_ok=True);b=Path('/cluster/dataX/xqyi/joint-migration-20261026/8765');api=next(b.rglob('lab_instrument_interaction/api.py'));sys.path.insert(0,str(api.parents[1]));from lab_instrument_interaction import InstrumentInteractionModule
config=next(p for p in b.rglob('instrument_006.json') if 'scheduler' in json.loads(p.read_text()));from lab_instrument_catalog import load_catalog
catalog_path=next(b.rglob('instrument_registry/catalog.json'));catalog=load_catalog(catalog_path);isolated=out/'isolated_catalog';isolated.mkdir(exist_ok=True);manifest=json.loads(catalog_path.read_text());runtime=dict(catalog.runtime);runtime['scheduler']=dict(runtime['scheduler'],buffer_seconds=6.5);(isolated/'runtime_profile.json').write_text(json.dumps(runtime));manifest['shared_runtime_profile']='runtime_profile.json';manifest['base_prompt']=str((catalog_path.parent/manifest['base_prompt']).resolve())
for entry in manifest['instruments']:
 key=Path(entry['archive']).stem;record=catalog.get(key);record.pop('profile_path',None);record.pop('archive_path',None);(isolated/(key+'.archive.json')).write_text(json.dumps(record));entry['archive']=key+'.archive.json';entry['profile']=key+'.json'
(isolated/'catalog.json').write_text(json.dumps(manifest));new_catalog=load_catalog(isolated/'catalog.json')
for entry in manifest['instruments']:
 key=Path(entry['archive']).name.split('.')[0];cfg=new_catalog.compatibility_projection(key);cfg['instrument_archive']={'catalog':'catalog.json','instrument_id':key};(isolated/(key+'.json')).write_text(json.dumps(cfg))
changed=isolated/'instrument_006.json'
cap=R/'runtime/questionnaire-20261104/centrifuge/captures';frames=[]
for p in cap.glob('*/request.json'):
 d=json.loads(p.read_text());raw=(p.parent/'original.jpg').read_bytes();assert hashlib.sha256(raw).hexdigest()==d['camera']['image_sha256'];frames.append((d,p.parent/'original.jpg'))
frames.sort(key=lambda x:x[0]['captured_at']);module=InstrumentInteractionModule(preload_models=True);report=json.loads((out/'comparison.json').read_text())
try:
 for label,path in [('after',changed)]:
  decisions=[];co=module.create_production_integration(config_path=path,on_result=lambda x:decisions.append(dict(x)),diagnostic_output_dir=out/label);responses=[]
  for d,p in frames:
   tracks=[{**t,'bbox_format':'xywh'} for t in d['camera']['tracks']];res=co.submit_frame(cv2.imread(str(p)),camera_id='cam_1',frame_id=d['snapshot_id'],snapshot_id=d['snapshot_id'],captured_at=d['captured_at'],tracks=tracks,receiver_received_wall=time.time(),receiver_received_monotonic=time.perf_counter(),frame_age_basis='receiver_arrival',image_url_source='isolated_recorded_replay:'+p.parent.name);responses.append({'captured_at':d['captured_at'],'response':res})
   if res['candidate_status']=='queued':
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
     snap=co.snapshot();m=snap['metrics']
     if not m.get('active_inference') and not m.get('pending_total'):break
     time.sleep(.05)
   print(label,d['captured_at'],res['candidate_status'],flush=True)
  closing=module.release_production_integration(co,drain_timeout_seconds=15);report[label]={'responses':responses,'decisions':decisions,'metrics':closing.get('metrics'),'counts':dict(collections.Counter(x['response']['candidate_status'] for x in responses)),'decision_counts':dict(collections.Counter(x['interaction_code'] for x in decisions))};(out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('SUMMARY',label,report[label]['counts'],report[label]['decision_counts'],flush=True)
finally:module.close()
print('REPLAY_DONE',flush=True)
