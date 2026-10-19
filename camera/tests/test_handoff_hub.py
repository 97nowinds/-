import json,tempfile,unittest
from pathlib import Path
import numpy as np
from handoff_hub import load_handoff_transitions,body_overlap_ratio
from association import BatchAssociator,AssociationObservation,AssociationTarget
from mtmc_config import load_mtmc_config

class HubTests(unittest.TestCase):
 def test_new_confirmed_face_can_reclaim_only_an_expired_global_owner(self):
  from yolo_person_tracker import CrossCameraTrackCoordinator
  extractor=type('Extractor',(),{'extract':lambda *args:np.array([1.,0.])})()
  coordinator=CrossCameraTrackCoordinator(extractor)
  coordinator.global_tracks={'old':{'appearance':np.array([1.,0.])},'new':{'appearance':np.array([1.,0.])}}
  identity={'known':True,'person_id':'registered','identity_source':'face'}
  self.assertIsNotNone(coordinator.set_identity('old',identity,'cam_entrance',now=1))
  self.assertIsNone(coordinator.set_identity('new',identity,'cam_entrance',now=2))
  coordinator.global_tracks.pop('old')
  self.assertIsNone(coordinator.set_identity('new',{**identity,'identity_source':'handoff'},'cam_entrance',now=2))
  self.assertIsNotNone(coordinator.set_identity('new',identity,'cam_entrance',now=2))
  self.assertEqual(coordinator.person_locks['registered'],'new')

 def test_strong_hub_candidate_needs_three_observed_frames(self):
  from yolo_person_tracker import CrossCameraTrackCoordinator
  from floor_map import FloorMapProjector
  floor=FloorMapProjector.from_path(Path(__file__).parents[1]/'config/floor_map.json')
  extractor=type('Extractor',(),{'extract':lambda *args:np.array([1.,0.])})()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'policy.json';p.write_text(json.dumps({'enabled':True,'hub_id':'door','cameras':['cam_1','cam_2','cam_entrance'],'minimum_body_overlap':.35}))
   coordinator=CrossCameraTrackCoordinator(extractor,floor,handoff_policy_path=p)
   frame=np.full((200,200,3),128,dtype=np.uint8)
   first=coordinator.update_batch('cam_1',[{'track_id':1,'box':(20,80,40,100),'image_anchor':{'x':.2,'y':.9}}],frame,monotonic_time=1)[1]
   target=[{'track_id':2,'box':(100,60,40,100),'image_anchor':{'x':.6,'y':.8}}]
   a=coordinator.update_batch('cam_2',target,frame,monotonic_time=1.1)[2]
   b=coordinator.update_batch('cam_2',target,frame,monotonic_time=1.21)[2]
   c=coordinator.update_batch('cam_2',target,frame,monotonic_time=1.32)[2]
   self.assertNotEqual(a,first);self.assertNotEqual(b,first);self.assertEqual(c,first)

 def test_duplicate_features_do_not_trigger_inference_every_frame(self):
  from yolo_person_tracker import CrossCameraTrackCoordinator
  calls=[]
  extractor=type('Extractor',(),{'extract':lambda *args:(calls.append(1) or np.array([1.,0.]))})()
  coordinator=CrossCameraTrackCoordinator(extractor)
  frame=np.full((200,200,3),128,dtype=np.uint8)
  track=[{'track_id':1,'box':(20,20,50,100)}]
  for i in range(30):coordinator.update_batch('cam_1',track,frame,monotonic_time=1+i*.04)
  self.assertEqual(len(calls),2)
  self.assertEqual(coordinator.local_features[('cam_1',1)][1],1)

 def config(self):
  floor=json.loads((Path(__file__).parents[1]/'config/floor_map.json').read_text(encoding='utf-8'))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'policy.json';p.write_text(json.dumps({'enabled':True,'hub_id':'door','cameras':['cam_1','cam_2','cam_entrance'],'minimum_body_overlap':.35}))
   return load_handoff_transitions(floor,p)
 def test_rectangle_intersection_is_area_based(self):
  polygon=[[0,0],[.5,0],[.5,1],[0,1]]
  self.assertAlmostEqual(body_overlap_ratio((.4,.2,.2,.2),polygon),.5)
  self.assertEqual(body_overlap_ratio((.6,.2,.2,.2),polygon),0)
  self.assertEqual(body_overlap_ratio(None,polygon),0)
 def test_hub_unifies_directions_but_requires_temporal_confirmation(self):
  engine=BatchAssociator(load_mtmc_config(),self.config(),'approximate')
  obs=AssociationObservation('cam_2',1,2,appearance=np.array([1.,0.]),image_anchor={'x':.65,'y':.7})
  target=AssociationTarget('p','cam_1',1.9,appearance=np.array([1.,0.]),image_anchor={'x':.2,'y':.9},active_camera_ids={'cam_1'})
  result=engine.evaluate(obs,target);self.assertTrue(result['eligible']);self.assertEqual(result['hub_id'],'door')
  decisions,_=engine.associate([obs],[target]);self.assertFalse(decisions[0]['accepted']);self.assertEqual(decisions[0]['confidence'],'medium')
 def test_visible_body_can_support_missing_feet_but_not_outside_hub(self):
  engine=BatchAssociator(load_mtmc_config(),self.config(),'approximate')
  obs=AssociationObservation('cam_2',1,2,appearance=np.array([1.,0.]),image_box=(.55,.3,.1,.15))
  target=AssociationTarget('p','cam_1',1.9,appearance=np.array([1.,0.]),image_box=(.05,.2,.1,.2))
  self.assertTrue(engine.evaluate(obs,target)['eligible'])
  obs.image_box=(.88,.05,.05,.1)
  self.assertFalse(engine.evaluate(obs,target)['eligible'])
 def test_loading_policy_does_not_mutate_published_map(self):
  floor={'camera_transitions':[]};before=json.dumps(floor)
  self.assertEqual(load_handoff_transitions(floor),[]);self.assertEqual(json.dumps(floor),before)
