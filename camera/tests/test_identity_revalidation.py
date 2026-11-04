import unittest
import time
from unittest.mock import patch
from identity_revalidation import can_replace_local_identity, face_owned_by_body
from camera.tests import test_camera_worker_identity as workers
from camera.tests import test_yolo_person_tracker as coordinators

class RevalidationTests(unittest.TestCase):
    def test_face_in_crossing_heads_is_ambiguous(self):
        self.assertTrue(face_owned_by_body((20,10,20,20),(0,0,80,200)))
        self.assertFalse(face_owned_by_body((20,10,20,20),(0,0,80,200),[(10,0,80,200)]))
        self.assertFalse(face_owned_by_body((90,10,20,20),(0,0,80,200)))

    def test_only_recent_same_camera_local_face_can_correct(self):
        target = {'camera_id':'cam_2','local_id':7,'last_seen':10}
        proof = {'identity_source':'face','_face_revalidation':True,'_face_camera_id':'cam_2',
                 '_face_local_id':7,'_face_votes':3,'_face_confirmed_monotonic':10}
        self.assertTrue(can_replace_local_identity(target,proof,'cam_2',10.1))
        for changes in [{'_face_local_id':8},{'_face_votes':2},{'_face_confirmed_monotonic':7},
                        {'_face_camera_id':'cam_1'},{'identity_source':'handoff'}, {'_face_revalidation':False}]:
            with self.subTest(changes=changes):
                self.assertFalse(can_replace_local_identity(target,{**proof,**changes},'cam_2',10.1))
        self.assertFalse(can_replace_local_identity(None,proof,'cam_2',10.1))

    def test_bound_track_is_rechecked_despite_unknown_person(self):
        worker = workers.CameraWorkerIdentityTests().worker()
        worker.track_identities[1] = worker.identity = {'known':True,'person_id':'old'}
        worker.face_attempts_by_track = {1:time.time()-4,2:time.time()-1}
        tracks = [{'track_id':n,'box':(n*100,0,80,200)} for n in (1,2)]
        self.assertEqual(worker.select_face_target(tracks)['track_id'],1)

    def test_new_identity_needs_three_consecutive_votes(self):
        worker = workers.CameraWorkerIdentityTests().worker()
        def face(person):return {'known':True,'person_id':person}
        for _ in range(7):worker.accept_face_result(face('old'),None,track_id=1)
        for _ in range(2):worker.accept_face_result(face('new'),None,track_id=1)
        self.assertEqual(worker.track_identities[1]['person_id'],'old')
        result=worker.accept_face_result(face('new'),None,track_id=1)
        self.assertEqual(result['person_id'],'new')
        self.assertEqual(result['_face_votes'],3)
        self.assertTrue(result['_face_revalidation'])

    def test_alternating_or_old_votes_cannot_confirm(self):
        worker = workers.CameraWorkerIdentityTests().worker()
        for person in ['a','b','a','b','a','b']:
            self.assertIsNone(worker.accept_face_result({'known':True,'person_id':person},None,track_id=1))
        worker = workers.CameraWorkerIdentityTests().worker()
        with patch('app.time.monotonic', return_value=1):
            for _ in range(2):worker.accept_face_result({'known':True,'person_id':'a'},None,track_id=1)
        with patch('app.time.monotonic', return_value=12):
            self.assertIsNone(worker.accept_face_result({'known':True,'person_id':'a'},None,track_id=1))

    def test_correction_preserves_tracks_and_live_other_owner(self):
        factory=coordinators.CrossCameraTrackCoordinatorTests()
        for occupied in (False,True):
            coordinator=factory.coordinator(factory.UncalibratedFloorMap())
            target=coordinator._new_global(10,'cam_2',None,None,local_id=7)
            coordinator.set_identity(target,{'known':True,'person_id':'wrong'},'cam_2',now=10)
            if occupied:
                other=coordinator._new_global(10,'cam_2',None,None,local_id=8)
                coordinator.set_identity(other,{'known':True,'person_id':'correct'},'cam_2',now=10)
            proof={'known':True,'person_id':'correct','identity_source':'face','_face_revalidation':True,
                   '_face_camera_id':'cam_2','_face_local_id':7,'_face_votes':3,'_face_confirmed_monotonic':10}
            result=coordinator.set_identity(target,proof,'cam_2',now=10.1)
            if occupied:
                self.assertIsNone(result)
                self.assertEqual(coordinator.person_locks['correct'],other)
                self.assertNotIn(target,coordinator.global_identities)
            else:
                self.assertEqual(result['person_id'],'correct')
                self.assertEqual(coordinator.person_locks['correct'],target)
            self.assertNotIn('wrong',coordinator.person_locks)
            self.assertIn(target,coordinator.global_tracks)
            self.assertEqual(coordinator.redirects,{})
