import unittest
from types import SimpleNamespace
from face_recovery import can_reclaim_identity, cached_identity_is_current


class FaceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.identity = {'identity_source':'face','person_id':'registered',
                         '_face_camera_id':'cam_2','_face_local_id':77,
                         '_face_votes':3,'_face_confirmed_monotonic':10.0}
        self.target = {'camera_id':'cam_2','local_id':77,'last_seen':10.0}
        self.old = {'camera_id':'cam_1','local_id':11,'last_seen':7.0}

    def test_retained_old_track_does_not_block_fresh_face(self):
        self.assertTrue(can_reclaim_identity(self.old,self.target,self.identity,'cam_2',10.1))

    def test_any_fresh_old_camera_blocks_transfer(self):
        self.old['camera_observations'] = {'cam_1':{'last_seen':10.0}}
        self.assertFalse(can_reclaim_identity(self.old,self.target,self.identity,'cam_2',10.1))

    def test_face_proof_must_be_current_and_same_local_camera(self):
        for changes in [{'_face_votes':2},{'_face_confirmed_monotonic':7.0},
                        {'_face_confirmed_monotonic':11.0},{'_face_camera_id':'cam_1'},
                        {'_face_local_id':78},{'identity_source':'handoff'}]:
            with self.subTest(changes=changes):
                self.assertFalse(can_reclaim_identity(self.old,self.target,{**self.identity,**changes},'cam_2',10.1))

    def test_new_body_cannot_be_stale_or_missing(self):
        self.assertFalse(can_reclaim_identity(self.old,None,self.identity,'cam_2',10.1))
        self.target['last_seen'] = 5.0
        self.assertFalse(can_reclaim_identity(self.old,self.target,self.identity,'cam_2',10.1))

    def test_unknown_old_body_allows_only_proven_face(self):
        self.assertTrue(can_reclaim_identity(None,self.target,self.identity,'cam_2',10.1))
        self.assertFalse(can_reclaim_identity(None,self.target,{'identity_source':'face'},'cam_2',10.1))

    def test_cached_name_must_follow_current_lock_including_redirects(self):
        coordinator = SimpleNamespace(redirects={'old':'new'},person_locks={'registered':'new'})
        identity = {'person_id':'registered','global_track_id':'old'}
        self.assertTrue(cached_identity_is_current(coordinator,identity,'new'))
        coordinator.person_locks['registered'] = 'other'
        self.assertFalse(cached_identity_is_current(coordinator,identity,'new'))


if __name__ == '__main__': unittest.main()
