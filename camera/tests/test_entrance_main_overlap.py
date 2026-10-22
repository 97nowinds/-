"""Synthetic identity-safety checks using the actual entrance/main route config."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from yolo_person_tracker import CrossCameraTrackCoordinator


class SyntheticAppearance:
    def extract(self, frame, box):
        x,y,w,h=box
        vector=frame[y:y+h,x:x+w].astype(np.float32).mean((0,1))
        return vector/np.linalg.norm(vector)


class EntranceMainOverlapTests(unittest.TestCase):
    def run_route(self, colour=(70,71,0), ambiguous=False, anchor=None, times=None):
        config=json.loads((Path(__file__).parents[1]/'config/floor_map.json').read_text(encoding='utf-8'))
        c=CrossCameraTrackCoordinator(SyntheticAppearance(),SimpleNamespace(config=config))
        source=np.full((120,220,3),(100,0,0),np.uint8)
        target=np.full_like(source,colour)
        originals=[{'track_id':1,'box':(20,20,40,80),'image_anchor':{'x':.64,'y':.8}}]
        if ambiguous:originals.append({'track_id':2,'box':(120,20,40,80),'image_anchor':{'x':.65,'y':.8}})
        old=c.update_batch('cam_entrance',originals,source,monotonic_time=1,unix_time=100)[1]
        c.set_identity(old,{'known':True,'person_id':'synthetic_person','name':'SYNTHETIC'},'cam_entrance',now=100)
        ids=[]
        for t in times or (2,2.2,2.4,2.6,2.8,3.1):
            c.update_batch('cam_entrance',originals,source,monotonic_time=t-.01,unix_time=99+t-.01)
            ids.append(c.update_batch('cam_2',[{'track_id':9,'box':(20,20,40,80),
                'image_anchor':anchor or {'x':.65,'y':.7}}],target,monotonic_time=t,unix_time=99+t)[9])
        return c,old,ids

    def test_medium_overlap_waits_then_transfers_the_registered_owner(self):
        c,old,ids=self.run_route()
        self.assertTrue(all(gid!=old for gid in ids[:-1]))
        self.assertEqual(ids[-1],old)
        identity=c.identity_for(old,'cam_2',now=102.1)
        self.assertEqual(identity['person_id'],'synthetic_person')

    def test_equal_appearance_candidates_do_not_transfer_a_name(self):
        _,old,ids=self.run_route(ambiguous=True)
        self.assertNotIn(old,ids)

    def test_appearance_below_route_floor_does_not_transfer(self):
        _,old,ids=self.run_route(colour=(60,80,0))
        self.assertNotIn(old,ids)

    def test_outside_entry_stays_anonymous_even_with_identical_appearance(self):
        c,old,ids=self.run_route(colour=(100,0,0),anchor={'x':.95,'y':.1})
        self.assertNotIn(old,ids)
        self.assertIsNone(c.identity_for(ids[-1],'cam_2',now=102.1))

    def test_one_observation_cannot_complete_medium_overlap(self):
        _,old,ids=self.run_route(times=[2.])
        self.assertNotIn(old,ids)


if __name__=='__main__':unittest.main()
