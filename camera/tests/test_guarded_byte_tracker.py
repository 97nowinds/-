import unittest
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import yaml
from ultralytics.engine.results import Boxes
from guarded_byte_tracker import GuardedByteTracker


class ColourEmbedder:
    def extract(self, frame, box):
        x,y,w,h = box
        feature = frame[y:y+h,x:x+w].astype(float).mean((0,1))
        norm = np.linalg.norm(feature)
        return feature/norm if norm else None


class GuardedLowTests(unittest.TestCase):
    def make_tracker(self):
        config = yaml.safe_load((Path(__file__).parents[1]/'config/bytetrack_guarded_lab.yaml').read_text())
        return GuardedByteTracker(SimpleNamespace(**config),ColourEmbedder())

    def update(self, tracker, detections, stamp, colour=(20,80,180)):
        frame = np.full((120,160,3),colour,dtype=np.uint8)
        boxes = Boxes(np.asarray(detections,dtype=np.float32).reshape(-1,6),frame.shape[:2]).numpy()
        return tracker.update(boxes,frame,observed_at=stamp)

    def test_three_fresh_low_observations_reactivate_same_lost_id(self):
        tracker = self.make_tracker()
        original = int(self.update(tracker,[[20,20,60,100,.9,0]],1)[0,4])
        self.update(tracker,[],1.04)
        low = [[20,20,60,65,.2,0]]
        self.assertEqual(len(self.update(tracker,low,1.08)),0)
        self.assertEqual(len(self.update(tracker,low,1.12)),0)
        output = self.update(tracker,low,1.17)
        self.assertEqual(int(output[0,4]),original)
        self.assertEqual(tracker.audit[-1]['reason'],'recovered')
        self.assertAlmostEqual(float(output[0,5]),.2,places=5)
        self.assertIn(original,tracker.partial_tracks)

    def test_changed_appearance_cannot_inherit_by_position(self):
        tracker = self.make_tracker()
        self.update(tracker,[[20,20,60,100,.9,0]],1,colour=(0,0,180))
        self.update(tracker,[],1.04)
        for stamp in (1.08,1.12,1.17):
            self.assertEqual(len(self.update(tracker,[[20,20,60,65,.2,0]],stamp,colour=(180,0,0))),0)

    def recovered_tracker(self):
        tracker=self.make_tracker()
        tid=int(self.update(tracker,[[20,20,60,100,.9,0]],1)[0,4])
        self.update(tracker,[],1.04)
        for stamp in (1.08,1.12,1.17):
            self.update(tracker,[[20,20,60,65,.2,0]],stamp)
        return tracker,tid

    def test_partial_low_to_high_shape_change_keeps_original_id(self):
        tracker,tid=self.recovered_tracker()
        for stamp in (1.21,1.25,1.29):
            output=self.update(tracker,[[20,20,60,55,.35,0]],stamp)
            self.assertEqual([int(row[4]) for row in output],[tid])
            self.assertIn(tid,tracker.partial_tracks)
            self.assertEqual(tracker.references[tid][0],stamp)
            self.assertEqual(tracker.audit[-1]['stage'],'high')
            self.assertEqual(tracker.audit[-1]['reason'],'continued')
        output=self.update(tracker,[[20,20,60,100,.9,0]],1.33)
        self.assertEqual(int(output[0,4]),tid)
        self.assertNotIn(tid,tracker.partial_tracks)

    def test_partial_high_colour_change_cannot_overwrite_reference(self):
        tracker,tid=self.recovered_tracker()
        stamp,box,reference=tracker.references[tid]
        self.update(tracker,[[20,20,60,65,.9,0]],1.21,colour=(180,0,0))
        self.assertNotIn(tid,tracker.confirmed_current)
        self.assertEqual(tracker.references[tid][0],stamp)
        np.testing.assert_array_equal(tracker.references[tid][2],reference)

    def test_high_partial_lost_confirmation_does_not_create_competing_id(self):
        tracker,tid=self.recovered_tracker()
        self.update(tracker,[],1.21)
        high=[[20,20,60,65,.35,0]]
        self.assertEqual(len(self.update(tracker,high,1.25)),0)
        self.assertEqual(len(self.update(tracker,high,1.29)),0)
        output=self.update(tracker,high,1.34)
        self.assertEqual([int(row[4]) for row in output],[tid])

    def test_head_turn_motion_and_appearance_can_continue_but_not_reacquire(self):
        class TurnEmbedder:
            def extract_crops(self,crops):
                return [np.array([1.,0.]) if i%2==0 else np.array([.82,np.sqrt(1-.82**2)]) for i in range(len(crops))]
        tracker,tid=self.recovered_tracker()
        tracker.embedder=TurnEmbedder()
        high=[[21,20,61,65,.35,0]]
        output=self.update(tracker,high,1.21)
        self.assertEqual([int(row[4]) for row in output],[tid])
        self.assertTrue(tracker.audit[-1]['pose_continuity_verified'])
        self.update(tracker,[],1.25)
        for stamp in (1.29,1.33,1.38):
            output=self.update(tracker,high,stamp)
            self.assertNotIn(tid,[int(row[4]) for row in output])

    def test_tiny_head_without_embedding_requires_real_texture_and_cannot_reacquire(self):
        class NoFeatureEmbedder:
            def extract_crops(self,crops):
                return [None]*len(crops)
        tracker,tid=self.recovered_tracker()
        tracker.embedder=NoFeatureEmbedder()
        gray=np.random.default_rng(21).integers(0,256,(120,160),dtype=np.uint8)
        frame=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        tracker.references[tid]=(1.17,(20,20,60,65),frame[20:65,20:60].copy())
        tracker.reference_gray[tid]=gray.copy()
        boxes=Boxes(np.array([[20,20,60,48,.2,0]],dtype=np.float32),frame.shape[:2]).numpy()
        for stamp in (1.21,1.25):
            output=tracker.update(boxes,frame,observed_at=stamp)
            self.assertEqual([int(row[4]) for row in output],[tid])
            self.assertTrue(tracker.audit[-1]['texture_only_continuity'])
            self.assertIsNone(tracker.audit[-1]['appearance_similarity'])
            self.assertEqual(tracker.references[tid][0],stamp)
        self.update(tracker,[],1.29)
        for stamp in (1.33,1.37,1.42):
            output=tracker.update(boxes,frame,observed_at=stamp)
            self.assertNotIn(tid,[int(row[4]) for row in output])

    def test_passing_body_containing_head_is_not_same_visible_person(self):
        tracker,tid=self.recovered_tracker()
        from ultralytics.trackers.byte_tracker import STrack
        passing=STrack(np.array([40,50,60,95,0]),.9,0)
        track=tracker.tracked_stracks[0]
        tracker.observed_at=1.21
        patches,reason=tracker._patches(track,passing)
        self.assertIsNone(patches)
        self.assertEqual(reason,'head_alignment')

    def test_repeated_partial_observation_does_not_refresh_reference(self):
        tracker,tid=self.recovered_tracker()
        stamp=tracker.references[tid][0]
        self.update(tracker,[[20,20,60,65,.35,0]],stamp)
        self.assertNotIn(tid,tracker.confirmed_current)
        self.assertEqual(tracker.references[tid][0],stamp)

    def test_equally_plausible_lost_ids_remain_unresolved(self):
        tracker = self.make_tracker()
        self.update(tracker,[[20,20,60,100,.9,0],[20,20,60,100,.9,0]],1)
        self.update(tracker,[],1.04)
        for stamp in (1.08,1.12,1.17):
            self.assertEqual(len(self.update(tracker,[[20,20,60,65,.2,0]],stamp)),0)

    def test_low_detection_never_creates_new_person(self):
        tracker = self.make_tracker()
        self.assertEqual(len(self.update(tracker,[[20,20,60,65,.2,0]],1)),0)

    def test_normal_active_match_cannot_be_reused_for_lost_person(self):
        tracker = self.make_tracker()
        first = self.update(tracker,[[20,20,60,100,.9,0],[22,20,62,100,.9,0]],1)
        active_id, lost_id = int(first[0,4]), int(first[1,4])
        active, other = tracker.tracked_stracks
        other.mark_lost()
        tracker.tracked_stracks = [active]
        tracker.lost_stracks = [other]
        for stamp in (1.04,1.08,1.13):
            output = self.update(tracker,[[20,20,60,100,.2,0]],stamp)
            self.assertEqual([int(row[4]) for row in output],[active_id])
            self.assertFalse(any(item['reason']=='recovered' for item in tracker.audit))
        # Native ByteTrack may remove the deliberately overlapping Lost
        # duplicate; the invariant is that it never consumes this detection.
        self.assertNotIn(lost_id,tracker.partial_tracks)

    def test_lost_candidate_rois_use_one_batch_per_frame(self):
        class BatchEmbedder(ColourEmbedder):
            def __init__(self):
                self.batch_sizes = []

            def extract_crops(self,crops):
                self.batch_sizes.append(len(crops))
                return [super(BatchEmbedder,self).extract(crop,(0,0,crop.shape[1],crop.shape[0])) for crop in crops]

        tracker = self.make_tracker()
        tracker.embedder = BatchEmbedder()
        self.update(tracker,[[20,20,60,100,.9,0]],1)
        self.update(tracker,[],1.04)
        self.update(tracker,[[20,20,60,65,.2,0],[22,20,62,100,.2,0]],1.08)
        self.assertEqual(tracker.embedder.batch_sizes,[4])

    def test_stale_reference_and_repeated_timestamp_cannot_confirm(self):
        tracker = self.make_tracker()
        self.update(tracker,[[20,20,60,100,.9,0]],1)
        self.update(tracker,[],1.04)
        low = [[20,20,60,65,.2,0]]
        for _ in range(5):
            self.assertEqual(len(self.update(tracker,low,1.08)),0)
        self.assertEqual(len(self.update(tracker,low,2)),0)

    def test_reset_clears_reference_and_confirmation(self):
        tracker = self.make_tracker()
        self.update(tracker,[[20,20,60,100,.9,0]],1)
        tracker.reset()
        self.assertFalse(tracker.references)
        self.assertFalse(tracker.evidence)

    def test_lk_requires_actual_texture_correspondence(self):
        tracker = self.make_tracker()
        tracker.current_gray = np.random.default_rng(12).integers(0,256,(120,160),dtype=np.uint8)
        tracker.reference_gray[1] = tracker.current_gray.copy()
        tracker.references[1] = (1,(20,20,60,80),np.zeros((60,40,3),dtype=np.uint8))
        tracker.observed_at=1.04
        same = tracker._flow(1,(20,20,60,80))
        self.assertTrue(same['flow_verified'])
        tracker.current_gray = np.random.default_rng(45).integers(0,256,(120,160),dtype=np.uint8)
        self.assertFalse(tracker._flow(1,(20,20,60,80))['flow_verified'])

    def test_motion_subset_cannot_reacquire_lost_identity(self):
        from unittest.mock import patch
        from ultralytics.trackers.basetrack import TrackState
        tracker,tid=self.recovered_tracker()
        tracker.current_gray=np.random.default_rng(12).integers(0,256,(120,160),dtype=np.uint8)
        tracker.reference_gray[tid]=tracker.current_gray.copy()
        tracker.references[tid]=(1.17,(20,20,60,65),np.zeros((45,40,3),dtype=np.uint8))
        tracker.observed_at=1.21
        def mixed_motion(points,*args,**kwargs):
            mask=np.zeros((len(points),1),dtype=np.uint8)
            mask[:int(len(points)*.70)]=1
            return None,mask
        with patch('guarded_byte_tracker.cv2.estimateAffinePartial2D',side_effect=mixed_motion):
            continuing=tracker._flow(tid,(20,20,60,65))
            self.assertTrue(continuing['flow_continuation_verified'])
            self.assertFalse(continuing['flow_strict_verified'])
            tracker.tracked_stracks[0].state=TrackState.Lost
            lost=tracker._flow(tid,(20,20,60,65))
            self.assertFalse(lost['flow_verified'])

    def test_small_head_texture_rule_only_continues_fresh_tracked_identity(self):
        from unittest.mock import patch
        from ultralytics.trackers.basetrack import TrackState
        tracker,tid=self.recovered_tracker()
        tracker.current_gray=np.random.default_rng(17).integers(0,256,(120,160),dtype=np.uint8)
        tracker.reference_gray[tid]=tracker.current_gray.copy()
        tracker.references[tid]=(1.17,(20,20,60,65),np.zeros((45,40,3),dtype=np.uint8))
        tracker.observed_at=1.21
        points=np.asarray([[x,y] for y in (24,35,47) for x in (24,38,55)],dtype=np.float32).reshape(-1,1,2)
        with patch('guarded_byte_tracker.cv2.goodFeaturesToTrack',return_value=points):
            result=tracker._flow(tid,(20,20,60,65),head_only=True)
            self.assertTrue(result['flow_continuation_verified'])
            self.assertFalse(result['flow_strict_verified'])
            tracker.tracked_stracks[0].state=TrackState.Lost
            self.assertFalse(tracker._flow(tid,(20,20,60,65),head_only=True)['flow_verified'])
            tracker.tracked_stracks[0].state=TrackState.Tracked
            tracker.observed_at=1.40
            self.assertFalse(tracker._flow(tid,(20,20,60,65),head_only=True)['flow_verified'])

    def test_nested_visible_hypotheses_do_not_include_adjacent_people(self):
        from types import SimpleNamespace
        head = SimpleNamespace(xyxy=np.array([20,20,60,65]))
        body = SimpleNamespace(xyxy=np.array([21,20,60,100]))
        neighbour = SimpleNamespace(xyxy=np.array([50,20,90,100]))
        self.assertTrue(GuardedByteTracker._same_visible_hypothesis(head,body))
        self.assertFalse(GuardedByteTracker._same_visible_hypothesis(head,neighbour))


if __name__ == '__main__':
    unittest.main()
