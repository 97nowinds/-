import unittest
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import yaml,numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from scoped_byte_tracker import ScopedByteTracker,TrackIdAllocator

class ScopedIdTests(unittest.TestCase):
 def args(self):return SimpleNamespace(**yaml.safe_load((Path(__file__).parents[1]/'config/bytetrack_lab.yaml').read_text()))
 def activate(self,tracker):
  boxes=Boxes(np.asarray([[10,10,50,100,.9,0]],dtype=np.float32),(120,160)).numpy()
  target=tracker.init_track(boxes)[0];target.activate(tracker.kalman_filter,1);return target.track_id
 def test_installed_shared_counter_collision_and_owned_counter_fix(self):
  ordinary=BYTETracker(self.args());first=self.activate(ordinary)
  BYTETracker(self.args())
  self.assertEqual(self.activate(ordinary),first)
  owned=ScopedByteTracker(self.args());first=self.activate(owned)
  BYTETracker(self.args())
  self.assertEqual(self.activate(owned),first+1)
 def test_reconnect_never_reuses_own_or_resets_other_camera_ids(self):
  a=ScopedByteTracker(self.args());b=ScopedByteTracker(self.args())
  self.assertEqual(self.activate(a),1);self.assertEqual(self.activate(b),1)
  a.reset();self.assertEqual(self.activate(a),2);self.assertEqual(self.activate(b),2)
 def test_one_allocator_is_unique_under_concurrent_calls(self):
  owner=TrackIdAllocator()
  with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(lambda _:owner.next(),range(200)))
  self.assertEqual(sorted(values),list(range(1,201)))
