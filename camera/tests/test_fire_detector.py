import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fire_detector import FireDetector
from fire_worker import authority_scene


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'test.onnx';self.path.write_bytes(b'synthetic model placeholder')
        self.session=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images',shape=[1,3,320,320])],
            get_modelmeta=lambda:SimpleNamespace(custom_metadata_map={'names':"{0:'fire',1:'smoke'}"}))
        # Overlapping fire boxes must be suppressed, a smoke box must not erase fire.
        self.output=np.array([[[160,160,100,50,.95,.1],[161,160,100,50,.90,.1],
                               [160,160,100,50,.1,.99]]],dtype=np.float32).transpose(0,2,1)
        self.session.run=lambda *args:[self.output]
        self.detector=FireDetector(self.path,hashlib.sha256(self.path.read_bytes()).hexdigest(),session=self.session)

    def tearDown(self):self.temp.cleanup()

    def test_letterbox_inverse_matches_original_pixels_and_preserves_class_nms(self):
        result=self.detector.predict(np.zeros((360,640,3),np.uint8))
        self.assertEqual(len(result['detections']),2)
        fire=next(d for d in result['detections'] if d['class_name']=='fire')
        self.assertEqual(fire['bbox'],[220.,130.,200.,100.])
        self.assertEqual(self.detector.verdict(result)['fire_state'],'detected')

    def test_preprocessing_rgb_normalization_and_padding(self):
        image=np.full((360,640,3),[10,20,30],dtype=np.uint8)
        tensor,_=self.detector.preprocess(image)
        np.testing.assert_allclose(tensor[0,:,100,100],np.array([30,20,10])/255,rtol=1e-6)
        np.testing.assert_allclose(tensor[0,:,0,0],114/255,rtol=1e-6)

    def test_smoke_or_no_box_cannot_be_declared_safe_or_fire(self):
        self.output[:,4,:]=.1
        result=self.detector.predict(np.zeros((360,640,3),np.uint8))
        self.assertEqual(self.detector.verdict(result),{'fire_state':'unknown','confidence':0.0})

    def test_reversed_smoke_fire_class_order_uses_actual_metadata(self):
        self.session.get_modelmeta=lambda:SimpleNamespace(custom_metadata_map={'names':"{0:'smoke',1:'fire'}"})
        detector=FireDetector(self.path,hashlib.sha256(self.path.read_bytes()).hexdigest(),session=self.session)
        self.assertEqual(detector.fire_ids,{1})
        self.assertAlmostEqual(detector.verdict(detector.predict(np.zeros((360,640,3),np.uint8)))['confidence'],.99,places=5)

    def test_checkpoint_export_rejects_unrecognized_code_before_loading(self):
        from unittest.mock import patch
        from scripts.download_fire_model import export_checkpoint
        with patch('torch.serialization.get_unsafe_globals_in_checkpoint',return_value=['os.system']),patch('torch.load') as loader:
            with self.assertRaises(ValueError):export_checkpoint(self.path,self.path.with_suffix('.export.onnx'),{})
            loader.assert_not_called()

    def test_checksum_and_invalid_outputs_fail(self):
        with self.assertRaises(ValueError):FireDetector(self.path,'bad',session=self.session)
        self.output[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.detector.predict(np.zeros((360,640,3),np.uint8))
        self.output=np.zeros((1,7,1),np.float32)
        with self.assertRaises(ValueError):self.detector.predict(np.zeros((360,640,3),np.uint8))

    def frozen(self):
        return {'snapshot_id':'s1','camera_id':'c1','source_frame_id':'raw1','source_clock':'gpu1',
            'timestamp_source':'host_receive','source_captured_at':90.,'source_occurred_at':91.,
            'image_sha256':'x','image_url':'http://127.0.0.1:5175/api/handoff/snapshots/s1.jpg',
            'frame_width':640,'frame_height':360,'sync_mapping':{'source_clock':'gpu1','authority_clock':'main1',
                'offset_ms':10000.,'uncertainty_ms':10.,'sync_valid_from':95.,'sync_valid_until':120.}}

    def test_authority_conversion_preserves_source_and_uses_distinct_scene_key(self):
        frozen=self.frozen();event=authority_scene(frozen,'main1',102.)
        self.assertEqual(event['snapshot_id'],'fire:s1');self.assertEqual(event['camera']['tracks'],[])
        self.assertEqual(event['captured_at'],'1970-01-01T00:01:40.000+00:00')
        self.assertEqual(frozen['source_captured_at'],90.)

    def test_missing_old_or_unrelated_clock_mapping_blocks_submission(self):
        with self.assertRaises(ValueError):authority_scene({**self.frozen(),'sync_mapping':None},'main1',102.)
        with self.assertRaises(ValueError):authority_scene(self.frozen(),'restarted-main',102.)
        with self.assertRaises(ValueError):authority_scene(self.frozen(),'main1',121.)
        frozen=self.frozen();frozen['sync_mapping']['uncertainty_ms']=101
        with self.assertRaises(ValueError):authority_scene(frozen,'main1',102.)
        frozen=self.frozen();frozen['sync_mapping']['sync_valid_until']=100.5
        with self.assertRaises(ValueError):authority_scene(frozen,'main1',100.25)


if __name__=='__main__':unittest.main()
