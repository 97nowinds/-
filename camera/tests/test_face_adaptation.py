import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from face_adaptation import SiteSamples


class SiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.now=2000000000.
        self.store=SiteSamples(self.tmp.name,wall=lambda:self.now);self.sid=self.store.start()
        self.people={'p1':{'name':'合成人员'}}
    def tearDown(self):self.store.db.close();self.tmp.cleanup()
    def add(self,i=0,person='p1',camera='cam_1',feature=None):
        q=feature if feature is not None else np.eye(512,dtype=np.float32)[i]
        return self.store.add(self.sid,camera,'frame'+str(i),1,person,.6,'合成姿态',self.now,{'sharpness':800},b'synthetic',q)
    def test_sharpness_boundary_and_existing_candidates(self):
        from face_adaptation import sharp_enough
        for value in (799.99, None, float('nan'), float('inf')):
            self.assertFalse(sharp_enough({'sharpness':value}))
        self.assertTrue(sharp_enough({'sharpness':800}))
        self.assertEqual(self.store.add(self.sid,'cam_1','low',1,None,.6,'pose',self.now,{'sharpness':799},b'synthetic',np.eye(512)[0]),'low_sharpness')
        self.assertEqual(self.add(),'saved')
        self.store.db.execute("UPDATE samples SET quality=?", ('{"sharpness":60}',));self.store.db.commit();self.store.db.close()
        self.store=SiteSamples(self.tmp.name,wall=lambda:self.now)
        self.assertEqual(self.store.status(self.people)['samples'][0]['status'],'rejected')
    def test_expiry_and_stop(self):
        self.assertEqual(self.add(),'saved');self.now+=86401
        self.assertIsNone(self.store.active());self.assertEqual(self.add(1),'inactive')
        self.sid=self.store.start();self.store.stop();self.assertEqual(self.add(1),'inactive')
    def test_interval_and_per_person_cap_across_cameras(self):
        self.assertEqual(self.add(),'saved');self.assertEqual(self.add(1),'interval')
        for i in range(1,10):self.now+=61;self.assertEqual(self.add(i),'saved')
        self.now+=61;self.assertEqual(self.add(10),'camera_limit')
        for i in range(10,20):self.now+=61;self.assertEqual(self.add(i,camera='cam_2'),'saved')
        self.now+=61;self.assertEqual(self.add(21,camera='cam_entrance'),'person_limit')
    def test_suggestion_never_automatically_enrolled(self):
        self.add();row=self.store.status(self.people)['samples'][0]
        self.assertEqual(row['status'],'pending');self.assertIsNone(row['person'])
        self.assertNotIn('feature',row);self.assertNotIn('embedding',row)
    def test_review_requires_existing_person(self):
        self.add();sid=self.store.status(self.people)['samples'][0]['id']
        with self.assertRaises(ValueError):self.store.review([sid],'not-registered',self.people)
        self.store.review([sid],'p1',self.people)
        self.assertEqual(self.store.status(self.people)['samples'][0]['status'],'approved')
    def test_similar_same_pose_rejected(self):
        self.add();self.now+=61
        self.assertEqual(self.add(1,feature=np.eye(512,dtype=np.float32)[0]),'similar')
    def test_unreview_only_pending_gallery_samples(self):
        self.add();sid=self.store.status(self.people)['samples'][0]['id']
        self.store.review([sid],'p1',self.people);self.store.unreview([sid])
        row=self.store.status(self.people)['samples'][0]
        self.assertEqual(row['status'],'pending');self.assertIsNone(row['person'])
    def test_invalid_vector_and_path_traversal(self):
        self.assertEqual(self.add(feature=np.zeros(512)),'invalid_feature')
        with self.assertRaises(ValueError):self.store.review(['../x'],'p1',self.people)
    def test_enrollment_requires_review_and_keeps_original(self):
        import cv2
        self.add();row=self.store.status(self.people)['samples'][0];sid=row['id']
        faces=Path(self.tmp.name)/'gallery';old=faces/'p1_original';old.mkdir(parents=True);(old/'keep.txt').write_text('original')
        with self.assertRaises(ValueError):self.store.enroll('p1',faces,self.people)
        self.store.review([sid],'p1',self.people)
        cv2.imwrite(str(self.store.root/(sid+'.jpg')),np.full((100,100,3),128,np.uint8))
        result=self.store.enroll('p1',faces,self.people)
        self.assertEqual(result['added'],1);self.assertTrue((old/'keep.txt').exists())
        with self.assertRaises(ValueError):self.store.enroll('p1',faces,self.people)


class LocalAccessTests(unittest.TestCase):
    def test_local_origin_allowed_without_token_remote_and_cross_site_rejected(self):
        from types import SimpleNamespace
        from face_adaptation_api import operator
        runtime=SimpleNamespace(service=SimpleNamespace(authenticate=lambda h:None))
        request=SimpleNamespace(client_address=('127.0.0.1',123),headers={'Host':'127.0.0.1:5173','Origin':'http://127.0.0.1:5173'})
        self.assertEqual(operator(request,runtime)['id'],'local-face-calibration')
        request.client_address=('192.168.31.5',123);self.assertIsNone(operator(request,runtime))
        request.client_address=('127.0.0.1',123);request.headers['Origin']='https://other.example';self.assertIsNone(operator(request,runtime))
        request.headers.pop('Origin');request.headers['Sec-Fetch-Site']='cross-site';self.assertIsNone(operator(request,runtime))
        request.headers={'Host':'other.example'};self.assertIsNone(operator(request,runtime))


if __name__=='__main__':unittest.main()
