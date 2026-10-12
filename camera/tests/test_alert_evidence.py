import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alert_evidence import AlertEvidence,EvidenceUnavailable,group_evidence_payload
from handoff_result_index import SnapshotTrackIndex
from feishu_app import protect


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        stream=BytesIO();Image.new('RGB',(200,100),'white').save(stream,format='JPEG');self.jpeg=stream.getvalue()
        self.index=SnapshotTrackIndex()
        self.event={'snapshot_id':'s1','camera':{'camera_id':'c1','image_url':'http://127.0.0.1:5175/api/handoff/snapshots/s1.jpg',
            'image_sha256':hashlib.sha256(self.jpeg).hexdigest(),'frame_width':200,'frame_height':100,
            'source_frame_id':'source1','tracks':[{'track_id':'t1','person_id':'p1','display_name':'合成人员',
            'observed_at':'2026-10-12T08:00:00+08:00','bbox':[20,10,50,70],'bbox_format':'xywh_pixels'}]}}
        self.index.remember(self.event)
        self.row={'camera_id':'c1','snapshot_id':'s1','track_id':'t1','person_id':'p1','captured_at':'2026-10-12T08:00:00+08:00'}
        self.store=AlertEvidence(self.temp.name,['http://127.0.0.1:5175'],fetch=lambda url:self.jpeg)

    def tearDown(self):self.temp.cleanup()

    def test_exact_image_box_and_encrypted_storage(self):
        key=self.store.capture(self.row,self.index)
        annotated=self.store.read(key)
        self.assertTrue(annotated.startswith(b'\xff\xd8'))
        image=Image.open(BytesIO(annotated));red,green,blue=image.getpixel((20,30))
        self.assertGreater(red,green+60)
        metadata=json.loads(protect((Path(self.temp.name)/key/'metadata.dpapi').read_bytes(),True))
        self.assertEqual(metadata['source_frame_id'],'source1')
        self.assertEqual(metadata['source_sha256'],hashlib.sha256(self.jpeg).hexdigest())
        self.assertNotEqual((Path(self.temp.name)/key/'original.dpapi').read_bytes(),self.jpeg)

    def test_unmatched_wrong_digest_dimensions_and_bbox_rejected(self):
        with self.assertRaises(EvidenceUnavailable):self.store.capture({**self.row,'track_id':'other'},self.index)
        for field,value in [('image_sha256','wrong'),('frame_width',640)]:
            original=self.event['camera'][field];self.event['camera'][field]=value;self.index.remember(self.event)
            with self.assertRaises(EvidenceUnavailable):self.store.capture(self.row,self.index)
            self.event['camera'][field]=original
        self.event['camera']['tracks'][0]['bbox']=[180,0,50,30];self.index.remember(self.event)
        with self.assertRaises(EvidenceUnavailable):self.store.capture(self.row,self.index)

    def test_receiver_url_not_used_and_unapproved_origin_rejected(self):
        self.row['image_url']='http://evil.example/secret'
        self.store.capture(self.row,self.index)
        self.event['camera']['image_url']='http://evil.example/api/handoff/snapshots/s1.jpg';self.index.remember(self.event)
        with self.assertRaises(EvidenceUnavailable):self.store.capture(self.row,self.index)

    def test_mutated_same_snapshot_rejected(self):
        self.store.capture(self.row,self.index)
        self.event['camera']['tracks'][0]['bbox']=[30,10,50,70];self.index.remember(self.event)
        with self.assertRaises(EvidenceUnavailable):self.store.capture(self.row,self.index)

    def test_post_contains_one_text_and_image_and_signature(self):
        payload=group_evidence_payload('实验室合成测试','img_synthetic','secret',now=123)
        self.assertEqual(payload['msg_type'],'post')
        body=payload['content']['post']['zh_cn']['content']
        self.assertEqual(body[1][0],{'tag':'img','image_key':'img_synthetic'})
        self.assertIn('sign',payload)

    def test_image_upload_uses_multipart_and_app_token(self):
        from feishu_app import FeishuAppClient
        client=FeishuAppClient({'app_id':'cli_synthetic','app_secret':'synthetic'},
            request=lambda *args:{'code':0,'expire':7200,'tenant_access_token':'synthetic'})
        class Response(BytesIO):
            def __enter__(self):return self
            def __exit__(self,*args):self.close()
        requests=[]
        def upload(req,timeout):
            requests.append(req);return Response(b'{"code":0,"data":{"image_key":"img_synthetic"}}')
        with patch('feishu_app.urlopen',side_effect=upload):
            self.assertEqual(client.upload_image(self.jpeg),'img_synthetic')
        self.assertIn(self.jpeg,requests[0].data)
        self.assertIn(b'name="image_type"',requests[0].data)
        self.assertEqual(requests[0].get_header('Authorization'),'Bearer synthetic')

    def test_mask_result_to_one_group_post_with_exact_evidence(self):
        from management_notifications import ManagementNotifications,EvidenceAdapter,iso
        now=2000000000.
        event=self.event
        event['camera']['tracks'][0]['observed_at']=iso(now-1);self.index.remember(event)
        config={'lab_id':'lab','principals':[], 'evidence_required_events':['safety.no_mask'],
                'silent_events':['person.entered','instrument.started'],
                'routes':[{'id':'experimenters','labs':['lab'],'event_types':['safety.no_mask'],
                           'webhook_env':'HOOK'}]}
        sent=[]
        service=ManagementNotifications(Path(self.temp.name)/'events.db',config,
                    {'HOOK':'https://open.feishu.cn/open-apis/bot/v2/hook/synthetic'},wall=lambda:now,
                    sender=lambda url,payload:sent.append(payload))
        service.evidence=self.store
        row={**self.row,'captured_at':iso(now-1),'published_at':iso(now),'expires_at':iso(now+5),
             'time_domain':'authority','analysis_id':'analysis1','status':'valid','mask_state':'not_worn','confidence':.95}
        try:
            self.assertTrue(EvidenceAdapter(service).analysis(row,self.index,mask=True))
            picture_key='img_synthetic'
            from types import SimpleNamespace
            client=SimpleNamespace(credentials={'app_id':'cli_synthetic'},upload_image=lambda jpeg:picture_key)
            path=Path(self.temp.name)/'image-app.dpapi';path.write_bytes(b'synthetic')
            with patch('feishu_app.EVIDENCE_CREDENTIALS',path),patch('feishu_app.load_credentials',return_value={}), \
                    patch('feishu_app.FeishuAppClient',return_value=client):
                self.assertTrue(service.deliver_once())
            self.assertEqual(len(sent),1)
            self.assertEqual(sent[0]['msg_type'],'post')
            self.assertIn('合成人员',sent[0]['content']['post']['zh_cn']['content'][0][0]['text'])
            self.assertEqual(sent[0]['content']['post']['zh_cn']['content'][1][0]['image_key'],picture_key)
        finally:service.db.close()

    def test_silent_normal_event_recorded_without_delivery(self):
        from management_notifications import ManagementNotifications,iso
        config={'lab_id':'lab','principals':[],'silent_events':['person.entered'],
            'routes':[{'id':'group','labs':['lab'],'event_types':['person.entered'],'webhook_env':'HOOK'}]}
        service=ManagementNotifications(Path(self.temp.name)/'events.db',config,wall=lambda:2000000000.)
        try:
            service.record({'event_id':'normal1','event_type':'person.entered','lab_id':'lab',
                            'person_id':'p1','observed_at':iso(2000000000.)})
            self.assertEqual(len(service.events({'role':'admin'})),1)
            self.assertEqual(service.events({'role':'admin'})[0]['deliveries'],[])
        finally:service.db.close()


if __name__=='__main__':unittest.main()
