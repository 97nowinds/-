"""Synthetic observations only. No serial device or real Feishu endpoint is used."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alert_evidence import AlertEvidence
from handoff_result_index import SnapshotTrackIndex
from management_notifications import ManagementNotifications,iso
from visual_fire import VisualFireAdapter


class FireTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.now=2000000000.;self.admin={'role':'admin','id':'maintenance'}
        self.config={'lab_id':'lab','principals':[{'id':'detector','role':'producer','labs':['lab'],'cameras':['c1'],'token_env':'TOKEN'}],
            'routes':[{'id':name,'labs':['lab'],'event_types':['environment.flame','environment.flame_recovered'],
                       'webhook_env':name} for name in ['experimenters','managers','administrators']],
            'fire_fusion':{'device_zones':{'sensor1':'zoneA'},'camera_zones':{'c1':'zoneA'},
                'confirmation_window_seconds':10,'visual_minimum_frames':2,'visual_minimum_seconds':1}}
        self.service=ManagementNotifications(Path(self.temp.name)/'db',self.config,{'TOKEN':'synthetic-producer'},wall=lambda:self.now,sender=lambda *args:None)
        self.index=SnapshotTrackIndex();self.visual=VisualFireAdapter(self.service)
        stream=BytesIO();Image.new('RGB',(200,100),'white').save(stream,format='JPEG');self.jpeg=stream.getvalue()
        self.service.evidence=AlertEvidence(Path(self.temp.name)/'evidence',['http://127.0.0.1:5175'],fetch=lambda url:self.jpeg)
        self.number=0

    def tearDown(self):self.service.db.close();self.temp.cleanup()

    def hardware(self,alarm=True,device='sensor1'):
        return self.service.incidents.observe_environment({'connected':True,'status':'alarm' if alarm else 'online',
            'device_id':device,'flame_alarm':alarm,'flame_observed_at':self.now})

    def row(self,state='detected',camera='c1',domain='authority',**updates):
        self.number+=1;snapshot='synthetic'+str(self.number)
        self.index.remember({'snapshot_id':snapshot,'captured_at':iso(self.now),'time_domain':domain,'camera':{
            'camera_id':camera,'source_frame_id':'source'+str(self.number),'tracks':[],
            'frame_width':200,'frame_height':100,'image_sha256':hashlib.sha256(self.jpeg).hexdigest(),
            'image_url':'http://127.0.0.1:5175/api/handoff/snapshots/'+snapshot+'.jpg'}})
        return dict({'evidence_type':'fire','analysis_id':'analysis'+str(self.number),'camera_id':camera,
            'snapshot_id':snapshot,'captured_at':iso(self.now),'published_at':iso(self.now),'expires_at':iso(self.now+5),
            'time_domain':'authority','status':'valid','fire_state':state,'confidence':.99,
            'fire_bbox':[20,10,30,40],'bbox_format':'xywh_pixels'},**updates)

    def feed(self,state='detected',**updates):return self.visual.analysis(self.row(state,**updates),self.index)

    def events(self):return self.service.events(self.admin)

    def test_hardware_first_then_visual_upgrades_same_incident_once(self):
        self.assertTrue(self.hardware());incident=self.events()[0]['incident_id']
        self.assertFalse(self.feed());self.now+=1
        self.assertTrue(self.feed());self.assertEqual(len(self.events()),2)
        confirmed=self.events()[0];self.assertEqual(confirmed['incident_id'],incident)
        self.assertEqual(confirmed['fire_stage'],'dual_confirmed');self.assertIn('evidence_id',confirmed)
        self.assertEqual(len(confirmed['deliveries']),3)
        self.now+=1;self.assertFalse(self.feed());self.assertFalse(self.hardware())
        self.assertEqual(len(self.events()),2)

    def test_visual_without_person_then_hardware_keeps_picture_and_real_time(self):
        self.assertFalse(self.feed());self.now+=1;self.assertTrue(self.feed())
        first=self.events()[0];self.assertNotIn('person_id',first)
        self.assertEqual(first['fire_stage'],'single_source')
        self.now+=1;self.assertTrue(self.hardware())
        event=self.events()[0];self.assertEqual(event['evidence_id'],first['evidence_id'])
        self.assertEqual(event['image_observed_at'],first['image_observed_at'])
        self.assertNotEqual(event['observed_at'],event['image_observed_at'])
        self.assertIn('双源',self.service.message(event))

    def test_different_coverage_is_not_fused(self):
        self.hardware();self.assertFalse(self.feed(camera='c2'));self.now+=1;self.assertTrue(self.feed(camera='c2'))
        self.assertEqual(len(self.service.incidents.list(self.admin)),2)
        self.assertFalse(any(e.get('fire_stage')=='dual_confirmed' for e in self.events()))

    def test_stale_hardware_cannot_confirm_visual(self):
        self.hardware();self.now+=11;self.feed();self.now+=1;self.feed()
        self.assertFalse(any(e.get('fire_stage')=='dual_confirmed' for e in self.events()))
        self.assertTrue(self.hardware());self.assertEqual(self.events()[0]['fire_stage'],'dual_confirmed')
        self.assertIn('evidence_id',self.events()[0])

    def test_hardware_five_second_freshness_and_disconnect_are_enforced(self):
        self.hardware();self.now+=5;self.feed();self.now+=1;self.feed()
        self.assertFalse(any(e.get('fire_stage')=='dual_confirmed' for e in self.events()))
        self.hardware()
        self.now+=1
        self.service.incidents.observe_environment({'device_id':'sensor1','connected':False})
        self.feed('clear');self.now+=1;self.feed('clear')
        self.assertTrue(self.service.incidents.list(self.admin)[0]['active'])
        self.assertTrue(self.hardware(False))

    def test_existing_hardware_incident_is_not_reissued_when_mapping_is_added(self):
        self.service.config['fire_fusion']['device_zones']={}
        self.hardware();incident=self.events()[0]['incident_id']
        self.service.config['fire_fusion']['device_zones']={'sensor1':'zoneA'}
        self.now+=1;self.assertFalse(self.hardware())
        self.feed();self.now+=1;self.assertTrue(self.feed())
        self.assertEqual(self.events()[0]['incident_id'],incident)
        self.assertEqual(len(self.service.incidents.list(self.admin)),1)

    def test_recovery_requires_fresh_clear_from_each_alarmed_source(self):
        self.hardware();self.feed();self.now+=1;self.feed()
        self.now+=1;self.hardware(False)
        self.assertTrue(self.service.incidents.list(self.admin)[0]['active'])
        self.feed('unknown');self.now+=1;self.feed('clear')
        self.assertTrue(self.service.incidents.list(self.admin)[0]['active'])
        self.now+=1;self.assertTrue(self.feed('clear'))
        incident=self.service.incidents.list(self.admin)[0]
        self.assertFalse(incident['active']);self.assertEqual(incident['state'],'recovered')
        self.assertEqual(self.events()[0]['event_type'],'environment.flame_recovered')
        self.service.incidents.transition(self.admin,incident['id'],'close','合成现场核查')

    def test_expired_clear_and_unknown_do_not_resolve_incident(self):
        self.feed();self.now+=1;self.feed();self.now+=1;self.hardware()
        self.now+=1;self.feed('clear');self.now+=1;self.feed('clear')
        self.now+=6;self.assertFalse(self.feed('unknown'))
        self.assertTrue(self.service.incidents.list(self.admin)[0]['active'])
        self.hardware(False)
        self.assertTrue(self.service.incidents.list(self.admin)[0]['active'])

    def test_bad_time_domain_and_duplicate_frame_cannot_generate_visual_alarm(self):
        self.assertFalse(self.feed(domain='source'));self.assertEqual(self.events(),[])
        row=self.row();self.assertFalse(self.visual.analysis(row,self.index))
        self.now+=1;self.assertFalse(self.visual.analysis(row,self.index))
        self.assertEqual(self.events(),[])
        self.assertTrue(self.feed())

    def test_missing_picture_and_invalid_box_do_not_emit_visual_event(self):
        self.feed();self.now+=1;self.assertFalse(self.feed(fire_bbox=[190,0,50,30]))
        self.assertEqual(self.events(),[])
        self.service.evidence.fetch=lambda url:b'bad JPEG'
        self.now+=1;self.assertFalse(self.feed());self.assertEqual(self.events(),[])
        self.service.evidence.fetch=lambda url:self.jpeg
        self.now+=1;self.assertTrue(self.feed())

    def test_low_confidence_breaks_visual_sequence(self):
        self.feed();self.now+=1;self.assertFalse(self.feed(confidence=.4))
        self.now+=1;self.assertFalse(self.feed());self.now+=1;self.assertTrue(self.feed())

    def test_visual_alarm_persists_without_rewriting_image_every_frame(self):
        from unittest.mock import patch
        capture=self.service.evidence.capture_scene
        with patch.object(self.service.evidence,'capture_scene',wraps=capture) as mock:
            self.feed();self.now+=1;self.feed()
            self.now+=1;self.feed();self.now+=1;self.feed()
            self.assertEqual(mock.call_count,1)

    def test_visual_scene_picture_delivered_to_three_fake_groups_without_identity(self):
        from unittest.mock import patch
        from feishu_app import protect
        self.feed();self.now+=1;self.assertTrue(self.feed(fire_bbox=None))
        event=self.events()[0]
        metadata=json.loads(protect((Path(self.temp.name)/'evidence'/event['evidence_id']/'metadata.dpapi').read_bytes(),True))
        self.assertIsNone(metadata['person_id']);self.assertIsNone(metadata['track_id'])
        sent=[];uploads=[]
        self.service.sender=lambda url,payload:sent.append((url,payload))
        for key in ['experimenters','managers','administrators']:self.service.env[key]='https://synthetic.invalid/'+key
        fake_path=Path(self.temp.name)/'synthetic.dpapi';fake_path.write_bytes(b'synthetic')
        client=SimpleNamespace(credentials={'app_id':'cli_synthetic'},upload_image=lambda jpeg:uploads.append(jpeg) or 'img_synthetic')
        with patch('feishu_app.EVIDENCE_CREDENTIALS',fake_path),patch('feishu_app.load_credentials',return_value={}),patch('feishu_app.FeishuAppClient',return_value=client):
            for _ in range(3):self.now+=2;self.assertTrue(self.service.deliver_once())
        # Emergency text is immediate; the durable three-camera worker now
        # owns scene uploads instead of blocking this first notification.
        self.assertEqual(len(sent),3);self.assertEqual(len(uploads),0)
        self.assertTrue(all(payload['msg_type']=='text' for _,payload in sent))
        self.assertEqual(self.service.db.execute('SELECT count(*) FROM fire_snapshot_jobs').fetchone()[0],1)

    def test_fire_http_keeps_producer_and_camera_scope(self):
        from management_api import handle_management
        def request(row,token='synthetic-producer'):
            data=json.dumps(row).encode();output=[]
            handler=SimpleNamespace(path='/api/management/evidence',headers={'Authorization':'Bearer '+token,'Content-Length':str(len(data))},
                rfile=BytesIO(data),server=SimpleNamespace(management=SimpleNamespace(service=self.service,fire=self.visual,index=lambda:self.index)))
            handler._send_json=lambda data,status=200:output.append((status,data))
            handle_management(handler,'POST');return output[0]
        self.assertEqual(request(self.row(),token='wrong')[0],401)
        self.assertEqual(request(self.row(camera='c2'))[0],403)
        code,data=request(self.row());self.assertEqual(code,200);self.assertFalse(data['accepted'])
        self.now+=1;code,data=request(self.row());self.assertEqual(code,200);self.assertTrue(data['accepted'])


if __name__=='__main__':unittest.main()
