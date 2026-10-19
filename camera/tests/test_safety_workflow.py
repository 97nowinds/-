import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from management_notifications import ManagementNotifications,iso
from snapshot_registry import SnapshotRegistry
from handoff_result_index import SnapshotTrackIndex


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.now=2000000000.;self.sent=[]
        self.config={'lab_id':'lab','principals':[], 'routes':[{'id':name,'audience':name,'labs':['lab'],
            'event_types':['environment.flame','environment.flame_recovered','safety.no_mask'],
            'webhook_env':name} for name in ['experimenters','managers','administrators']]}
        self.env={name:'https://open.feishu.cn/open-apis/bot/v2/hook/'+name for name in ['experimenters','managers','administrators']}
        self.service=ManagementNotifications(Path(self.temp.name)/'state.db',self.config,self.env,
            wall=lambda:self.now,sender=lambda url,payload:self.sent.append((url,payload)))
        self.admin={'id':'admin','role':'admin'}
        self.manager={'id':'manager','role':'manager','labs':['lab']}

    def tearDown(self):
        self.service.db.close();self.temp.cleanup()

    def sensor(self,alarm=True,**updates):
        return dict({'device_id':'sensor1','connected':True,'status':'alarm' if alarm else 'online',
                     'flame_alarm':alarm,'flame_observed_at':self.now},**updates)

    def test_fire_once_fanout_priority_and_recovery(self):
        self.assertTrue(self.service.incidents.observe_environment(self.sensor()))
        self.now+=1;self.assertFalse(self.service.incidents.observe_environment(self.sensor()))
        events=self.service.events(self.admin);self.assertEqual(len(events),1)
        self.assertEqual(len(events[0]['deliveries']),3)
        for _ in range(3):self.now+=2;self.service.deliver_once()
        self.assertEqual(len(self.sent),3)
        incident=events[0]['incident_id']
        self.service.incidents.transition(self.admin,incident,'ack')
        with self.assertRaises(ValueError):self.service.incidents.transition(self.admin,incident,'close','checked')
        self.now+=1;self.assertTrue(self.service.incidents.observe_environment(self.sensor(False)))
        with self.assertRaises(PermissionError):self.service.incidents.transition(self.manager,incident,'close','checked')
        self.service.incidents.transition(self.admin,incident,'close','现场核查完成（合成测试）')
        self.assertEqual(self.service.incidents.list(self.admin)[0]['state'],'closed')

    def test_stale_unknown_and_temperature_do_not_clear_fire(self):
        for sample in [self.sensor(connected=False),self.sensor(flame_observed_at=self.now-10),self.sensor(flame_alarm='false')]:
            self.assertFalse(self.service.incidents.observe_environment(sample))
        self.service.incidents.observe_environment(self.sensor())
        self.now+=10
        self.assertFalse(self.service.incidents.observe_environment(self.sensor(False,flame_observed_at=self.now-10)))
        self.assertTrue(self.service.incidents.list(self.admin)[0]['active'])

    def test_new_fire_not_suppressed_by_ordinary_cooldown(self):
        self.service.incidents.observe_environment(self.sensor())
        self.now+=1;self.service.incidents.observe_environment(self.sensor(False))
        self.now+=1;self.service.incidents.observe_environment(self.sensor())
        self.assertEqual(len(self.service.incidents.list(self.admin)),2)
        old=self.service.events(self.admin)[-1]
        self.assertTrue(all(d['state']=='cancelled' for d in old['deliveries']))

    def test_critical_retry_not_dropped_after_five_minutes(self):
        self.service.incidents.observe_environment(self.sensor())
        self.now+=1000
        self.assertTrue(self.service.deliver_once())
        self.assertEqual(len(self.sent),1)

    def test_slow_group_does_not_lock_other_group_or_api(self):
        self.service.incidents.observe_environment(self.sensor())
        entered=threading.Event();release=threading.Event()
        def sender(url,payload):
            if url.endswith('experimenters'):entered.set();release.wait(3)
            self.sent.append(url)
        self.service.sender=sender
        thread=threading.Thread(target=self.service.deliver_once);thread.start()
        try:
            self.assertTrue(entered.wait(1));self.now+=2
            self.assertTrue(self.service.deliver_once())
            self.assertEqual(len(self.service.events(self.admin)),1)
            self.assertTrue(any(url.endswith('managers') for url in self.sent))
        finally:release.set();thread.join(3)



    def test_group_delivery_does_not_require_personal_accounts(self):
        self.service.record({'event_id':'synthetic-group-only','event_type':'safety.no_mask','lab_id':'lab',
            'person_id':'synthetic-person','audiences':'experimenters','observed_at':iso(self.now)})
        self.assertTrue(self.service.deliver_once())
        self.assertEqual(len(self.sent),1)
        self.assertTrue(self.sent[0][0].endswith('experimenters'))


    def test_ppe_recovery_requires_observed_worn_and_policy_keeps_confirmed_incident(self):
        from management_policy import set_policy
        policy={'ppe_rules':{'mask':{'enabled':True,'minimum_confidence':.9}},'ppe_confirmation':{
            'minimum_frames':2,'minimum_seconds':1,'maximum_gap_seconds':5,'escalation_seconds':6}}
        set_policy(self.service,self.admin,policy)
        p=self.service.ppe_policy
        p.observe('p1','safety.no_mask',None,self.now,'f1','not_worn')
        result=p.observe('p1','safety.no_mask',None,self.now+1,'f2','not_worn')
        self.service.record({**result,'event_id':'test-ppe','event_type':'safety.no_mask','lab_id':'lab',
            'person_id':'p1','observed_at':iso(self.now)})
        set_policy(self.service,self.admin,policy)
        p.observe('p1','safety.no_mask',None,self.now+2,'f3','unknown')
        self.assertTrue(self.service.incidents.list(self.admin)[0]['active'])
        p.observe('p1','safety.no_mask',None,self.now+3,'f4','worn')
        self.assertFalse(self.service.incidents.list(self.admin)[0]['active'])
        self.service.incidents.transition(self.admin,result['incident_id'],'close','合成整改核查')

    def test_outbox_rechecks_audience_after_route_change(self):
        self.service.record({'event_id':'one-group','event_type':'safety.no_mask','lab_id':'lab','person_id':'p1',
            'audiences':'experimenters','observed_at':iso(self.now)})
        self.service.config['routes'][0]['audience']='administrators'
        self.service.deliver_once()
        self.assertEqual(self.sent,[])
        self.assertEqual(self.service.events(self.admin)[0]['deliveries'][0]['state'],'cancelled')

    def test_three_group_migration_is_once_and_keeps_later_admin_changes(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from management_api import ManagementRuntime
        import feishu_app
        root=Path(self.temp.name)/'isolated';(root/'config').mkdir(parents=True)
        path=root/'config'/'management.json'
        path.write_text(json.dumps({'lab_id':'lab','principals':[],'routes':[{
            'id':'laboratory_group','labs':['lab'],'event_types':['safety.no_mask'],'webhook_env':'OLD'}]}))
        secret_path=root/'synthetic.dpapi';secret_path.write_bytes(b'synthetic')
        credentials={'management_admin_token':'synthetic-admin','groups':{name:{'webhook':'https://synthetic.invalid'}
            for name in ['experimenters','managers','administrators']}}
        server=SimpleNamespace(api_base='http://127.0.0.1:15001',live_publisher=None)
        with patch.object(feishu_app,'EVIDENCE_CREDENTIALS',secret_path),patch.object(feishu_app,'load_credentials',return_value=credentials):
            runtime=ManagementRuntime(server,path)
            self.assertEqual({r['id'] for r in runtime.service.config['routes']},{'experimenters','managers','administrators'})
            custom=[runtime.service.config['routes'][0]]
            runtime.service.set_routes(self.admin,custom);runtime.service.db.close()
            restored=ManagementRuntime(server,path)
            try:self.assertEqual(restored.service.config['routes'],custom)
            finally:restored.service.db.close()

    def test_ppe_distinct_frames_unknown_reset_and_escalation(self):
        self.service.config['ppe_confirmation']={'minimum_frames':3,'minimum_seconds':2,'maximum_gap_seconds':5,'escalation_seconds':6}
        policy=self.service.ppe_policy
        self.assertIsNone(policy.observe('p1','safety.no_mask',None,self.now,'f1','not_worn'))
        self.assertIsNone(policy.observe('p1','safety.no_mask',None,self.now+2,'f1','not_worn'))
        policy.observe('p1','safety.no_mask',None,self.now+3,'f2','unknown')
        self.assertIsNone(policy.observe('p1','safety.no_mask',None,self.now+4,'f3','not_worn'))
        policy.observe('p1','safety.no_mask',None,self.now+5,'f4','not_worn')
        first=policy.observe('p1','safety.no_mask',None,self.now+6,'f5','not_worn');self.assertEqual(first['severity'],'L1')
        policy.observe('p1','safety.no_mask',None,self.now+8,'f6','not_worn')
        escalation=policy.observe('p1','safety.no_mask',None,self.now+10,'f7','not_worn')
        self.assertEqual(escalation['severity'],'L2');self.assertEqual(escalation['incident_id'],first['incident_id'])
        self.assertIsNone(policy.observe('p1','safety.no_mask',None,self.now+11,'f8','worn'))

    def test_registry_shared_immutable_encrypted_and_expiring(self):
        path=Path(self.temp.name)/'registry.db'
        writer=SnapshotRegistry(path,wall=lambda:self.now,retention=5)
        event={'snapshot_id':'s1','time_domain':'source','camera':{'camera_id':'c1',
               'tracks':[{'track_id':'t1','person_id':'p1','display_name':'合成测试姓名'}]}}
        writer.publish(event);reader=SnapshotRegistry(path,wall=lambda:self.now)
        index=SnapshotTrackIndex();self.assertEqual(reader.populate(index),1)
        self.assertEqual(index.resolve({'camera_id':'c1','snapshot_id':'s1','track_id':'t1'})['person_id'],'p1')
        self.assertEqual(index.evidence_track('c1','s1','t1')['time_domain'],'source')
        self.assertNotIn('合成测试姓名'.encode(),path.read_bytes())
        event['camera']['tracks'][0]['person_id']='wrong'
        with self.assertRaises(ValueError):writer.publish(event)
        self.now+=6;self.assertEqual(reader.populate(SnapshotTrackIndex()),0)


if __name__=='__main__':unittest.main()
