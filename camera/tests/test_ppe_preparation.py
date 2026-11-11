"""Synthetic workflow verification; no network or production database writes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from management_notifications import ManagementNotifications, EvidenceAdapter, iso
from handoff_result_index import SnapshotTrackIndex


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.now=2000000000.
        self.path=Path(self.temp.name)/'events.db'
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config/management.json').read_text(encoding='utf-8'))
        self.service=ManagementNotifications(self.path,self.config,environ={},wall=lambda:self.now,
            sender=lambda *_:self.fail('real send forbidden'))
        self.service.preparation.see('student',self.now,entrance_face=True)

    def tearDown(self):
        self.service.db.close();self.temp.cleanup()

    def observation(self,seconds,state='not_worn'):
        return self.service.ppe_policy.observe('student','safety.no_mask',None,
            self.now+seconds,'synthetic-frame-'+str(seconds),state)

    def test_preparation_samples_never_make_an_alert_or_count_after_deadline(self):
        for sec in [0,3,6,30,60,119]:self.assertIsNone(self.observation(sec))
        self.assertIsNone(self.observation(120))
        self.assertIsNone(self.observation(123))
        result=self.observation(125)
        self.assertEqual(result['severity'],'L1')
        self.assertEqual(result['audiences'],'experimenters')

    def test_prepared_before_deadline_and_unknown_after_deadline_do_not_alert(self):
        self.observation(0);self.observation(5)
        self.assertIsNone(self.observation(30,'worn'))
        for sec in [120,125,130]:self.assertIsNone(self.observation(sec,'unknown'))

    def test_repeated_face_or_tracker_changes_do_not_extend_window(self):
        self.service.preparation.see('student',self.now+100,entrance_face=True)
        self.assertFalse(self.service.preparation.active('student',self.now+120))
        self.service.db.close()
        self.service=ManagementNotifications(self.path,self.config,environ={},wall=lambda:self.now+121)
        self.service.preparation.see('student',self.now+121,entrance_face=True)
        self.assertFalse(self.service.preparation.active('student',self.now+121))

    def test_other_camera_identity_does_not_grant_a_window(self):
        self.service.preparation.see('other-student',self.now,entrance_face=False)
        self.assertFalse(self.service.preparation.active('other-student',self.now+1))

    def test_actual_camera_adapter_starts_window_only_for_live_confirmed_entrance_face(self):
        adapter=EvidenceAdapter(self.service)
        camera={'id':'cam_entrance','role':'entrance_identity','status':'running',
            'detection_age_seconds':.1,'capture_age_seconds':.1,
            'identities':[{'local_id':1,'global_track_id':'g2','person_id':'student2',
                'name':'合成测试人员','identity_lock_status':'locked','identity_source':'face'}],
            'yolo_tracks':[{'local_id':1,'track_id':'g2','position_estimated':False}]}
        adapter.cameras({'cameras':[{**camera,'replay':{'enabled':True}}]})
        self.assertFalse(self.service.preparation.active('student2',self.now+1))
        adapter.cameras({'cameras':[camera]})
        self.assertTrue(self.service.preparation.active('student2',self.now+1))

    def test_only_valid_true_instrument_result_can_end_window(self):
        index=SnapshotTrackIndex()
        index.remember({'snapshot_id':'synthetic-s1','time_domain':'authority',
            'camera':{'camera_id':'cam_2','tracks':[{'track_id':'t1','global_id':'g1',
                'person_id':'student','observed_at':iso(self.now-1)}]}})
        row={'snapshot_id':'synthetic-s1','camera_id':'cam_2','track_id':'t1','analysis_id':'a1',
            'instrument_id':'synthetic-instrument','captured_at':iso(self.now-1),
            'published_at':iso(self.now),'expires_at':iso(self.now+5),'time_domain':'authority',
            'status':'valid','is_interacting':False}
        adapter=EvidenceAdapter(self.service)
        self.assertFalse(adapter.analysis(row,index))
        self.assertTrue(self.service.preparation.active('student',self.now+1))
        self.assertTrue(adapter.analysis({**row,'is_interacting':True},index))
        self.assertFalse(self.service.preparation.active('student',self.now+1))
        # Instrument notifications are independently configurable. Ending the
        # preparation window must never synthesize a PPE violation.
        queued = self.service.db.execute(
            'SELECT e.payload FROM outbox o JOIN events e ON e.id=o.event_id').fetchall()
        self.assertTrue(all(json.loads(row[0])['event_type']=='instrument.started'
                            for row in queued))

    def test_confirmed_operation_ends_preparation_without_making_a_violation(self):
        self.service.preparation.finish('student',self.now+40,'confirmed_instrument_interaction')
        self.assertIsNone(self.observation(41))
        self.assertIsNone(self.observation(44))
        self.assertEqual(self.observation(46)['severity'],'L1')

    def record_first_warning(self):
        self.observation(120)
        self.observation(123)
        first = self.observation(125)
        recorded_at = self.now + 125
        self.service.wall = lambda: recorded_at
        self.assertTrue(self.service.record({**first, 'event_id':first['incident_id']+':yellow:1',
            'event_type':'safety.no_mask', 'lab_id':self.config['lab_id'], 'person_id':'student',
            'observed_at':iso(recorded_at)}))
        return first

    def test_escalation_waits_sixty_seconds_after_first_reminder(self):
        first=self.record_first_warning()
        for sec in range(130,185,5):self.assertIsNone(self.observation(sec))
        result=self.observation(185)
        self.assertEqual((result['severity'], result['warning_number']), ('L1', '2'))
        self.assertEqual(result['incident_id'],first['incident_id'])
        self.assertEqual(result['audiences'],'experimenters,managers')

    def test_rectification_cancels_unstarted_messages(self):
        self.observation(120);self.observation(123);first=self.observation(125)
        self.now+=125
        self.service.record({**first,'event_id':'synthetic-reminder','event_type':'safety.no_mask',
            'lab_id':self.config['lab_id'],'person_id':'student','evidence_id':'synthetic-evidence',
            'observed_at':iso(self.now)})
        self.service.ppe_policy.observe('student','safety.no_mask',None,self.now+1,'worn-frame','worn')
        self.assertEqual(self.service.db.execute("SELECT state FROM outbox WHERE event_id='synthetic-reminder'").fetchone()[0],'cancelled')

    def test_unknown_interrupts_continuous_escalation_time(self):
        self.record_first_warning()
        self.observation(130,'unknown')
        for sec in range(180,241,5):
            result=self.observation(sec)
            if sec<240:self.assertIsNone(result)
        self.assertEqual((result['severity'], result['warning_number']), ('L1', '2'))

    def test_operation_before_first_face_cannot_later_grant_preparation(self):
        self.service.preparation.finish('working-student',self.now,'confirmed_instrument_interaction')
        self.service.preparation.see('working-student',self.now+1,entrance_face=True)
        self.assertFalse(self.service.preparation.active('working-student',self.now+2))

    def test_fire_bypasses_personal_preparation_and_reaches_all_groups(self):
        self.service.incidents.observe_environment({'connected':True,'status':'alarm',
            'device_id':'synthetic-fire-sensor','flame_alarm':True,'flame_observed_at':self.now})
        routes={r[0] for r in self.service.db.execute('SELECT route_id FROM outbox')}
        self.assertEqual(routes,{'experimenters','managers','administrators'})

    def test_only_confirmed_new_visit_can_reset_preparation(self):
        self.service.preparation.see('student',self.now+3600,entrance_face=True)
        self.assertFalse(self.service.preparation.active('student',self.now+3601))
        self.service.preparation.close_visit('student')
        self.service.preparation.see('student',self.now+4000,entrance_face=True)
        self.assertTrue(self.service.preparation.active('student',self.now+4001))


if __name__=='__main__':unittest.main()
