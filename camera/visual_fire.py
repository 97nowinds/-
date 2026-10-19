"""Validate visual fire results against original scene snapshots, without person identity."""
import json
import math


class VisualFireAdapter:
    def __init__(self,service):self.service=service

    def analysis(self,row,index):
        from management_notifications import stamp
        s=self.service;now=s.wall();policy=s.config.get('fire_fusion',{})
        if (row.get('evidence_type')!='fire' or row.get('status')!='valid' or
            row.get('time_domain')!='authority' or not row.get('analysis_id')):return False
        camera=row['camera_id'];snapshot=row['snapshot_id']
        frame=index.evidence_frame(camera,snapshot)
        if not frame or frame.get('time_domain')!='authority' or not frame.get('captured_at'):return False
        observed=stamp(row['captured_at'])
        if (abs(observed-stamp(frame['captured_at']))>.005 or
            not -2<=now-observed<=s.config.get('event_max_age_seconds',15) or
            min(stamp(row['expires_at']),stamp(row['published_at'])+5)<=now or stamp(row['published_at'])>now+2):return False
        confidence=row.get('confidence');state=row.get('fire_state')
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:return False
        if state not in {'detected','clear','unknown'}:raise ValueError('invalid fire state')
        if confidence<policy.get('visual_minimum_confidence',.9):state='unknown'
        # Serialize per-camera sequence and scene capture so concurrent repeated
        # POSTs cannot count the same source frame more than once.
        with s.fire_fusion.visual_lock:
            with s.lock,s.db:
                saved=s.db.execute('SELECT payload FROM fire_visual_sequences WHERE camera=?',(camera,)).fetchone()
                seq=json.loads(saved['payload']) if saved else {'last':-1,'frames':[],'last_frame':None}
                source=frame.get('source_frame_id') or snapshot
                if observed<=seq['last'] or source==seq['last_frame'] or source in [r[1] for r in seq['frames']]:return False
                if observed-seq['last']>policy.get('visual_maximum_gap_seconds',5) or seq.get('state')!=state:seq['frames']=[]
                seq.update(last=observed,last_frame=source,state=state)
                if state in {'detected','clear'}:seq['frames']=(seq['frames']+[[observed,source]])[-30:]
                else:seq['frames']=[]
                s.db.execute('INSERT OR REPLACE INTO fire_visual_sequences VALUES(?,?)',(camera,json.dumps(seq)))
                ready=(len(seq['frames'])>=policy.get('visual_minimum_frames',2) and
                       observed-seq['frames'][0][0]>=policy.get('visual_minimum_seconds',1))
            zone=policy.get('camera_zones',{}).get(camera,'camera:'+camera)
            if state=='unknown':
                s.fire_fusion.invalidate(zone,'visual:'+camera,observed)
                return False
            if not ready:return False
            evidence=None
            if state=='detected' and s.fire_fusion.needs_evidence(zone,'visual:'+camera,observed):
                try:
                    if s.evidence is None:return False
                    evidence={'evidence_id':s.evidence.capture_scene(row,index),'camera_id':camera,
                              'snapshot_id':snapshot,'image_observed_at':row['captured_at']}
                except Exception:
                    with s.lock,s.db:s._audit('system','fire.evidence_unavailable',str(snapshot)[:128])
                    return False
            return s.fire_fusion.observe(zone,'visual:'+camera,'visual',observed,state=='detected','camera:'+camera,evidence)
