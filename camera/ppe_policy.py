"""Configurable PPE confirmation and escalation using distinct observed frames."""
import json
import uuid


class PPEPolicy:
    def __init__(self,service):
        self.service=service
        with service.lock,service.db:
            service.db.execute('CREATE TABLE IF NOT EXISTS ppe_states(key TEXT PRIMARY KEY,payload TEXT)')

    def observe(self,person,kind,instrument,observed,frame,state):
        s=self.service;policy=s.config.get('ppe_confirmation')
        if not policy:return {} if state=='not_worn' else None
        key=json.dumps([person,kind,instrument])
        with s.lock,s.db:
            saved=s.db.execute('SELECT payload FROM ppe_states WHERE key=?',(key,)).fetchone()
            session=json.loads(saved['payload']) if saved else None
            if session and observed<session['last']:return None
            if session and session['emitted'] and 'first_alert_at' not in session:
                session['first_alert_at']=observed
            if state=='worn':
                if session and session['emitted']:s.incidents.recover_ppe(session['id'],observed)
                s.db.execute('DELETE FROM ppe_states WHERE key=?',(key,));return None
            if s.preparation.active(person,observed):
                # Preparation samples cannot become post-deadline evidence.
                if session and not session['emitted']:
                    s.db.execute('DELETE FROM ppe_states WHERE key=?',(key,))
                return None
            if state!='not_worn':
                if session:
                    session['frames']=[];session['continuous_since']=None;session['escalation_since']=None
                    s.db.execute('UPDATE ppe_states SET payload=? WHERE key=?',(json.dumps(session),key))
                return None
            if not session:
                session={'id':'ppe:'+uuid.uuid4().hex,'last':observed,'frames':[],'continuous_since':observed,
                         'emitted':False,'escalated':False}
            if frame in [row[1] for row in session['frames']]:return None
            if session['continuous_since'] is None or observed-session['last']>policy['maximum_gap_seconds']:
                session['frames']=[];session['continuous_since']=observed;session['escalation_since']=None
            if session['emitted'] and session.get('escalation_since') is None:
                session['escalation_since']=observed
            session['last']=observed
            if frame not in [row[1] for row in session['frames']]:session['frames'].append([observed,frame])
            session['frames']=session['frames'][-100:]
            ready=(len(session['frames'])>=policy['minimum_frames'] and
                   observed-session['continuous_since']>=policy['minimum_seconds'])
            output=None
            if ready and not session['emitted']:
                session['emitted']=True
                session['first_alert_at']=observed
                session['escalation_since']=observed
                output={'incident_id':session['id'],'severity':'L1','audiences':'experimenters'}
            elif ready and session['emitted'] and not session['escalated'] and observed-session.get('escalation_since',observed)>=policy['escalation_seconds']:
                session['escalated']=True
                output={'incident_id':session['id'],'severity':'L2','audiences':'experimenters,managers'}
            s.db.execute('INSERT OR REPLACE INTO ppe_states VALUES(?,?)',(key,json.dumps(session)))
            return output

    def retry(self,incident,severity='L1'):
        """A missing picture must not consume the only alert for an episode."""
        s=self.service
        with s.lock,s.db:
            for row in s.db.execute('SELECT key,payload FROM ppe_states'):
                data=json.loads(row['payload'])
                if data['id']==incident:
                    if severity=='L2':data['escalated']=False
                    else:data['emitted']=False
                    s.db.execute('UPDATE ppe_states SET payload=? WHERE key=?',(json.dumps(data),row['key']))
