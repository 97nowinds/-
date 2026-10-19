"""Merge fresh visual/hardware signals only inside explicitly mapped coverage zones."""
import json
import math
import uuid
import threading


def validate_config(config):
    policy=config.get('fire_fusion',{})
    for key in ('camera_zones','device_zones'):
        mapping=policy.get(key,{})
        if not isinstance(mapping,dict) or any(not isinstance(k,str) or not k or len(k)>64 or
            not isinstance(v,str) or not v or len(v)>64 for k,v in mapping.items()):
            raise ValueError('invalid fire coverage mapping')
    for key,default in [('confirmation_window_seconds',10),('visual_minimum_seconds',1),('visual_maximum_gap_seconds',5)]:
        value=policy.get(key,default)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=60:
            raise ValueError('invalid fire observation window')
    count=policy.get('visual_minimum_frames',2)
    if type(count) is not int or not 2<=count<=30:raise ValueError('distinct visual frames required')
    confidence=policy.get('visual_minimum_confidence',.9)
    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not .5<=confidence<=1:
        raise ValueError('invalid visual fire confidence')


class FireFusion:
    def __init__(self,service):
        self.service=service
        self.visual_lock=threading.RLock()
        validate_config(service.config)
        with service.lock,service.db:
            service.db.executescript('''CREATE TABLE IF NOT EXISTS fire_sources(
                zone TEXT,source TEXT,kind TEXT,observed REAL,alarm INTEGER,incident TEXT,
                PRIMARY KEY(zone,source));
                CREATE TABLE IF NOT EXISTS fire_visual_sequences(camera TEXT PRIMARY KEY,payload TEXT);
                CREATE TABLE IF NOT EXISTS fire_scene_evidence(incident TEXT,source TEXT,payload TEXT,PRIMARY KEY(incident,source));
            ''')

    def needs_evidence(self,zone,source,observed):
        s=self.service
        with s.lock:
            incident=s.db.execute("SELECT id FROM incidents WHERE lab=? AND zone_id=? AND kind='fire' AND active=1",
                                 (s.config['lab_id'],zone)).fetchone()
            if not incident:return True
            if not s.db.execute('SELECT 1 FROM fire_scene_evidence WHERE incident=? AND source=?',(incident['id'],source)).fetchone():return True
            if s.db.execute('SELECT 1 FROM events WHERE id=?',(incident['id']+':dual_confirmed',)).fetchone():return False
            window=s.config.get('fire_fusion',{}).get('confirmation_window_seconds',10)
            return any(row['alarm'] and abs(observed-row['observed'])<=min(window,5) for row in s.db.execute(
                "SELECT alarm,observed FROM fire_sources WHERE incident=? AND kind='hardware'",(incident['id'],)))

    def invalidate(self,zone,source,observed=None):
        """An explicit unknown invalidates a previous clear; it never means safe."""
        s=self.service
        with s.lock,s.db:
            if observed is None:
                s.db.execute('UPDATE fire_sources SET alarm=NULL WHERE zone=? AND source=?',(zone,source))
            else:s.db.execute('UPDATE fire_sources SET alarm=NULL,observed=? WHERE zone=? AND source=? AND observed<?',
                              (observed,zone,source,observed))

    def observe(self,zone,source,kind,observed,alarm,device,evidence=None):
        s=self.service;now=s.wall()
        if kind not in {'hardware','visual'} or type(alarm) is not bool:return False
        if not -2<=now-observed<=(5 if kind=='hardware' else s.config.get('event_max_age_seconds',15)):return False
        with s.lock,s.db:
            previous=s.db.execute('SELECT * FROM fire_sources WHERE zone=? AND source=?',(zone,source)).fetchone()
            if previous and observed<=previous['observed']:return False
            current=s.db.execute("SELECT * FROM incidents WHERE lab=? AND zone_id=? AND kind='fire' AND active=1",
                                 (s.config['lab_id'],zone)).fetchone()
            if not alarm and (not current or not previous or previous['incident']!=current['id']):return False
            opened=False
            if not current:
                incident='fire:'+uuid.uuid4().hex;opened=True
                s.db.execute('INSERT INTO incidents(id,lab,device,kind,active,state,opened,last_observed,zone_id) VALUES(?,?,?,?,?,?,?,?,?)',
                    (incident,s.config['lab_id'],device,'fire',1,'open',now,observed,zone))
            else:incident=current['id']
            s.db.execute('INSERT OR REPLACE INTO fire_sources VALUES(?,?,?,?,?,?)',(zone,source,kind,observed,int(alarm),incident))
            if evidence:s.db.execute('INSERT OR REPLACE INTO fire_scene_evidence VALUES(?,?,?)',(incident,source,json.dumps(evidence)))
            s.db.execute('UPDATE incidents SET last_observed=MAX(last_observed,?) WHERE id=?',(observed,incident))
            rows=s.db.execute('SELECT * FROM fire_sources WHERE incident=?',(incident,)).fetchall()
            window=s.config.get('fire_fusion',{}).get('confirmation_window_seconds',10)
            positive=[r for r in rows if r['alarm'] and -2<=now-r['observed']<=min(window,5 if r['kind']=='hardware' else 15)]
            dual=({r['kind'] for r in positive}=={'visual','hardware'} and
                  max(r['observed'] for r in positive)-min(r['observed'] for r in positive)<=window)
            if opened:return self.emit(incident,zone,device,'single_source',observed,rows,evidence)
            if dual:
                return self.emit(incident,zone,device,'dual_confirmed',observed,rows,evidence)
            if all(r['alarm']==0 and -2<=now-r['observed']<=(5 if r['kind']=='hardware' else 15) for r in rows):
                s.db.execute("UPDATE incidents SET active=0,state='recovered',recovered=? WHERE id=?",(now,incident))
                for row in s.db.execute('SELECT id,payload FROM events').fetchall():
                    if json.loads(row['payload']).get('incident_id')==incident:
                        s.db.execute("UPDATE outbox SET state='cancelled' WHERE event_id=? AND state='pending'",(row['id'],))
                return self.emit(incident,zone,device,'recovered',observed,rows)
        return False

    def emit(self,incident,zone,device,stage,observed,rows,evidence=None):
        from management_notifications import iso
        s=self.service;event_id=incident+':'+stage
        # A repeated dual signal is not a new notification and may carry a new
        # observation timestamp; check idempotency before constructing its body.
        if s.db.execute('SELECT 1 FROM events WHERE id=?',(event_id,)).fetchone():return False
        if stage=='dual_confirmed' and not evidence:
            row=s.db.execute('SELECT payload FROM fire_scene_evidence WHERE incident=? ORDER BY rowid DESC LIMIT 1',(incident,)).fetchone()
            if row:evidence=json.loads(row['payload'])
        if stage=='dual_confirmed' and not evidence:
            for row in s.db.execute('SELECT payload FROM events ORDER BY received DESC').fetchall():
                prior=json.loads(row['payload'])
                if prior.get('incident_id')==incident and prior.get('evidence_id'):
                    evidence={key:prior[key] for key in ('evidence_id','camera_id','snapshot_id','image_observed_at') if key in prior}
                    break
        event={'event_id':event_id,'incident_id':incident,'event_type':
            'environment.flame_recovered' if stage=='recovered' else 'environment.flame',
            'severity':'L3','device_id':device,'lab_id':s.config['lab_id'],'zone_id':zone,
            'fire_stage':stage,'sources':','.join(sorted({r['kind'] for r in rows})),
            'observed_at':iso(observed),'evidence_source':'fire_signal_fusion'}
        for kind in ('hardware','visual'):
            times=[r['observed'] for r in rows if r['kind']==kind]
            if times:event[kind+'_observed_at']=iso(max(times))
        if evidence:event.update(evidence)
        return s.record(event)
