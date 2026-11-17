"""Persistent fire incident lifecycle driven only by local fresh hardware samples."""
import json
import math
import uuid


class SafetyIncidents:
    def __init__(self,service):
        self.service=service
        with service.lock,service.db:
            service.db.execute('''CREATE TABLE IF NOT EXISTS incidents(
                id TEXT PRIMARY KEY,lab TEXT,device TEXT,kind TEXT,active INTEGER,
                state TEXT,opened REAL,last_observed REAL,recovered REAL,
                acknowledged_by TEXT,closed_by TEXT,note TEXT)''')
            columns={row[1] for row in service.db.execute('PRAGMA table_info(incidents)')}
            if 'person_id' not in columns:service.db.execute('ALTER TABLE incidents ADD COLUMN person_id TEXT')
            if 'instrument_id' not in columns:service.db.execute('ALTER TABLE incidents ADD COLUMN instrument_id TEXT')
            if 'camera_id' not in columns:service.db.execute('ALTER TABLE incidents ADD COLUMN camera_id TEXT')
            if 'warning_event_id' not in columns:
                service.db.execute('ALTER TABLE incidents ADD COLUMN warning_event_id TEXT')
            if 'zone_id' not in columns:service.db.execute('ALTER TABLE incidents ADD COLUMN zone_id TEXT')

    def observe_environment(self,sample):
        s=self.service;now=s.wall()
        stamp=sample.get('flame_observed_at')
        device=sample.get('device_id')
        if not isinstance(device,str) or not device or len(device)>64:return False
        zone=s.config.get('fire_fusion',{}).get('device_zones',{}).get(device)
        if (not sample.get('connected') or sample.get('status') not in {'online','alarm'}
                or type(sample.get('flame_alarm')) is not bool or type(stamp) not in (int,float)
                or not math.isfinite(stamp) or not -2<=now-stamp<=5):
            if zone:s.fire_fusion.invalidate(zone,'hardware:'+device)
            return False
        if zone:
            # Carry an existing hardware incident into its newly confirmed zone;
            # a config change must not manufacture a second fire notification.
            with s.lock,s.db:
                legacy=s.db.execute("SELECT id,last_observed FROM incidents WHERE lab=? AND device=? AND kind='fire' AND active=1 AND zone_id IS NULL",
                                    (s.config['lab_id'],device)).fetchone()
                if legacy and not s.db.execute("SELECT 1 FROM incidents WHERE lab=? AND zone_id=? AND kind='fire' AND active=1",
                                              (s.config['lab_id'],zone)).fetchone():
                    s.db.execute('UPDATE incidents SET zone_id=? WHERE id=?',(zone,legacy['id']))
                    s.db.execute('INSERT OR IGNORE INTO fire_sources VALUES(?,?,?,?,?,?)',
                        (zone,'hardware:'+device,'hardware',legacy['last_observed'],1,legacy['id']))
            return s.fire_fusion.observe(zone,'hardware:'+device,'hardware',stamp,sample['flame_alarm'],device)
        with s.lock,s.db:
            current=s.db.execute('SELECT * FROM incidents WHERE lab=? AND device=? AND kind=? AND active=1',
                                (s.config['lab_id'],device,'fire')).fetchone()
            if current and stamp<=current['last_observed']:return False
            if sample['flame_alarm']:
                if current:
                    s.db.execute('UPDATE incidents SET last_observed=? WHERE id=?',(stamp,current['id']))
                    return False
                incident='fire:'+uuid.uuid4().hex
                s.db.execute('INSERT INTO incidents(id,lab,device,kind,active,state,opened,last_observed,recovered,acknowledged_by,closed_by,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (incident,s.config['lab_id'],device,'fire',1,'open',now,stamp,None,None,None,None))
                self.emit(incident,device,'environment.flame',stamp)
                return True
            if current:
                s.db.execute("UPDATE incidents SET active=0,state='recovered',recovered=?,last_observed=? WHERE id=?",
                             (now,stamp,current['id']))
                # Old pending alarms must not appear as new fires after network recovery.
                for row in s.db.execute('SELECT id,payload FROM events'):
                    item=json.loads(row['payload'])
                    if item.get('incident_id')==current['id'] and item['event_type']=='environment.flame':
                        s.db.execute("UPDATE outbox SET state='cancelled' WHERE event_id=? AND state='pending'",(row['id'],))
                self.emit(current['id'],device,'environment.flame_recovered',stamp)
                return True
        return False

    def emit(self,incident,device,kind,observed):
        from management_notifications import iso
        return self.service.record({'event_id':incident+':'+kind,'incident_id':incident,
            'event_type':kind,'severity':'L3','device_id':device,'lab_id':self.service.config['lab_id'],
            'observed_at':iso(observed),'evidence_source':'local_serial'})

    def list(self,principal):
        from management_notifications import scoped
        with self.service.lock:
            return [dict(row) for row in self.service.db.execute('SELECT * FROM incidents ORDER BY opened DESC LIMIT 200')
                    if scoped(principal,{'lab_id':row['lab'],'severity':'L3' if row['kind']=='fire' else 'L1',
                                         'person_id':row['person_id'],'instrument_id':row['instrument_id'],'camera_id':row['camera_id']})]

    def note_ppe(self,event):
        s=self.service
        s.db.execute('INSERT OR IGNORE INTO incidents(id,lab,kind,active,state,opened,last_observed,person_id,instrument_id,camera_id) VALUES(?,?,?,?,?,?,?,?,?,?)',
            (event['incident_id'],event['lab_id'],'ppe',1,'open',s.wall(),s.wall(),event['person_id'],event.get('instrument_id'),event.get('camera_id')))

        if event.get('warning_number'):
            from management_notifications import stamp
            s.db.execute('UPDATE incidents SET warning_event_id=?,last_observed=? WHERE id=?',
                         (event['event_id'], stamp(event['observed_at']), event['incident_id']))

    def ppe_open(self, incident, before_red=False):
        row = self.service.db.execute(
            "SELECT warning_event_id FROM incidents WHERE id=? AND kind='ppe' AND active=1 AND state='open'",
            (incident,)).fetchone()
        return row is not None and not (before_red and (row['warning_event_id'] or '').endswith(':red'))

    def display_warnings(self):
        # Include the confirmed name for mask warnings without exposing tracking data.
        with self.service.lock:
            rows = self.service.db.execute(
                "SELECT e.payload FROM incidents i JOIN events e ON e.id=i.warning_event_id "
                "WHERE i.kind='ppe' AND i.active=1 AND i.state='open' "
                "ORDER BY i.warning_event_id LIKE '%:red' DESC,i.last_observed DESC LIMIT 9").fetchall()
        fields = ('event_type', 'severity', 'observed_at', 'warning_number')
        warnings = []
        for row in rows:
            item = json.loads(row[0])
            warning = {field: item[field] for field in fields}
            if item["event_type"] == "safety.no_mask":
                name = item.get("display_name")
                person_id = item.get("person_id")
                if (isinstance(name, str) and name.strip() and isinstance(person_id, str)
                        and person_id and not person_id.startswith("track:")):
                    warning["display_name"] = name.strip()
            warnings.append(warning)
        return warnings

    def recover_ppe(self,incident,observed):
        s=self.service
        with s.lock,s.db:
            s.db.execute("UPDATE incidents SET active=0,state='recovered',recovered=?,last_observed=? WHERE id=? AND kind='ppe'",
                         (observed,observed,incident))
            for row in s.db.execute('SELECT id,payload FROM events').fetchall():
                event=json.loads(row['payload'])
                if event.get('incident_id')==incident and event.get('event_type','').startswith('safety.'):
                    s.db.execute("UPDATE outbox SET state='cancelled' WHERE event_id=? AND state='pending'",(row['id'],))

    def transition(self,principal,incident,action,note=''):
        from management_notifications import scoped
        if principal['role']!='admin':raise PermissionError('maintenance administrator required')
        s=self.service
        with s.lock,s.db:
            row=s.db.execute('SELECT * FROM incidents WHERE id=?',(incident,)).fetchone()
            if not row or not scoped(principal,{'lab_id':row['lab'],'severity':'L3' if row['kind']=='fire' else 'L1',
                                               'person_id':row['person_id'],'instrument_id':row['instrument_id'],'camera_id':row['camera_id']}):raise PermissionError('incident scope denied')
            if action=='ack':
                if row['state']=='closed':raise ValueError('incident already closed')
                s.db.execute('UPDATE incidents SET acknowledged_by=?,state=? WHERE id=?',
                    (principal['id'],'acknowledged' if row['active'] else 'recovered',incident))
            elif action=='close':
                if row['active']:raise ValueError('active alarm cannot be closed')
                if not isinstance(note,str) or not note.strip() or len(note)>1000:raise ValueError('inspection note required')
                s.db.execute("UPDATE incidents SET state='closed',closed_by=?,note=? WHERE id=?",(principal['id'],note.strip(),incident))
            else:raise ValueError('unknown incident action')
            if action == 'ack' and row['kind'] == 'ppe':
                s.db.execute("UPDATE outbox SET state='cancelled' WHERE state='pending' AND event_id LIKE ?",
                             (incident+':%',))
            s._audit(principal['id'],'incident.'+action,incident)
