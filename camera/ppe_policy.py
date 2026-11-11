"""Confirm and escalate PPE incidents using distinct observed frames."""
import json
import uuid

WARNING_LIMIT = 5


class PPEPolicy:
    def __init__(self, service):
        self.service = service
        with service.lock, service.db:
            service.db.execute('CREATE TABLE IF NOT EXISTS ppe_states(key TEXT PRIMARY KEY,payload TEXT)')

    def _session(self, key):
        row = self.service.db.execute('SELECT payload FROM ppe_states WHERE key=?', (key,)).fetchone()
        return json.loads(row['payload']) if row else None

    def _save(self, key, session):
        self.service.db.execute('INSERT OR REPLACE INTO ppe_states VALUES(?,?)', (key, json.dumps(session)))

    def observe(self, person, kind, instrument, observed, frame, state):
        s = self.service
        policy = s.config.get('ppe_confirmation')
        if not policy:
            return {} if state == 'not_worn' else None
        key = json.dumps([person, kind, instrument])
        with s.lock, s.db:
            session = self._session(key)
            if session and observed < session['last']:
                return None
            if state == 'worn':
                if session and session['emitted']:
                    s.incidents.recover_ppe(session['id'], observed)
                s.db.execute('DELETE FROM ppe_states WHERE key=?', (key,))
                return None
            if s.preparation.active(person, observed) or state != 'not_worn':
                if session:
                    session.update(frames=[], continuous_since=None, last=observed)
                    self._save(key, session)
                return None
            if not session:
                session = dict(id='ppe:'+uuid.uuid4().hex, last=observed, frames=[],
                               continuous_since=observed, emitted=False)
            if any(frame == previous[1] for previous in session['frames']):
                return None
            if session['continuous_since'] is None or observed-session['last'] > policy['maximum_gap_seconds']:
                session.update(frames=[], continuous_since=observed)
            session['last'] = observed
            session['frames'] = (session['frames'] + [[observed, frame]])[-100:]
            self._save(key, session)
            if (len(session['frames']) < policy['minimum_frames'] or
                    observed-session['continuous_since'] < policy['minimum_seconds']):
                return None
            count = session.get('warning_count', 0)
            if count > WARNING_LIMIT:
                return None
            if session['emitted'] and not s.incidents.ppe_open(session['id']):
                return None
            if count and observed-max(session['continuous_since'], session['warning_at']) < policy['escalation_seconds']:
                return None
            red = count == WARNING_LIMIT
            return {'incident_id': session['id'], 'severity': 'L3' if red else 'L1',
                    'warning_number': str(min(count+1, WARNING_LIMIT)),
                    'audiences': 'administrators' if red else
                                 'experimenters,managers' if count else 'experimenters'}

    def commit_warning(self, event, observed):
        """Advance the sequence in the same transaction as event and outbox writes."""
        key = json.dumps([event['person_id'], event['event_type'], event.get('instrument_id')])
        session = self._session(key)
        if (not session or session['id'] != event['incident_id'] or
                session['continuous_since'] is None or observed < session['continuous_since']):
            return False
        count = session.get('warning_count', 0)
        if session['emitted'] and not self.service.incidents.ppe_open(session['id']):
            return False
        red = event['severity'] == 'L3'
        if (count > WARNING_LIMIT or red != (count == WARNING_LIMIT) or
                int(event['warning_number']) != min(count+1, WARNING_LIMIT)):
            return False
        session.update(emitted=True, warning_count=count+1, warning_at=observed)
        self._save(key, session)
        return True
