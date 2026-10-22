"""Persistent preparation windows, keyed by registered person, not tracker IDs."""
import math


class PPEPreparation:
    def __init__(self, service):
        self.service = service
        with service.lock, service.db:
            service.db.execute('''CREATE TABLE IF NOT EXISTS ppe_preparation(
                person TEXT PRIMARY KEY,started REAL,deadline REAL,last_seen REAL,
                ended REAL,reason TEXT)''')

    def see(self, person, observed, entrance_face=False):
        s = self.service
        policy = s.config.get('ppe_preparation', {})
        if not policy.get('enabled') or not person or not math.isfinite(observed):
            return
        with s.lock, s.db:
            row = s.db.execute('SELECT * FROM ppe_preparation WHERE person=?', (person,)).fetchone()
            # Only the first confirmed entrance face
            # grants a window; another visit requires an explicit session end.
            if entrance_face and row is None:
                active = s.db.execute("SELECT 1 FROM incidents WHERE person_id=? AND kind='ppe' AND active=1",
                                      (person,)).fetchone()
                if active:
                    return
                s.db.execute('INSERT INTO ppe_preparation VALUES(?,?,?,?,?,?)',
                    (person, observed, observed + policy['grace_seconds'], observed, None, None))
                s._audit('system', 'ppe.preparation.start', person)
            elif row is not None and observed > row['last_seen']:
                s.db.execute('UPDATE ppe_preparation SET last_seen=? WHERE person=?', (observed, person))

    def active(self, person, observed):
        s = self.service
        if not s.config.get('ppe_preparation', {}).get('enabled'):
            return False
        with s.lock:
            row = s.db.execute('SELECT * FROM ppe_preparation WHERE person=?', (person,)).fetchone()
            return bool(row and row['ended'] is None and row['started'] <= observed < row['deadline'])

    def finish(self, person, observed, reason):
        s = self.service
        with s.lock, s.db:
            s.db.execute('INSERT OR IGNORE INTO ppe_preparation VALUES(?,?,?,?,?,?)',
                         (person, observed, observed, observed, observed, reason))
            s.db.execute('UPDATE ppe_preparation SET ended=?,reason=? WHERE person=? AND ended IS NULL',
                         (observed, reason, person))

    def close_visit(self, person):
        # Only call with independently confirmed departure or an explicit
        # maintenance correction, never because a detector lost a track.
        s = self.service
        with s.lock, s.db:
            s.db.execute('DELETE FROM ppe_preparation WHERE person=?', (person,))
