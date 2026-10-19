"""Protected immutable snapshot metadata shared by local sender and notifier."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import threading
from weakref import WeakKeyDictionary
from contextlib import contextmanager

from feishu_app import protect


class SnapshotRegistry:
    def __init__(self,path,wall=time.time,retention=120,max_entries=1200):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.wall,self.retention,self.max_entries=wall,retention,max_entries
        self.loaded=WeakKeyDictionary()
        self.lock=threading.RLock()
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS snapshots(camera TEXT,snapshot TEXT,digest TEXT,payload BLOB,expires REAL,PRIMARY KEY(camera,snapshot))')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=5)
        try:
            with db:yield db
        finally:db.close()

    def publish(self,event):
        camera=event['camera']
        allowed_camera=('camera_id','image_url','image_sha256','frame_width','frame_height','source_frame_id','timing')
        allowed_track=('track_id','global_id','person_id','display_name','observed_at','bbox','bbox_format')
        safe={key:event.get(key) for key in ('snapshot_id','captured_at','occurred_at','time_domain')}
        safe['camera']={key:camera.get(key) for key in allowed_camera}
        safe['camera']['tracks']=[{key:track.get(key) for key in allowed_track} for track in camera.get('tracks',[])]
        raw=json.dumps(safe,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
        digest=hashlib.sha256(raw).hexdigest()
        with self.connect() as db:
            old=db.execute('SELECT digest FROM snapshots WHERE camera=? AND snapshot=?',
                (camera['camera_id'],event['snapshot_id'])).fetchone()
            if old:
                if old[0]!=digest:raise ValueError('snapshot metadata is immutable')
                return
            db.execute('DELETE FROM snapshots WHERE expires<=?',(self.wall(),))
            if db.execute('SELECT count(*) FROM snapshots').fetchone()[0]>=self.max_entries:
                raise BufferError('snapshot registry capacity reached')
            db.execute('INSERT INTO snapshots VALUES(?,?,?,?,?)',
                (camera['camera_id'],event['snapshot_id'],digest,protect(raw),self.wall()+self.retention))

    def populate(self,index):
        with self.connect() as db:
            rows=db.execute('SELECT camera,snapshot,digest,payload FROM snapshots WHERE expires>? ORDER BY expires',
                            (self.wall(),)).fetchall()
        with self.lock:
            loaded=self.loaded.setdefault(index,{})
            active={(camera,snapshot) for camera,snapshot,_,_ in rows}
            for key in list(loaded):
                if key not in active:del loaded[key]
            for camera,snapshot,digest,encrypted in rows:
                key=(camera,snapshot)
                if loaded.get(key)==digest and index.evidence_frame(camera,snapshot) is not None:continue
                raw=protect(encrypted,True)
                if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('snapshot registry integrity failure')
                index.remember(json.loads(raw))
                loaded[key]=digest
        return len(rows)
