"""Bounded, reviewed site samples. No model-predicted identity is auto-enrolled."""
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid


MIN_SHARPNESS = 800.0


def sharp_enough(quality):
    try:
        value = float(quality.get("sharpness", 0))
        return math.isfinite(value) and value >= MIN_SHARPNESS
    except (TypeError, ValueError, AttributeError):
        return False


class SiteSamples:
    def __init__(self, root, wall=time.time):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.wall=wall; self.lock=threading.RLock()
        self.db=sqlite3.connect(self.root/'samples.sqlite3',check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.executescript('''CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,started REAL,expires REAL,active INTEGER);
            CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY,session TEXT,camera TEXT,source_frame TEXT,
                track TEXT,bucket TEXT,suggested TEXT,score REAL,pose TEXT,captured REAL,quality TEXT,
                status TEXT DEFAULT 'pending',person TEXT,UNIQUE(session,camera,source_frame,track));''')
        with self.db:
            for row in self.db.execute("SELECT id,quality FROM samples WHERE status IN ('pending','approved')").fetchall():
                if not sharp_enough(json.loads(row['quality'])):
                    self.db.execute("UPDATE samples SET status='rejected',person=NULL WHERE id=?", (row['id'],))
        self.error=None

    def start(self):
        with self.lock,self.db:
            self.db.execute('UPDATE sessions SET active=0')
            sid=uuid.uuid4().hex; now=self.wall()
            self.db.execute('INSERT INTO sessions VALUES(?,?,?,1)',(sid,now,now+86400))
        return sid

    def stop(self):
        with self.lock,self.db:self.db.execute('UPDATE sessions SET active=0')

    def active(self):
        with self.lock:
            row=self.db.execute('SELECT * FROM sessions WHERE active=1 AND expires>? ORDER BY started DESC LIMIT 1',(self.wall(),)).fetchone()
            return dict(row) if row else None

    def add(self, session, camera, source, track, suggested, score, pose, captured, quality, jpeg, feature):
        import numpy as np
        bucket='suggested:'+suggested if suggested else camera+':track:'+str(track)
        now=self.wall()
        with self.lock,self.db:
            current=self.active()
            if not current or current['id']!=session:return 'inactive'
            if not sharp_enough(quality):return 'low_sharpness'
            if self.db.execute('SELECT count(*) FROM samples').fetchone()[0]>=1000:return 'storage_limit'
            rows=self.db.execute("SELECT * FROM samples WHERE session=? AND bucket=? AND status!='rejected'",(session,bucket)).fetchall()
            if len(rows)>=20:return 'person_limit'
            same=[r for r in rows if r['camera']==camera]
            if len(same)>=10:return 'camera_limit'
            if same and now-max(r['captured'] for r in same)<60:return 'interval'
            if self.db.execute('SELECT 1 FROM samples WHERE session=? AND camera=? AND source_frame=? AND track=?',(session,camera,source,str(track))).fetchone():return 'duplicate'
            vector=np.asarray(feature,dtype=np.float32).reshape(-1)
            if vector.size!=512 or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-8:return 'invalid_feature'
            vector/=np.linalg.norm(vector)
            for row in rows:
                old=np.load(self.root/(row['id']+'.npy'),allow_pickle=False)
                if float(old@vector)>.97 and row['pose']==pose:return 'similar'
            sid=uuid.uuid4().hex
            (self.root/(sid+'.jpg')).write_bytes(jpeg);np.save(self.root/(sid+'.npy'),vector)
            self.db.execute('INSERT INTO samples(id,session,camera,source_frame,track,bucket,suggested,score,pose,captured,quality) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (sid,session,camera,source,str(track),bucket,suggested,float(score),pose,float(captured),json.dumps(quality)))
            return 'saved'

    def review(self, ids, person, people, reject=False):
        if not isinstance(ids,list) or not 1<=len(ids)<=20 or any(not re.fullmatch('[a-f0-9]{32}',str(x)) for x in ids):raise ValueError('请选择1至20张候选图')
        if not reject and person not in people:raise ValueError('请选择已有注册人员')
        with self.lock,self.db:
            if not reject:
                count=self.db.execute("SELECT count(*) FROM samples WHERE person=? AND status IN ('approved','enrolled')",(person,)).fetchone()[0]
                pending=self.db.execute('SELECT count(*) FROM samples WHERE id IN ('+','.join('?'*len(ids))+') AND status=\'pending\'',ids).fetchone()[0]
                if count+pending>20:raise ValueError('每位人员最多20张现场补充样本')
            for sid in ids:
                self.db.execute("UPDATE samples SET status=?,person=? WHERE id=? AND status='pending'",('rejected' if reject else 'approved',None if reject else person,sid))

    def status(self, people):
        with self.lock:
            rows=[dict(r) for r in self.db.execute('SELECT * FROM samples ORDER BY captured DESC LIMIT 1000')]
        for r in rows:
            r.pop('bucket',None);r['quality']=json.loads(r['quality'])
        return {'session':self.active(),'samples':rows,'people':[{'id':p,'name':v.get('name',p)} for p,v in people.items()],
            'last_error':self.error,'limits':{'min_sharpness':MIN_SHARPNESS,'per_person':20,'interval_seconds':60,'poll_seconds':15,'hours':24,'total_samples':1000},
            'meaning':'建议身份仅供审核；批准前不写注册库。姿态标签为关键点比例的近似分组。'}

    def unreview(self, ids):
        if not isinstance(ids,list) or not 1<=len(ids)<=20 or any(not re.fullmatch('[a-f0-9]{32}',str(x)) for x in ids):raise ValueError('请选择1至20张已审核照片')
        with self.lock,self.db:
            for sid in ids:
                self.db.execute("UPDATE samples SET status='pending',person=NULL WHERE id=? AND status='approved'",(sid,))

    def enroll(self, person, faces_dir, people):
        if person not in people or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',person):raise ValueError('人员不存在')
        import cv2
        import shutil
        with self.lock,self.db:
            rows=self.db.execute("SELECT * FROM samples WHERE person=? AND status='approved' ORDER BY captured",(person,)).fetchall()
            if not rows:raise ValueError('没有已审核的样本')
            target=Path(faces_dir)/(person+'_site_'+uuid.uuid4().hex)
            target.mkdir(parents=True)
            try:
                for i,row in enumerate(rows):
                    image=cv2.imread(str(self.root/(row['id']+'.jpg')),cv2.IMREAD_GRAYSCALE)
                    if image is None:raise ValueError('样本文件缺失')
                    if not cv2.imwrite(str(target/(str(i)+'.png')),image):raise ValueError('写入失败')
                    shutil.copy2(self.root/(row['id']+'.npy'),target/(str(i)+'.npy'))
                for row in rows:self.db.execute("UPDATE samples SET status='enrolled' WHERE id=?",(row['id'],))
            except Exception:
                shutil.rmtree(target);raise
        return {'added':len(rows),'directory':target.name,'person_id':person,'reload_required':True}
