"""Exact frozen-frame evidence, protected at rest and never fetched from result URLs."""
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from PIL import Image, ImageDraw
from feishu_app import protect


class EvidenceUnavailable(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class AlertEvidence:
    def __init__(self, root, allowed_origins, token='', fetch=None):
        self.root=Path(root)
        self.allowed_origins=set(allowed_origins)
        self.token=token
        self.fetch=fetch or self._fetch

    def _fetch(self,url):
        headers={'Authorization':'Bearer '+self.token} if self.token else {}
        try:
            with build_opener(NoRedirect()).open(Request(url,headers=headers),timeout=5) as response:
                if response.headers.get_content_type()!='image/jpeg':
                    raise EvidenceUnavailable('snapshot MIME mismatch')
                return response.read(10*1024*1024+1)
        except Exception:
            raise EvidenceUnavailable('exact snapshot unavailable') from None

    def capture_scene(self,row,index):
        """Fire evidence is a scene observation, never a fabricated person track."""
        return self.capture(row,index,scene=True)

    def capture(self, row, index, scene=False):
        frame=index.evidence_frame(row['camera_id'],row['snapshot_id'])
        if scene:
            track=({'bbox':row['fire_bbox'] if row.get('fire_bbox') is not None else [0,0,frame['frame_width'],frame['frame_height']],
                    'bbox_format':row.get('bbox_format') if row.get('fire_bbox') is not None else 'xywh_pixels'} if frame else None)
        else:track=index.evidence_track(row['camera_id'],row['snapshot_id'],row['track_id'])
        if not frame or not track:
            raise EvidenceUnavailable('original snapshot binding missing')
        url=frame.get('image_url') or ''
        parsed=urlsplit(url)
        origin=f'{parsed.scheme}://{parsed.netloc}'
        if (origin not in self.allowed_origins or parsed.username or parsed.password or parsed.query or parsed.fragment
                or not parsed.path.startswith('/api/handoff/') or not parsed.path.endswith('.jpg')):
            raise EvidenceUnavailable('snapshot origin not allowlisted')
        jpeg=self.fetch(url)
        return self._store(row, frame, track, jpeg, scene)

    def store_scene_jpeg(self, row, jpeg, width, height):
        """Store a trusted observer or recording frame already received as bytes."""
        frame = {'image_sha256': hashlib.sha256(jpeg).hexdigest(), 'frame_width': width,
                 'frame_height': height, 'source_frame_id': row.get('source_frame_id')}
        track = {'bbox': row.get('fire_bbox') or [0, 0, width, height], 'bbox_format': 'xywh_pixels'}
        return self._store(row, frame, track, jpeg, True, circle=True)

    def _store(self, row, frame, track, jpeg, scene, circle=False):
        if len(jpeg)>10*1024*1024 or hashlib.sha256(jpeg).hexdigest()!=frame.get('image_sha256'):
            raise EvidenceUnavailable('snapshot digest mismatch')
        try:
            with Image.open(BytesIO(jpeg)) as image:
                if image.format!='JPEG' or image.size!=(frame['frame_width'],frame['frame_height']):
                    raise EvidenceUnavailable('snapshot dimensions mismatch')
                if image.width*image.height>32*1024*1024:
                    raise EvidenceUnavailable('snapshot too large')
                box=track.get('bbox')
                if track.get('bbox_format')!='xywh_pixels' or not isinstance(box,list) or len(box)!=4:
                    raise EvidenceUnavailable('bbox space unknown')
                if any(type(v) not in (int,float) or not math.isfinite(v) for v in box):
                    raise EvidenceUnavailable('invalid bbox')
                x,y,w,h=box
                if min(x,y)<0 or min(w,h)<=0 or x+w>image.width+1 or y+h>image.height+1:
                    raise EvidenceUnavailable('bbox outside snapshot')
                # Preserve original; create a separate annotated copy for group viewing.
                annotated=image.convert('RGB')
                if not scene or row.get('fire_bbox') is not None:
                    painter = ImageDraw.Draw(annotated)
                    if circle:
                        cx, cy = x + w / 2, y + h / 2
                        radius = max(24, math.hypot(w, h) / 2 + 10)
                        painter.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(255, 50, 30), width=7)
                    else:
                        painter.rectangle((x,y,x+w,y+h),outline=(255,60,30),width=4)
                output=BytesIO();annotated.save(output,format='JPEG',quality=90)
        except EvidenceUnavailable:
            raise
        except Exception:
            raise EvidenceUnavailable('snapshot decode failed') from None
        evidence_id=hashlib.sha256((row['camera_id']+'|'+row['snapshot_id']+'|'+('scene:fire' if scene else row['track_id'])).encode()).hexdigest()
        directory=self.root/evidence_id;directory.mkdir(parents=True,exist_ok=True)
        if (directory/'metadata.dpapi').exists():
            old=json.loads(protect((directory/'metadata.dpapi').read_bytes(),decrypt=True))
            if old['source_sha256']!=frame['image_sha256'] or old['bbox']!=box:
                raise EvidenceUnavailable('snapshot evidence changed')
        metadata={key:row.get(key) for key in ('camera_id','snapshot_id','track_id','person_id','captured_at')}
        if scene:metadata.update(track_id=None,person_id=None,kind=row.get('kind','fire_scene'))
        metadata.update(source_sha256=frame['image_sha256'],source_frame_id=frame.get('source_frame_id'),
                        bbox=box,frame_width=frame['frame_width'],frame_height=frame['frame_height'])
        for name,content in [('original.dpapi',jpeg),('annotated.dpapi',output.getvalue()),
                             ('metadata.dpapi',json.dumps(metadata,ensure_ascii=False).encode())]:
            path=directory/name
            if not path.exists():
                import uuid
                temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
                temporary.write_bytes(protect(content));temporary.replace(path)
        return evidence_id

    def read(self,evidence_id):
        if len(evidence_id)!=64 or any(c not in '0123456789abcdef' for c in evidence_id):
            raise EvidenceUnavailable('invalid evidence id')
        try:
            return protect((self.root/evidence_id/'annotated.dpapi').read_bytes(),decrypt=True)
        except Exception:
            raise EvidenceUnavailable('stored evidence unavailable') from None


def group_evidence_payload(text,image_key,secret='',now=None,title='实验室防护告警｜证据快照'):
    from management_notifications import feishu_payload
    image_keys = image_key if isinstance(image_key, list) else [image_key]
    payload={'msg_type':'post','content':{'post':{'zh_cn':{
        'title':title,
        'content':[[{'tag':'text','text':text}]] + [[{'tag':'img','image_key':key}] for key in image_keys]}}}}
    signed=feishu_payload('',secret,now)
    for key in ('timestamp','sign'):
        if key in signed:payload[key]=signed[key]
    return payload
