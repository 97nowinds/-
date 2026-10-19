"""Evaluate labelled D-Fire image samples; no notification or live service calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fire_detector import FireDetector


def iou(box,truth):
    x,y,w,h=box;a,b,c,d=truth
    intersection=max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b))
    union=w*h+(c-a)*(d-b)-intersection
    return intersection/union if union>0 else 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'));root=args.manifest.parent.resolve()
    detector=FireDetector(args.model,args.sha256);rows=[]
    for sample in manifest['samples']:
        path=(root/sample['image']).resolve()
        if not path.is_relative_to(root):raise ValueError('sample path escapes benchmark')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=sample['sha256']:raise ValueError('sample checksum mismatch')
        image=cv2.imdecode(np.frombuffer(data,np.uint8),1)
        if image is None:raise ValueError('invalid benchmark image')
        h,w=image.shape[:2];truth=[]
        for cls,cx,cy,bw,bh in sample['annotations']:
            if cls==1:truth.append([max(0,(cx-bw/2)*w),max(0,(cy-bh/2)*h),min(w,(cx+bw/2)*w),min(h,(cy+bh/2)*h)])
        prediction=detector.predict(image);fire=[d for d in prediction['detections'] if d['class_id'] in detector.fire_ids]
        rows.append({'image':sample['image'],'group':sample['group'],'fire_truth':bool(truth),
            'max_fire_score':max((d['confidence'] for d in fire),default=0),
            'max_matched_score':max((d['confidence'] for d in fire if any(iou(d['bbox'],g)>=.5 for g in truth)),default=0),
            'inference_ms':prediction['inference_ms']})
    positive=sum(r['fire_truth'] for r in rows);negative=len(rows)-positive;thresholds=[]
    for t in [.25,.4,.5,.6,.7,.8,.9]:
        thresholds.append({'threshold':t,'positive_images':positive,'negative_images':negative,
            'true_positive_images':sum(r['fire_truth'] and r['max_fire_score']>=t for r in rows),
            'matched_positive_images_iou_05':sum(r['fire_truth'] and r['max_matched_score']>=t for r in rows),
            'false_positive_images':sum(not r['fire_truth'] and r['max_fire_score']>=t for r in rows)})
    result={'model':str(args.model),'sha256':args.sha256,'dataset_revision':manifest['revision'],
        'sample_count':len(rows),'thresholds':thresholds,'frames':rows}
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'model':str(args.model),'thresholds':thresholds}))


if __name__=='__main__':main()
