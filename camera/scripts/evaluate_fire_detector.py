"""Offline video evaluation. Never calls live capture or notification endpoints."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fire_detector import FireDetector
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('videos',nargs='+',type=Path)
    parser.add_argument('--samples',type=int,default=30);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    provenance=json.loads((ROOT/'config/fire_model.json').read_text(encoding='utf-8'))
    detector=FireDetector(ROOT/provenance['path'],provenance['sha256'],threads=provenance['cpu_threads'])
    minimum=json.loads((ROOT/'config/management.json').read_text(encoding='utf-8')).get('fire_fusion',{}).get('visual_minimum_confidence',.9)
    cv2.setNumThreads(1);reports=[]
    for video in args.videos:
        cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or count<=0:raise ValueError('cannot decode evaluation video')
        rows=[]
        for frame_number in np.unique(np.linspace(0,count-1,min(args.samples,count),dtype=int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_number));ok,frame=cap.read()
            if not ok:continue
            prediction=detector.predict(frame)
            rows.append({'frame':int(frame_number),'seconds':round(frame_number/fps,3),**prediction,
                         'verdict':detector.verdict(prediction,minimum)})
        cap.release()
        times=[row['inference_ms'] for row in rows]
        reports.append({'video':str(video.resolve()),'sampled_frames':len(rows),
            'candidate_fire_frames':sum(any(d['class_id'] in detector.fire_ids for d in r['detections']) for r in rows),
            'eligible_fire_frames':sum(r['verdict']['fire_state']=='detected' for r in rows),
            'median_inference_ms':float(np.median(times)),'p95_inference_ms':float(np.percentile(times,95)),'frames':rows})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'model':provenance,'reports':reports},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='frames'} for r in reports]))


if __name__=='__main__':main()
