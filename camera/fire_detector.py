"""Bounded CPU ONNX fire inference, independent of person tracking."""
import ast
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


class FireDetector:
    def __init__(self,model,sha256,threads=2,session=None):
        if not 1<=threads<=4:raise ValueError('fire CPU thread limit must be 1..4')
        path=Path(model)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:raise ValueError('fire model checksum mismatch')
        if session is None:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.session=session;self.input=session.get_inputs()[0]
        shape=self.input.shape
        if len(shape)!=4 or shape[:2]!=[1,3] or not all(type(v) is int and 32<=v<=1280 for v in shape[2:]):
            raise ValueError('unsupported fire model input')
        self.height,self.width=shape[2:]
        names=ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])
        if not isinstance(names,dict) or any(type(k) is not int or not isinstance(v,str) for k,v in names.items()):
            raise ValueError('fire model class metadata required')
        if set(names)!=set(range(len(names))):raise ValueError('contiguous model class IDs required')
        self.names=names
        self.fire_ids={k for k,v in names.items() if v.lower() in {'fire','flame'}}
        if not self.fire_ids:raise ValueError('model has no explicit fire class')

    def preprocess(self,image):
        if image is None or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('uint8 BGR frame required')
        h,w=image.shape[:2];scale=min(self.width/w,self.height/h)
        resized=(round(w*scale),round(h*scale))
        left=(self.width-resized[0])//2;top=(self.height-resized[1])//2
        canvas=np.full((self.height,self.width,3),114,np.uint8)
        canvas[top:top+resized[1],left:left+resized[0]]=cv2.resize(image,resized)
        tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        return tensor,(left,top,resized[0]/w,resized[1]/h)

    def predict(self,image,threshold=.25):
        if not 0<threshold<1:raise ValueError('invalid candidate threshold')
        started=time.perf_counter();tensor,transform=self.preprocess(image)
        output=self.session.run(None,{self.input.name:tensor})[0]
        expected=4+len(self.names)
        if output.ndim!=3 or output.shape[0]!=1 or output.shape[1]!=expected:
            raise ValueError('unsupported fire model output')
        rows=output[0].T
        if not np.isfinite(rows).all():raise ValueError('non-finite model output')
        left,top,sx,sy=transform;h,w=image.shape[:2];detections=[]
        for class_id,name in self.names.items():
            selected=rows[rows[:,4+class_id]>=threshold]
            boxes=[];scores=[]
            for row in selected:
                cx,cy,bw,bh=row[:4]
                x1=max(0.,min(float(w),(cx-bw/2-left)/sx));y1=max(0.,min(float(h),(cy-bh/2-top)/sy))
                x2=max(0.,min(float(w),(cx+bw/2-left)/sx));y2=max(0.,min(float(h),(cy+bh/2-top)/sy))
                if x2<=x1 or y2<=y1:continue
                boxes.append([x1,y1,x2-x1,y2-y1]);scores.append(float(row[4+class_id]))
            indices=cv2.dnn.NMSBoxes(boxes,scores,threshold,.45) if boxes else []
            for index in np.asarray(indices,dtype=int).reshape(-1):
                detections.append({'class_id':class_id,'class_name':name,'confidence':scores[index],
                                   'bbox':[round(v,3) for v in boxes[index]],'bbox_format':'xywh_pixels'})
        detections.sort(key=lambda d:d['confidence'],reverse=True)
        return {'detections':detections[:100],'inference_ms':round((time.perf_counter()-started)*1000,3),
                'image_width':w,'image_height':h}

    def verdict(self,prediction,minimum=.9):
        fire=[d for d in prediction['detections'] if d['class_id'] in self.fire_ids and d['confidence']>=minimum]
        if fire:return {'fire_state':'detected','confidence':fire[0]['confidence'],
                        'fire_bbox':fire[0]['bbox'],'bbox_format':'xywh_pixels'}
        # Absence of a box is not calibrated evidence that a fire is extinguished.
        return {'fire_state':'unknown','confidence':0.0}
