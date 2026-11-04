"""GPU mask observations. Missing/ambiguous faces remain unknown.

Requires the official AIZOOTech/FaceMaskDetection checkout, including weights.
This adapter never publishes notifications or assigns a person's identity.
"""
from pathlib import Path
import importlib.util
import sys


class MaskDetector:
    def __init__(self, vendor_root, device="cuda:0", confidence=0.9):
        import torch
        import numpy as np
        self.root = Path(vendor_root).resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PPE inference requires a cluster CUDA GPU")
        self.confidence = float(confidence)
        if not 0 < self.confidence <= 1:
            raise ValueError("invalid confidence")
        # Official weights serialize MainModel.KitModel. Only load the trusted
        # upstream checkpoint; do not accept uploaded arbitrary pickle files.
        sys.path.insert(0, str(self.root / "models"))
        try:
            self.model = torch.load(self.root / "models/model360.pth",
                                    map_location=self.device, weights_only=False).eval()
        finally:
            sys.path.pop(0)
        def load(name):
            spec = importlib.util.spec_from_file_location("ppe_aizoo_" + name,
                                                          self.root / "utils" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        self.decode = load("anchor_decode").decode_bbox
        self.nms = load("nms").single_class_non_max_suppression
        anchors = load("anchor_generator").generate_anchors(
            [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]],
            [[.04, .056], [.08, .11], [.16, .22], [.32, .45], [.64, .72]],
            [[1, .62, .42]] * 5)
        self.anchors = np.expand_dims(anchors, 0)

    def observe(self, person_bgr, *, face_crop=False, verify_absence=True):
        import cv2
        import numpy as np
        import torch
        unknown = {"state": "unknown", "confidence": None, "faces": []}
        if person_bgr is None or person_bgr.size == 0:
            return unknown
        # Limit the search to this person's upper body. Coordinates returned
        # below are relative to the supplied person crop, never the full frame.
        head = person_bgr if face_crop else person_bgr[:max(1, round(person_bgr.shape[0] * .55))]
        h, w = head.shape[:2]
        pad_x = round(w * .4) if face_crop else 0
        pad_y = round(h * .4) if face_crop else 0
        if face_crop:
            head = cv2.copyMakeBorder(head, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        input_h, input_w = head.shape[:2]
        if min(h, w) < 32:
            return unknown
        rgb = cv2.cvtColor(cv2.resize(head, (360, 360)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].to(self.device) / 255
        with torch.inference_mode():
            boxes, scores = self.model(tensor)
        boxes = self.decode(self.anchors, boxes.cpu().numpy())[0]
        scores = scores.cpu().numpy()[0]
        maxima = np.max(scores, axis=1)
        classes = np.argmax(scores, axis=1)
        keep = self.nms(boxes, maxima, conf_thresh=self.confidence, iou_thresh=.4)
        faces = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i] * [input_w, input_h, input_w, input_h]
            x1, x2 = x1 - pad_x, x2 - pad_x
            y1, y2 = y1 - pad_y, y2 - pad_y
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            if min(x2 - x1, y2 - y1) < 24:
                continue
            faces.append({"state": "worn" if int(classes[i]) == 0 else "not_worn",
                          "confidence": float(maxima[i]),
                          "bbox": [float(v) for v in (x1, y1, x2, y2)]})
        if len(faces) != 1:
            return {**unknown, "faces": faces}
        if verify_absence and not face_crop and faces[0]['state'] == 'not_worn':
            # A partial upper-face detection can misclassify a visible mask.
            # Require the same negative conclusion on a contextual face crop.
            # Conflicting evidence remains unknown and cannot trigger an alert.
            a, b, c, d = faces[0]['bbox']
            margin_x, margin_y = (c-a)*.25, (d-b)*.25
            face = person_bgr[max(0,int(b-margin_y)):min(person_bgr.shape[0],int(d+margin_y)),
                              max(0,int(a-margin_x)):min(person_bgr.shape[1],int(c+margin_x))]
            verification = self.observe(face, face_crop=True, verify_absence=False)
            if verification['state'] != 'not_worn':
                return {**unknown, 'faces': faces, 'reason': 'negative_not_confirmed',
                        'verification': verification}
        return {**faces[0], "faces": faces}
