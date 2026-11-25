"""Same-frame filtering for recurring laboratory smoke false positives."""

from collections import Counter

import cv2
import numpy as np

from fire_detector import FireDetector


def coverage(candidate, region):
    x, y, w, h = candidate
    a, b, rw, rh = region
    if min(w, h, rw, rh) <= 0:
        return 0.0
    overlap = max(0, min(x+w, a+rw)-max(x, a)) * max(0, min(y+h, b+rh)-max(y, b))
    return overlap / (w*h)


class SmokeGuard:
    def __init__(self, root, config, camera):
        person = config['person_model']
        self.person = FireDetector(root / person['path'], person['sha256'], threads=1, classes={'person'})
        self.reflection = config.get('reflections', {}).get(camera)
        if self.reflection is not None:
            self.template = np.array(self.reflection['grayscale_16x12'], dtype=np.float32)
            if self.template.shape != (12, 16):
                raise ValueError('invalid smoke reflection template')

    def reflection_region(self, frame):
        if self.reflection is None:
            return None
        h, w = frame.shape[:2]
        x, y, rw, rh = self.reflection['area_xywh_normalized']
        left, top, right, bottom = int(x*w), int(y*h), int((x+rw)*w), int((y+rh)*h)
        if not 0 <= left < right <= w or not 0 <= top < bottom <= h:
            return None
        region = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(region, (16, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
        correlation = cv2.matchTemplate(patch, self.template, cv2.TM_CCOEFF_NORMED)[0, 0]
        if correlation >= .93 and np.mean(np.abs(patch - self.template)) <= 8:
            return left, top, right-left, bottom-top
        return None

    def screen(self, frame, candidates):
        if not candidates:
            return candidates, Counter()
        thumbnail = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
        if 115 <= gray.mean() <= 140 and gray.std() < 5 and thumbnail.std(axis=2).mean() < 4:
            return [], Counter(gray_frame=len(candidates))

        people = self.person.predict(frame, .75)['detections']
        reflection = self.reflection_region(frame)
        kept, rejected = [], Counter()
        for candidate in candidates:
            box = candidate['bbox']
            if any(coverage(box, person['bbox']) >= .85 for person in people):
                rejected['person'] += 1
            elif reflection is not None and coverage(box, reflection) >= .65:
                rejected['reflection'] += 1
            else:
                kept.append(candidate)
        return kept, rejected
