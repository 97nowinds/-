"""Cluster Qwen flame inference and same-region confirmation."""
from collections import deque
import json
import math
import re
import time


MODEL_NAME = 'Qwen3-VL-4B-Instruct'
PROMPT = (
    'Analyze this camera image for visible open flame and visible smoke. '
    'Ignore glare, reflections, lights, skin, hair, shadows and fog-like compression artifacts. '
    'List only directly visible evidence. Use a tight box around each actual flame or smoke region. '
    'If neither is clearly visible, return []. Output only a JSON array. '
    'Each item must be {"label":"fire" or "smoke","bbox_2d":[x1,y1,x2,y2]}. '
    'Coordinates are normalized to 0–1000, origin top left. '
    'Do not return boxes for equipment or people. At most three items.'
)
CONFIRM_SECONDS = 2.5
REPEAT_SECONDS = 10.0


def flame_boxes(text, width, height):
    match = re.search(r'\[.*\]', text.strip(), re.DOTALL)
    if match is None:
        raise ValueError('Qwen did not return an array')
    rows = json.loads(match[0])
    if not isinstance(rows, list) or len(rows) > 3:
        raise ValueError('invalid Qwen detections')
    result = []
    for row in rows:
        if not isinstance(row, dict) or row.get('label') not in ('fire', 'smoke'):
            raise ValueError('invalid Qwen label')
        if row['label'] == 'smoke':
            continue
        box = row.get('bbox_2d')
        if not isinstance(box, list) or len(box) != 4 or not all(
                type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in box):
            raise ValueError('invalid Qwen fire box')
        left, top, right, bottom = box
        if left >= right or top >= bottom:
            raise ValueError('empty Qwen fire box')
        result.append([left * width / 1000, top * height / 1000,
                       (right - left) * width / 1000, (bottom - top) * height / 1000])
    return result


def covers_same_area(a, b):
    x, y, w, h = a
    left, top, width, height = b
    overlap = max(0, min(x+w, left+width)-max(x, left)) * max(0, min(y+h, top+height)-max(y, top))
    return overlap >= 0.25 * max(w*h, width*height)


class QwenFlameWindow:
    def __init__(self):
        self.recent = deque(maxlen=2)
        self.last_match = None
        self.last_alert = None

    def observe(self, boxes, captured_at):
        while self.recent and captured_at - self.recent[0][0] > CONFIRM_SECONDS:
            self.recent.popleft()
        if self.last_match is not None and captured_at - self.last_match > CONFIRM_SECONDS:
            self.last_alert = None
        match = next((b for b in boxes if any(covers_same_area(previous, b)
                      for _, detections in self.recent for previous in detections)), None)
        self.recent.append((captured_at, boxes))
        if match is None:
            return None
        self.last_match = captured_at
        if self.last_alert is None:
            stage = 'visual_initial'
        elif captured_at - self.last_alert >= REPEAT_SECONDS:
            stage = 'visual_ongoing'
        else:
            return None
        self.last_alert = captured_at
        return stage, match


class QwenFlameDetector:
    def __init__(self, path):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        if not free or max(free) < 12 * 1024**3:
            raise RuntimeError('Qwen needs 12 GiB of free GPU memory')
        gpu = max(range(len(free)), key=free.__getitem__)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16, device_map={'': gpu},
            attn_implementation='sdpa').eval()

    def predict(self, frame):
        import cv2
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((1280, 1280))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                {'type': 'text', 'text': PROMPT}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors='pt').to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = self.processor.decode(ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        height, width = frame.shape[:2]
        return flame_boxes(text, width, height)


def run_qwen(cameras, path):
    windows = {camera.name: QwenFlameWindow() for camera in cameras}
    seen = {}
    detector = None
    while True:
        if detector is None:
            try:
                detector = QwenFlameDetector(path)
            except (ImportError, OSError, RuntimeError, ValueError):
                for camera in cameras:
                    with camera.lock:
                        camera.counts['qwen_load_errors'] += 1
                time.sleep(30)
                continue
        for camera in cameras:
            with camera.lock:
                frame, captured, generation = camera.latest, camera.received, camera.generation
                frame_id = camera.counts['frames']
            prior = seen.get(camera.name)
            if frame is None or prior == (generation, frame_id) or time.monotonic() - captured > 2:
                continue
            if prior is None or prior[0] != generation:
                windows[camera.name] = QwenFlameWindow()
            seen[camera.name] = generation, frame_id
            try:
                boxes = detector.predict(frame)
            except (RuntimeError, ValueError):
                with camera.lock:
                    camera.counts['qwen_inference_errors'] += 1
                continue
            with camera.lock:
                if camera.generation != generation or time.monotonic() - captured > CONFIRM_SECONDS:
                    continue
                camera.counts['qwen_samples'] += 1
                camera.counts['qwen_candidates'] += bool(boxes)
                event = windows[camera.name].observe(boxes, captured)
                if event:
                    stage, box = event
                    camera.record_flame(frame, frame_id, box, stage, None, 'qwen', captured)
        time.sleep(.05)
