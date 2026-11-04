"""Offline coat VLM + specialist mask baseline. No alerts, gallery or ReID writes."""
import argparse
import json
import re
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT = (
    '请看图判断此人的穿戴。实验服是穿在身上的长款实验室白大褂，不能把普通白色T恤算作实验服。'
    '看不清或无法区分衣服类型时回答未知。只输出：实验服=已穿戴或未穿戴或未知。'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--person-model', type=Path, required=True)
    parser.add_argument('--coat-model', type=Path, required=True)
    parser.add_argument('--mask-vendor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sample-seconds', type=float, default=2.)
    args = parser.parse_args()
    if args.sample_seconds <= 0:
        parser.error('--sample-seconds must be positive')
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from ultralytics import YOLO
    from ppe_detector import MaskDetector
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Run this evaluator on a cluster CUDA GPU')
    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.coat_model, local_files_only=True)
    coat = AutoModelForImageTextToText.from_pretrained(
        args.coat_model, dtype=torch.bfloat16, device_map=args.device,
        local_files_only=True, attn_implementation='sdpa').eval()
    person = YOLO(str(args.person_model))
    mask = MaskDetector(args.mask_vendor, args.device)
    info = json.loads((args.session / 'info.json').read_text(encoding='utf-8'))
    states = {'已穿戴': 'worn', '未穿戴': 'not_worn', '未知': 'unknown'}
    with (args.output / 'observations.jsonl').open('w', encoding='utf-8') as output:
        for camera, stream in info['streams'].items():
            cap = cv2.VideoCapture(str(args.session / stream['video_file']))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {camera}')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise RuntimeError(f'Invalid FPS: {camera}')
            stride = max(1, round(args.sample_seconds * fps))
            try:
                for index in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), stride):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f'Cannot decode {camera} frame {index}')
                    detected = person.predict(frame, classes=[0], conf=.3,
                                              device=args.device, verbose=False)[0]
                    for n, box in enumerate(detected.boxes.xyxy.cpu().tolist()):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        crop = frame[max(y1, 0):min(y2, frame.shape[0]),
                                     max(x1, 0):min(x2, frame.shape[1])]
                        if crop.size == 0:
                            continue
                        crop_name = f'{camera}_{index}_{n}.jpg'
                        if not cv2.imwrite(str(args.output / crop_name), crop):
                            raise RuntimeError('Cannot save evidence crop')
                        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        inputs = processor.apply_chat_template(
                            [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                        {'type': 'text', 'text': PROMPT}]}],
                            tokenize=True, add_generation_prompt=True, return_dict=True,
                            return_tensors='pt', processor_kwargs={'downsample_mode': '4x',
                            'max_slice_nums': 1, 'use_image_id': False}).to(coat.device)
                        started = time.perf_counter()
                        with torch.inference_mode():
                            generated = coat.generate(**inputs, downsample_mode='4x',
                                                      max_new_tokens=32, do_sample=False)
                        raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                                     skip_special_tokens=True)[0].strip()
                        match = re.fullmatch(r'实验服\s*[=：:]\s*(已穿戴|未穿戴|未知)[。.]?', raw)
                        observation = {'offline': True, 'camera_id': camera,
                            'source_frame_index': index, 'video_seconds': index / fps,
                            'bbox': box, 'crop': crop_name, 'person_id': None,
                            'lab_coat': states[match[1]] if match else 'unknown',
                            'coat_raw_response': raw, 'coat_valid_response': bool(match),
                            'mask': mask.observe(crop),
                            'coat_latency_seconds': time.perf_counter() - started,
                            'notification_eligible': False}
                        output.write(json.dumps(observation, ensure_ascii=False) + '\n')
                        output.flush()
            finally:
                cap.release()


if __name__ == '__main__':
    main()
