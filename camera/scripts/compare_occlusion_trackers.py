"""Offline comparison on identical YOLO detections; never changes live config.

Uses the installed Ultralytics ByteTrack / BoT-SORT / Deep OC-SORT variants.
BoT-SORT and Deep OC-SORT receive the project's OSNet embeddings BEFORE
local assignment. This is not a reproduction of the authors' benchmark.
No ground truth => no IDSW/IDF1/HOTA claim. Outputs detections and local IDs
for subsequent annotation, including detections without active tracks.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track import TRACKER_MAP
import yaml
from reid_embedder import ReIDEmbedder


def build_trackers():
    cfg_root = Path(ultralytics.__file__).parent / 'cfg/trackers'
    trackers = {}
    for name in ('bytetrack', 'botsort', 'deepocsort'):
        if name not in TRACKER_MAP:
            raise RuntimeError(f'{name} unavailable in Ultralytics {ultralytics.__version__}')
        cfg = yaml.safe_load((cfg_root / f'{name}.yaml').read_text())
        cfg.update(track_high_thresh=.25, track_low_thresh=.10, new_track_thresh=.25,
                   track_buffer=60, match_thresh=.80, fuse_score=True)
        if name != 'bytetrack':
            # Avoid loading a classifier or auto detector features. Our OSNet
            # features are supplied through the documented per-detection input.
            cfg.update(with_reid=False, model='auto', gmc_method='none',
                       proximity_thresh=.5, appearance_thresh=.75)
        if name == 'deepocsort':
            cfg['use_byte'] = True
        args = SimpleNamespace(**cfg)
        tracker = TRACKER_MAP[name](args=args)
        if name != 'bytetrack':
            args.with_reid = True
            tracker.encoder = lambda supplied_features, boxes: supplied_features
        trackers[name] = tracker
    return trackers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detector', type=Path, default=Path('models/yolov8n.pt'))
    parser.add_argument('--reid-model', type=Path, default=Path('models/osnet_x1_0_msmt17.pth'))
    parser.add_argument('--frames', type=int, default=0, help='0 means entire clip')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--nms-iou', type=float, default=.50, help='Hold fixed across methods; test .70 separately')
    args = parser.parse_args()
    if args.stride < 1 or args.frames < 0:
        parser.error('stride must be positive and frames nonnegative')
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError('Cannot read the supplied video')
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise RuntimeError('Video has no valid FPS; real per-frame timestamps are required')
    detector = YOLO(str(args.detector))
    embedder = ReIDEmbedder(args.reid_model)
    trackers = build_trackers()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in trackers}
    frames = 0
    try:
        with args.output.open('w', encoding='utf-8') as out:
            index = -1
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                if index % args.stride:
                    continue
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                started = time.perf_counter()
                result = detector.predict(frame, conf=.10, iou=args.nms_iou,
                    classes=[0], imgsz=args.imgsz, device=str(embedder.device), verbose=False)[0]
                det = result.boxes.cpu().numpy()
                detector_ms = (time.perf_counter() - started) * 1000
                corners = np.asarray(det.xyxy)
                boxes = [(int(x), int(y), max(1, int(r-x)), max(1, int(b-y))) for x,y,r,b in corners]
                reid_started = time.perf_counter()
                features = embedder.extract_batch(frame, boxes) if boxes else []
                # Keep the original detections in the audit. All three methods
                # receive the same subset of crops that support a real feature;
                # never give an invalid crop a fabricated embedding.
                raw_detections = det.data.tolist()
                valid_indices = [i for i, feature in enumerate(features) if feature is not None]
                invalid_indices = [i for i, feature in enumerate(features) if feature is None]
                det = det[valid_indices]
                features = np.asarray([features[i] for i in valid_indices], dtype=np.float32) if valid_indices else np.empty((0, embedder.feature_dim), dtype=np.float32)
                reid_ms = (time.perf_counter() - reid_started) * 1000
                results, timings = {}, {}
                for name, tracker in trackers.items():
                    before = time.perf_counter()
                    tracked = tracker.update(det, frame, feats=features if name != 'bytetrack' else None)
                    timings[name] = round((time.perf_counter()-before)*1000, 3)
                    results[name] = tracked.tolist()
                    counts[name] += len(tracked)
                out.write(json.dumps({'frame_index':index, 'video_seconds':index/source_fps,
                    'frame_width':frame.shape[1], 'frame_height':frame.shape[0],
                    'detections_xyxy_conf_cls':raw_detections,
                    'association_detection_indices':valid_indices,
                    'invalid_reid_crop_indices':invalid_indices,
                    'local_tracks_xyxy_id_conf_cls_detection_index':results,
                    'detector_ms':round(detector_ms,3), 'shared_osnet_ms':round(reid_ms,3),
                    'association_ms':timings})+'\n')
                frames += 1
                if args.frames and frames >= args.frames:
                    break
    finally:
        capture.release()
    metadata = {'ultralytics_version':ultralytics.__version__, 'reid':embedder.backend,
        'frames':frames, 'source_fps':source_fps, 'stride':args.stride,
        'ground_truth_available':False, 'paper_reproduction':False,
        'note':'Track counts are not IDSW/IDF1/HOTA; frame_index/FPS is video time, not capture UTC. Crowded crops are not visibility-masked. All methods share valid ReID crops; raw detections and excluded indices are retained; tracker detection_index refers to the filtered subset.',
        'detector_conf':.10, 'nms_iou':args.nms_iou, 'track_output_rows':counts}
    args.output.with_suffix('.meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == '__main__':
    main()
