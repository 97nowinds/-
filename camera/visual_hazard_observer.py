"""Offline spark and smoke candidate alarms for one local video."""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--live-relay", help="read an existing loopback RTSP relay instead of the file")
    parser.add_argument("--seconds", type=float, default=20, help="bounded live observation duration")
    parser.add_argument("--camera-id", default="offline")
    parser.add_argument("--max-frames", type=int, help="stop after this many decoded frames")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--smoke-weights", type=Path)
    model.add_argument("--project-camera-root", type=Path,
                       help="reuse the original project's pinned CPU ONNX model")
    parser.add_argument("--smoke-label", default="smoke")
    parser.add_argument("--smoke-device", default="cpu")
    parser.add_argument("--smoke-imgsz", type=int, default=640)
    parser.add_argument("--smoke-score", type=float, default=0.35)
    parser.add_argument("--smoke-hz", type=float, default=5)
    parser.add_argument("--smoke-window", type=int, default=5)
    parser.add_argument("--smoke-votes", type=int, default=3)
    parser.add_argument("--spark-luma", type=int, default=180)
    parser.add_argument("--spark-delta", type=int, default=45)
    parser.add_argument("--spark-min-area", type=int, default=1)
    parser.add_argument("--spark-max-area", type=int, default=120)
    parser.add_argument("--spark-global-fraction", type=float, default=0.08)
    parser.add_argument("--spark-merge-s", type=float, default=0.25)
    args = parser.parse_args()
    if args.live_relay:
        if args.video is not None:
            parser.error("choose either a local recording or the loopback RTSP relay")
        relay = urlsplit(args.live_relay)
        if (relay.scheme != "rtsp" or relay.hostname not in ("127.0.0.1", "::1", "localhost")
                or relay.username is not None or relay.query or relay.fragment
                or not math.isfinite(args.seconds) or not 0 < args.seconds <= 300):
            parser.error("use an existing loopback RTSP relay without embedded credentials")
    elif args.video is None or not args.video.is_file():
        parser.error("video must be a local file")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max frames must be positive")
    if args.smoke_weights and not args.smoke_weights.is_file():
        parser.error("smoke weights must be a local file")
    if not (1 <= args.smoke_votes <= args.smoke_window and
            0 < args.smoke_hz <= 120 and 0 < args.smoke_score <= 1 and args.smoke_imgsz > 0):
        parser.error("invalid smoke settings")
    if not (0 <= args.spark_luma <= 255 and 1 <= args.spark_delta <= 255 and
            0 < args.spark_min_area <= args.spark_max_area and
            0 < args.spark_global_fraction <= 1 and
            math.isfinite(args.spark_merge_s) and args.spark_merge_s >= 0):
        parser.error("invalid spark settings")
    return args


def spark_boxes(gray, previous, args, cv2):
    rise = cv2.subtract(gray, previous)
    bright = cv2.inRange(gray, args.spark_luma, 255)
    changed = cv2.inRange(rise, args.spark_delta, 255)
    mask = cv2.bitwise_and(bright, changed)
    active = cv2.countNonZero(mask)
    if active == 0 or active / mask.size >= args.spark_global_fraction:
        return []
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [
        [int(x), int(y), int(w), int(h)]
        for x, y, w, h, area in stats[1:]
        if args.spark_min_area <= area <= args.spark_max_area
    ]


def load_project_smoke(root, label, score):
    root = root.resolve()
    sys.path.insert(0, str(root))
    from fire_detector import FireDetector

    data = json.loads((root / "config/fire_model.json").read_text(encoding="utf-8"))
    model = FireDetector(root / data["path"], data["sha256"], threads=data["cpu_threads"])
    if label not in model.names.values():
        raise ValueError(f"missing smoke class {label!r}. Model classes: {model.names}")

    def detect(frame):
        detections = model.predict(frame, threshold=score)["detections"]
        return [
            ([round(value) for value in d["bbox"]], d["confidence"])
            for d in detections if d["class_name"] == label
        ]

    return detect


def smoke_boxes(bgr_roi, model, args):
    result = model.predict(
        bgr_roi, imgsz=args.smoke_imgsz, conf=args.smoke_score,
        device=args.smoke_device, verbose=False,
    )[0]
    names = result.names
    if args.smoke_label not in names.values():
        raise ValueError(f"missing smoke class {args.smoke_label!r}. Model classes: {names}")
    boxes = []
    for xyxy, class_id, score in zip(
        result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
    ):
        if names[int(class_id)] == args.smoke_label:
            x1, y1, x2, y2 = map(round, xyxy)
            boxes.append(([x1, y1, x2 - x1, y2 - y1], float(score)))
    return boxes


def emit(kind, frame_idx, seconds, boxes, origin, args, score=None):
    ox, oy = origin
    record = {
        "type": kind,
        "camera_id": args.camera_id,
        "video": args.video.name if args.video is not None else None,
        "frame_idx": frame_idx,
        "video_seconds": round(seconds, 6),
        "boxes_xywh": [[x + ox, y + oy, w, h] for x, y, w, h in boxes],
    }
    if score is not None:
        record["score"] = round(score, 6)
    if getattr(args, "live_relay", None):
        record["observed_at_unix"] = time.time()
    print(json.dumps(record, ensure_ascii=True), flush=True)


def replay(cap, args, cv2, smoke):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video FPS unavailable")
    if smoke and args.smoke_hz > fps:
        raise ValueError("smoke sampling frequency exceeds video FPS")
    ok, first = cap.read()
    if not ok:
        raise ValueError("first frame could not be decoded")
    height, width = first.shape[:2]
    x, y, w, h = args.roi or (0, 0, width, height)
    if not (0 <= x < width and 0 <= y < height and 0 < w <= width - x and 0 < h <= height - y):
        raise ValueError(f"ROI exceeds the video dimensions {width}x{height}")

    origin = (x, y)
    last_spark_hit = -math.inf
    smoke_history = deque(maxlen=args.smoke_window)
    smoke_active = False
    smoke_sample_count = 0
    next_smoke_sample = 0.0
    previous = None
    frame_idx = 0
    frame = first
    real_time = bool(getattr(args, "live_relay", None))
    started = time.monotonic()
    while ok:
        seconds = time.monotonic() - started if real_time else frame_idx / fps
        if frame.shape[:2] != (height, width):
            raise ValueError(f"frame dimensions changed at frame {frame_idx}")
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            boxes = spark_boxes(gray, previous, args, cv2)
            if boxes:
                if seconds - last_spark_hit > args.spark_merge_s:
                    emit("spark_candidate", frame_idx, seconds, boxes, origin, args)
                last_spark_hit = seconds
        previous = gray

        if smoke and seconds + 1e-9 >= next_smoke_sample:
            candidates = smoke(roi)
            smoke_history.append(bool(candidates))
            if not smoke_active and candidates and sum(smoke_history) >= args.smoke_votes:
                boxes, scores = zip(*candidates)
                emit("smoke_candidate", frame_idx, seconds, boxes, origin, args, max(scores))
                smoke_active = True
            elif smoke_active and not any(smoke_history):
                smoke_active = False
            smoke_sample_count += 1
            next_smoke_sample = (seconds + 1 / args.smoke_hz if real_time
                                 else smoke_sample_count / args.smoke_hz)

        frame_idx += 1
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        if real_time and time.monotonic() - started >= args.seconds:
            break
        ok, frame = cap.read()
        if real_time and not ok:
            raise ValueError("RTSP relay disconnected during observation")
    print(json.dumps({"frames": frame_idx, "nominal_fps": fps,
                      "smoke_samples": smoke_sample_count}), file=sys.stderr)


def main():
    args = arguments()
    import cv2

    smoke = None
    if args.project_camera_root:
        smoke = load_project_smoke(args.project_camera_root, args.smoke_label, args.smoke_score)
    elif args.smoke_weights:
        from ultralytics import YOLO

        model = YOLO(str(args.smoke_weights), task="detect")

        def smoke(frame):
            return smoke_boxes(frame, model, args)
    if args.live_relay:
        cap = cv2.VideoCapture(args.live_relay, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000,
                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    else:
        cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            raise ValueError("cannot open loopback RTSP relay" if args.live_relay
                             else f"cannot open {args.video}")
        replay(cap, args, cv2, smoke)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
