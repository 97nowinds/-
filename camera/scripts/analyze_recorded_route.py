import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def flush_batch(model, frames, indices, output, args):
    if not frames:
        return
    results = model.predict(
        frames,
        classes=[0],
        conf=args.confidence,
        imgsz=args.image_size,
        device=args.device,
        verbose=False,
    )
    for frame_index, result in zip(indices, results):
        detections = []
        if result.boxes is not None:
            for coordinates, confidence in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                detections.append(
                    {
                        "confidence": round(float(confidence), 4),
                        "box": [round(float(value), 1) for value in coordinates],
                    }
                )
        output.append({"frame_index": frame_index, "detections": detections})
    frames.clear()
    indices.clear()


def analyze_video(model, path, args):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    stride = max(1, round(fps / args.sample_fps))
    output = []
    frames = []
    indices = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            frames.append(frame)
            indices.append(frame_index)
            if len(frames) >= args.batch_size:
                flush_batch(model, frames, indices, output, args)
        frame_index += 1
    flush_batch(model, frames, indices, output, args)
    capture.release()
    return {
        "video_file": path.name,
        "fps": fps,
        "frames": frame_index,
        "sample_stride": stride,
        "samples": output,
    }


def main():
    parser = argparse.ArgumentParser(description="Scan recorded routes for person boundaries")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    metadata = json.loads((args.session / "info.json").read_text(encoding="utf-8"))
    model = YOLO(str(args.model))
    report = {"session": str(args.session), "cameras": {}}
    for camera_id, stream in metadata.get("streams", {}).items():
        video_file = stream.get("video_file")
        if video_file:
            report["cameras"][camera_id] = analyze_video(
                model, args.session / video_file, args
            )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
