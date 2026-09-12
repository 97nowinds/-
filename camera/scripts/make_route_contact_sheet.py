import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (("cam_1", 25.0), ("cam_entrance", 20.0), ("cam_2", 25.0))


def fetch_frame(api, subject, session, camera_id, frame_index):
    url = (
        f"{api}/api/recordings/{subject}/{session}/frame/{camera_id}"
        f"?index={frame_index}"
    )
    with urllib.request.urlopen(url, timeout=20) as response:
        encoded = np.frombuffer(response.read(), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot decode {url}")
    return frame


def main():
    parser = argparse.ArgumentParser(description="Create a three-camera route contact sheet")
    parser.add_argument("--api", default="http://127.0.0.1:15000")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--step", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seconds = []
    value = args.start
    while value <= args.end + 1e-6:
        seconds.append(value)
        value += args.step
    rows = []
    for second in seconds:
        cells = []
        for camera_id, fps in CAMERAS:
            frame_index = round(second * fps)
            frame = fetch_frame(
                args.api.rstrip("/"), args.subject, args.session, camera_id, frame_index
            )
            height, width = frame.shape[:2]
            scale = min(480 / width, 270 / height)
            resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
            cell = np.zeros((310, 500, 3), dtype=np.uint8)
            left = (500 - resized.shape[1]) // 2
            cell[30 : 30 + resized.shape[0], left : left + resized.shape[1]] = resized
            cv2.putText(
                cell,
                f"{camera_id}  t={second:.1f}s  frame={frame_index}",
                (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (110, 235, 170),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
        rows.append(np.hstack(cells))
    sheet = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"cannot write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
