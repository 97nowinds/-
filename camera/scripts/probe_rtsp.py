"""Validate one RTSP stream before adding it to the Flask service."""

import argparse
import os
import sys

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-env", required=True, help="RTSP URL 环境变量名")
    parser.add_argument("--frames", type=int, default=30, help="读取的最少有效帧数")
    args = parser.parse_args()

    url = os.environ.get(args.url_env, "").strip()
    if not url:
        print(f"未设置环境变量 {args.url_env}", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        print("无法打开 RTSP 流，请检查 IP、账号密码、网络和摄像头 RTSP 设置", file=sys.stderr)
        return 1

    read_count = 0
    width = height = 0
    while read_count < args.frames:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        height, width = frame.shape[:2]
        read_count += 1
    capture.release()

    if read_count < args.frames:
        print(f"RTSP 流不稳定：仅读取到 {read_count}/{args.frames} 帧", file=sys.stderr)
        return 1
    print(f"RTSP 流可用：{width}x{height}，连续读取 {read_count} 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
