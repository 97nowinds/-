from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2


def probe_video(path: str | Path) -> dict:
    path = Path(path)
    container = av.open(str(path))
    try:
        stream = next((s for s in container.streams.video), None)
        if stream is None:
            return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size, "video": None}
        duration = None
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size,
            "duration": duration,
            "codec": stream.codec_context.name,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "pixel_format": stream.codec_context.pix_fmt,
            "frames": stream.frames,
            "time_base": str(stream.time_base),
        }
    finally:
        container.close()


def transcode_to_browser_mp4(
    source_path: str | Path,
    target_path: str | Path,
    fps: float,
) -> dict:
    source_path = Path(source_path)
    target_path = Path(target_path)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开待转码视频: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"待转码视频尺寸无效: {source_path}")

    out_width = width + (width % 2)
    out_height = height + (height % 2)
    rate = Fraction(max(1, round(float(fps) * 1000)), 1000)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(target_path), mode="w", format="mp4", options={"movflags": "faststart"})
    stream = container.add_stream("libx264", rate=rate)
    stream.width = out_width
    stream.height = out_height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23", "profile": "main"}

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if out_width != width or out_height != height:
                frame_bgr = cv2.copyMakeBorder(
                    frame_bgr,
                    0,
                    out_height - height,
                    0,
                    out_width - width,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        cap.release()
        container.close()

    info = probe_video(target_path)
    if not info.get("duration") or info["duration"] <= 0:
        raise RuntimeError(f"H.264 转码后视频时长无效: {target_path}")
    if info.get("codec") != "h264":
        raise RuntimeError(f"H.264 转码失败，当前 codec={info.get('codec')}: {target_path}")
    if info.get("pixel_format") != "yuv420p":
        raise RuntimeError(f"H.264 转码后像素格式不是 yuv420p: {info.get('pixel_format')}")
    return info
