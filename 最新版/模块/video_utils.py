from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image

from settings import MAX_CLIP_SECONDS, MAX_VIDEO_MB, OUTPUTS_DIR


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class VideoProcessingError(ValueError):
    pass


@dataclass
class VideoInfo:
    path: Path
    file_name: str
    size_mb: float
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class ExtractedFrame:
    image: Image.Image
    timestamp: float
    width: int
    height: int
    path: Path | None = None


def inspect_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    if not path.exists():
        raise VideoProcessingError("视频文件不存在。")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoProcessingError("视频格式不支持。请使用 MP4，或尝试 AVI、MOV、MKV。")

    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > MAX_VIDEO_MB:
        raise VideoProcessingError(f"视频文件过大：{size_mb:.1f} MB。当前限制为 {MAX_VIDEO_MB} MB。")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoProcessingError("视频格式无法读取，文件可能损坏或编码不受支持。")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise VideoProcessingError("视频文件损坏，或无法读取帧率/帧数信息。")

    duration = frame_count / fps
    if duration <= 0 or not math.isfinite(duration):
        raise VideoProcessingError("视频时长无法识别。")

    return VideoInfo(path, path.name, size_mb, duration, fps, frame_count, width, height)


def normalize_clip_range(
    info: VideoInfo,
    start_seconds: float | None,
    end_seconds: float | None,
) -> tuple[float, float]:
    start = 0.0 if start_seconds is None else float(start_seconds)
    end = info.duration_seconds if end_seconds is None else float(end_seconds)

    if start < 0 or end < 0:
        raise VideoProcessingError("开始时间和结束时间不能为负数。")
    if start >= end:
        raise VideoProcessingError("开始时间必须小于结束时间。")
    if start > info.duration_seconds:
        raise VideoProcessingError("开始时间超出视频长度。")
    if end > info.duration_seconds + 0.05:
        raise VideoProcessingError("结束时间超出视频长度。")
    if end - start > MAX_CLIP_SECONDS:
        raise VideoProcessingError(f"默认最多分析 {MAX_CLIP_SECONDS:.0f} 秒视频片段。请缩短裁剪范围。")
    return start, min(end, info.duration_seconds)


def _resize_keep_ratio(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _target_timestamps(start: float, end: float, frame_count: int) -> list[float]:
    if frame_count < 1 or frame_count > 12:
        raise VideoProcessingError("抽帧数量必须在 1 到 12 之间。")
    if frame_count == 1:
        return [(start + end) / 2]
    step = (end - start) / (frame_count - 1)
    return [start + i * step for i in range(frame_count)]


def extract_keyframes(
    video_path: str | Path,
    start_seconds: float | None,
    end_seconds: float | None,
    frame_count: int,
    max_side: int,
    output_dir: Path | None = None,
) -> tuple[VideoInfo, tuple[float, float], list[ExtractedFrame]]:
    info = inspect_video(video_path)
    start, end = normalize_clip_range(info, start_seconds, end_seconds)
    timestamps = _target_timestamps(start, end, int(frame_count))

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise VideoProcessingError("视频格式无法读取，文件可能损坏或编码不受支持。")

    frames: list[ExtractedFrame] = []
    for index, timestamp in enumerate(timestamps, start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            frame_index = min(max(int(timestamp * info.fps), 0), info.frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = _resize_keep_ratio(image, max_side)
        item = ExtractedFrame(image=image, timestamp=timestamp, width=image.width, height=image.height)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"frame_{index:02d}_{timestamp:.2f}s.jpg"
            image.save(out_path, quality=92)
            item.path = out_path
        frames.append(item)

    cap.release()

    if not frames:
        raise VideoProcessingError("无法抽取视频帧。请检查视频编码或缩短裁剪范围。")

    return info, (start, end), frames


def create_preview_dir() -> Path:
    preview_dir = OUTPUTS_DIR / "previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir, ignore_errors=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir


def frames_to_gallery(frames: list[ExtractedFrame]) -> list[tuple[str, str]]:
    gallery: list[tuple[str, str]] = []
    for frame in frames:
        if frame.path is None:
            continue
        gallery.append((str(frame.path), f"{frame.timestamp:.2f}s | {frame.width}x{frame.height}"))
    return gallery
