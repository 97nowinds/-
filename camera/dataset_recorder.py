import csv
import json
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


SUBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024


class RecordingError(RuntimeError):
    pass


def iso_time(timestamp):
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="milliseconds")


def validate_subject_id(value):
    subject_id = str(value or "").strip()
    if not SUBJECT_ID_PATTERN.fullmatch(subject_id):
        raise RecordingError("人员编号只能包含字母、数字、下划线和连字符，长度为1到40位")
    return subject_id


class RecordingStream:
    def __init__(self, directory, camera_id, frame, fps):
        self.camera_id = camera_id
        self.width = int(frame.shape[1])
        self.height = int(frame.shape[0])
        self.fps = float(fps) if 1.0 <= float(fps or 0) <= 60.0 else 20.0
        self.frames = 0
        self.first_frame_at = None
        self.last_frame_at = None
        self.error = None
        self.lock = threading.RLock()

        self.video_path, self.codec, self.writer = self._open_writer(directory)
        self.timestamps_path = directory / f"{camera_id}_timestamps.csv"
        self.timestamps_file = self.timestamps_path.open("w", encoding="utf-8", newline="")
        self.timestamps = csv.writer(self.timestamps_file)
        self.timestamps.writerow(("frame_index", "unix_time", "local_time"))

    def _open_writer(self, directory):
        candidates = (
            ("mp4v", ".mp4"),
            ("MJPG", ".avi"),
        )
        for codec, suffix in candidates:
            path = directory / f"{self.camera_id}{suffix}"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                self.fps,
                (self.width, self.height),
            )
            if writer.isOpened():
                return path, codec, writer
            writer.release()
            path.unlink(missing_ok=True)
        raise RecordingError(f"无法为{self.camera_id}创建视频编码器")

    def write(self, frame, captured_at):
        with self.lock:
            if self.writer is None or self.error:
                return
            try:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self.writer.write(frame)
                self.timestamps.writerow((self.frames, f"{captured_at:.6f}", iso_time(captured_at)))
                self.frames += 1
                self.first_frame_at = self.first_frame_at or captured_at
                self.last_frame_at = captured_at
                if self.frames % 100 == 0:
                    self.timestamps_file.flush()
            except (OSError, cv2.error) as error:
                self.error = str(error)

    def close(self):
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            if self.timestamps_file is not None:
                self.timestamps_file.flush()
                self.timestamps_file.close()
                self.timestamps_file = None

    def payload(self):
        with self.lock:
            return {
                "camera_id": self.camera_id,
                "video_file": self.video_path.name,
                "timestamps_file": self.timestamps_path.name,
                "codec": self.codec,
                "width": self.width,
                "height": self.height,
                "fps": round(self.fps, 2),
                "frames": self.frames,
                "first_frame_at": iso_time(self.first_frame_at) if self.first_frame_at else None,
                "last_frame_at": iso_time(self.last_frame_at) if self.last_frame_at else None,
                "error": self.error,
            }


class RecordingSession:
    def __init__(self, directory, relative_directory, subject_id, notes, camera_ids):
        self.directory = directory
        self.relative_directory = relative_directory
        self.subject_id = subject_id
        self.notes = notes
        self.camera_ids = tuple(camera_ids)
        self.session_id = directory.name
        self.started_at = time.time()
        self.stopped_at = None
        self.stop_reason = None
        self.streams = {}
        self.lock = threading.RLock()
        self.closed = False
        self.write_metadata("recording")

    def submit(self, camera_id, frame, fps, captured_at):
        if camera_id not in self.camera_ids:
            return
        with self.lock:
            if self.closed:
                return
            stream = self.streams.get(camera_id)
            if stream is None:
                stream = RecordingStream(self.directory, camera_id, frame, fps)
                self.streams[camera_id] = stream
        stream.write(frame, captured_at)

    def close(self, reason):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stopped_at = time.time()
            self.stop_reason = reason
            streams = list(self.streams.values())
        for stream in streams:
            stream.close()
        self.write_metadata("complete")

    def payload(self, active=None):
        with self.lock:
            is_active = not self.closed if active is None else active
            end_time = time.time() if is_active else (self.stopped_at or time.time())
            streams = {camera_id: stream.payload() for camera_id, stream in self.streams.items()}
            for camera_id in self.camera_ids:
                streams.setdefault(
                    camera_id,
                    {
                        "camera_id": camera_id,
                        "video_file": None,
                        "timestamps_file": None,
                        "codec": None,
                        "width": None,
                        "height": None,
                        "fps": None,
                        "frames": 0,
                        "first_frame_at": None,
                        "last_frame_at": None,
                        "error": None,
                    },
                )
            return {
                "active": is_active,
                "session_id": self.session_id,
                "subject_id": self.subject_id,
                "notes": self.notes,
                "directory": self.relative_directory.as_posix(),
                "started_at": iso_time(self.started_at),
                "stopped_at": iso_time(self.stopped_at) if self.stopped_at else None,
                "elapsed_seconds": round(max(0.0, end_time - self.started_at), 1),
                "stop_reason": self.stop_reason,
                "streams": streams,
            }

    def write_metadata(self, status):
        payload = self.payload(active=status == "recording")
        payload["status"] = status
        temporary = self.directory / "info.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "info.json")


class DatasetRecorder:
    def __init__(self, root, camera_ids, min_free_bytes=MIN_FREE_BYTES):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.camera_ids = tuple(camera_ids)
        self.min_free_bytes = min_free_bytes
        self.lock = threading.RLock()
        self.active_session = None
        self.last_session = None

    def start(self, subject_id, notes=""):
        subject_id = validate_subject_id(subject_id)
        notes = str(notes or "").strip()[:300]
        with self.lock:
            if self.active_session is not None:
                raise RecordingError("已有录像正在进行，请先停止当前录像")
            if shutil.disk_usage(self.root).free < self.min_free_bytes:
                raise RecordingError("剩余磁盘空间不足2 GB，不能开始录像")

            subject_directory = self.root / subject_id
            subject_directory.mkdir(parents=True, exist_ok=True)
            session_name = time.strftime("%Y%m%d_%H%M%S")
            directory = subject_directory / session_name
            suffix = 1
            while directory.exists():
                directory = subject_directory / f"{session_name}_{suffix:02d}"
                suffix += 1
            directory.mkdir()
            relative_directory = directory.relative_to(self.root.parent)
            self.active_session = RecordingSession(
                directory,
                relative_directory,
                subject_id,
                notes,
                self.camera_ids,
            )
            return self.active_session.payload()

    def submit(self, camera_id, frame, fps=20.0, captured_at=None):
        with self.lock:
            session = self.active_session
        if session is not None:
            session.submit(camera_id, frame, fps, captured_at or time.time())

    def stop(self, reason="operator"):
        with self.lock:
            session = self.active_session
            if session is None:
                raise RecordingError("当前没有正在进行的录像")
            self.active_session = None
            self.last_session = session
        session.close(reason)
        return session.payload(active=False)

    def stop_if_active(self, reason="shutdown"):
        with self.lock:
            active = self.active_session is not None
        if active:
            return self.stop(reason)
        return None

    def status(self):
        with self.lock:
            session = self.active_session or self.last_session
        if session is None:
            return {
                "active": False,
                "session_id": None,
                "subject_id": None,
                "directory": None,
                "elapsed_seconds": 0.0,
                "streams": {
                    camera_id: {"camera_id": camera_id, "frames": 0, "error": None}
                    for camera_id in self.camera_ids
                },
            }
        return session.payload(active=session is self.active_session)
