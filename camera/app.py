import atexit
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

try:
    import cv2
    import numpy as np
    from flask import Flask, Response, jsonify, render_template, request, send_file

    from camera_source import resolve_camera_source
    from dataset_recorder import DatasetRecorder, RecordingError
    from face_identity import (
        ARCFACE_MIN_GALLERY_FEATURES,
        REGISTRATION_MAX_SAMPLES,
        REGISTRATION_MIN_SAMPLES,
        FaceIdentityStore,
        registration_pose_plan,
    )
    from floor_map import FloorMapProjector
    from hardware_serial import EnvironmentSerialMonitor, discover_hardware_port
    from identity_handoff import IdentityHandoff
    from motion_person_detector import MotionPersonDetector
    from mtmc_config import load_mtmc_config, public_config
    from mtmc_events import AssociationEventStore
    from person_tracking import PersonTrack
    from reid_embedder import ReIDEmbedder
    from slam_localizer import SlamLocalizer
    from yolo_person_tracker import (
        CrossCameraTrackCoordinator,
        TrackTrailStore,
        YoloPersonTracker,
    )
except ModuleNotFoundError as exc:
    missing = exc.name or "unknown"
    raise RuntimeError(
        "Missing Python dependency '%s'. Use camera\\.venv\\Scripts\\python.exe "
        "or run scripts\\start_*.ps1 from the camera directory, then install "
        "requirements with 'python -m pip install -r requirements.txt'." % missing
    ) from exc


# Keep FFmpeg's RTSP demuxer from accumulating old frames. UDP gives the lowest
# latency on a reliable LAN; TCP remains the safer default.
RTSP_TRANSPORT = os.environ.get("LAB_RTSP_TRANSPORT", "tcp").strip().lower()
if RTSP_TRANSPORT not in {"tcp", "udp"}:
    RTSP_TRANSPORT = "tcp"
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    (
        f"rtsp_transport;{RTSP_TRANSPORT}|stimeout;5000000|fflags;nobuffer|"
        "flags;low_delay|max_delay;0|analyzeduration;0|probesize;32768"
    ),
)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "cameras.json"
FLOOR_MAP_PATH = Path(
    os.environ.get("LAB_FLOOR_MAP_PATH", BASE_DIR / "config" / "floor_map.json")
).expanduser().resolve()
MTMC_CONFIG_PATH = BASE_DIR / "config" / "mtmc.json"
DATA_DIR = BASE_DIR / "data"
FACE_DATA_DIR = Path(os.environ.get("LAB_FACE_DATA_DIR", DATA_DIR)).expanduser().resolve()
PEOPLE_PATH = FACE_DATA_DIR / "people.json"
KNOWN_FACES_DIR = FACE_DATA_DIR / "known_faces"
RECORDINGS_DIR = Path(
    os.environ.get("LAB_RECORDINGS_DIR", FACE_DATA_DIR / "recordings")
).expanduser().resolve()
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNITION_MODEL_PATH = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
MODERN_FACE_MODEL_ROOT = BASE_DIR / "models" / "insightface"
FACE_RECOGNITION_ENGINE = os.environ.get("LAB_FACE_ENGINE", "arcface").strip().lower()
REID_MODEL_PATH = BASE_DIR.parent / "person_track" / "weights" / "reID" / "719rank1.pth"
FRAME_WIDTH = 960
RTSP_OPEN_TIMEOUT_MS = 5000
RTSP_READ_TIMEOUT_MS = 3000
RECONNECT_SECONDS = 2.0
FRAME_STALE_SECONDS = 4.0
# Face cascades are CPU-heavy. Recognition does not need to run on every
# tracking frame; the current track remains valid between these attempts.
FACE_DETECT_INTERVAL_SECONDS = 2.0
BODY_DETECT_INTERVAL = 5
IDENTITY_VOTES_REQUIRED = 3
HANDOFF_UNKNOWN_LIMIT = 6
IDENTITY_HANDOFF_TTL_SECONDS = 90.0
BODY_CONFIDENCE_MIN = 0.65
BODY_TRACK_IOU_MIN = 0.12
YOLO_CONFIDENCE_MIN = 0.45
YOLO_IMAGE_SIZE = 640
# 每 5 帧才运行一次 YOLO；中间帧复用上一检测结果，降低 CPU/GPU 压力。
YOLO_FRAME_STRIDE = 5
YOLO_TRACK_HOLD_SECONDS = 0.8
# YuNet is a small neural face detector that is substantially more tolerant of
# small, oblique and partially lit faces than Haar cascades.
FACE_SCORE_THRESHOLD = 0.25
FACE_NMS_THRESHOLD = 0.30
FACE_TOP_K = 5000
# 地图状态允许最多约 200 ms 的显示延迟，避免每个视频帧都触发地图计算和接口刷新。
MAP_PUBLISH_INTERVAL_SECONDS = 0.2
# ORB feature matching is substantially more expensive than map publication.
# Keep the last valid transform for projection between SLAM updates.
SLAM_UPDATE_INTERVAL_SECONDS = 0.5
ANNOTATION_LOCK = threading.RLock()
REGISTRATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
RECORDING_PATH_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REPLAY_REALTIME = os.environ.get("LAB_REPLAY_REALTIME") == "1"
REPLAY_LOOP = os.environ.get("LAB_REPLAY_LOOP") == "1"

app = Flask(__name__)


def recording_session_directory(subject_id, session_id):
    if not RECORDING_PATH_PATTERN.fullmatch(str(subject_id or "")):
        raise ValueError("invalid recording subject")
    if not RECORDING_PATH_PATTERN.fullmatch(str(session_id or "")):
        raise ValueError("invalid recording session")
    directory = (RECORDINGS_DIR / subject_id / session_id).resolve()
    if RECORDINGS_DIR.resolve() not in directory.parents or not directory.is_dir():
        raise FileNotFoundError("recording session not found")
    return directory


def recording_session_payload(directory):
    info_path = directory / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError("recording metadata not found")
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    payload["subject_id"] = directory.parent.name
    payload["session_id"] = directory.name
    return payload


def recording_frame_at(directory, camera_id, relative_seconds):
    info = recording_session_payload(directory)
    stream = info.get("streams", {}).get(camera_id)
    if not stream or not stream.get("timestamps_file"):
        raise ValueError(f"camera timestamps unavailable: {camera_id}")
    path = (directory / stream["timestamps_file"]).resolve()
    if directory not in path.parents or not path.is_file():
        raise ValueError(f"camera timestamps missing: {camera_id}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"camera timestamps empty: {camera_id}")
    requested_seconds = float(relative_seconds)
    if not np.isfinite(requested_seconds) or requested_seconds < 0:
        raise ValueError("video time must be a finite non-negative number")
    fps = float(stream.get("fps") or 0)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"camera FPS unavailable: {camera_id}")
    first_unix = float(rows[0]["unix_time"])
    requested_frame = requested_seconds * fps
    nearest = min(rows, key=lambda row: abs(int(row["frame_index"]) - requested_frame))
    frame_index = int(nearest["frame_index"])
    unix_time = float(nearest["unix_time"])
    return {
        # relative_seconds remains the selected video's timeline for old clients.
        "relative_seconds": round(frame_index / fps, 3),
        "video_seconds": round(frame_index / fps, 3),
        "capture_relative_seconds": round(unix_time - first_unix, 3),
        "unix_time": unix_time,
        "frame_index": frame_index,
    }


@app.after_request
def add_api_cors(response):
    """Allow the separately served static frontend to call this API."""
    origin = os.environ.get("LAB_FRONTEND_ORIGIN", "*")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def load_cameras():
    cameras = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    normalized = []
    for index, camera in enumerate(cameras):
        source_env = camera.get("source_env")
        if camera.get("optional") and not os.environ.get(source_env or "", "").strip():
            continue
        normalized.append(
            {
                "id": camera.get("id", f"cam_{index + 1}"),
                "name": camera.get("name", f"Camera {index + 1}"),
                "role": camera.get("role", "tracking"),
                "face_recognition": bool(camera.get("face_recognition", True)),
                "face_interval_seconds": max(
                    0.1,
                    float(
                        camera.get(
                            "face_interval_seconds", FACE_DETECT_INTERVAL_SECONDS
                        )
                    ),
                ),
                "yolo_confidence": min(
                    0.95,
                    max(0.05, float(camera.get("yolo_confidence", YOLO_CONFIDENCE_MIN))),
                ),
                "yolo_image_size": max(
                    320, int(camera.get("yolo_image_size", YOLO_IMAGE_SIZE))
                ),
                "yolo_frame_stride": max(
                    1, int(camera.get("yolo_frame_stride", YOLO_FRAME_STRIDE))
                ),
                "track_hold_seconds": max(
                    0.1,
                    float(camera.get("track_hold_seconds", YOLO_TRACK_HOLD_SECONDS)),
                ),
                **resolve_camera_source(camera, fallback_index=index),
            }
        )
    return normalized


class CameraWorker:
    def __init__(
        self,
        camera,
        face_store,
        handoff,
        recorder,
        floor_map,
        coordinator,
    ):
        self.camera = camera
        self.face_store = face_store
        self.handoff = handoff
        self.recorder = recorder
        self.floor_map = floor_map
        self.coordinator = coordinator
        self.face_recognition_ready = (
            face_store.modern.available and bool(face_store.modern_features)
            if FACE_RECOGNITION_ENGINE == "arcface"
            else True
        )
        self.face_recognition_error = (
            None
            if self.face_recognition_ready
            else face_store.modern.error
            or "ArcFace gallery requires at least 5 valid features per person"
        )
        self.slam = SlamLocalizer(
            camera.get("id", "camera"),
            floor_map.config.get("slam", {}),
        )
        self.face_detector_lock = threading.RLock()
        self.yunet_detector = None
        self.face_detector_backend = "Haar fallback"
        self.face_detector_error = None
        if FACE_MODEL_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(FACE_MODEL_PATH),
                    "",
                    (320, 320),
                    FACE_SCORE_THRESHOLD,
                    FACE_NMS_THRESHOLD,
                    FACE_TOP_K,
                )
                self.face_detector_backend = "YuNet"
            except Exception as exc:
                self.face_detector_error = str(exc)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        self.face_detector_default = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        if self.face_store.modern.available:
            self.face_detector_backend = self.face_store.modern.engine
            self.face_detector_error = None
        self.body_detector = cv2.HOGDescriptor()
        self.body_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.motion_detector = MotionPersonDetector(warmup_frames=20)
        self.person_track = PersonTrack(lost_seconds=2.0, evidence_seconds=5.0)
        self.yolo_tracker = YoloPersonTracker(
            YOLO_MODEL_PATH,
            confidence=self.camera["yolo_confidence"],
            image_size=self.camera["yolo_image_size"],
        )
        self.yolo_trails = TrackTrailStore(max_points=90, stale_seconds=3.0)
        self.yolo_tracks = []
        self.last_yolo_tracks_at = None
        self.yolo_error = None
        self.yolo_generation = 0
        self.last_yolo_at = None
        self.last_map_publish_at = 0.0
        self.last_slam_at = 0.0
        self.slam_state = self.slam.status_payload()
        self.yolo_inference_ms = None
        self.roi_rejected_tracks = 0
        self.identity = None
        self.identity_track_id = None
        self.identity_votes = deque(maxlen=7)
        self.track_identities = {}
        self.identity_votes_by_track = defaultdict(lambda: deque(maxlen=7))
        self.active_local_ids = set()
        self.handoff_unknown_count = 0
        self.last_face_at = None
        self.last_face_attempt_at = 0.0
        self.last_face_box = None
        self.last_face_result = None
        self.last_face_box_at = 0.0
        self.last_handoff_refresh_at = 0.0
        self.last_handoff_attempt_at = 0.0
        self.track_sequence = 0
        self.local_track_id = None
        self.pending_body_box = None
        self.pending_body_hits = 0
        self.pending_body_at = 0.0
        self.pending_body_origin = None
        self.pending_body_travel = 0.0
        self.lock = threading.RLock()
        self.frame = None
        self.display_frame = None
        self.map_observation = None
        self.map_observations = []
        self.status = "starting"
        self.fps = 0.0
        self.capture_fps = 0.0
        self.processing_latency_ms = None
        self.inference_started_at = None
        self.inference_ended_at = None
        self.status_published_at = None
        self.last_capture_monotonic = None
        self.last_frame_monotonic = None
        self.current_frame_monotonic = None
        self.last_frame_at = None
        self.last_capture_at = None
        self.reconnect_count = 0
        self.frame_counter = 0
        self.capture_frame_counter = 0
        self.dropped_frames = 0
        self.running = True
        self._fps_started = time.monotonic()
        self._fps_frames = 0
        self._capture_fps_started = time.monotonic()
        self._capture_fps_frames = 0
        self.source_fps = 20.0
        self.replay_enabled = REPLAY_REALTIME and self.camera["source_type"] == "file_or_url"
        self.replay_loop = self.replay_enabled and REPLAY_LOOP
        self.replay_start_monotonic = None
        self.replay_start_unix = None
        self.replay_frame_index = 0
        self.replay_cycle = 0
        self.replay_duration_seconds = None
        self.replay_finished = False
        self.capture_condition = threading.Condition()
        self.latest_capture = None
        self.capture_sequence = 0
        self.capture_generation = 0
        self.display_condition = threading.Condition(self.lock)
        self.display_sequence = 0
        self.jpeg_condition = threading.Condition()
        self.jpeg_frames = {False: None, True: None}
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread = threading.Thread(target=self.process_loop, daemon=True)
        self.encoder_thread = threading.Thread(target=self.encode_loop, daemon=True)

    def start(self):
        self.capture_thread.start()
        self.thread.start()
        self.encoder_thread.start()

    def capture_loop(self):
        capture = None
        while self.running:
            if capture is None:
                if self.replay_finished:
                    time.sleep(0.25)
                    continue
                if self.camera["configuration_error"]:
                    self.status = "configuration_error"
                    self.publish_placeholder("RTSP environment variable is not configured")
                    time.sleep(RECONNECT_SECONDS)
                    continue
                self.status = "connecting" if self.reconnect_count == 0 else "reconnecting"
                capture = self.open_capture()
                if capture is None:
                    self.reconnect_count += 1
                    self.publish_placeholder("Camera connection failed")
                    time.sleep(RECONNECT_SECONDS)
                    continue

                source_fps = capture.get(cv2.CAP_PROP_FPS)
                self.source_fps = source_fps if 1.0 <= source_fps <= 60.0 else 20.0
                if self.replay_enabled:
                    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
                    self.replay_duration_seconds = frame_count / self.source_fps
                    self.replay_frame_index = 0
                with self.capture_condition:
                    self.capture_generation += 1
                    self.latest_capture = None
                    self.capture_condition.notify_all()

            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                capture = None
                if self.replay_enabled:
                    if self.replay_loop:
                        self.replay_cycle += 1
                        continue
                    self.replay_finished = True
                    self.status = "replay_complete"
                    continue
                self.reconnect_count += 1
                self.status = "reconnecting"
                with self.capture_condition:
                    self.latest_capture = None
                    self.capture_condition.notify_all()
                self.publish_placeholder("Camera reconnecting")
                time.sleep(RECONNECT_SECONDS)
                continue

            self.status = "running"
            if self.replay_enabled:
                cycle_offset = self.replay_cycle * self.replay_duration_seconds
                frame_offset = self.replay_frame_index / self.source_fps
                captured_monotonic = self.replay_start_monotonic + cycle_offset + frame_offset
                captured_at = self.replay_start_unix + cycle_offset + frame_offset
                delay = captured_monotonic - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                self.replay_frame_index += 1
            else:
                captured_at = time.time()
                captured_monotonic = time.monotonic()
            self.coordinator.register_stream_timestamp(
                self.camera["id"], captured_monotonic, captured_at
            )
            self.recorder.submit(
                self.camera["id"], frame, fps=self.source_fps, captured_at=captured_at
            )
            self.record_capture(captured_at, captured_monotonic)
            with self.capture_condition:
                self.capture_sequence += 1
                self.latest_capture = (
                    self.capture_sequence,
                    self.capture_generation,
                    captured_at,
                    captured_monotonic,
                    frame,
                )
                self.capture_condition.notify_all()

        if capture is not None:
            capture.release()

    def process_loop(self):
        processed_sequence = 0
        processed_generation = 0
        while self.running:
            with self.capture_condition:
                self.capture_condition.wait_for(
                    lambda: not self.running
                    or (
                        self.latest_capture is not None
                        and self.latest_capture[0] > processed_sequence
                    ),
                    timeout=1.0,
                )
                if not self.running:
                    return
                capture = self.latest_capture
            if capture is None:
                continue

            sequence, generation, captured_at, captured_monotonic, frame = capture
            if generation != processed_generation:
                self.reset_processing_state()
                processed_generation = generation
            if processed_sequence:
                self.dropped_frames += max(0, sequence - processed_sequence - 1)
            processed_sequence = sequence

            frame = self.resize(frame)
            self.frame_counter += 1
            inference_started_at = time.time()
            inference_started_monotonic = time.monotonic()
            self.inference_started_at = inference_started_at
            self.record_frame(inference_started_at, inference_started_monotonic)
            display = self.process(
                frame,
                frame_unix_time=captured_at,
                frame_monotonic_time=captured_monotonic,
            )
            self.inference_ended_at = time.time()
            if generation != self.capture_generation or self.status != "running":
                continue
            with self.display_condition:
                self.frame = frame
                self.display_frame = display
                self.processing_latency_ms = round(
                    (time.monotonic() - captured_monotonic) * 1000.0, 1
                )
                self.status_published_at = time.time()
                self.display_sequence += 1
                self.display_condition.notify_all()

    def reset_processing_state(self):
        self.motion_detector.reset()
        self.slam.reset()
        self.slam_state = self.slam.status_payload()
        self.yolo_tracker.reset()
        self.yolo_trails.clear()
        self.last_yolo_tracks_at = None
        self.coordinator.forget_camera(self.camera["id"])
        self.face_store.forget_camera(self.camera["id"])
        self.active_local_ids.clear()
        self.yolo_generation += 1
        self.last_yolo_at = None
        self.last_map_publish_at = 0.0
        self.yolo_inference_ms = None
        self.end_current_track()

    def encode_loop(self):
        encoded_sequence = 0
        while self.running:
            with self.display_condition:
                self.display_condition.wait_for(
                    lambda: not self.running or self.display_sequence > encoded_sequence,
                    timeout=1.0,
                )
                if not self.running:
                    return
                sequence = self.display_sequence
                frame = self.display_frame.copy() if self.display_frame is not None else None
            if frame is None:
                continue

            profiles = {}
            for remote, quality, max_width in ((False, 82, 960), (True, 68, 640)):
                output = frame
                if output.shape[1] > max_width:
                    scale = max_width / output.shape[1]
                    output = cv2.resize(output, (max_width, int(output.shape[0] * scale)))
                ok, encoded = cv2.imencode(
                    ".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                )
                if ok:
                    profiles[remote] = (sequence, encoded.tobytes())
            with self.jpeg_condition:
                self.jpeg_frames.update(profiles)
                self.jpeg_condition.notify_all()
            encoded_sequence = sequence

    def open_capture(self):
        source = self.camera["source"]
        if self.camera["source_type"] == "rtsp":
            try:
                capture = cv2.VideoCapture(
                    source,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        RTSP_OPEN_TIMEOUT_MS,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        RTSP_READ_TIMEOUT_MS,
                    ],
                )
            except (TypeError, cv2.error):
                capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            capture = cv2.VideoCapture(source)
        if capture.isOpened():
            return capture
        capture.release()
        return None

    @staticmethod
    def resize(frame):
        height, width = frame.shape[:2]
        if width <= FRAME_WIDTH:
            return frame
        scale = FRAME_WIDTH / width
        return cv2.resize(frame, (FRAME_WIDTH, int(height * scale)))

    def record_frame(self, unix_time=None, monotonic_time=None):
        self.last_frame_at = time.time() if unix_time is None else float(unix_time)
        self.last_frame_monotonic = (
            time.monotonic() if monotonic_time is None else float(monotonic_time)
        )
        self._fps_frames += 1
        elapsed = time.monotonic() - self._fps_started
        if elapsed >= 1.0:
            self.fps = round(self._fps_frames / elapsed, 1)
            self._fps_frames = 0
            self._fps_started = time.monotonic()

    def record_capture(self, captured_at, monotonic_time=None):
        self.last_capture_at = captured_at
        self.last_capture_monotonic = (
            time.monotonic() if monotonic_time is None else float(monotonic_time)
        )
        self.capture_frame_counter += 1
        self._capture_fps_frames += 1
        elapsed = time.monotonic() - self._capture_fps_started
        if elapsed >= 1.0:
            self.capture_fps = round(self._capture_fps_frames / elapsed, 1)
            self._capture_fps_frames = 0
            self._capture_fps_started = time.monotonic()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.face_store.modern.available:
            records = self.face_store.modern.detect(frame)
            return gray, [record["box"] for record in records]
        # YuNet works on BGR input and returns [x, y, w, h, landmarks..., score].
        # Small person crops are enlarged before inference so a distant face
        # still occupies enough pixels for the detector.
        if self.yunet_detector is not None:
            gray, records = self.detect_yunet_records(frame)
            return gray, [record["box"] for record in records]

        # Fallback for installations where the ONNX file or FaceDetectorYN is
        # unavailable. This keeps the service usable while clearly reporting
        # the active backend in /api/state.
        # A fixed ceiling previously shrank already-small surveillance faces.
        # Upscale only compact head crops so cascades see roughly 80-120 px
        # facial structure while keeping full-frame fallback work bounded.
        # (The legacy path below intentionally remains unchanged.)
        # A fixed ceiling previously shrank already-small surveillance faces.
        # Upscale only compact head crops so cascades see roughly 80-120 px
        # facial structure while keeping full-frame fallback work bounded.
        scale = min(2.0, max(1.0, 640.0 / max(gray.shape[1], 1)))
        detection = (
            gray
            if scale == 1.0
            else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        )
        equalized = cv2.equalizeHist(detection)
        candidates = []
        for detector, neighbors in (
            (self.face_detector, 4),
            (self.profile_face_detector, 4),
        ):
            candidates.extend(
                detector.detectMultiScale(
                    equalized,
                    scaleFactor=1.12,
                    minNeighbors=neighbors,
                    minSize=(28, 28),
                )
            )
        candidates.extend(
            self.profile_face_detector.detectMultiScale(
                equalized,
                scaleFactor=1.12,
                minNeighbors=4,
                minSize=(28, 28),
            )
        )
        flipped = cv2.flip(equalized, 1)
        for x, y, width, height in self.profile_face_detector.detectMultiScale(
            flipped,
            scaleFactor=1.12,
            minNeighbors=4,
            minSize=(28, 28),
        ):
            candidates.append((equalized.shape[1] - x - width, y, width, height))

        inverse = 1.0 / scale
        faces = [
            tuple(int(round(value * inverse)) for value in face)
            for face in candidates
        ]
        plausible = []
        for face in faces:
            box = tuple(map(int, face))
            if self.is_plausible_face(frame, box) and all(
                self.box_iou(box, existing) < 0.45 for existing in plausible
            ):
                plausible.append(box)
        return gray, plausible

    def detect_yunet_records(self, frame):
        """Return YuNet boxes plus landmarks for SFace alignment."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        scale = min(3.0, max(1.0, 640.0 / max(width, 1)))
        detection = (
            frame
            if scale == 1.0
            else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        )
        with self.face_detector_lock:
            self.yunet_detector.setInputSize((detection.shape[1], detection.shape[0]))
            _, detections = self.yunet_detector.detect(detection)
        records = []
        if detections is None:
            return gray, records
        for detection_row in detections:
            x, y, box_width, box_height = detection_row[:4]
            score = float(detection_row[-1])
            if score < FACE_SCORE_THRESHOLD:
                continue
            scaled_box = [x, y, box_width, box_height]
            box = tuple(int(round(value / scale)) for value in scaled_box)
            if self.is_valid_face_box(frame, box) and all(
                self.box_iou(box, existing["box"]) < 0.45 for existing in records
            ):
                row = np.asarray(detection_row, dtype=np.float32).copy()
                row[:14] /= scale
                records.append({"box": box, "detection": row, "score": score})
        return gray, records

    def _legacy_detect_faces(self, gray, frame):
        """Retained only for readable fallback code paths."""
        # This method is intentionally unused; fallback logic lives in
        # detect_faces so existing installations need no extra dependency.
        return gray, []

    @staticmethod
    def is_valid_face_box(frame, box):
        x, y, width, height = map(int, box)
        frame_height, frame_width = frame.shape[:2]
        if width < 16 or height < 16 or x + width <= 0 or y + height <= 0:
            return False
        if x >= frame_width or y >= frame_height:
            return False
        aspect = width / max(height, 1)
        return 0.45 <= aspect <= 1.8

    def select_face_target(self, tracks):
        if not tracks:
            return None
        unidentified = [
            track
            for track in tracks
            if int(track["track_id"]) not in self.track_identities
        ]
        return max(
            unidentified or tracks,
            key=lambda track: (
                float(track.get("confidence", 0.0)) * track["box"][2] * track["box"][3],
                track["box"][2] * track["box"][3],
            ),
        )

    def detect_face_in_person_box(self, frame, person_box):
        x, y, width, height = person_box
        pad_x = max(4, int(round(width * 0.10)))
        # YOLO has already established the person track. Search only the top
        # head-and-shoulder portion so clothing and nearby instruments cannot
        # become face candidates.
        face_limit = max(1, int(round(height * 0.64)))
        left = max(0, x - pad_x * 2)
        top = max(0, y - int(round(height * 0.04)))
        right = min(frame.shape[1], x + width + pad_x * 2)
        bottom = min(frame.shape[0], y + face_limit)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        records = []
        if self.face_store.modern.available:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            records = self.face_store.modern.detect(crop)
            faces = [record["box"] for record in records]
        elif self.yunet_detector is not None:
            gray, records = self.detect_yunet_records(crop)
            faces = [record["box"] for record in records]
        else:
            gray, faces = self.detect_faces(crop)
        if not faces:
            return None

        crop_center_x = crop.shape[1] / 2.0
        crop_center_y = crop.shape[0] * 0.30

        def face_score(box):
            fx, fy, fw, fh = box
            face_area = fw * fh
            center_x = fx + fw / 2.0
            center_y = fy + fh / 2.0
            center_penalty = abs(center_x - crop_center_x) + abs(center_y - crop_center_y)
            return face_area - center_penalty * 2.0

        face_box = max(faces, key=face_score)
        selected_record = next(
            (record for record in records if record["box"] == face_box),
            None,
        )
        fx, fy, fw, fh = face_box
        face = gray[fy : fy + fh, fx : fx + fw]
        if face.size == 0:
            return None
        aligned_face = None
        embedding = None
        if selected_record is not None and hasattr(self.face_store, "align_face"):
            aligned_face = self.face_store.align_face(crop, selected_record["detection"])
            embedding = selected_record.get("embedding")
        return (left + fx, top + fy, fw, fh), face, aligned_face, embedding

    def confirm_face_identity(self, frame, person_box, track_id=None):
        detection = self.detect_face_in_person_box(frame, person_box)
        if detection is None:
            return None
        face_box, face, aligned_face, embedding = detection
        appearance = self.handoff.appearance(frame, person_box)
        result = self.face_store.predict(
            face,
            aligned_face=aligned_face,
            track_key=(self.camera["id"], track_id),
            query_embedding=embedding,
        )
        self.accept_face_result(result, appearance, track_id=track_id)
        self.last_face_at = time.time()
        return face_box, result

    @staticmethod
    def is_plausible_face(frame, box):
        x, y, width, height = box
        crop = frame[y : y + height, x : x + width]
        if crop.size == 0 or width < 28 or height < 28:
            return False
        aspect = width / max(height, 1)
        if not 0.72 <= aspect <= 1.38:
            return False
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(
            ycrcb,
            np.array((0, 132, 76), dtype=np.uint8),
            np.array((255, 178, 132), dtype=np.uint8),
        )
        skin_ratio = cv2.countNonZero(skin) / max(width * height, 1)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        texture = float(np.std(gray_crop))
        return skin_ratio >= 0.055 and texture >= 8.0

    def detect_bodies(self, frame):
        detection = frame
        scale = min(1.0, 640.0 / max(frame.shape[1], 1))
        if scale < 1.0:
            detection = cv2.resize(frame, None, fx=scale, fy=scale)
        try:
            boxes, weights = self.body_detector.detectMultiScale(
                detection, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
        except cv2.error:
            return []
        candidates = []
        inverse = 1.0 / scale
        for box, weight in zip(boxes, weights):
            if float(weight) < BODY_CONFIDENCE_MIN:
                continue
            transformed = tuple(int(round(value * inverse)) for value in box)
            candidates.append((float(weight), transformed))
        return sorted(candidates, key=lambda item: item[0], reverse=True)

    @staticmethod
    def box_iou(first, second):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / max(union, 1)

    @staticmethod
    def motion_to_person_box(box, frame_shape):
        """Expand a moving body part into a conservative whole-person box."""
        x, y, width, height = box
        aspect = width / max(height, 1)
        width_scale = 1.25 if aspect < 0.48 else 1.4
        height_scale = 1.25 if aspect < 0.48 else 2.05
        person_width = max(width, int(round(width * width_scale)))
        person_height = max(
            height,
            int(round(height * height_scale)),
            int(round(person_width * 2.0)),
        )
        person_x = int(round(x + width / 2 - person_width / 2))
        person_y = int(round(y - height * 0.12))
        return PersonTrack.tracker_box(
            (person_x, person_y, person_width, person_height), frame_shape
        )

    def accept_face_result(self, result, appearance, track_id=None):
        if track_id is not None:
            track_id = int(track_id)
            votes_for_track = self.identity_votes_by_track[track_id]
            if result["known"]:
                votes_for_track.append(result["person_id"])
                person_id, votes = Counter(votes_for_track).most_common(1)[0]
                if votes >= IDENTITY_VOTES_REQUIRED and person_id == result["person_id"]:
                    identity = {**result, "identity_source": "face"}
                    self.track_identities[track_id] = identity
                    self.identity = identity
                    self.identity_track_id = track_id
                    self.handoff_unknown_count = 0
                    self.handoff.confirm(self.camera["id"], identity, appearance)
                    return identity
            else:
                votes_for_track.append(None)
            return None
        if result["known"]:
            self.identity_votes.append(result["person_id"])
            person_id, votes = Counter(self.identity_votes).most_common(1)[0]
            if votes >= IDENTITY_VOTES_REQUIRED and person_id == result["person_id"]:
                self.identity = {**result, "identity_source": "face"}
                self.handoff_unknown_count = 0
                self.handoff.confirm(self.camera["id"], self.identity, appearance)
            return

        self.identity_votes.append(None)
        # A confirmed identity is owned by the active ByteTrack target. Do not
        # clear a handoff identity merely because the person turns away or the
        # face is temporarily too small; the target-loss check in process()
        # clears it when the tracked person actually disappears.
        if self.identity and self.identity.get("identity_source") == "handoff":
            self.handoff_unknown_count += 1
        return None

    def _sync_primary_identity(self):
        if self.identity_track_id in self.track_identities:
            self.identity = self.track_identities[self.identity_track_id]
            return
        if self.track_identities:
            self.identity_track_id = sorted(self.track_identities)[0]
            self.identity = self.track_identities[self.identity_track_id]
            return
        self.identity = None
        self.identity_track_id = None

    def clear_identity(self, track_id=None):
        if track_id is not None:
            track_id = int(track_id)
            self.track_identities.pop(track_id, None)
            self.identity_votes_by_track.pop(track_id, None)
            if self.identity_track_id == track_id:
                self.identity_track_id = None
            self._sync_primary_identity()
            return
        self.track_identities.clear()
        self.identity_votes_by_track.clear()
        self.identity = None
        self.identity_track_id = None
        self.identity_votes.clear()
        self.handoff_unknown_count = 0

    def begin_local_track(self):
        if self.local_track_id is None:
            self.track_sequence += 1
            self.local_track_id = f"unknown:{self.camera['id']}:{self.track_sequence}"
        return self.local_track_id

    def end_current_track(self):
        self.person_track.stop()
        self.clear_identity()
        self.last_face_box = None
        self.last_face_result = None
        self.last_face_box_at = 0.0
        self.local_track_id = None
        self.pending_body_box = None
        self.pending_body_hits = 0
        self.pending_body_at = 0.0
        self.pending_body_origin = None
        self.pending_body_travel = 0.0
        self.last_handoff_refresh_at = 0.0
        self.last_handoff_attempt_at = 0.0
        with self.lock:
            self.map_observation = None
            self.map_observations = []
            self.yolo_tracks = []
        self.yolo_trails.clear()
        self.coordinator.forget_camera(self.camera["id"])

    def body_candidate_is_stable(self, box, frame_shape, require_displacement=False):
        now = time.monotonic()
        center = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        if (
            self.pending_body_box is not None
            and now - self.pending_body_at <= 1.5
            and self.box_iou(self.pending_body_box, box) >= 0.20
        ):
            self.pending_body_hits += 1
        else:
            self.pending_body_hits = 1
            self.pending_body_origin = center
            self.pending_body_travel = 0.0
        if self.pending_body_origin is not None:
            dx = center[0] - self.pending_body_origin[0]
            dy = center[1] - self.pending_body_origin[1]
            self.pending_body_travel = max(
                self.pending_body_travel, (dx * dx + dy * dy) ** 0.5
            )
        self.pending_body_box = box
        self.pending_body_at = now
        minimum_travel = max(10.0, min(frame_shape[:2]) * 0.025)
        return self.pending_body_hits >= 3 and (
            not require_displacement or self.pending_body_travel >= minimum_travel
        )

    def refresh_handoff(self, frame, box, identity=None):
        identity = identity or self.identity
        if not identity or not identity.get("person_id"):
            return
        now = time.monotonic()
        if now - self.last_handoff_refresh_at < 0.75:
            return
        appearance = self.handoff.appearance(frame, box)
        self.handoff.confirm(self.camera["id"], identity, appearance)
        self.last_handoff_refresh_at = now

    def body_candidate_is_walkable(self, box, frame_shape):
        position = self.project_box(box, frame_shape)
        return self.floor_map.is_walkable(position)

    def project_box(self, box, frame_shape):
        """Project a person's foot point, preferring the active visual SLAM pose."""
        x, y, width, height = box
        foot = ((x + width * 0.5), (y + height))
        slam_position = self.slam.project_pixel(
            foot, monotonic_time=self.current_frame_monotonic
        )
        if slam_position is not None:
            return self.floor_map.clamp_position(slam_position)
        return self.floor_map.project(self.camera["id"], box, frame_shape)

    @staticmethod
    def yolo_color(track_id):
        return (
            70 + (track_id * 53) % 150,
            90 + (track_id * 97) % 140,
            80 + (track_id * 31) % 160,
        )

    def process(self, frame, frame_unix_time=None, frame_monotonic_time=None):
        display = frame.copy()
        frame_unix_time = time.time() if frame_unix_time is None else float(frame_unix_time)
        now = time.monotonic() if frame_monotonic_time is None else float(frame_monotonic_time)
        self.current_frame_monotonic = now
        if now - self.last_slam_at >= SLAM_UPDATE_INTERVAL_SECONDS:
            self.slam_state = self.slam.update(
                frame,
                baseline_transform=self.floor_map.pixel_transform(
                    self.camera["id"], frame.shape
                ),
                monotonic_time=now,
                unix_time=frame_unix_time,
            )
            self.last_slam_at = now
        # 只在每路摄像头配置的抽样帧运行 YOLO。非抽样帧使用上一批检测框绘制画面，
        # 这样视频流仍然连续，而检测模型不再被每一帧调用。
        should_infer = (
            self.frame_counter == 1
            or self.frame_counter % self.camera["yolo_frame_stride"] == 0
            or self.last_yolo_at is None
        )
        cached_global_ids = {}
        if should_infer:
            started_at = time.perf_counter()
            try:
                tracks = self.yolo_tracker.track(frame)
                self.yolo_error = None
            except Exception as error:
                tracks = []
                self.yolo_error = str(error)
            self.yolo_inference_ms = round((time.perf_counter() - started_at) * 1000.0, 1)
            self.last_yolo_at = now
        else:
            # 将已发布的缓存结果转换为检测器返回的字段，统一走绘制逻辑。
            with self.lock:
                cached_tracks = [dict(item) for item in self.yolo_tracks]
            tracks = [
                {
                    "track_id": item["local_id"],
                    "box": item["box"],
                    "confidence": item["confidence"],
                }
                for item in cached_tracks
            ]
            cached_global_ids = {
                item["local_id"]: item["track_id"] for item in cached_tracks
            }

        # Filter the detector result before deciding whether the previous
        # trustworthy tracks should be held. A transient bench/equipment false
        # positive must not suppress the short occlusion hold for a real person.
        unfiltered_track_count = len(tracks)
        tracks = [
            track
            for track in tracks
            if self.floor_map.track_footpoint_allowed(
                self.camera["id"], track.get("box"), frame.shape
            )
        ]
        self.roi_rejected_tracks = unfiltered_track_count - len(tracks)

        # A detector can miss a person for one inference cycle because of blur,
        # occlusion, or RTSP jitter. Keep the last ByteTrack boxes briefly so
        # the map trajectory remains continuous; a stale target is still
        # removed after the hold window.
        hold_cached_tracks = bool(
            should_infer
            and not tracks
            and self.yolo_tracks
            and self.last_yolo_tracks_at is not None
            and now - self.last_yolo_tracks_at <= self.camera["track_hold_seconds"]
        )
        if hold_cached_tracks:
            with self.lock:
                cached_tracks = [dict(item) for item in self.yolo_tracks]
            cached_global_ids = {
                item["local_id"]: item["track_id"] for item in cached_tracks
            }
            tracks = [
                {
                    "track_id": item["local_id"],
                    "box": item["box"],
                    "confidence": item["confidence"],
                }
                for item in cached_tracks
            ]
            cached_global_ids = {
                item["local_id"]: item["track_id"] for item in cached_tracks
            }

        # A confirmed identity belongs to the active ByteTrack target. Keep
        # that name while the target moves, but clear it when the ID is gone.
        active_local_ids = {int(item["track_id"]) for item in tracks}
        for missing_local_id in self.active_local_ids - active_local_ids:
            self.clear_identity(missing_local_id)
            self.face_store.forget_track((self.camera["id"], missing_local_id))
            self.coordinator.forget_local(self.camera["id"], missing_local_id)
        self.active_local_ids = active_local_ids

        observations = []
        published_tracks = []
        global_ids_by_local_id = {}
        face_result = None
        face_box = None
        prepared_tracks = []
        for track in tracks:
            local_id = track["track_id"]
            x, y, width, height = track["box"]
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            width = max(1, min(width, frame.shape[1] - x))
            height = max(1, min(height, frame.shape[0] - y))
            box = (x, y, width, height)
            foot = (x + width // 2, y + height)
            # A fixed tracking anchor is only meaningful for one person at a
            # doorway. Publishing that same coordinate for a crowd makes the
            # map falsely collapse multiple workers into one point.
            position = (
                None
                if len(tracks) > 1
                and not self.floor_map.has_projection(self.camera["id"])
                else self.project_box(box, frame.shape)
            )
            prepared_tracks.append(
                {
                    **track,
                    "track_id": local_id,
                    "box": box,
                    "position": position,
                }
            )
        if should_infer and not hold_cached_tracks:
            batch_global_ids = self.coordinator.update_batch(
                self.camera["id"],
                prepared_tracks,
                frame,
                active_local_ids=active_local_ids,
                monotonic_time=now,
                unix_time=frame_unix_time,
                stream_skew_seconds=self.coordinator.stream_skew_seconds(now),
            )
        else:
            batch_global_ids = cached_global_ids

        for track in prepared_tracks:
            local_id = track["track_id"]
            x, y, width, height = track["box"]
            box = track["box"]
            foot = (x + width // 2, y + height)
            position = track["position"]
            global_id = batch_global_ids[local_id]
            global_ids_by_local_id[local_id] = global_id
            # A confirmed identity stored by cam_1 is adopted by cam_2 only
            # after this target is physically inside the overlap zone.
            if local_id not in self.track_identities:
                inherited = self.coordinator.identity_for(
                    global_id, self.camera["id"], position, now=now
                )
                if inherited is not None:
                    self.track_identities[local_id] = inherited
                    self.identity_votes_by_track[local_id].clear()
                    self.identity = inherited
                    self.identity_track_id = local_id
                    self.handoff_unknown_count = 0
            trail = self.yolo_trails.update(global_id, foot, now=now)
            global_number = int(global_id.rsplit("_", 1)[-1])
            color = self.yolo_color(global_number)
            track_identity = (
                self.track_identities.get(local_id)
            )
            mtmc_diagnostics = self.coordinator.track_diagnostics(
                self.camera["id"], local_id
            )

            if len(trail) >= 2:
                cv2.polylines(
                    display,
                    [np.asarray(trail, dtype=np.int32).reshape((-1, 1, 2))],
                    False,
                    color,
                    3,
                    cv2.LINE_AA,
                )
            cv2.circle(display, foot, 5, color, -1, cv2.LINE_AA)
            cv2.rectangle(display, (x, y), (x + width, y + height), color, 3)
            if track_identity:
                person_number = track_identity.get("person_number")
                display_label = (
                    f"{person_number} {track_identity.get('name', '')}".strip()
                    if person_number
                    else str(track_identity.get("name", "IDENTIFIED"))
                )
                label_color = (80, 220, 130)
            else:
                display_label = (
                    "UNREGISTERED"
                    if mtmc_diagnostics["counted"]
                    else "PENDING"
                )
                label_color = color
            cv2.putText(
                display,
                f"{display_label} | {track['confidence']:.0%}",
                (x, max(62, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                label_color,
                2,
                cv2.LINE_AA,
            )

            if position is not None:
                observations.append(
                    {
                        "camera_id": self.camera["id"],
                        "track_id": global_id,
                        "local_id": local_id,
                        "position": position,
                        "in_overlap": self.coordinator.is_overlap_position(
                            position, self.camera["id"]
                        ),
                        "person_id": track_identity.get("person_id") if track_identity else None,
                        "person_number": track_identity.get("person_number") if track_identity else None,
                        "name": track_identity.get("name") if track_identity else None,
                        "identity_source": track_identity.get("identity_source", "face") if track_identity else "yolo_handoff",
                        "confidence": track["confidence"],
                        "counted": mtmc_diagnostics["counted"],
                        "count_status": mtmc_diagnostics["count_status"],
                        "count_duplicate_of": mtmc_diagnostics["count_duplicate_of"],
                        "count_duplicate_reid_score": mtmc_diagnostics[
                            "count_duplicate_reid_score"
                        ],
                        "track_age_seconds": mtmc_diagnostics["track_age_seconds"],
                        "observed_at": frame_unix_time,
                        "observed_monotonic": now,
                        "timestamp_source": (
                            "recording_replay" if self.replay_enabled else "host_receive"
                        ),
                    }
                )
            published_tracks.append(
                {
                    "track_id": global_id,
                    "local_id": local_id,
                    "box": box,
                    "confidence": track["confidence"],
                    "reid_gallery": mtmc_diagnostics["gallery"],
                    "association_confidence": mtmc_diagnostics["confidence"],
                    "association_score": mtmc_diagnostics["final_match_score"],
                    "association_pending": mtmc_diagnostics["pending"],
                    "counted": mtmc_diagnostics["counted"],
                    "count_status": mtmc_diagnostics["count_status"],
                    "count_duplicate_of": mtmc_diagnostics["count_duplicate_of"],
                    "count_duplicate_reid_score": mtmc_diagnostics[
                        "count_duplicate_reid_score"
                    ],
                    "track_age_seconds": mtmc_diagnostics["track_age_seconds"],
                    "observation_count": mtmc_diagnostics["observation_count"],
                }
            )

        should_check_face = bool(
            self.camera["face_recognition"]
            and self.face_recognition_ready
            and tracks
            and now - self.last_face_attempt_at
            >= self.camera["face_interval_seconds"]
        )
        if should_check_face:
            self.last_face_attempt_at = now
            target_track = self.select_face_target(tracks)
            if target_track is not None:
                candidate = self.confirm_face_identity(
                    frame, target_track["box"], target_track.get("track_id")
                )
                if candidate is not None:
                    face_box, face_result = candidate
                    self.last_face_box = face_box
                    self.last_face_result = face_result
                    self.last_face_box_at = now

        # Publish a name only after the coordinator grants the single-owner
        # identity lock. Conflicts remain anonymous instead of being mislabelled.
        for identity_local_id, identity in list(self.track_identities.items()):
            bound_track = next(
                (item for item in tracks if item["track_id"] == identity_local_id),
                None,
            )
            if bound_track is not None:
                current_global_id = global_ids_by_local_id.get(identity_local_id)
                current_position = self.project_box(
                    bound_track["box"], frame.shape
                )
                locked_identity = self.coordinator.set_identity(
                    current_global_id,
                    identity,
                    self.camera["id"],
                    current_position,
                    now=now,
                )
                if locked_identity is None:
                    self.clear_identity(identity_local_id)
                else:
                    self.track_identities[identity_local_id] = locked_identity
                    if self.identity_track_id == identity_local_id:
                        self.identity = locked_identity
                    for observation in observations:
                        if observation.get("local_id") != identity_local_id:
                            continue
                        observation.update(
                            {
                                "person_id": locked_identity.get("person_id"),
                                "person_number": locked_identity.get("person_number"),
                                "name": locked_identity.get("name"),
                                "identity_source": locked_identity.get("identity_source", "face"),
                                "identity_lock_status": locked_identity.get("identity_lock_status"),
                                "global_track_id": locked_identity.get("global_track_id"),
                                "handoff_from_camera": locked_identity.get("handoff_from_camera"),
                            }
                        )
                    self.refresh_handoff(frame, bound_track["box"], locked_identity)
        self._sync_primary_identity()

        self.yolo_trails.prune(now=now)
        if not should_infer or hold_cached_tracks:
            # 非检测帧沿用上一检测结果，不重新生成目标 ID，避免轨迹抖动。
            published_tracks = cached_tracks
        should_publish_map = (
            should_infer
            or now - self.last_map_publish_at >= MAP_PUBLISH_INTERVAL_SECONDS
        )
        with self.lock:
            self.yolo_tracks = published_tracks
            if should_infer and not hold_cached_tracks and published_tracks:
                self.last_yolo_tracks_at = now
            elif not published_tracks and self.last_yolo_tracks_at is not None and now - self.last_yolo_tracks_at > self.camera["track_hold_seconds"]:
                self.last_yolo_tracks_at = None
            if should_publish_map:
                self.map_observations = observations
                self.map_observation = observations[0] if observations else None
                self.last_map_publish_at = now

        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (20, 23, 27), -1)
        status = "ERROR" if self.yolo_error else f"{len(published_tracks)} PEOPLE"
        header = f"{self.camera['id']} | YOLOv8 + ByteTrack | {status}"
        cv2.putText(
            display,
            header,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 238, 242) if not self.yolo_error else (80, 120, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    def process_legacy(self, frame):
        display = frame.copy()
        face_boxes = []
        gray = None
        if (self.frame_counter - 1) % FACE_DETECT_INTERVAL == 0:
            gray, face_boxes = self.detect_faces(frame)

        # Motion is only a scene-change hint. It must never create a person
        # track by itself because screens, reflections and moving instruments
        # are common in a laboratory.
        self.motion_detector.detect(frame)
        body_candidates = []
        if self.frame_counter % BODY_DETECT_INTERVAL == 0:
            body_candidates = [
                candidate
                for candidate in self.detect_bodies(frame)
                if self.body_candidate_is_walkable(candidate[1], frame.shape)
            ]

        track_box = self.person_track.update(frame)
        if (
            track_box is None
            and self.person_track.last_seen_at is not None
            and time.time() - self.person_track.last_seen_at > self.person_track.lost_seconds
        ):
            self.end_current_track()
        face_result = None
        if face_boxes:
            face_box = max(face_boxes, key=lambda box: box[2] * box[3])
            person_box = PersonTrack.face_to_person(face_box, frame.shape)
            if self.person_track.observation_is_distinct(person_box):
                self.end_current_track()
            track_id = self.begin_local_track()
            track_box = self.person_track.observe(
                frame,
                person_box,
                self.identity["person_id"] if self.identity else track_id,
            )
            self.pending_body_box = None
            self.pending_body_hits = 0
            self.pending_body_origin = None
            self.pending_body_travel = 0.0
            x, y, width, height = face_box
            face = gray[y : y + height, x : x + width]
            if face.size:
                appearance = self.handoff.appearance(frame, track_box)
                face_result = self.face_store.predict(face)
                self.accept_face_result(face_result, appearance)
                self.last_face_at = time.time()
                cv2.rectangle(display, (x, y), (x + width, y + height), (255, 190, 70), 2)

        # A detector confirmation anchors CamShift to a real person and prevents
        # a color track from slowly walking onto a wall or piece of furniture.
        if track_box is not None and body_candidates:
            _, matched_body = max(
                body_candidates,
                key=lambda item: self.box_iou(track_box, item[1]),
            )
            if self.box_iou(track_box, matched_body) >= BODY_TRACK_IOU_MIN:
                self.person_track.mark_evidence()

        # A local visual tracker can follow any textured object indefinitely.
        # Require periodic face/body confirmation so instruments cannot remain
        # published as people after an initial false proposal.
        if self.person_track.evidence_expired():
            self.end_current_track()
            track_box = None

        # Only a verified body detector result may start a body-only track.
        if track_box is None and not face_boxes and body_candidates:
            _, body_box = body_candidates[0]
            stable = self.body_candidate_is_stable(
                body_box,
                frame.shape,
                require_displacement=False,
            )
            if stable:
                appearance = self.handoff.appearance(frame, body_box)
                inherited = self.handoff.inherit(self.camera["id"], appearance)
                if inherited:
                    self.identity = inherited
                    self.handoff_unknown_count = 0
                    self.begin_local_track()
                    track_box = self.person_track.seed(
                        frame, body_box, inherited["person_id"], validated=True
                    )
                else:
                    track_id = self.begin_local_track()
                    track_box = self.person_track.seed(
                        frame, body_box, track_id, validated=True
                    )
                self.pending_body_box = None
                self.pending_body_hits = 0
                self.pending_body_origin = None
                self.pending_body_travel = 0.0

        if track_box is not None:
            self.refresh_handoff(frame, track_box)

        self.update_map_observation(track_box, frame.shape)
        self.draw_overlay(display, track_box, face_result)
        return display

    def update_map_observation(self, box, frame_shape):
        position = self.project_box(box, frame_shape)
        if position is None:
            with self.lock:
                self.map_observation = None
            return

        identity = self.identity or {}
        observation = {
            "camera_id": self.camera["id"],
            "track_id": self.local_track_id,
            "position": position,
            "person_id": identity.get("person_id"),
            "person_number": identity.get("person_number"),
            "name": identity.get("name"),
            "identity_source": identity.get("identity_source"),
            "observed_at": time.time(),
        }
        with self.lock:
            self.map_observation = observation

    def draw_overlay(self, frame, box, face_result):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 23, 27), -1)
        if box is not None:
            x, y, width, height = box
            if self.identity:
                source = self.identity.get("identity_source", "face")
                color = (70, 225, 120) if source == "face" else (60, 220, 245)
                number = self.identity.get("person_number")
                identity_label = f"{number} {self.identity['name']}" if number else self.identity["name"]
                mode = "FACE+TRACK" if source == "face" else "HANDOFF+TRACK"
                label = f"{identity_label} | {mode}"
            else:
                color = (80, 160, 255)
                label = face_result["name"] if face_result else "SEARCHING IDENTITY"
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
            cv2.putText(
                frame,
                label,
                (x, max(62, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        header = f"{self.camera['id']} | {self.status.upper()}"
        cv2.putText(frame, header, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 238, 242), 2)

    def get_frame(self):
        with self.lock:
            if self.display_frame is not None:
                return self.display_frame.copy()
        return self.placeholder("Waiting for camera")

    def wait_for_jpeg(self, remote, after_sequence, timeout=1.0):
        with self.jpeg_condition:
            self.jpeg_condition.wait_for(
                lambda: not self.running
                or (
                    self.jpeg_frames[remote] is not None
                    and self.jpeg_frames[remote][0] > after_sequence
                ),
                timeout=timeout,
            )
            return self.jpeg_frames[remote]

    def registration_face(self):
        """Return one quality-checkable face constrained by the current YOLO person box."""
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            tracks = [dict(item) for item in self.yolo_tracks]
        if frame is None or self.status != "running":
            return None, "camera_not_ready"
        if not tracks:
            return None, "no_person"
        if len(tracks) > 1:
            return None, "multiple_people"
        detection = self.detect_face_in_person_box(frame, tracks[0]["box"])
        if detection is None:
            return None, "face_not_found"
        face_box, face, _, embedding = detection
        return {
            "face": face,
            "embedding": embedding,
            "face_box": [int(value) for value in face_box],
            "person_box": [int(value) for value in tracks[0]["box"]],
            "captured_at": time.time(),
        }, None

    def status_payload(self):
        now_monotonic = time.monotonic()
        age = (
            now_monotonic - self.last_frame_monotonic
            if self.last_frame_monotonic is not None
            else None
        )
        capture_age = (
            now_monotonic - self.last_capture_monotonic
            if self.last_capture_monotonic is not None
            else None
        )
        status = (
            "stale"
            if self.status == "running"
            and age is not None
            and age > FRAME_STALE_SECONDS
            else self.status
        )
        identity = None
        if self.identity:
            identity = {
                key: self.identity.get(key)
                for key in (
                    "person_id",
                    "person_number",
                    "name",
                    "identity_source",
                    "handoff_from_camera",
                    "appearance_score",
                    "handoff_match_source",
                    "identity_lock_status",
                    "global_track_id",
                )
            }
        identities = []
        for local_id, track_identity in sorted(self.track_identities.items()):
            identities.append(
                {
                    "local_id": local_id,
                    **{
                        key: track_identity.get(key)
                        for key in (
                            "person_id",
                            "person_number",
                            "name",
                            "identity_source",
                            "identity_lock_status",
                            "global_track_id",
                        )
                    },
                }
            )
        with self.lock:
            map_observation = dict(self.map_observation) if self.map_observation else None
            map_observations = [dict(item) for item in self.map_observations]
            yolo_tracks = [dict(item) for item in self.yolo_tracks]
        tracking = bool(yolo_tracks)
        confidence = max(
            (track["confidence"] for track in yolo_tracks), default=0.0
        )
        stream_skew = self.coordinator.stream_skew_seconds(now_monotonic)
        max_skew = self.coordinator.mtmc_config["calibration"]["max_stream_skew_seconds"]
        localization = self.floor_map.localization_status(
            self.camera["id"],
            slam_state=self.slam_state,
            stream_skew_seconds=stream_skew,
            max_stream_skew_seconds=max_skew,
        )
        return {
            "id": self.camera["id"],
            "name": self.camera["name"],
            "source": self.camera["source_display"],
            "source_type": self.camera["source_type"],
            "role": self.camera["role"],
            "face_recognition_enabled": self.camera["face_recognition"],
            "face_recognition_ready": self.face_recognition_ready,
            "face_recognition_error": self.face_recognition_error,
            "status": status,
            "fps": self.fps,
            "capture_fps": self.capture_fps,
            "frame_age_seconds": round(age, 2) if age is not None else None,
            "capture_age_seconds": round(capture_age, 2) if capture_age is not None else None,
            "processing_latency_ms": self.processing_latency_ms,
            "timestamp_source": "recording_replay" if self.replay_enabled else "host_receive",
            "capture_received_at": self.last_capture_at,
            "inference_started_at": self.inference_started_at,
            "inference_ended_at": self.inference_ended_at,
            "status_published_at": self.status_published_at,
            "estimated_stream_skew_seconds": round(stream_skew, 4),
            "dropped_frames": self.dropped_frames,
            "pipeline_mode": "replay_latest_frame" if self.replay_enabled else "latest_frame",
            "replay": {
                "enabled": self.replay_enabled,
                "loop": self.replay_loop,
                "cycle": self.replay_cycle,
                "frame_index": self.replay_frame_index,
                "duration_seconds": self.replay_duration_seconds,
                "finished": self.replay_finished,
            },
            "rtsp_transport": RTSP_TRANSPORT if self.camera["source_type"] == "rtsp" else None,
            "reconnect_count": self.reconnect_count,
            "identity": identity,
            "identities": identities,
            "tracking": tracking,
            "tracked_people": len(yolo_tracks),
            "counted_people": sum(
                1 for track in yolo_tracks if track.get("counted", True)
            ),
            "provisional_people": sum(
                1 for track in yolo_tracks if not track.get("counted", True)
            ),
            "roi_rejected_tracks": self.roi_rejected_tracks,
            "yolo_tracks": yolo_tracks,
            "tracker_engine": "YOLOv8n + ByteTrack",
            "face_detector": self.face_detector_backend,
            "face_detector_error": self.face_detector_error,
            "track_status": "error" if self.yolo_error else "tracking" if tracking else "searching",
            "track_confidence": round(confidence, 4),
            "track_failures": 1 if self.yolo_error else 0,
            "tracker_error": self.yolo_error,
            "inference_frame_stride": self.camera["yolo_frame_stride"],
            "detector_confidence_threshold": self.camera["yolo_confidence"],
            "detector_image_size": self.camera["yolo_image_size"],
            "last_inference_age_seconds": (
                round(now_monotonic - self.last_yolo_at, 2)
                if self.last_yolo_at is not None
                else None
            ),
            "inference_ms": self.yolo_inference_ms,
            "track_hold_ms": int(self.camera["track_hold_seconds"] * 1000),
            "map_publish_interval_ms": int(MAP_PUBLISH_INTERVAL_SECONDS * 1000),
            "map_observation": map_observation,
            "map_observations": map_observations,
            "slam": dict(self.slam_state),
            "localization": localization,
        }

    def publish_placeholder(self, message):
        frame = self.placeholder(message)
        with self.display_condition:
            self.frame = frame.copy()
            self.display_frame = frame
            self.display_sequence += 1
            self.display_condition.notify_all()

    def stop(self):
        self.running = False
        for condition in (
            self.capture_condition,
            self.display_condition,
            self.jpeg_condition,
        ):
            with condition:
                condition.notify_all()

    def placeholder(self, message):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(frame, self.camera["name"], (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 225, 230), 2)
        cv2.putText(frame, message, (28, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 170, 255), 2)
        return frame


class CameraManager:
    def __init__(self, cameras, face_store):
        self.face_store = face_store
        self.floor_map = FloorMapProjector.from_path(FLOOR_MAP_PATH)
        self.mtmc_config = load_mtmc_config(MTMC_CONFIG_PATH)
        event_config = self.mtmc_config["events"]
        event_path = Path(
            os.environ.get("LAB_MTMC_EVENT_PATH", BASE_DIR / event_config["path"])
        ).expanduser().resolve()
        self.event_store = AssociationEventStore(
            event_path,
            memory_limit=event_config["memory_limit"],
            retention_days=event_config["retention_days"],
        )
        self.reid_embedder = ReIDEmbedder(REID_MODEL_PATH)
        self.handoff = IdentityHandoff(
            self.reid_embedder,
            ttl_seconds=IDENTITY_HANDOFF_TTL_SECONDS,
            similarity_threshold=self.mtmc_config["reid"]["high_similarity"],
            match_margin=self.mtmc_config["reid"]["match_margin"],
        )
        self.coordinator = CrossCameraTrackCoordinator(
            self.reid_embedder,
            self.floor_map,
            mtmc_config=self.mtmc_config,
            event_store=self.event_store,
            camera_roles={camera["id"]: camera["role"] for camera in cameras},
        )
        camera_ids = [camera["id"] for camera in cameras]
        self.recorder = DatasetRecorder(RECORDINGS_DIR, camera_ids)
        configured_hardware_port = os.environ.get("LAB_HARDWARE_PORT", "").strip()
        hardware_auto_detect = os.environ.get("LAB_HARDWARE_AUTO", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        hardware_port = configured_hardware_port or (
            discover_hardware_port() if hardware_auto_detect else None
        )
        self.environment = EnvironmentSerialMonitor(
            port=hardware_port,
            baudrate=int(os.environ.get("LAB_HARDWARE_BAUD", "115200")),
            stale_seconds=float(os.environ.get("LAB_HARDWARE_STALE_SECONDS", "30")),
        )
        self.workers = {
            camera["id"]: CameraWorker(
                camera,
                face_store,
                self.handoff,
                self.recorder,
                self.floor_map,
                self.coordinator,
            )
            for camera in cameras
        }
        replay_start_monotonic = time.monotonic() + 2.0
        replay_start_unix = time.time() + 2.0
        for worker in self.workers.values():
            worker.replay_start_monotonic = replay_start_monotonic
            worker.replay_start_unix = replay_start_unix
            worker.start()
        self.environment.start()

    def stop(self):
        self.environment.stop()
        for worker in self.workers.values():
            worker.stop()
        self.recorder.stop_if_active()

    def state(self):
        cameras = [worker.status_payload() for worker in self.workers.values()]
        observations = [
            observation
            for camera in cameras
            for observation in camera.get("map_observations", [])
        ]
        active_tracks_by_id = {}
        for camera in cameras:
            for track in camera.get("yolo_tracks", []):
                track_id = track.get("track_id")
                if not track_id:
                    continue
                active = active_tracks_by_id.setdefault(
                    str(track_id),
                    {"track_id": str(track_id), "cameras": [], "observed_at": 0.0},
                )
                active["cameras"].append(camera["id"])
                active["observed_at"] = max(
                    active["observed_at"],
                    float(camera.get("inference_ended_at") or camera.get("status_published_at") or time.time()),
                )
        return {
            "cameras": cameras,
            "environment": self.environment.status(),
            "people": list(face_store.people.values()),
            "recording": self.recorder.status(),
            "processing": {
                "inference_device": str(self.reid_embedder.device),
                "cpu_fallback_allowed": os.environ.get("LAB_ALLOW_CPU_FALLBACK") == "1",
                "inference_frame_stride": YOLO_FRAME_STRIDE,
                "map_publish_interval_ms": int(MAP_PUBLISH_INTERVAL_SECONDS * 1000),
                "frontend_poll_interval_ms": 200,
                "face_recognition_engine": (
                    self.face_store.modern.engine
                    if self.face_store.modern.available
                    else f"ArcFace unavailable: {self.face_store.modern.error}"
                ),
                "face_quality_gate": "brightness + contrast + sharpness + open-set margin",
                "face_feature_fusion_frames": 5,
                "arcface_gallery_people": len(self.face_store.modern_features),
                "arcface_gallery_required_features": ARCFACE_MIN_GALLERY_FEATURES,
                "arcface_gallery_feature_counts": dict(
                    self.face_store.modern_feature_counts
                ),
                "arcface_gallery_missing_ids": sorted(
                    set(self.face_store.people) - set(self.face_store.modern_features)
                ),
                "reid_engine": "ResNet50-IBN",
                "reid_feature_dimensions": 2048,
                "reid_similarity_threshold": self.mtmc_config["reid"]["high_similarity"],
                "reid_match_margin": self.mtmc_config["reid"]["match_margin"],
                "reid_feature_refresh_ms": int(self.mtmc_config["reid"]["feature_refresh_seconds"] * 1000),
                "identity_handoff_ttl_seconds": IDENTITY_HANDOFF_TTL_SECONDS,
                "identity_handoff_path": "entrance ArcFace -> directed Re-ID transition -> indoor tracks",
                "identity_lock_policy": "single active global track per registered person; conflicts stay anonymous",
                "late_identity_sync_policy": "directed entrance transition + <=3s + single target + Re-ID >=0.35; cam_1 distance gate deferred until map calibration",
                "slam_engine": "ArUco anchors + ORB monocular SLAM",
                "slam_marker_reacquire": "automatic",
                "track_hold_ms": int(YOLO_TRACK_HOLD_SECONDS * 1000),
                "algorithm_version": self.mtmc_config["algorithm_version"],
                "effective_mtmc_config": public_config(self.mtmc_config),
                "mtmc_diagnostics": self.coordinator.diagnostics(),
            },
            "floor_map": self.floor_map.state(
                observations,
                stream_skew_seconds=self.coordinator.stream_skew_seconds(),
                max_stream_skew_seconds=self.mtmc_config["calibration"]["max_stream_skew_seconds"],
                observation_max_age_seconds=self.mtmc_config["calibration"]["observation_max_age_seconds"],
                fusion_max_distance_m=self.mtmc_config["calibration"]["fusion_max_distance_m"],
                max_map_speed_mps=self.mtmc_config["calibration"]["max_map_speed_mps"],
                max_motion_gap_seconds=self.mtmc_config["calibration"]["max_motion_gap_seconds"],
                active_tracks=list(active_tracks_by_id.values()),
                map_position_hold_seconds=self.mtmc_config["calibration"]["map_position_hold_seconds"],
            ),
        }

    def start_recording(self, subject_id, notes=""):
        if REPLAY_REALTIME:
            raise RecordingError("录像回放模式不能再次开始数据录制")
        unavailable = [
            worker.camera["id"]
            for worker in self.workers.values()
            if worker.status_payload()["status"] != "running"
        ]
        if unavailable:
            raise RecordingError(f"摄像头未就绪：{', '.join(unavailable)}")
        return self.recorder.start(subject_id, notes)


class FaceRegistrationManager:
    def __init__(self, store, cameras):
        self.store = store
        self.cameras = cameras
        self.lock = threading.RLock()
        self.session = None
        self.camera_id = None
        self.pose_plan = []
        self.pose_index = 0
        self.pose_counts = {}
        self.last_sample_at = 0.0
        self.last_normalized = None
        self.last_result = None
        self.last_preview_at = 0.0
        self.last_preview_payload = None
        self.replace_existing = True

    @staticmethod
    def guidance(reason):
        return {
            "camera_not_ready": "摄像头未就绪，请等待画面恢复",
            "no_person": "画面中未检测到人，请站到摄像头正前方",
            "multiple_people": "画面中只能保留一名采集人员",
            "face_not_found": "未捕捉到正脸，请按当前姿态调整并露出完整面部",
            "face_too_small": "脸部过小，请靠近摄像头",
            "face_too_dark": "脸部过暗，请面向光源或增加正面照明",
            "face_too_bright": "脸部过亮，请避开强光或降低正面照明",
            "face_low_contrast": "脸部对比度不足，请调整位置并露出完整五官",
            "face_blurry": "人脸模糊，请保持头部稳定并等待画面清晰",
            "duplicate_frame": "与上一张过于相似，请轻微转动头部或改变表情",
            "capture_too_fast": "正在控制采样间隔，请保持当前姿态",
            "sample_limit_reached": "样本数量已达到设定值，可以完成注册",
        }.get(reason, "本帧未计入，请按引导重新采集")

    def _payload(self):
        if self.session is None:
            return {
                "active": False,
                "cameras": [worker.status_payload() for worker in self.cameras.values()],
            }
        pose = self.pose_plan[min(self.pose_index, len(self.pose_plan) - 1)]
        return {
            "active": True,
            **self.session.summary(),
            "camera_id": self.camera_id,
            "pose": pose,
            "pose_index": self.pose_index,
            "pose_total": len(self.pose_plan),
            "pose_count": self.pose_counts.get(pose["id"], 0),
            "pose_plan": [
                {**item, "count": self.pose_counts.get(item["id"], 0)}
                for item in self.pose_plan
            ],
            "last_result": self.last_result,
            "ready": self.session.samples >= self.session.target_samples,
            "cameras": [worker.status_payload() for worker in self.cameras.values()],
        }

    def status(self):
        with self.lock:
            return self._payload()

    def start(self, payload):
        person_id = str(payload.get("person_id", "")).strip()
        name = str(payload.get("name", "")).strip()
        camera_id = str(payload.get("camera_id", "")).strip()
        try:
            label = int(payload.get("label"))
            target_samples = int(payload.get("target_samples", 60))
        except (TypeError, ValueError) as error:
            raise ValueError("人员序号和样本数量必须是整数") from error
        if not REGISTRATION_ID_PATTERN.fullmatch(person_id):
            raise ValueError("人员ID仅允许字母、数字、下划线和短横线")
        if not name:
            raise ValueError("请输入人员姓名")
        if label <= 0:
            raise ValueError("人员序号必须大于 0")
        if camera_id not in self.cameras:
            raise ValueError("请选择可用摄像头")
        if not REGISTRATION_MIN_SAMPLES <= target_samples <= REGISTRATION_MAX_SAMPLES:
            raise ValueError("正式注册需要采集 50～80 张有效样本")
        for existing_id, person in self.store.people.items():
            if existing_id != person_id and int(person.get("label", 0)) == label:
                raise ValueError(f"人员序号 {label} 已被 {person.get('name', existing_id)} 使用")
        with self.lock:
            if self.session is not None:
                raise RuntimeError("已有正在进行的人脸采集会话")
            self.session = self.store.start_registration(
                person_id,
                name,
                label,
                target_samples=target_samples,
                min_samples=target_samples,
                max_samples=target_samples,
            )
            self.camera_id = camera_id
            self.pose_plan = registration_pose_plan(target_samples)
            self.pose_index = 0
            self.pose_counts = {item["id"]: 0 for item in self.pose_plan}
            self.last_sample_at = 0.0
            self.last_normalized = None
            self.last_result = None
            self.last_preview_at = 0.0
            self.last_preview_payload = None
            self.replace_existing = bool(payload.get("replace_existing", True))
            return self._payload()

    def capture(self):
        with self.lock:
            if self.session is None:
                raise RuntimeError("请先开始人脸采集")
            now = time.time()
            if now - self.last_sample_at < 0.35:
                reason = "capture_too_fast"
                self.last_result = {"accepted": False, "reason": reason, "message": self.guidance(reason)}
                return self._payload()
            worker = self.cameras[self.camera_id]
            captured, error = worker.registration_face()
            if error:
                self.last_result = {"accepted": False, "reason": error, "message": self.guidance(error)}
                return self._payload()
            normalized = self.store.normalize(captured["face"])
            if self.last_normalized is not None:
                difference = float(cv2.absdiff(normalized, self.last_normalized).mean())
                if difference < 1.4:
                    reason = "duplicate_frame"
                    self.last_result = {
                        "accepted": False,
                        "reason": reason,
                        "difference": round(difference, 2),
                        "message": self.guidance(reason),
                    }
                    self.last_sample_at = now
                    return self._payload()
            pose = self.pose_plan[self.pose_index]
            result = self.session.add_sample(
                captured["face"],
                metadata={"pose": pose["id"], "face_box": captured["face_box"]},
                embedding=captured.get("embedding"),
            )
            self.last_sample_at = now
            if not result["accepted"]:
                result["message"] = self.guidance(result["reason"])
                self.last_result = result
                return self._payload()
            self.last_normalized = normalized
            self.pose_counts[pose["id"]] += 1
            if self.pose_counts[pose["id"]] >= pose["target"] and self.pose_index < len(self.pose_plan) - 1:
                self.pose_index += 1
            result["message"] = "样本有效，已计入当前姿态"
            self.last_result = result
            return self._payload()

    def preview(self, camera_id=None):
        """Inspect the latest frame without writing or advancing registration."""
        with self.lock:
            selected_camera = self.camera_id if self.session is not None else camera_id
            now = time.time()
            # The browser polls frequently for responsive guidance, but face
            # cascades are expensive. Reuse the last inspection briefly instead
            # of running a new detector for every poll.
            if (
                self.last_preview_payload is not None
                and now - self.last_preview_at < 2.0
                and (self.session is not None or not self.last_preview_payload.get("active"))
            ):
                cached = dict(self.last_preview_payload)
                cached["sample_count"] = self.session.samples if self.session is not None else 0
                return cached
            if not selected_camera:
                result = {"active": False, "ready": False, "reason": "session_not_started", "message": "请选择入口摄像头并填写资料后开始采集"}
                self.last_preview_payload = result
                self.last_preview_at = now
                return result
            if selected_camera in self.cameras:
                worker = self.cameras[selected_camera]
            else:
                result = {"active": False, "ready": False, "reason": "camera_not_found", "message": "采集摄像头不可用"}
                self.last_preview_payload = result
                self.last_preview_at = now
                return result
            captured, error = worker.registration_face()
            if error:
                result = {
                    "active": self.session is not None,
                    "ready": False,
                    "reason": error,
                    "message": self.guidance(error),
                    "sample_count": self.session.samples if self.session is not None else 0,
                }
                self.last_preview_payload = result
                self.last_preview_at = now
                return result
            quality = self.store.capture_policy.assess(captured["face"])
            if not quality["accepted"]:
                result = {
                    "active": self.session is not None,
                    "ready": False,
                    "reason": quality["reason"],
                    "message": self.guidance(quality["reason"]),
                    "quality": quality,
                    "face_box": captured["face_box"],
                    "sample_count": self.session.samples if self.session is not None else 0,
                }
                self.last_preview_payload = result
                self.last_preview_at = now
                return result
            result = {
                "active": self.session is not None,
                "ready": True,
                "reason": None,
                "message": "人脸质量合格，可以采集",
                "quality": quality,
                "face_box": captured["face_box"],
                "sample_count": self.session.samples if self.session is not None else 0,
            }
            self.last_preview_payload = result
            self.last_preview_at = now
            return result

    def finalize(self):
        with self.lock:
            if self.session is None:
                raise RuntimeError("没有正在进行的人脸采集")
            if self.session.samples < self.session.target_samples:
                raise ValueError(f"还需采集 {self.session.target_samples - self.session.samples} 张有效样本")
            result = self.session.finalize(replace_existing=self.replace_existing)
            self.session = None
            self.last_result = None
            self.last_preview_payload = None
            return result

    def cancel(self):
        with self.lock:
            if self.session is not None:
                shutil.rmtree(self.session.directory, ignore_errors=True)
            self.session = None
            self.last_result = None
            self.last_preview_payload = None
            return {"ok": True}


def mjpeg_stream(worker, remote=False):
    sequence = -1
    while worker.running:
        result = worker.wait_for_jpeg(remote, sequence)
        if result is None or result[0] <= sequence:
            continue
        sequence, encoded = result
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded + b"\r\n"


def _validate_annotation_point(point, name, *, normalized=False):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    try:
        values = [float(point[0]), float(point[1])]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains invalid coordinates") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains invalid coordinates")
    if normalized and not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{name} must be normalized to 0..1")
    return values


def _validate_annotation_payload(payload, config):
    camera_id = payload.get("camera_id")
    cameras = config.get("cameras", {})
    if camera_id not in cameras:
        raise ValueError("unknown camera_id")

    regions = payload.get("regions") or {}
    if not isinstance(regions, dict):
        raise ValueError("regions must be an object")
    allowed_types = {"main_aisle", "secondary_aisle", "rear_service", "overlap"}
    clean_regions = {}
    for region_type, region in regions.items():
        if region_type not in allowed_types:
            raise ValueError(f"unsupported region type: {region_type}")
        if not isinstance(region, dict):
            raise ValueError(f"{region_type} must be an object")
        if "points" in region:
            points = region.get("points")
            if not isinstance(points, list) or len(points) != 4:
                raise ValueError(f"{region_type}.points must contain four points")
            clean_points = [
                _validate_annotation_point(
                    point, f"{region_type}.points", normalized=True
                )
                for point in points
            ]
            polygon = np.asarray(clean_points, dtype=np.float32)
            if not cv2.isContourConvex(polygon):
                raise ValueError(f"{region_type}.points must form a convex quadrilateral")
            area = abs(float(cv2.contourArea(polygon)))
            if region_type == "main_aisle":
                # Opposite diagonals of a parallelogram share one midpoint.
                diagonal_error = np.linalg.norm(
                    polygon[0] + polygon[2] - polygon[1] - polygon[3]
                )
                if diagonal_error > 0.002:
                    raise ValueError("main_aisle.points must form a parallelogram")
            if area < 0.0004:
                raise ValueError(f"{region_type} quadrilateral is too small")
            clean_regions[region_type] = {"points": clean_points}
            continue
        values = {}
        for key in ("x", "y", "width", "height"):
            try:
                values[key] = float(region[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{region_type}.{key} is invalid") from error
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"{region_type} contains invalid coordinates")
        if not (0.0 <= values["x"] < 1.0 and 0.0 <= values["y"] < 1.0):
            raise ValueError(f"{region_type} origin must be normalized to 0..1")
        if values["width"] <= 0.0 or values["height"] <= 0.0:
            raise ValueError(f"{region_type} size must be positive")
        if values["x"] + values["width"] > 1.0 or values["y"] + values["height"] > 1.0:
            raise ValueError(f"{region_type} must stay inside the image")
        clean_regions[region_type] = values

    raw_calibration_changed = payload.get("calibration_points_changed")
    calibration_points_changed = (
        bool(raw_calibration_changed)
        if raw_calibration_changed is not None
        else bool(payload.get("image_points") or payload.get("map_points"))
    )
    if calibration_points_changed:
        image_points = payload.get("image_points") or []
        map_points = payload.get("map_points") or []
    else:
        image_points = cameras[camera_id].get("image_points") or []
        map_points = cameras[camera_id].get("map_points") or []
    if image_points or map_points:
        if len(image_points) != 4 or len(map_points) != 4:
            raise ValueError("image_points and map_points must both contain four points")
        clean_image_points = [
            _validate_annotation_point(point, "image_points", normalized=True)
            for point in image_points
        ]
        clean_map_points = [
            _validate_annotation_point(point, "map_points") for point in map_points
        ]
        width_m = float(config.get("width_m", 0.0))
        height_m = float(config.get("height_m", 0.0))
        if any(
            point[0] < 0.0
            or point[1] < 0.0
            or point[0] > width_m
            or point[1] > height_m
            for point in clean_map_points
        ):
            raise ValueError("map_points exceed the configured map size")
    else:
        clean_image_points = []
        clean_map_points = []

    return {
        "camera_id": camera_id,
        "regions": clean_regions,
        "image_points": clean_image_points,
        "map_points": clean_map_points,
        "calibration_points_changed": calibration_points_changed,
        "activate_calibration": bool(payload.get("activate_calibration", False)),
    }


def _save_floor_map_config(config):
    FLOOR_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=FLOOR_MAP_PATH.parent,
            prefix=f"{FLOOR_MAP_PATH.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(config, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary_name = temporary.name
        os.replace(temporary_name, FLOOR_MAP_PATH)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _floor_map_revision(content):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


face_store = FaceIdentityStore(
    PEOPLE_PATH,
    KNOWN_FACES_DIR,
    recognition_model_path=FACE_RECOGNITION_MODEL_PATH,
    modern_model_root=MODERN_FACE_MODEL_ROOT,
    required_engine=FACE_RECOGNITION_ENGINE,
)
camera_manager = CameraManager(load_cameras(), face_store)
face_registration = FaceRegistrationManager(face_store, camera_manager.workers)
atexit.register(camera_manager.stop)


@app.route("/")
def index():
    if os.environ.get("LAB_API_ONLY") == "1":
        return jsonify({"service": "lab-api", "frontend": "http://127.0.0.1:5173/"})
    return render_template("index.html")


@app.route("/faces")
def faces():
    if os.environ.get("LAB_API_ONLY") == "1":
        return jsonify({"service": "lab-api", "frontend": "http://127.0.0.1:5173/faces"})
    return render_template("faces.html")


@app.route("/annotate")
def annotate():
    if os.environ.get("LAB_API_ONLY") == "1":
        return jsonify({"service": "lab-api", "frontend": "http://127.0.0.1:5173/annotate"})
    return render_template("annotate.html")


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "lab-api"})


@app.route("/video/<camera_id>")
def video(camera_id):
    worker = camera_manager.workers.get(camera_id)
    if worker is None:
        return "camera not found", 404
    remote = request.args.get("remote") == "1"
    response = Response(
        mjpeg_stream(worker, remote=remote),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/state")
def api_state():
    return jsonify(camera_manager.state())


@app.route("/api/environment/ingest", methods=["POST"])
def api_environment_ingest():
    try:
        return jsonify(camera_manager.environment.ingest(request.get_json(silent=True) or {}))
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.route("/api/mtmc/events")
def api_mtmc_events():
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be an integer"}), 400
    return jsonify(
        {
            "events": camera_manager.event_store.recent(limit),
            "summary": camera_manager.event_store.summary(),
            "algorithm_version": camera_manager.mtmc_config["algorithm_version"],
        }
    )


@app.route("/api/annotation/config")
def api_annotation_config():
    with ANNOTATION_LOCK:
        content = FLOOR_MAP_PATH.read_text(encoding="utf-8")
        config = json.loads(content)
    return jsonify(
        {
            "name": config.get("name", "实验室平面图"),
            "width_m": float(config.get("width_m", 0.0)),
            "height_m": float(config.get("height_m", 0.0)),
            "projection_domain_margin_normalized": float(
                config.get("projection_domain_margin_normalized", 0.03)
            ),
            "calibrated": bool(config.get("calibrated", False)),
            "calibration": config.get("calibration", {}),
            "zones": config.get("zones", []),
            "fixtures": config.get("fixtures", []),
            "cameras": config.get("cameras", {}),
            "annotations": config.get("annotations", {}),
            "revision": _floor_map_revision(content),
            "persistent": bool(os.environ.get("LAB_FLOOR_MAP_PATH")),
        }
    )


@app.route("/api/annotation/save", methods=["POST"])
def api_annotation_save():
    payload = request.get_json(silent=True) or {}
    try:
        with ANNOTATION_LOCK:
            content = FLOOR_MAP_PATH.read_text(encoding="utf-8")
            config = json.loads(content)
            current_revision = _floor_map_revision(content)
            base_revision = payload.get("base_revision")
            if base_revision and base_revision != current_revision:
                return jsonify(
                    {
                        "ok": False,
                        "error": "地图配置已被其他页面或进程更新，请刷新后再保存",
                        "revision": current_revision,
                    }
                ), 409
            annotation = _validate_annotation_payload(payload, config)
            camera_id = annotation["camera_id"]
            config.setdefault("annotations", {})[camera_id] = {
                "regions": annotation["regions"],
                "image_points": annotation["image_points"],
                "map_points": annotation["map_points"],
            }
            if annotation["calibration_points_changed"]:
                config["cameras"][camera_id]["image_points"] = annotation["image_points"]
                config["cameras"][camera_id]["map_points"] = annotation["map_points"]
            if annotation["activate_calibration"]:
                all_calibrated = all(
                    len(camera.get("image_points", [])) == 4
                    and len(camera.get("map_points", [])) == 4
                    for camera in config.get("cameras", {}).values()
                )
                if not all_calibrated:
                    raise ValueError("请先为每路摄像头完成四点标定")
                config["calibrated"] = True
                config.setdefault("calibration", {})["status"] = "formal"
                config["calibration"]["geometry_fusion_allowed"] = True
                config["calibration"]["reason"] = "all camera control points activated by operator"
                for camera in config.get("cameras", {}).values():
                    camera["localization_mode"] = "formal_homography"
            _save_floor_map_config(config)
            saved_revision = _floor_map_revision(
                FLOOR_MAP_PATH.read_text(encoding="utf-8")
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    return jsonify(
        {
            "ok": True,
            "camera_id": camera_id,
            "requires_restart": True,
            "calibrated": bool(config.get("calibrated", False)),
            "calibration_status": config.get("calibration", {}).get("status", "uncalibrated"),
            "message": "标注已保存；重启后端后新的四点映射才会应用",
            "revision": saved_revision,
        }
    )


@app.route("/api/people")
def api_people():
    return jsonify(list(face_store.people.values()))


@app.route("/api/face-registration", methods=["GET"])
def api_face_registration():
    return jsonify(face_registration.status())


@app.route("/api/face-registration/start", methods=["POST"])
def api_face_registration_start():
    try:
        return jsonify(face_registration.start(request.get_json(silent=True) or {})), 201
    except (ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 409


@app.route("/api/face-registration/capture", methods=["POST"])
def api_face_registration_capture():
    try:
        return jsonify(face_registration.capture())
    except (ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 409


@app.route("/api/face-registration/preview", methods=["GET"])
def api_face_registration_preview():
    return jsonify(face_registration.preview(request.args.get("camera_id")))


@app.route("/api/face-registration/finalize", methods=["POST"])
def api_face_registration_finalize():
    try:
        return jsonify(face_registration.finalize())
    except (ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 409


@app.route("/api/face-registration/cancel", methods=["POST"])
def api_face_registration_cancel():
    return jsonify(face_registration.cancel())


@app.route("/api/recording", methods=["GET"])
def api_recording():
    return jsonify(camera_manager.recorder.status())


@app.route("/api/recording/start", methods=["POST"])
def api_recording_start():
    payload = request.get_json(silent=True) or {}
    try:
        recording = camera_manager.start_recording(
            payload.get("subject_id"), payload.get("notes", "")
        )
        return jsonify(recording), 201
    except RecordingError as error:
        return jsonify({"error": str(error)}), 409


@app.route("/api/recording/stop", methods=["POST"])
def api_recording_stop():
    try:
        return jsonify(camera_manager.recorder.stop("operator"))
    except RecordingError as error:
        return jsonify({"error": str(error)}), 409


@app.route("/api/recordings", methods=["GET"])
def api_recordings():
    sessions = []
    for info_path in RECORDINGS_DIR.glob("*/*/info.json"):
        try:
            session = recording_session_payload(info_path.parent)
            if session.get("status") == "complete":
                sessions.append(session)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    sessions.sort(key=lambda item: item.get("started_at") or "", reverse=True)
    return jsonify({"sessions": sessions[:100]})


@app.route(
    "/api/recordings/<subject_id>/<session_id>/video/<camera_id>",
    methods=["GET"],
)
def api_recording_video(subject_id, session_id, camera_id):
    try:
        directory = recording_session_directory(subject_id, session_id)
        info = recording_session_payload(directory)
        stream = info.get("streams", {}).get(camera_id)
        if not stream or not stream.get("video_file"):
            raise FileNotFoundError("recorded camera video not found")
        video_path = (directory / stream["video_file"]).resolve()
        if directory not in video_path.parents or not video_path.is_file():
            raise FileNotFoundError("recorded camera video not found")
        return send_file(video_path, conditional=True)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 404


@app.route(
    "/api/recordings/<subject_id>/<session_id>/frame/<camera_id>",
    methods=["GET"],
)
def api_recording_frame(subject_id, session_id, camera_id):
    capture = None
    try:
        directory = recording_session_directory(subject_id, session_id)
        info = recording_session_payload(directory)
        stream = info.get("streams", {}).get(camera_id)
        if not stream or not stream.get("video_file"):
            raise FileNotFoundError("recorded camera video not found")
        frame_index = int(request.args.get("index", 0))
        frame_count = int(stream.get("frames") or 0)
        if frame_index < 0 or frame_index >= frame_count:
            raise ValueError(f"frame index must be between 0 and {max(0, frame_count - 1)}")
        video_path = (directory / stream["video_file"]).resolve()
        if directory not in video_path.parents or not video_path.is_file():
            raise FileNotFoundError("recorded camera video not found")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("recorded camera video cannot be decoded")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"recorded frame cannot be decoded: {frame_index}")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("recorded frame cannot be encoded")
        response = Response(encoded.tobytes(), mimetype="image/jpeg")
        response.headers["Cache-Control"] = "private, max-age=3600"
        response.headers["X-Frame-Index"] = str(frame_index)
        return response
    except (FileNotFoundError, ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 400
    finally:
        if capture is not None:
            capture.release()


def manual_handoff_payload(directory):
    path = directory / "manual_handoffs.json"
    if not path.exists():
        annotations = []
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotations = payload.get("annotations", [])
    by_pair = defaultdict(list)
    for item in annotations:
        by_pair[f"{item['from_camera']}->{item['to_camera']}"].append(
            float(item["gap_seconds"])
        )
    suggestions = {}
    for pair, values in sorted(by_pair.items()):
        suggestions[pair] = {
            "samples": len(values),
            "median_gap_seconds": round(float(np.median(values)), 3),
            "p90_gap_seconds": round(float(np.percentile(values, 90)), 3),
            "automatic_config_update": False,
        }
    return {
        "format_version": "manual-mtmc-handoffs-v1",
        "session_id": directory.name,
        "annotations": annotations,
        "suggestions": suggestions,
    }


def save_manual_handoffs(directory, document, annotations):
    path = directory / "manual_handoffs.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "format_version": document["format_version"],
                "session_id": document["session_id"],
                "annotations": annotations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


@app.route(
    "/api/recordings/<subject_id>/<session_id>/handoffs",
    methods=["GET", "POST"],
)
def api_recording_handoffs(subject_id, session_id):
    try:
        directory = recording_session_directory(subject_id, session_id)
        if request.method == "GET":
            return jsonify(manual_handoff_payload(directory))
        payload = request.get_json(silent=True) or {}
        person_id = str(payload.get("person_id") or "").strip()
        source_camera = str(payload.get("from_camera") or "").strip()
        target_camera = str(payload.get("to_camera") or "").strip()
        if not RECORDING_PATH_PATTERN.fullmatch(person_id):
            raise ValueError("person_id must use letters, numbers, underscore or dash")
        info = recording_session_payload(directory)
        cameras = set(info.get("streams", {}))
        if source_camera == target_camera or {source_camera, target_camera} - cameras:
            raise ValueError("select two different cameras recorded in this session")
        source = recording_frame_at(
            directory, source_camera, payload.get("source_end_seconds", 0.0)
        )
        target = recording_frame_at(
            directory, target_camera, payload.get("target_start_seconds", 0.0)
        )
        with ANNOTATION_LOCK:
            document = manual_handoff_payload(directory)
            sequence = max(
                [int(item.get("id", 0)) for item in document["annotations"]] + [0]
            ) + 1
            annotation = {
                "id": sequence,
                "person_id": person_id,
                "from_camera": source_camera,
                "to_camera": target_camera,
                "source_end": source,
                "target_start": target,
                "gap_seconds": round(target["unix_time"] - source["unix_time"], 3),
                "notes": str(payload.get("notes") or "").strip()[:300],
                "created_at": time.time(),
            }
            document["annotations"].append(annotation)
            save_manual_handoffs(directory, document, document["annotations"])
        return jsonify(manual_handoff_payload(directory)), 201
    except (FileNotFoundError, ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 400


@app.route(
    "/api/recordings/<subject_id>/<session_id>/handoffs/<int:annotation_id>",
    methods=["DELETE"],
)
def api_recording_handoff_delete(subject_id, session_id, annotation_id):
    try:
        directory = recording_session_directory(subject_id, session_id)
        with ANNOTATION_LOCK:
            document = manual_handoff_payload(directory)
            kept = [
                item
                for item in document["annotations"]
                if int(item.get("id", 0)) != annotation_id
            ]
            if len(kept) == len(document["annotations"]):
                raise FileNotFoundError("handoff annotation not found")
            save_manual_handoffs(directory, document, kept)
        return jsonify(manual_handoff_payload(directory))
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 404


if __name__ == "__main__":
    app.run(
        host=os.environ.get("LAB_APP_HOST", "0.0.0.0"),
        port=int(os.environ.get("LAB_APP_PORT", "5000")),
        threaded=True,
    )
