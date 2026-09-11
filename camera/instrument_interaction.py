"""Remote client for the instrument-interaction inference service."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import cv2

LOGGER = logging.getLogger(__name__)


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


class InstrumentInteractionManager:
    """Push camera frames to one remote inference service per process."""

    def __init__(self, base_dir: Path, data_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.data_dir = Path(data_dir)
        self.server_url = os.environ.get("LAB_INTERACTION_SERVER_URL", "").strip().rstrip("/")
        requested = _truthy("LAB_INTERACTION_ENABLED")
        self.enabled = requested and bool(self.server_url)
        self.request_timeout = max(1.0, float(os.environ.get("LAB_INTERACTION_TIMEOUT_SECONDS", "5")))
        self.auth_token = os.environ.get("LAB_INTERACTION_TOKEN", "")
        self.lock = threading.RLock()
        self.sessions: dict[str, dict[str, Any]] = {}
        self.camera_errors: dict[str, str] = {}
        self.startup_error = None if self.enabled else (
            "未配置 LAB_INTERACTION_SERVER_URL" if requested else None
        )
        self.event_log_path = self.data_dir / "interaction_events.jsonl"
        self.cursor_path = self.data_dir / "interaction_cursors.json"

    @staticmethod
    def _default_state(camera: dict[str, Any]) -> dict[str, Any]:
        configured_camera_id = camera.get("interaction_camera_id")
        return {
            "configured": bool(configured_camera_id),
            "camera_id": configured_camera_id,
            "status": "disabled",
            "current_interactions": [],
            "error": None,
            "event_gap_detected": False,
            "analysis_lag_seconds": None,
            "metrics": {},
            "last_event_sequence": 0,
            "events_saved": 0,
            "transport": "remote_http",
        }

    def _session_for(self, camera: dict[str, Any], frame: Any) -> dict[str, Any]:
        camera_id = camera["id"]
        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        with self.lock:
            session = self.sessions.get(camera_id)
            if session is not None and session["frame_size"] == frame_size:
                return session
            if session is not None:
                session["stop"].set()
            stop = threading.Event()
            session = {
                "frame_size": frame_size,
                "queue": queue.Queue(maxsize=1),
                "stop": stop,
                "thread": None,
                "started_at": time.monotonic(),
                "last_timestamp": -1.0,
                "event_cursor": 0,
                "dropped_frames": 0,
                "state": {**self._default_state(camera), "status": "warming_up"},
            }
            thread = threading.Thread(
                target=self._send_loop,
                args=(camera, session),
                name=f"interaction-push-{camera_id}",
                daemon=True,
            )
            session["thread"] = thread
            self.sessions[camera_id] = session
            thread.start()
            return session

    def submit(self, camera: dict[str, Any], frame: Any, persons: list[dict[str, Any]]) -> None:
        """Encode and enqueue the latest frame without blocking the RTSP worker."""
        if not camera.get("interaction_camera_id") or not self.enabled:
            return
        try:
            success, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if not success:
                raise RuntimeError("无法编码交互识别视频帧")
            session = self._session_for(camera, frame)
            timestamp = max(
                time.monotonic() - session["started_at"],
                session["last_timestamp"] + 0.000001,
            )
            session["last_timestamp"] = timestamp
            item = {
                "jpeg": encoded.tobytes(),
                "timestamp": timestamp,
                "persons": persons,
                "frame_size": session["frame_size"],
            }
            try:
                session["queue"].put_nowait(item)
            except queue.Full:
                session["queue"].get_nowait()
                session["dropped_frames"] += 1
                session["queue"].put_nowait(item)
        except Exception as exc:
            self._record_error(camera, f"{type(exc).__name__}: {exc}")

    def _send_loop(self, camera: dict[str, Any], session: dict[str, Any]) -> None:
        while not session["stop"].is_set():
            try:
                item = session["queue"].get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                response = self._post_frame(camera, session, item)
                self._consume_response(camera, session, response)
            except Exception as exc:
                self._record_error(camera, f"{type(exc).__name__}: {exc}")
                time.sleep(0.5)

    def _post_frame(self, camera: dict[str, Any], session: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        boundary = f"----LabInteraction{uuid.uuid4().hex}"
        fields = {
            "camera_id": camera["id"],
            "interaction_camera_id": camera["interaction_camera_id"],
            "timestamp": str(item["timestamp"]),
            "persons": json.dumps(item["persons"], ensure_ascii=False),
            "after_event_sequence": str(session["event_cursor"]),
            "frame_width": str(item["frame_size"][0]),
            "frame_height": str(item["frame_size"][1]),
        }
        body = bytearray()
        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(value.encode("utf-8"))
            body.extend(b"\r\n")
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="frame"; filename="frame.jpg"\r\n')
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(item["jpeg"])
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.auth_token:
            headers["X-Lab-Interaction-Token"] = self.auth_token
        request = urllib.request.Request(
            f"{self.server_url}/api/interaction/frame",
            data=bytes(body),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as result:
            payload = json.loads(result.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("交互识别服务器返回了无效 JSON")
        return payload

    def _consume_response(self, camera: dict[str, Any], session: dict[str, Any], response: dict[str, Any]) -> None:
        state = {
            **self._default_state(camera),
            "status": response.get("status", "error"),
            "current_interactions": list(response.get("current_interactions") or []),
            "error": response.get("error"),
            "event_gap_detected": bool(response.get("event_gap_detected")),
            "analysis_lag_seconds": response.get("analysis_lag_seconds"),
            "metrics": {
                **dict(response.get("metrics") or {}),
                "local_dropped_frames": session["dropped_frames"],
            },
            "last_event_sequence": int(response.get("last_event_sequence", 0)),
            "events_saved": session["state"].get("events_saved", 0),
            "session_id": response.get("session_id"),
            "accepted_frame": response.get("accepted_frame"),
        }
        events = list(response.get("events") or [])
        if events:
            self._append_events(camera, response.get("session_id"), events)
            state["events_saved"] += len(events)
        previous_cursor = session["event_cursor"]
        session["event_cursor"] = state["last_event_sequence"]
        session["state"] = state
        if events or session["event_cursor"] != previous_cursor:
            self._persist_cursors()
        if state["error"]:
            LOGGER.error("Remote instrument interaction error for %s: %s", camera["id"], state["error"])
        if state["event_gap_detected"]:
            LOGGER.warning("Remote instrument interaction event gap for %s", camera["id"])

    def _append_events(self, camera: dict[str, Any], session_id: Any, events: list[dict[str, Any]]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.lock, self.event_log_path.open("a", encoding="utf-8") as output:
            for event in events:
                output.write(json.dumps({
                    "recorded_at": time.time(),
                    "camera_source_id": camera["id"],
                    "interaction_camera_id": camera["interaction_camera_id"],
                    "session_id": session_id,
                    "event": event,
                }, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def _persist_cursors(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            camera_id: {
                "event_cursor": session["event_cursor"],
                "last_event_sequence": session["state"].get("last_event_sequence", 0),
                "status": session["state"].get("status"),
                "updated_at": time.time(),
            }
            for camera_id, session in self.sessions.items()
        }
        temporary_path = self.cursor_path.with_suffix(".tmp")
        with self.lock:
            temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary_path, self.cursor_path)

    def _record_error(self, camera: dict[str, Any], error: str) -> None:
        with self.lock:
            session = self.sessions.get(camera.get("id"))
            if session is None:
                self.camera_errors[camera.get("id", "unknown")] = error
                return
            state = dict(session["state"])
            state.update({"status": "error", "error": error})
            session["state"] = state
        LOGGER.error("Remote instrument interaction failed for %s: %s", camera.get("id"), error)

    def state_for(self, camera: dict[str, Any]) -> dict[str, Any]:
        default = self._default_state(camera)
        if not self.enabled:
            if camera.get("interaction_camera_id") and self.startup_error:
                default.update({"status": "error", "error": self.startup_error})
            return default
        with self.lock:
            session = self.sessions.get(camera.get("id"))
            if session is None:
                error = self.camera_errors.get(camera.get("id"))
                return {**default, "status": "error" if error else "warming_up", "error": error}
            return dict(session["state"])

    def status(self) -> dict[str, Any]:
        with self.lock:
            connected = any(item["state"].get("status") not in {"error", "disabled"} for item in self.sessions.values())
        return {
            "enabled": self.enabled,
            "mode": "remote_http",
            "server_url": self.server_url or None,
            "connected": connected,
            "startup_error": self.startup_error,
            "event_log": str(self.event_log_path),
            "cursor_store": str(self.cursor_path),
        }

    def stop(self) -> None:
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            session["stop"].set()
        for session in sessions:
            thread = session.get("thread")
            if thread is not None:
                thread.join(timeout=2.0)
