"""HTTP inference service for the instrument-interaction module.

Run this file on the GPU server. The camera host sends JPEG frames and person
tracks; this process owns the plug-in, its model, and its persistent streams.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request


logging.basicConfig(level=os.environ.get("LAB_LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("interaction-server")


class InteractionInferenceService:
    def __init__(self, plugin_root: Path) -> None:
        module_dir = plugin_root / "模块"
        if not module_dir.is_dir():
            raise RuntimeError(f"仪器交互插件目录不存在：{module_dir}")
        sys.path.insert(0, str(module_dir.resolve()))
        from lab_instrument_interaction import InstrumentInteractionModule

        self.module = InstrumentInteractionModule(preload_models=True)
        health = self.module.health_check(require_loaded=True)
        if not health.get("ok"):
            self.module.close()
            raise RuntimeError(f"仪器交互模块健康检查失败：{health}")
        self.plugin_info = {**self.module.plugin_info(), "health": health}
        self.lock = threading.RLock()
        self.streams: dict[str, dict[str, Any]] = {}

    def process(self, payload: dict[str, Any], frame: np.ndarray) -> dict[str, Any]:
        source_id = str(payload["camera_id"])
        plugin_camera_id = str(payload["interaction_camera_id"])
        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        with self.lock:
            session = self.streams.get(source_id)
            if session is None or session["frame_size"] != frame_size:
                if session is not None:
                    session["stream"].close()
                stream = self.module.create_stream(
                    camera_id=plugin_camera_id,
                    frame_size=frame_size,
                )
                session = {"stream": stream, "frame_size": frame_size, "plugin_camera_id": plugin_camera_id}
                self.streams[source_id] = session
            stream = session["stream"]
            return stream.process_frame(
                frame,
                timestamp=float(payload["timestamp"]),
                persons=payload["persons"],
                after_event_sequence=int(payload.get("after_event_sequence", 0)),
            )

    def close(self) -> None:
        with self.lock:
            sessions = list(self.streams.values())
            self.streams.clear()
            module = self.module
            self.module = None
        for session in sessions:
            try:
                session["stream"].close()
            except Exception:
                LOGGER.exception("关闭交互识别流失败")
        if module is not None:
            module.close()


def _json_field(name: str, default: Any = None) -> Any:
    raw = request.form.get(name)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"字段 {name} 不是有效 JSON") from exc


def create_app(service: InteractionInferenceService) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("LAB_INTERACTION_MAX_FRAME_BYTES", str(8 * 1024 * 1024)))
    expected_token = os.environ.get("LAB_INTERACTION_TOKEN", "")

    @app.before_request
    def authenticate() -> None:
        if expected_token and request.headers.get("X-Lab-Interaction-Token") != expected_token:
            return jsonify({"error": "unauthorized"}), 401
        return None

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "instrument-interaction", "plugin": service.plugin_info})

    @app.get("/api/interaction/status")
    def status():
        return jsonify({
            "status": "ok",
            "service": "instrument-interaction",
            "plugin": service.plugin_info,
            "streams": list(service.streams),
        })

    @app.post("/api/interaction/frame")
    def frame():
        required = ["camera_id", "interaction_camera_id", "timestamp", "persons"]
        missing = [name for name in required if name not in request.form or not request.form[name]]
        if missing or "frame" not in request.files:
            return jsonify({"error": f"missing fields: {missing or ['frame']}"}), 400
        try:
            persons = _json_field("persons")
            if not isinstance(persons, list):
                raise ValueError("persons 必须是列表")
            raw = request.files["frame"].read()
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("无法解码 JPEG 帧")
            payload = {
                "camera_id": request.form["camera_id"],
                "interaction_camera_id": request.form["interaction_camera_id"],
                "timestamp": float(request.form["timestamp"]),
                "persons": persons,
                "after_event_sequence": int(request.form.get("after_event_sequence", 0)),
            }
            response = service.process(payload, decoded)
            return jsonify(response)
        except (ValueError, TypeError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            LOGGER.exception("远程交互识别请求失败")
            return jsonify({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), 500

    return app


def main() -> None:
    plugin_root = Path(os.environ.get("LAB_INTERACTION_PLUGIN_DIR", Path(__file__).parent / "plugins" / "instrument_interaction"))
    service = InteractionInferenceService(plugin_root)
    atexit.register(service.close)
    app = create_app(service)
    host = os.environ.get("LAB_INTERACTION_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("LAB_INTERACTION_SERVER_PORT", "6000"))
    LOGGER.info("Instrument interaction inference server listening on %s:%s", host, port)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
