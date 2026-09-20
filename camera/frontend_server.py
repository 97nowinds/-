import argparse
import base64
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from hardware_serial import EnvironmentSerialMonitor, discover_hardware_port
from member_integration import (
    MemberIntegrationConfigError,
    MemberIntegrationTransportError,
    MemberServiceClient,
    build_connection_test_event,
    build_monitoring_snapshot_frame_event,
    build_synthetic_person_handoff_event,
    build_synthetic_snapshot_frame_event,
    build_synthetic_snapshot_ready_event,
    extract_receipt_id,
    extract_results,
)


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "index.html"
ANNOTATE_TEMPLATE = ROOT / "templates" / "annotate.html"
FACES_TEMPLATE = ROOT / "templates" / "faces.html"
RECORDINGS_TEMPLATE = ROOT / "templates" / "recordings.html"
STATIC = ROOT / "static"

# Contains no camera content or people. This lets the handoff consumer verify
# its allowlist, MIME checks and download path before real frames are shared.
HANDOFF_TEST_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjL/wAARCAAkAEADASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEA"
    "EAAAAAAAAAAAAAAAAAAAAA/8QAFgEBAQEAAAAAAAAAAAAAAAAAAAID/8QAFBEBAAAAAAAA"
    "AAAAAAAAAAAAAP/aAAwDAQACEQMRAD8Ai4DRAAAAAAAAAAAAAAAAAAAAAAD/2Q=="
)


def _read_live_jpeg(api_base, camera_id, timeout=5.0, max_bytes=8 * 1024 * 1024):
    url = f"{api_base}/video/{camera_id}?remote=1"
    data = bytearray()
    with urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"camera stream returned HTTP {response.status}")
        while len(data) < max_bytes:
            chunk = response.read(min(4096, max_bytes - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            start = data.find(b"\xff\xd8")
            if start >= 0:
                end = data.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    return bytes(data[start : end + 2])
    raise RuntimeError("camera stream did not produce a complete JPEG")


def _jpeg_dimensions(data):
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("invalid JPEG start marker")
    offset = 2
    while offset + 9 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width > 0 and height > 0:
                return width, height
        offset += length
    raise ValueError("JPEG dimensions are unavailable")


class LiveSnapshotPublisher:
    def __init__(
        self,
        server,
        camera_id="cam_2",
        interval_seconds=1.0,
        max_frames=None,
    ):
        self.server = server
        self.camera_id = camera_id
        self.interval_seconds = max(0.25, float(interval_seconds))
        self.max_frames = int(max_frames) if max_frames else None
        self.stop_event = threading.Event()
        self.thread = None
        self._lock = threading.Lock()
        self._status = {
            "enabled": True,
            "camera_id": camera_id,
            "interval_seconds": self.interval_seconds,
            "state": "starting",
            "sent_frames": 0,
            "last_error": None,
            "last_receipt_id": None,
            "target_frames": self.max_frames,
            "frames": [],
        }

    @property
    def public_status(self):
        with self._lock:
            return dict(self._status)

    def _update(self, **values):
        with self._lock:
            self._status.update(values)

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="member-live-snapshot-publisher",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def _state(self):
        with urlopen(f"{self.server.api_base}/api/state", timeout=5.0) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _tracks(camera, observed_at):
        result = []
        for track in camera.get("yolo_tracks") or []:
            box = track.get("box") or []
            if len(box) != 4:
                continue
            x1, y1, width, height = (max(0.0, float(value)) for value in box)
            if width <= 0 or height <= 0:
                continue
            global_id = str(track.get("track_id") or f"anonymous-{track.get('local_id')}")[:128]
            result.append(
                {
                    "track_id": global_id,
                    "global_id": global_id,
                    "local_track_id": track.get("local_id"),
                    "person_id": None,
                    "display_name": None,
                    "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                    "bbox_format": "xywh_pixels",
                    "confidence": min(1.0, max(0.0, float(track.get("confidence") or 0.0))),
                    "observed_at": observed_at,
                }
            )
        return result

    def _publish_once(self):
        state = self._state()
        camera = next(
            (item for item in state.get("cameras", []) if item.get("id") == self.camera_id),
            None,
        )
        if not camera or camera.get("status") != "running":
            raise RuntimeError(f"{self.camera_id} is not running")
        jpeg = _read_live_jpeg(self.server.api_base, self.camera_id)
        width, height = _jpeg_dimensions(jpeg)
        now = datetime.now(timezone.utc)
        captured_at = now.isoformat(timespec="milliseconds")
        unique = now.strftime("%Y%m%dT%H%M%S%fZ")
        snapshot_id = f"live-{self.camera_id}-{unique}"
        image_url = f"{self.server.live_public_base}/api/handoff/live/{snapshot_id}.jpg"
        event = build_monitoring_snapshot_frame_event(
            event_id=f"snapshot.frame:{snapshot_id}",
            snapshot_id=snapshot_id,
            camera_id=self.camera_id,
            captured_at=captured_at,
            image_url=image_url,
            frame_width=width,
            frame_height=height,
            tracks=self._tracks(camera, captured_at),
        )
        if not event["camera"]["tracks"]:
            self._update(
                state="idle_no_tracks",
                last_error=None,
                last_tracks=0,
                frame_width=width,
                frame_height=height,
            )
            return
        with self.server.live_snapshot_lock:
            self.server.live_snapshots[snapshot_id] = jpeg
            while len(self.server.live_snapshots) > 12:
                self.server.live_snapshots.pop(next(iter(self.server.live_snapshots)))
        response = self.server.member_client.send_handoff(event)
        receipt_id = extract_receipt_id(response.body)
        if response.status != 200 or not receipt_id:
            raise RuntimeError(f"member receiver rejected live frame with HTTP {response.status}")
        current_status = self.public_status
        frame_reports = list(current_status["frames"])
        frame_reports.append(
            {
                "snapshot_id": snapshot_id,
                "http_status": response.status,
                "receipt_id": receipt_id,
            }
        )
        frame_reports = frame_reports[-20:]
        sent_frames = current_status["sent_frames"] + 1
        self._update(
            state="running",
            sent_frames=sent_frames,
            last_error=None,
            last_receipt_id=receipt_id,
            frames=frame_reports,
            last_sent_at=captured_at,
            last_tracks=len(event["camera"]["tracks"]),
            frame_width=width,
            frame_height=height,
        )
        if self.max_frames is not None and sent_frames >= self.max_frames:
            self._update(state="complete")
            self.stop_event.set()

    def _run(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self._publish_once()
            except Exception as error:
                self._update(state="error", last_error=f"{type(error).__name__}: {error}"[:500])
            delay = max(0.0, self.interval_seconds - (time.monotonic() - started))
            self.stop_event.wait(delay)


class FrontendHandler(SimpleHTTPRequestHandler):
    server_version = "LabFrontend/1.0"

    def _send_html(self, path, api_base=None):
        html = path.read_text(encoding="utf-8")
        api_base = (api_base or self.server.api_base).replace("\\", "\\\\").replace('"', '\\"')
        bootstrap = (
            f'<script>window.__LAB_API_BASE__ = "{api_base}"; '
            'window.__LAB_HARDWARE_URL__ = "/api/environment";</script>'
        )
        html = html.replace("</head>", f"  {bootstrap}\n  </head>", 1)
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self):
        parsed = urlsplit(self.path)
        route = parsed.path
        requested_api = parse_qs(parsed.query).get("api", [None])[0]
        if requested_api and re.fullmatch(r"https?://[^\\s]+", requested_api):
            requested_api = requested_api.rstrip("/")
        else:
            requested_api = None
        live_match = re.fullmatch(
            r"/api/handoff/live/([A-Za-z0-9_-]{1,128})\.jpg",
            route,
        )
        if live_match:
            if self.client_address[0] not in self.server.live_snapshot_allowed_clients:
                self._send_json({"error": "live snapshot access denied"}, status=403)
                return
            snapshot_id = live_match.group(1)
            with self.server.live_snapshot_lock:
                jpeg = self.server.live_snapshots.get(snapshot_id)
            if jpeg is None:
                self._send_json({"error": "live snapshot not found or expired"}, status=404)
                return
            self._send_bytes(jpeg, "image/jpeg", cache_control="no-store")
            return
        if route == "/api/handoff/test.jpg":
            self._send_bytes(
                HANDOFF_TEST_JPEG,
                "image/jpeg",
                cache_control="public, max-age=30",
            )
            return
        if route == "/api/environment":
            data = json.dumps(self.server.environment.status(), ensure_ascii=False).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8")
            return
        if route == "/api/integration/status":
            status = dict(self.server.member_integration_status)
            publisher = self.server.live_publisher
            status["live_publisher"] = (
                publisher.public_status
                if publisher is not None
                else {"enabled": False, "state": "disabled"}
            )
            self._send_json(status)
            return
        if route == "/api/interaction-results/latest":
            self._proxy_interaction_results(parsed.query)
            return
        if route == "/":
            self._send_html(TEMPLATE, requested_api)
            return
        if route == "/annotate":
            self._send_html(ANNOTATE_TEMPLATE, requested_api)
            return
        if route == "/faces":
            self._send_html(FACES_TEMPLATE, requested_api)
            return
        if route == "/recordings":
            self._send_html(RECORDINGS_TEMPLATE, requested_api)
            return
        if route.startswith("/static/"):
            relative = route.removeprefix("/static/")
            candidate = (STATIC / relative).resolve()
            if STATIC in candidate.parents and candidate.is_file():
                content_type = {
                    ".css": "text/css; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                }.get(candidate.suffix, "application/octet-stream")
                self._send_file(candidate, content_type)
                return
        self.send_error(404, "frontend resource not found")

    def do_POST(self):
        route = urlsplit(self.path).path
        allowed_routes = {
            "/api/integration/synthetic-round2": self._run_synthetic_round2,
            "/api/integration/synthetic-round3-frames": self._run_synthetic_round3_frames,
            "/api/integration/synthetic-round4-track": self._run_synthetic_round4_track,
            "/api/integration/synthetic-round5-crossing-tracks": self._run_synthetic_round5_crossing_tracks,
            "/api/integration/synthetic-round6-camera-handoff": self._run_synthetic_round6_camera_handoff,
        }
        operation = allowed_routes.get(route)
        if operation is None:
            self.send_error(404, "frontend resource not found")
            return
        # This operation writes fixed synthetic events to the peer service.  It
        # is intentionally unavailable to other LAN hosts and accepts no event
        # body, so it cannot become a generic authenticated relay.
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            self._send_json({"error": "local access only"}, status=403)
            return
        if int(self.headers.get("Content-Length", "0") or 0) != 0:
            self._send_json({"error": "request body is not allowed"}, status=400)
            return
        try:
            operation()
        except Exception as error:
            # These fixed synthetic operations are localhost-only. Return a
            # bounded diagnostic instead of dropping the HTTP connection, and
            # never include request headers or the member-service token.
            self._send_json(
                {
                    "success": False,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                },
                status=500,
            )

    @staticmethod
    def _member_step(response):
        return response.public_dict()

    def _run_synthetic_round2(self):
        client = self.server.member_client
        if client is None:
            self._send_json(
                {"success": False, "error": "member integration is not configured"},
                status=503,
            )
            return
        connection_event = build_connection_test_event(
            "connection.test:synthetic-round2-001"
        )
        snapshot_event = build_synthetic_snapshot_ready_event()
        try:
            health = client.health()
            connection_first = client.send_handoff(connection_event)
            connection_second = client.send_handoff(connection_event)
            snapshot_first = client.send_handoff(snapshot_event)
            snapshot_second = client.send_handoff(snapshot_event)
            latest = client.latest_interactions()
            cam_2 = client.latest_interactions("cam_2")
        except MemberIntegrationTransportError as error:
            self._send_json({"success": False, "error": str(error)}, status=502)
            return

        connection_receipts = [
            extract_receipt_id(connection_first.body),
            extract_receipt_id(connection_second.body),
        ]
        snapshot_receipts = [
            extract_receipt_id(snapshot_first.body),
            extract_receipt_id(snapshot_second.body),
        ]

        def acknowledged(response):
            return bool(
                response.ok
                and isinstance(response.body, dict)
                and response.body.get("status") == "acknowledged"
            )

        success = bool(
            health.ok
            and acknowledged(connection_first)
            and acknowledged(connection_second)
            and acknowledged(snapshot_first)
            and acknowledged(snapshot_second)
            and connection_receipts[0]
            and connection_receipts[0] == connection_receipts[1]
            and snapshot_receipts[0]
            and snapshot_receipts[0] == snapshot_receipts[1]
            and latest.ok
            and cam_2.ok
            and extract_results(latest.body) == []
            and extract_results(cam_2.body) == []
        )
        report = {
            "success": success,
            "contains_real_person_data": False,
            "health": self._member_step(health),
            "connection_test": {
                "first": self._member_step(connection_first),
                "duplicate": self._member_step(connection_second),
                "receipt_ids": connection_receipts,
                "same_receipt_id": bool(
                    connection_receipts[0]
                    and connection_receipts[0] == connection_receipts[1]
                ),
            },
            "snapshot_ready": {
                "first": self._member_step(snapshot_first),
                "duplicate": self._member_step(snapshot_second),
                "receipt_ids": snapshot_receipts,
                "same_receipt_id": bool(
                    snapshot_receipts[0]
                    and snapshot_receipts[0] == snapshot_receipts[1]
                ),
            },
            "interaction_results_latest": self._member_step(latest),
            "interaction_results_cam_2": self._member_step(cam_2),
        }
        self._send_json(report, status=200 if success else 502)

    def _run_synthetic_round3_frames(self):
        client = self.server.member_client
        if client is None:
            self._send_json(
                {"success": False, "error": "member integration is not configured"},
                status=503,
            )
            return

        frames = []
        started = time.monotonic()
        for index in range(1, 6):
            target_time = started + (index - 1)
            delay = target_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            event_id = f"snapshot.frame:synthetic-round3:cam_2:{index:06d}"
            snapshot_id = f"synthetic-round3-cam_2-{index:06d}"
            event = build_synthetic_snapshot_frame_event(
                event_id,
                snapshot_id=snapshot_id,
                include_test_track=False,
            )
            request_started = time.monotonic()
            try:
                response = client.send_handoff(event)
            except MemberIntegrationTransportError as error:
                frames.append(
                    {
                        "index": index,
                        "event_id": event_id,
                        "snapshot_id": snapshot_id,
                        "captured_at": event["captured_at"],
                        "elapsed_ms": round(
                            (time.monotonic() - request_started) * 1000, 1
                        ),
                        "error": str(error),
                    }
                )
                continue
            frames.append(
                {
                    "index": index,
                    "event_id": event_id,
                    "snapshot_id": snapshot_id,
                    "captured_at": event["captured_at"],
                    "tracks_count": len(event["camera"]["tracks"]),
                    "elapsed_ms": round(
                        (time.monotonic() - request_started) * 1000, 1
                    ),
                    **self._member_step(response),
                    "receipt_id": extract_receipt_id(response.body),
                }
            )

        try:
            latest = client.latest_interactions()
            cam_2 = client.latest_interactions("cam_2")
        except MemberIntegrationTransportError as error:
            self._send_json(
                {
                    "success": False,
                    "contains_real_person_data": False,
                    "frames": frames,
                    "error": str(error),
                },
                status=502,
            )
            return

        frames_ok = all(
            frame.get("http_status") == 200
            and isinstance(frame.get("response"), dict)
            and frame["response"].get("status") == "acknowledged"
            and bool(frame.get("receipt_id"))
            and frame.get("tracks_count") == 0
            for frame in frames
        )
        results_ok = bool(
            latest.ok
            and cam_2.ok
            and extract_results(latest.body) == []
            and extract_results(cam_2.body) == []
            and isinstance(latest.body, dict)
            and latest.body.get("errors") == []
            and isinstance(cam_2.body, dict)
            and cam_2.body.get("errors") == []
        )
        report = {
            "success": frames_ok and results_ok,
            "contains_real_person_data": False,
            "requested_frequency_hz": 1,
            "requested_frame_count": 5,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "frames": frames,
            "interaction_results_latest": self._member_step(latest),
            "interaction_results_cam_2": self._member_step(cam_2),
        }
        self._send_json(report, status=200 if report["success"] else 502)

    def _run_synthetic_round4_track(self):
        client = self.server.member_client
        if client is None:
            self._send_json(
                {"success": False, "error": "member integration is not configured"},
                status=503,
            )
            return

        frames = []
        started = time.monotonic()
        for index in range(1, 6):
            target_time = started + (index - 1)
            delay = target_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            event_id = f"snapshot.frame:synthetic-round4:cam_2:{index:06d}"
            snapshot_id = f"synthetic-round4-cam_2-{index:06d}"
            bbox = (8 + ((index - 1) * 4), 4, 12, 24)
            event = build_synthetic_snapshot_frame_event(
                event_id,
                snapshot_id=snapshot_id,
                include_test_track=True,
                test_bbox=bbox,
            )
            request_started = time.monotonic()
            try:
                response = client.send_handoff(event)
            except MemberIntegrationTransportError as error:
                frames.append(
                    {
                        "index": index,
                        "event_id": event_id,
                        "snapshot_id": snapshot_id,
                        "captured_at": event["captured_at"],
                        "bbox": list(bbox),
                        "elapsed_ms": round(
                            (time.monotonic() - request_started) * 1000, 1
                        ),
                        "error": str(error),
                    }
                )
                continue
            frames.append(
                {
                    "index": index,
                    "event_id": event_id,
                    "snapshot_id": snapshot_id,
                    "captured_at": event["captured_at"],
                    "track_id": event["camera"]["tracks"][0]["track_id"],
                    "bbox": list(bbox),
                    "elapsed_ms": round(
                        (time.monotonic() - request_started) * 1000, 1
                    ),
                    **self._member_step(response),
                    "receipt_id": extract_receipt_id(response.body),
                }
            )

        # Analysis is asynchronous. Give the receiver one normal result TTL
        # interval to publish either an empty result or a non-sensitive error.
        time.sleep(1.0)
        try:
            latest = client.latest_interactions()
            cam_2 = client.latest_interactions("cam_2")
        except MemberIntegrationTransportError as error:
            self._send_json(
                {
                    "success": False,
                    "contains_real_person_data": False,
                    "frames": frames,
                    "error": str(error),
                },
                status=502,
            )
            return

        frames_ok = all(
            frame.get("http_status") == 200
            and isinstance(frame.get("response"), dict)
            and frame["response"].get("status") == "acknowledged"
            and bool(frame.get("receipt_id"))
            for frame in frames
        )
        results_ok = bool(
            latest.ok
            and cam_2.ok
            and extract_results(latest.body) == []
            and extract_results(cam_2.body) == []
            and isinstance(latest.body, dict)
            and latest.body.get("errors") == []
            and isinstance(cam_2.body, dict)
            and cam_2.body.get("errors") == []
        )
        report = {
            "success": frames_ok and results_ok,
            "contains_real_person_data": False,
            "synthetic_track_id": "track_test_1",
            "requested_frequency_hz": 1,
            "requested_frame_count": 5,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "frames": frames,
            "interaction_results_latest": self._member_step(latest),
            "interaction_results_cam_2": self._member_step(cam_2),
        }
        self._send_json(report, status=200 if report["success"] else 502)

    def _run_synthetic_round5_crossing_tracks(self):
        client = self.server.member_client
        if client is None:
            self._send_json(
                {"success": False, "error": "member integration is not configured"},
                status=503,
            )
            return

        frames = []
        started = time.monotonic()
        for index in range(1, 6):
            target_time = started + (index - 1)
            delay = target_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            event_id = f"snapshot.frame:synthetic-round5:cam_2:{index:06d}"
            snapshot_id = f"synthetic-round5-cam_2-{index:06d}"
            first_bbox = (4 + ((index - 1) * 6), 4, 10, 24)
            second_bbox = (34 - ((index - 1) * 6), 4, 10, 24)
            event = build_synthetic_snapshot_frame_event(
                event_id,
                snapshot_id=snapshot_id,
                include_test_track=True,
                test_bbox=first_bbox,
                second_test_bbox=second_bbox,
            )
            request_started = time.monotonic()
            try:
                response = client.send_handoff(event)
            except MemberIntegrationTransportError as error:
                frames.append(
                    {
                        "index": index,
                        "event_id": event_id,
                        "snapshot_id": snapshot_id,
                        "captured_at": event["captured_at"],
                        "track_bboxes": {
                            "track_test_1": list(first_bbox),
                            "track_test_2": list(second_bbox),
                        },
                        "elapsed_ms": round(
                            (time.monotonic() - request_started) * 1000, 1
                        ),
                        "error": str(error),
                    }
                )
                continue
            frames.append(
                {
                    "index": index,
                    "event_id": event_id,
                    "snapshot_id": snapshot_id,
                    "captured_at": event["captured_at"],
                    "track_ids": [
                        track["track_id"] for track in event["camera"]["tracks"]
                    ],
                    "track_bboxes": {
                        track["track_id"]: track["bbox"]
                        for track in event["camera"]["tracks"]
                    },
                    "elapsed_ms": round(
                        (time.monotonic() - request_started) * 1000, 1
                    ),
                    **self._member_step(response),
                    "receipt_id": extract_receipt_id(response.body),
                }
            )

        time.sleep(1.0)
        try:
            latest = client.latest_interactions()
            cam_2 = client.latest_interactions("cam_2")
        except MemberIntegrationTransportError as error:
            self._send_json(
                {
                    "success": False,
                    "contains_real_person_data": False,
                    "frames": frames,
                    "error": str(error),
                },
                status=502,
            )
            return

        expected_track_ids = ["track_test_1", "track_test_2"]
        frames_ok = all(
            frame.get("http_status") == 200
            and isinstance(frame.get("response"), dict)
            and frame["response"].get("status") == "acknowledged"
            and bool(frame.get("receipt_id"))
            and frame.get("track_ids") == expected_track_ids
            for frame in frames
        )
        results_ok = bool(
            latest.ok
            and cam_2.ok
            and extract_results(latest.body) == []
            and extract_results(cam_2.body) == []
            and isinstance(latest.body, dict)
            and latest.body.get("errors") == []
            and isinstance(cam_2.body, dict)
            and cam_2.body.get("errors") == []
        )
        report = {
            "success": frames_ok and results_ok,
            "contains_real_person_data": False,
            "synthetic_track_ids": expected_track_ids,
            "scenario": "two_tracks_crossing",
            "requested_frequency_hz": 1,
            "requested_frame_count": 5,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "frames": frames,
            "interaction_results_latest": self._member_step(latest),
            "interaction_results_cam_2": self._member_step(cam_2),
        }
        self._send_json(report, status=200 if report["success"] else 502)

    def _run_synthetic_round6_camera_handoff(self):
        client = self.server.member_client
        if client is None:
            self._send_json(
                {"success": False, "error": "member integration is not configured"},
                status=503,
            )
            return

        source_event = build_synthetic_snapshot_frame_event(
            "snapshot.frame:synthetic-round6:cam_2:000001",
            snapshot_id="synthetic-round6-cam_2-000001",
            camera_id="cam_2",
            track_id="track_test_source_1",
            global_id="global_test_handoff_1",
            local_track_id=101,
            person_id="person_test_handoff_1",
            test_bbox=(24, 4, 10, 24),
        )
        handoff_event = build_synthetic_person_handoff_event()
        target_event = build_synthetic_snapshot_frame_event(
            "snapshot.frame:synthetic-round6:cam_entrance:000001",
            snapshot_id="synthetic-round6-cam_entrance-000001",
            camera_id="cam_entrance",
            track_id="track_test_target_1",
            global_id="global_test_handoff_1",
            local_track_id=201,
            person_id="person_test_handoff_1",
            test_bbox=(8, 4, 10, 24),
        )

        started = time.monotonic()
        try:
            source_response = client.send_handoff(source_event)
            handoff_first = client.send_handoff(handoff_event)
            handoff_duplicate = client.send_handoff(handoff_event)
            target_response = client.send_handoff(target_event)
            time.sleep(1.0)
            latest = client.latest_interactions()
            cam_2 = client.latest_interactions("cam_2")
            cam_entrance = client.latest_interactions("cam_entrance")
        except MemberIntegrationTransportError as error:
            self._send_json(
                {
                    "success": False,
                    "contains_real_person_data": False,
                    "error": str(error),
                },
                status=502,
            )
            return

        def acknowledged(response):
            return bool(
                response.status == 200
                and isinstance(response.body, dict)
                and response.body.get("status") == "acknowledged"
                and extract_receipt_id(response.body)
            )

        handoff_receipts = [
            extract_receipt_id(handoff_first.body),
            extract_receipt_id(handoff_duplicate.body),
        ]
        responses_ok = all(
            acknowledged(response)
            for response in (
                source_response,
                handoff_first,
                handoff_duplicate,
                target_response,
            )
        )
        results_ok = all(
            response.ok
            and extract_results(response.body) == []
            and isinstance(response.body, dict)
            and response.body.get("errors") == []
            for response in (latest, cam_2, cam_entrance)
        )
        same_handoff_receipt = bool(
            handoff_receipts[0]
            and handoff_receipts[0] == handoff_receipts[1]
        )
        report = {
            "success": responses_ok and results_ok and same_handoff_receipt,
            "contains_real_person_data": False,
            "scenario": "cam_2_to_cam_entrance_handoff",
            "identity_continuity": {
                "person_id": "person_test_handoff_1",
                "global_id": "global_test_handoff_1",
                "source_camera_id": "cam_2",
                "source_track_id": "track_test_source_1",
                "source_local_track_id": 101,
                "target_camera_id": "cam_entrance",
                "target_track_id": "track_test_target_1",
                "target_local_track_id": 201,
            },
            "source_frame": self._member_step(source_response),
            "handoff": {
                "first": self._member_step(handoff_first),
                "duplicate": self._member_step(handoff_duplicate),
                "receipt_ids": handoff_receipts,
                "same_receipt_id": same_handoff_receipt,
            },
            "target_frame": self._member_step(target_response),
            "interaction_results_latest": self._member_step(latest),
            "interaction_results_cam_2": self._member_step(cam_2),
            "interaction_results_cam_entrance": self._member_step(cam_entrance),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        self._send_json(report, status=200 if report["success"] else 502)

    def _proxy_interaction_results(self, query):
        client = self.server.member_client
        if client is None:
            self._send_json(
                {
                    "results": [],
                    "integration": {"configured": False, "status": "disabled"},
                },
                status=503,
            )
            return
        camera_id = parse_qs(query).get("camera_id", [None])[0]
        if camera_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
            self._send_json({"error": "invalid camera_id"}, status=400)
            return
        try:
            response = client.latest_interactions(camera_id)
        except MemberIntegrationTransportError as error:
            self._send_json(
                {"results": [], "integration": {"status": "unavailable", "error": str(error)}},
                status=502,
            )
            return
        self._send_json(response.body, status=response.status)

    def _send_file(self, path, content_type):
        self._send_bytes(path.read_bytes(), content_type)

    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status=status)

    def _send_bytes(self, data, content_type, cache_control="no-store", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"[frontend] {self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="Standalone laboratory frontend server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--api", default="http://127.0.0.1:15000")
    args = parser.parse_args()
    print(f"Frontend: http://127.0.0.1:{args.port}/?api={args.api}")
    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    server.api_base = args.api.rstrip("/")
    server.live_public_base = os.environ.get(
        "LAB_MEMBER_LIVE_PUBLIC_BASE",
        f"http://192.168.31.127:{args.port}",
    ).rstrip("/")
    server.live_snapshot_lock = threading.Lock()
    server.live_snapshots = {}
    server.live_snapshot_allowed_clients = {
        "127.0.0.1",
        "::1",
        os.environ.get("LAB_MEMBER_RECEIVER_IP", "192.168.31.189").strip(),
    }
    server.live_publisher = None
    configured_port = os.environ.get("LAB_HARDWARE_PORT", "").strip()
    auto_detect = os.environ.get("LAB_HARDWARE_AUTO", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    server.environment = EnvironmentSerialMonitor(
        port=configured_port or (discover_hardware_port() if auto_detect else None),
        baudrate=int(os.environ.get("LAB_HARDWARE_BAUD", "115200")),
        stale_seconds=float(os.environ.get("LAB_HARDWARE_STALE_SECONDS", "30")),
    )
    member_url = os.environ.get("LAB_MEMBER_SERVICE_URL", "").strip()
    member_token = os.environ.get("LAB_MEMBER_SERVICE_TOKEN", "").strip()
    server.member_client = None
    server.member_integration_status = {"configured": False, "status": "disabled"}
    if member_url or member_token:
        try:
            server.member_client = MemberServiceClient.from_environment()
        except (MemberIntegrationConfigError, ValueError) as error:
            raise SystemExit(f"Member integration configuration error: {error}") from error
        server.member_integration_status = {
            **server.member_client.public_config,
            "status": "configured",
        }
        live_enabled = os.environ.get(
            "LAB_MEMBER_LIVE_PUBLISH_ENABLED", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        if live_enabled:
            server.live_publisher = LiveSnapshotPublisher(
                server,
                camera_id=os.environ.get("LAB_MEMBER_LIVE_CAMERA_ID", "cam_2"),
                interval_seconds=float(
                    os.environ.get("LAB_MEMBER_LIVE_INTERVAL_SECONDS", "1")
                ),
                max_frames=(
                    int(os.environ["LAB_MEMBER_LIVE_MAX_FRAMES"])
                    if os.environ.get("LAB_MEMBER_LIVE_MAX_FRAMES", "").strip()
                    else None
                ),
            )
    server.environment.start()
    if server.live_publisher is not None:
        server.live_publisher.start()
    print(f"Hardware: {server.environment.port or 'not detected'}")
    print(
        "Member integration: "
        + (server.member_client.base_url if server.member_client else "disabled")
    )
    try:
        server.serve_forever()
    finally:
        if server.live_publisher is not None:
            server.live_publisher.stop()
        server.environment.stop()
        server.server_close()


if __name__ == "__main__":
    main()
