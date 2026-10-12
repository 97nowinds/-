import argparse
import base64
import json
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen
from handoff_transport import iso_time, load_demo_config
from joint_demo import build_joint_event
from joint_results import JointResultProjection
from management_api import ManagementRuntime, handle_management

from hardware_serial import EnvironmentSerialMonitor, discover_hardware_port
from map_handoff import read_snapshot
from handoff_result_index import SnapshotTrackIndex
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
DISPLAY_TEMPLATE = ROOT / "templates" / "display.html"
ANNOTATE_TEMPLATE = ROOT / "templates" / "annotate.html"
FACES_TEMPLATE = ROOT / "templates" / "faces.html"
RECORDINGS_TEMPLATE = ROOT / "templates" / "recordings.html"
STATIC = ROOT / "static"
INSTRUMENT_CONFIG = ROOT / "config" / "instruments.json"

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
        self.position_match_tolerance_seconds = load_demo_config()['location_max_delta_seconds']
        self.position_history = defaultdict(lambda: deque(maxlen=120))
        self.result_index = SnapshotTrackIndex()
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
            "position_match_tolerance_seconds": self.position_match_tolerance_seconds,
            "last_tracks_with_location": 0,
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
    def _location_observed_at(value):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return datetime.fromtimestamp(float(value), timezone.utc).isoformat(
                timespec="milliseconds"
            )
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _unix_time(value):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    @staticmethod
    def _person_aliases(person):
        aliases = {
            person.get("global_id"),
            person.get("global_track_id"),
            person.get("track_id"),
            *(person.get("global_track_ids") or []),
        }
        return {str(alias) for alias in aliases if alias is not None}

    def _remember_floor_people(self, floor_people):
        for person in floor_people or []:
            if not isinstance(person, dict):
                continue
            observed_unix = self._unix_time(
                person.get("position_observed_at", person.get("observed_at"))
            )
            try:
                x_m = float(person["x"])
                y_m = float(person["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if observed_unix is None or not math.isfinite(x_m) or not math.isfinite(y_m):
                continue
            quality = str(person.get("localization_quality") or "unknown").lower()
            if quality not in {"high", "medium", "low", "unknown"}:
                quality = "unknown"
            sample = {
                "unix_time": observed_unix,
                "x_m": x_m,
                "y_m": y_m,
                "quality": quality,
                "localization_state": str(person.get("localization_state") or ""),
                "association_conflict": bool(person.get("association_conflict")),
            }
            for alias in self._person_aliases(person):
                history = self.position_history[alias]
                if history and abs(history[-1]["unix_time"] - observed_unix) < 1e-6:
                    history[-1] = sample
                else:
                    history.append(sample)

    @staticmethod
    def _trajectory_status(sample, history):
        if sample.get("association_conflict"):
            return "unknown"
        if sample.get("localization_state") == "held":
            return "lost"
        previous = next(
            (
                item
                for item in reversed(history)
                if item["unix_time"] < sample["unix_time"] - 1e-6
            ),
            None,
        )
        if previous is None:
            return "unknown"
        elapsed = sample["unix_time"] - previous["unix_time"]
        if elapsed <= 0.0 or elapsed > 3.0:
            return "unknown"
        distance = math.hypot(
            sample["x_m"] - previous["x_m"],
            sample["y_m"] - previous["y_m"],
        )
        return "moving" if distance / elapsed >= 0.15 else "stationary"

    @classmethod
    def _tracks(
        cls,
        camera,
        observed_at,
        floor_people=None,
        position_history=None,
        max_location_delta_seconds=0.75,
    ):
        candidates_by_id = defaultdict(list)
        alias_owners = defaultdict(set)
        for person in floor_people or []:
            if not isinstance(person, dict):
                continue
            observed_unix = cls._unix_time(
                person.get("position_observed_at", person.get("observed_at"))
            )
            try:
                x_m = float(person["x"])
                y_m = float(person["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if observed_unix is None or not math.isfinite(x_m) or not math.isfinite(y_m):
                continue
            quality = str(person.get("localization_quality") or "unknown").lower()
            if quality not in {"high", "medium", "low", "unknown"}:
                quality = "unknown"
            sample = {
                "unix_time": observed_unix,
                "x_m": x_m,
                "y_m": y_m,
                "quality": quality,
                "localization_state": str(person.get("localization_state") or ""),
                "association_conflict": bool(person.get("association_conflict")),
            }
            for alias in cls._person_aliases(person):
                alias_owners[alias].add(str(person.get("track_id") or id(person)))
                candidates_by_id[alias].append(sample)
        for alias, history in (position_history or {}).items():
            candidates_by_id[str(alias)].extend(history)

        captured_unix = cls._unix_time(observed_at)
        result = []
        for track in camera.get("yolo_tracks") or []:
            box = track.get("box") or []
            if len(box) != 4:
                continue
            x1, y1, width, height = (max(0.0, float(value)) for value in box)
            if width <= 0 or height <= 0:
                continue
            global_id = str(
                track.get("global_id")
                or track.get("track_id")
                or f"anonymous-{track.get('local_id')}"
            )[:128]
            item = {
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
            identities = [identity for identity in camera.get("identities") or []
                          if identity.get("local_id") == track.get("local_id")
                          and identity.get("global_track_id") == global_id
                          and identity.get("identity_lock_status") == "locked"
                          and identity.get("person_id")]
            if len(identities) == 1:
                item["person_id"] = identities[0]["person_id"]
                item["display_name"] = identities[0].get("name")
            candidates = candidates_by_id.get(global_id) or []
            if len(alias_owners.get(global_id, ())) > 1:
                candidates = []
            # Held/conflicted positions are not current instrument candidates.
            # Keep their original timestamps in the map API, omit them here.
            candidates = [sample for sample in candidates
                          if not sample.get("association_conflict")
                          and sample.get("localization_state") not in {"held", "predicted"}]
            nearest = (
                min(candidates, key=lambda sample: abs(sample["unix_time"] - captured_unix))
                if candidates and captured_unix is not None
                else None
            )
            if (
                nearest is not None
                and abs(nearest["unix_time"] - captured_unix)
                <= float(max_location_delta_seconds)
            ):
                history = sorted(candidates, key=lambda sample: sample["unix_time"])
                # Detector confidence describes the image-space person box,
                # not the map projection. Do not copy it into location.
                item["location"] = {
                    "x_m": round(nearest["x_m"], 3),
                    "y_m": round(nearest["y_m"], 3),
                    "quality": nearest["quality"],
                    "trajectory_status": cls._trajectory_status(nearest, history),
                    "observed_at": cls._location_observed_at(nearest["unix_time"]),
                }
            result.append(item)
        return result

    def _publish_once(self):
        config = load_demo_config()
        if not config.get('automatic_enabled'):
            self._update(state='awaiting_alignment_acceptance', last_error=None)
            return
        token = os.environ.get('LAB_HANDOFF_IMAGE_READ_TOKEN', '')
        if not token:
            raise ValueError('snapshot read credential required')
        pending = getattr(self, '_pending_event', None)
        if pending and time.monotonic() >= self._pending_deadline:
            self._last_snapshot_id = pending['snapshot_id']
            self._pending_event = pending = None
        if pending:
            sync_until = self._unix_time(pending['camera']['timing']['sync_valid_until'])
            captured = self._unix_time(pending['captured_at'])
            if time.time() >= sync_until or time.time() - captured > config['inference_start_max_age_seconds']:
                self._last_snapshot_id = pending['snapshot_id']
                self._pending_event = None
                self._update(state='stale', last_error=None)
                return
            event = pending
        else:
            req = Request(config['image_base'] + '/api/handoff/frozen/' + self.camera_id,
                          headers={'Authorization': 'Bearer ' + token})
            with urlopen(req, timeout=5) as response:
                frozen = json.load(response)
            if frozen['snapshot_id'] == getattr(self, '_last_snapshot_id', None):
                return
            captured_at = iso_time(frozen['source_captured_at'])
            people = [{**obs, 'x': obs['position']['x'], 'y': obs['position']['y'],
                       'position_observed_at': obs['observed_at']}
                      for obs in frozen['map_observations']
                      if not obs.get('position_estimated') and obs.get('position')]
            tracks = self._tracks(frozen['camera'], captured_at, people,
                                 max_location_delta_seconds=config['location_max_delta_seconds'])
            event = build_joint_event(frozen, tracks, config)
            if time.time() - self._unix_time(event['captured_at']) > config['inference_start_max_age_seconds']:
                self._last_snapshot_id = frozen['snapshot_id']
                self._update(state='stale', last_error=None)
                return
            # Check the actual image route separately from map availability.
            with urlopen(Request(frozen['image_url'],
                         headers={'Authorization': 'Bearer ' + token}), timeout=5) as response:
                jpeg = response.read(16 * 1024 * 1024 + 1)
                if response.headers.get_content_type() != 'image/jpeg':
                    raise ValueError('snapshot route did not return JPEG')
            import hashlib
            if hashlib.sha256(jpeg).hexdigest() != frozen['image_sha256']:
                raise ValueError('frozen JPEG digest mismatch')
            if _jpeg_dimensions(jpeg) != (frozen['frame_width'], frozen['frame_height']):
                raise ValueError('frozen JPEG dimensions mismatch')
            from jsonschema import Draft202012Validator, FormatChecker
            schema = json.loads(Path(config['receiver_schema_file']).read_text(encoding='utf-8'))
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema, format_checker=FormatChecker()).validate(event)
            remaining = frozen['image_expires_at'] - time.time()
            if remaining <= 0:
                raise ValueError('snapshot image expired')
            self._pending_event = event
            self._pending_deadline = time.monotonic() + remaining
            # The index must exist even when the result arrives before ACK.
            self.result_index.remember(event)
        width, height = event['camera']['frame_width'], event['camera']['frame_height']
        snapshot_id, captured_at = event['snapshot_id'], event['captured_at']
        if not event["camera"]["tracks"]:
            self._last_snapshot_id = snapshot_id
            self._pending_event = None
            self._update(
                state="idle_no_tracks",
                last_error=None,
                last_tracks=0,
                last_tracks_with_location=0,
                frame_width=width,
                frame_height=height,
            )
            return
        response = self.server.member_client.send_handoff(event)
        receipt_id = extract_receipt_id(response.body)
        if response.status != 200 or not receipt_id:
            raise RuntimeError(f"member receiver rejected live frame with HTTP {response.status}")
        self._last_snapshot_id = snapshot_id
        self._pending_event = None
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
            last_tracks_with_location=sum(
                1 for track in event["camera"]["tracks"] if "location" in track
            ),
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


class CameraPublishers:
    """Independent latest-only sender per analyzed camera, shared exact index."""
    def __init__(self, server, camera_ids, interval_seconds, max_frames=None):
        if not camera_ids or any(camera not in {'cam_1', 'cam_2'} for camera in camera_ids):
            raise ValueError('demo publisher supports cam_1 and cam_2 only')
        self.result_index = SnapshotTrackIndex()
        self.publishers = [LiveSnapshotPublisher(server, camera, interval_seconds, max_frames)
                           for camera in dict.fromkeys(camera_ids)]
        for publisher in self.publishers:
            publisher.result_index = self.result_index

    @property
    def public_status(self):
        return {'enabled': True, 'cameras': [publisher.public_status for publisher in self.publishers]}

    def start(self):
        for publisher in self.publishers:
            publisher.start()

    def stop(self):
        for publisher in self.publishers:
            publisher.stop()


class FrontendHandler(SimpleHTTPRequestHandler):
    server_version = "LabFrontend/1.0"

    def _send_html(self, path, api_base=None):
        html = path.read_text(encoding="utf-8")
        api_base = (api_base or self.server.api_base).replace("\\", "\\\\").replace('"', '\\"')
        bootstrap = (
            f'<script>window.__LAB_API_BASE__ = "{api_base}"; '
            'window.__LAB_HARDWARE_URL__ = "/api/environment"; '
            f'window.__LAB_RESULT_CONTRACT__ = {json.dumps(os.environ.get("LAB_MEMBER_RESULT_CONTRACT", ""))};</script>'
        )
        html = html.replace("</head>", f"  {bootstrap}\n  </head>", 1)
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self):
        if handle_management(self, 'GET'):
            return
        parsed = urlsplit(self.path)
        route = parsed.path
        if route in {"/api/handoff/map", "/api/handoff/positions"}:
            try:
                payload = read_snapshot(self.server.api_base)
                if route.endswith("/positions"):
                    payload.pop("layout")
                self._send_json(payload)
            except Exception:
                self._send_json({"error": "camera backend unavailable", "people": []}, status=503)
            return
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
        if route == "/api/instruments/anchors":
            self._send_json(json.loads(INSTRUMENT_CONFIG.read_text(encoding="utf-8")))
            return
        if route == "/display":
            self._send_html(DISPLAY_TEMPLATE, requested_api)
            return
        if route in {"/", "/monitor"}:
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
                    ".json": "application/json; charset=utf-8",
                }.get(candidate.suffix, "application/octet-stream")
                self._send_file(candidate, content_type)
                return
        self.send_error(404, "frontend resource not found")

    def do_POST(self):
        if handle_management(self, 'POST'):
            return
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
        contract = parse_qs(query).get("contract", [None])[0]
        if contract is not None and contract != 'lab-interaction-results-1.2':
            self._send_json({'error': 'unsupported result contract'}, status=400)
            return
        if camera_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
            self._send_json({"error": "invalid camera_id"}, status=400)
            return
        try:
            response = (client.latest_interactions(camera_id, contract=contract)
                        if contract else client.latest_interactions(camera_id))
        except MemberIntegrationTransportError as error:
            self._send_json(
                {"results": [], "integration": {"status": "unavailable", "error": str(error)}},
                status=502,
            )
            return
        body = response.body
        if contract and (not isinstance(body, dict) or body.get('result_contract') != contract):
            self._send_json({'error': 'receiver result contract unavailable',
                             'results': [], 'analyses': []}, status=502)
            return
        if contract:
            if not all(isinstance(body.get(field), list) for field in ('results', 'analyses', 'errors')):
                self._send_json({'error': 'invalid result contract shape'}, status=502)
                return
            for analysis in body['analyses']:
                if (not isinstance(analysis, dict) or analysis.get('status') not in
                        {'valid', 'pending', 'unavailable', 'error', 'stale', 'expired'}
                        or (analysis['status'] == 'valid' and type(analysis.get('is_interacting')) is not bool)
                        or (analysis['status'] != 'valid' and analysis.get('is_interacting') is not None)):
                    self._send_json({'error': 'invalid analysis status or judgment'}, status=502)
                    return
        # This opt-in affects only the local result proxy. It sends no new
        # fields to the receiver and defaults off during existing joint tests.
        if contract or os.environ.get("LAB_MEMBER_EXACT_RESULT_MATCH", "0") == "1":
            results = extract_results(body)
            publisher = self.server.live_publisher
            if results is not None:
                index = publisher.result_index if publisher else SnapshotTrackIndex()
                resolved = [index.resolve(item) for item in results if isinstance(item, dict)]
                body = resolved if isinstance(body, list) else {**body, "results": resolved}
                if contract:
                    body['analyses'] = [index.resolve(item) for item in body['analyses']]
                    try:
                        body = self.server.joint_result_projection.project(body)
                    except (ValueError, TypeError, KeyError):
                        self._send_json({'error': 'invalid result timestamps or association'}, status=502)
                        return
        self._send_json(body, status=response.status)

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
    server.joint_result_projection = JointResultProjection(
        ROOT / 'runtime' / 'joint-demo-analyses.jsonl')
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
            "LAB_MEMBER_LIVE_PUBLISH_ENABLED", "0"
        ).strip().lower() not in {"0", "false", "no", "off"}
        if live_enabled:
            camera_ids = os.environ.get('LAB_MEMBER_LIVE_CAMERA_IDS',
                os.environ.get('LAB_MEMBER_LIVE_CAMERA_ID', 'cam_1,cam_2')).split(',')
            server.live_publisher = CameraPublishers(
                server,
                camera_ids=[camera.strip() for camera in camera_ids],
                interval_seconds=float(
                    os.environ.get("LAB_MEMBER_LIVE_INTERVAL_SECONDS", "1")
                ),
                max_frames=(
                    int(os.environ["LAB_MEMBER_LIVE_MAX_FRAMES"])
                    if os.environ.get("LAB_MEMBER_LIVE_MAX_FRAMES", "").strip()
                    else None
                ),
            )
    server.management = None
    if os.environ.get('LAB_MANAGEMENT_ENABLED', '0') == '1':
        server.management = ManagementRuntime(server, ROOT / 'config' / 'management.json')
        server.management.start()
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
        if server.management is not None:
            server.management.stop()
        if server.live_publisher is not None:
            server.live_publisher.stop()
        server.environment.stop()
        server.server_close()


if __name__ == "__main__":
    main()
