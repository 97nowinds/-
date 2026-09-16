import argparse
import base64
import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from hardware_serial import EnvironmentSerialMonitor, discover_hardware_port
from member_integration import (
    MemberIntegrationConfigError,
    MemberIntegrationTransportError,
    MemberServiceClient,
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


class FrontendHandler(SimpleHTTPRequestHandler):
    server_version = "LabFrontend/1.0"

    def _send_html(self, path):
        html = path.read_text(encoding="utf-8")
        api_base = self.server.api_base.replace("\\", "\\\\").replace('"', '\\"')
        bootstrap = (
            f'<script>window.__LAB_API_BASE__ = "{api_base}"; '
            'window.__LAB_HARDWARE_URL__ = "/api/environment";</script>'
        )
        html = html.replace("</head>", f"  {bootstrap}\n  </head>", 1)
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self):
        parsed = urlsplit(self.path)
        route = parsed.path
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
            self._send_json(self.server.member_integration_status)
            return
        if route == "/api/interaction-results/latest":
            self._proxy_interaction_results(parsed.query)
            return
        if route == "/":
            self._send_html(TEMPLATE)
            return
        if route == "/annotate":
            self._send_html(ANNOTATE_TEMPLATE)
            return
        if route == "/faces":
            self._send_html(FACES_TEMPLATE)
            return
        if route == "/recordings":
            self._send_html(RECORDINGS_TEMPLATE)
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
    parser.add_argument("--api", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    print(f"Frontend: http://127.0.0.1:{args.port}/?api={args.api}")
    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    server.api_base = args.api.rstrip("/")
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
    server.environment.start()
    print(f"Hardware: {server.environment.port or 'not detected'}")
    print(
        "Member integration: "
        + (server.member_client.base_url if server.member_client else "disabled")
    )
    try:
        server.serve_forever()
    finally:
        server.environment.stop()
        server.server_close()


if __name__ == "__main__":
    main()
