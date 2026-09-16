import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from member_integration import (  # noqa: E402
    MemberIntegrationConfigError,
    MemberServiceClient,
    build_connection_test_event,
    extract_receipt_id,
    extract_results,
)


class FakeMemberHandler(BaseHTTPRequestHandler):
    token = "unit-test-token"
    receipts = {}
    requests = []

    def _json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        return self.headers.get("Authorization") == f"Bearer {self.token}"

    def do_GET(self):
        self.__class__.requests.append(("GET", self.path, self.headers.get("Authorization")))
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
        elif self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path.startswith("/api/interaction-results/latest"):
            self._json(200, {"results": []})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append(("POST", self.path, self.headers.get("Authorization"), payload))
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        if self.path != "/api/handoffs":
            self._json(404, {"error": "not found"})
            return
        event_id = payload["event_id"]
        receipt_id = self.__class__.receipts.setdefault(event_id, f"receipt-{len(self.receipts) + 1}")
        self._json(200, {"receipt_id": receipt_id})

    def log_message(self, *_args):
        pass


class MemberIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeMemberHandler.receipts = {}
        FakeMemberHandler.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMemberHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.client = MemberServiceClient(self.base_url, FakeMemberHandler.token, 1)

    def test_configuration_rejects_missing_token_and_url_credentials(self):
        with self.assertRaises(MemberIntegrationConfigError):
            MemberServiceClient(self.base_url, "")
        with self.assertRaises(MemberIntegrationConfigError):
            MemberServiceClient("http://user:password@127.0.0.1:8765", "token")

    def test_authorization_is_server_side_and_public_config_is_redacted(self):
        response = self.client.health()
        self.assertTrue(response.ok)
        self.assertEqual(FakeMemberHandler.requests[-1][2], "Bearer unit-test-token")
        self.assertNotIn("unit-test-token", json.dumps(self.client.public_config))

    def test_connection_event_contains_no_person_or_image_fields(self):
        event = build_connection_test_event("connection.test:fixed")
        serialized = json.dumps(event).lower()
        self.assertEqual(event["schema_version"], "lab-handoff-1.1")
        self.assertEqual(event["event_type"], "connection.test")
        self.assertFalse(event["test"]["contains_real_data"])
        for forbidden in ("person_id", "person_name", "face", "position", "image_url", "embedding"):
            self.assertNotIn(forbidden, serialized)

    def test_duplicate_event_id_returns_same_receipt(self):
        event = build_connection_test_event("connection.test:idempotent")
        first = self.client.send_handoff(event)
        second = self.client.send_handoff(event)
        self.assertTrue(first.ok and second.ok)
        self.assertEqual(extract_receipt_id(first.body), extract_receipt_id(second.body))

    def test_latest_results_supports_empty_global_and_camera_queries(self):
        latest = self.client.latest_interactions()
        cam_2 = self.client.latest_interactions("cam_2")
        self.assertEqual(extract_results(latest.body), [])
        self.assertEqual(extract_results(cam_2.body), [])
        self.assertIn("camera_id=cam_2", FakeMemberHandler.requests[-1][1])

    def test_http_error_preserves_status_and_complete_json_response(self):
        wrong = MemberServiceClient(self.base_url, "wrong-token", 1)
        response = wrong.health()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.body, {"error": "unauthorized"})


if __name__ == "__main__":
    unittest.main()
