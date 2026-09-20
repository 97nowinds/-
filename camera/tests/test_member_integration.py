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
    build_monitoring_snapshot_frame_event,
    build_connection_test_event,
    build_synthetic_person_handoff_event,
    build_synthetic_snapshot_frame_event,
    build_synthetic_snapshot_ready_event,
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
        if self.path != "/health" and not self._authorized():
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
        self._json(200, {"status": "acknowledged", "receipt_id": receipt_id})

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
        response = self.client.latest_interactions()
        self.assertTrue(response.ok)
        self.assertEqual(FakeMemberHandler.requests[-1][2], "Bearer unit-test-token")
        self.assertNotIn("unit-test-token", json.dumps(self.client.public_config))

    def test_health_does_not_send_authorization(self):
        response = self.client.health()
        self.assertTrue(response.ok)
        self.assertIsNone(FakeMemberHandler.requests[-1][2])

    def test_connection_event_contains_no_person_or_image_fields(self):
        event = build_connection_test_event("connection.test:fixed")
        serialized = json.dumps(event).lower()
        self.assertEqual(event["schema_version"], "lab-handoff-1.1")
        self.assertEqual(event["event_type"], "connection.test")
        self.assertEqual(event["data_classification"], "synthetic")
        self.assertEqual(event["status"], "test")
        self.assertNotIn("test", event)
        for forbidden in ("person_id", "person_name", "face", "position", "image_url", "embedding"):
            self.assertNotIn(forbidden, serialized)

    def test_synthetic_snapshot_ready_matches_round_two_contract(self):
        event = build_synthetic_snapshot_ready_event()
        self.assertEqual(event["event_type"], "snapshot.ready")
        self.assertEqual(event["data_classification"], "synthetic")
        self.assertEqual(event["snapshot"]["camera_id"], "cam_2")
        self.assertEqual(event["snapshot"]["frame_width"], 64)
        self.assertEqual(event["snapshot"]["frame_height"], 36)
        self.assertEqual(event["people"][0]["bbox_format"], "xywh_pixels")
        serialized = json.dumps(event).lower()
        for forbidden in ("embedding", "rtsp://", "authorization", "bearer"):
            self.assertNotIn(forbidden, serialized)

    def test_synthetic_snapshot_frame_is_ready_for_low_rate_analysis(self):
        event = build_synthetic_snapshot_frame_event()
        self.assertEqual(event["event_type"], "snapshot.frame")
        self.assertEqual(event["status"], "test")
        self.assertEqual(event["camera"]["tracks"][0]["track_id"], "track_test_1")
        self.assertEqual(event["camera"]["tracks"][0]["bbox"], [8, 4, 12, 24])

    def test_synthetic_snapshot_frame_can_have_no_tracks(self):
        event = build_synthetic_snapshot_frame_event(
            "snapshot.frame:synthetic-round3:cam_2:000001",
            snapshot_id="synthetic-round3-cam_2-000001",
            include_test_track=False,
        )
        self.assertEqual(event["snapshot_id"], "synthetic-round3-cam_2-000001")
        self.assertEqual(event["camera"]["camera_id"], "cam_2")
        self.assertEqual(event["camera"]["tracks"], [])

    def test_synthetic_snapshot_frame_accepts_a_moving_test_bbox(self):
        event = build_synthetic_snapshot_frame_event(
            "snapshot.frame:synthetic-round4:cam_2:000005",
            snapshot_id="synthetic-round4-cam_2-000005",
            test_bbox=(24, 4, 12, 24),
        )
        self.assertEqual(event["camera"]["tracks"][0]["bbox"], [24, 4, 12, 24])
        self.assertEqual(event["camera"]["tracks"][0]["track_id"], "track_test_1")

    def test_synthetic_snapshot_frame_accepts_two_distinct_crossing_tracks(self):
        event = build_synthetic_snapshot_frame_event(
            "snapshot.frame:synthetic-round5:cam_2:000003",
            snapshot_id="synthetic-round5-cam_2-000003",
            test_bbox=(16, 4, 10, 24),
            second_test_bbox=(22, 4, 10, 24),
        )
        tracks = event["camera"]["tracks"]
        self.assertEqual([track["track_id"] for track in tracks], ["track_test_1", "track_test_2"])
        self.assertEqual([track["local_track_id"] for track in tracks], [1, 2])
        self.assertEqual(tracks[0]["bbox"], [16, 4, 10, 24])
        self.assertEqual(tracks[1]["bbox"], [22, 4, 10, 24])

    def test_synthetic_person_handoff_preserves_global_identity(self):
        event = build_synthetic_person_handoff_event()
        handoff = event["handoff"]
        self.assertEqual(event["event_type"], "person.handoff")
        self.assertEqual(event["data_classification"], "synthetic")
        self.assertEqual(handoff["global_id"], "global_test_handoff_1")
        self.assertEqual(handoff["source_camera_id"], "cam_2")
        self.assertEqual(handoff["target_camera_id"], "cam_entrance")
        self.assertNotEqual(
            handoff["source_local_track_id"], handoff["target_local_track_id"]
        )

    def test_synthetic_snapshot_frame_supports_handoff_camera_and_identity(self):
        event = build_synthetic_snapshot_frame_event(
            "snapshot.frame:synthetic-round6:cam_entrance:000001",
            camera_id="cam_entrance",
            track_id="track_test_target_1",
            global_id="global_test_handoff_1",
            local_track_id=201,
            person_id="person_test_handoff_1",
        )
        track = event["camera"]["tracks"][0]
        self.assertEqual(event["camera"]["camera_id"], "cam_entrance")
        self.assertEqual(track["track_id"], "track_test_target_1")
        self.assertEqual(track["global_id"], "global_test_handoff_1")
        self.assertEqual(track["local_track_id"], 201)

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
        response = wrong.latest_interactions()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.body, {"error": "unauthorized"})

    def test_member_token_rejects_non_ascii_and_whitespace(self):
        with self.assertRaisesRegex(MemberIntegrationConfigError, "ASCII"):
            MemberServiceClient(self.base_url, "错误-token", 1)
        with self.assertRaisesRegex(MemberIntegrationConfigError, "whitespace"):
            MemberServiceClient(self.base_url, "two tokens", 1)

    def test_monitoring_frame_contains_anonymized_tracks_without_embeddings(self):
        event = build_monitoring_snapshot_frame_event(
            event_id="snapshot.frame:live-cam_2-test",
            snapshot_id="live-cam_2-test",
            camera_id="cam_2",
            captured_at="2026-09-17T09:00:00+08:00",
            image_url="http://192.168.31.127:5173/api/handoff/live/live-cam_2-test.jpg",
            frame_width=1280,
            frame_height=720,
            tracks=[
                {
                    "track_id": "person_1",
                    "global_id": "person_1",
                    "local_track_id": 7,
                    "person_id": None,
                    "display_name": None,
                    "bbox": [10, 20, 30, 40],
                    "bbox_format": "xywh_pixels",
                    "confidence": 0.9,
                    "observed_at": "2026-09-17T09:00:00+08:00",
                }
            ],
        )
        self.assertEqual(event["data_classification"], "anonymized_monitoring")
        self.assertEqual(event["status"], "active")
        self.assertEqual(event["camera"]["tracks"][0]["bbox"], [10, 20, 30, 40])
        self.assertNotIn("embedding", json.dumps(event).lower())


if __name__ == "__main__":
    unittest.main()
