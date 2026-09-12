import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendSeparationTests(unittest.TestCase):
    def test_frontend_server_only_serves_static_surface(self):
        source = (ROOT / "frontend_server.py").read_text(encoding="utf-8")
        self.assertIn("class FrontendHandler", source)
        self.assertIn("/static/", source)
        self.assertIn("/annotate", source)
        self.assertIn("/faces", source)
        self.assertIn("/recordings", source)
        self.assertNotIn("/api/", source)
        self.assertNotIn("VideoCapture", source)

    def test_annotation_surface_exists(self):
        template = (ROOT / "templates" / "annotate.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "annotate.js").read_text(encoding="utf-8")
        self.assertIn('id="annotationCanvas"', template)
        self.assertIn("/api/annotation/config", script)
        self.assertIn("/api/annotation/save", script)

    def test_main_aisle_uses_adjustable_parallelogram(self):
        template = (ROOT / "templates" / "annotate.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "annotate.js").read_text(encoding="utf-8")
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("主通道（平行四边形）", template)
        self.assertIn("drawPolygonRegion", script)
        self.assertIn("moveParallelogramHandle", script)
        self.assertIn('region_type == "main_aisle" and "points" in region', app_source)
        self.assertIn("must form a parallelogram", app_source)

    def test_face_registration_surface_exists(self):
        template = (ROOT / "templates" / "faces.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "faces.js").read_text(encoding="utf-8")
        self.assertIn('id="registrationForm"', template)
        self.assertIn("/api/face-registration/start", script)
        self.assertIn("/api/face-registration/capture", script)
        self.assertIn("/api/face-registration/preview", script)

    def test_frontend_calls_configured_backend(self):
        source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("__LAB_API_BASE__", source)
        self.assertIn("${API_BASE}/api/state", source)
        self.assertIn("${API_BASE}/video/", source)

    def test_frontend_server_injects_configured_backend(self):
        source = (ROOT / "frontend_server.py").read_text(encoding="utf-8")
        self.assertIn("window.__LAB_API_BASE__", source)
        self.assertIn("server.api_base = args.api", source)

    def test_monitoring_page_exposes_fixed_calibration_entry(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="calibrationLink"', template)
        self.assertIn('@app.route("/annotate")', app_source)
        self.assertIn("calibrationLink.href", script)

    def test_secondary_and_side_aisle_names_are_distinct(self):
        template = (ROOT / "templates" / "annotate.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "annotate.js").read_text(encoding="utf-8")
        monitor = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        floor_map = (ROOT / "config" / "floor_map.json").read_text(encoding="utf-8")
        self.assertIn("副通道", template)
        self.assertIn('value="rear_service">侧向通道（框选）', template)
        self.assertIn('secondary_aisle: "副通道"', script)
        self.assertIn('zone.id === "rear_service" ? "侧向通道"', script)
        self.assertIn('item?.id === "secondary_aisle"', monitor)
        self.assertIn('item?.id === "rear_service"', monitor)
        self.assertIn('"name": "副通道"', floor_map)
        self.assertIn('"name": "侧向通道"', floor_map)
        self.assertNotIn("后端操作区", floor_map)
        self.assertNotIn("后端仪器", floor_map)
        self.assertIn('fixture.id === "rear_console"', script)

    def test_top_fixture_and_corridor_keep_distinct_names(self):
        floor_map = (ROOT / "config" / "floor_map.json").read_text(encoding="utf-8")

        self.assertIn('"id": "wall_bench_zone"', floor_map)
        self.assertIn('"name": "沿墙实验区"', floor_map)
        self.assertIn('"id": "secondary_aisle"', floor_map)
        self.assertIn('"name": "副通道"', floor_map)

    def test_entrance_uses_tracking_pipeline_with_stronger_detection_and_face_checks(self):
        cameras = json.loads((ROOT / "config" / "cameras.json").read_text(encoding="utf-8"))
        entrance = next(camera for camera in cameras if camera["id"] == "cam_entrance")

        self.assertTrue(entrance["face_recognition"])
        self.assertLessEqual(entrance["face_interval_seconds"], 0.2)
        self.assertLess(entrance["yolo_confidence"], 0.45)
        self.assertLess(entrance["yolo_frame_stride"], 5)
        self.assertGreater(entrance["track_hold_seconds"], 0.8)

    def test_cam1_uses_replay_validated_small_person_detection_settings(self):
        cameras = json.loads((ROOT / "config" / "cameras.json").read_text(encoding="utf-8"))
        cam1 = next(camera for camera in cameras if camera["id"] == "cam_1")

        self.assertLessEqual(cam1["yolo_confidence"], 0.3)
        self.assertLessEqual(cam1["yolo_frame_stride"], 3)
        self.assertGreaterEqual(cam1["yolo_image_size"], 768)

    def test_recorded_route_transitions_allow_validated_bidirectional_overlap(self):
        floor_map = json.loads((ROOT / "config" / "floor_map.json").read_text(encoding="utf-8"))
        transitions = floor_map["camera_transitions"]

        self.assertEqual(len(transitions), 3)
        self.assertTrue(all(item["bidirectional"] for item in transitions))
        self.assertTrue(all(item["simultaneous_overlap_validated"] for item in transitions))

    def test_state_reports_arcface_gallery_completion_counts(self):
        backend = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn('"arcface_gallery_required_features"', backend)
        self.assertIn('"arcface_gallery_feature_counts"', backend)

    def test_monitoring_page_has_recording_dialog_and_api_controls(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="recordingControlButton"', template)
        self.assertIn('id="recordingDialog"', template)
        self.assertIn('id="recordingSubjectId"', template)
        self.assertIn('id="recordingStopButton"', template)
        self.assertIn('recordingRequest("/api/recording/start"', script)
        self.assertIn('recordingRequest("/api/recording/stop"', script)
        self.assertIn("renderRecording(state.recording, state.cameras)", script)

    def test_route_recording_review_surface_exists(self):
        template = (ROOT / "templates" / "recordings.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "recordings.js").read_text(encoding="utf-8")
        monitor = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        backend = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn('id="recordingReviewLink"', monitor)
        self.assertIn('id="recordingSessionSelect"', template)
        self.assertIn('id="recordedVideoGrid"', template)
        self.assertIn('id="captureSourceButton"', template)
        self.assertIn('id="captureTargetButton"', template)
        self.assertIn('api("/api/recordings")', script)
        self.assertIn('sessionPath("/handoffs")', script)
        self.assertIn('data-step-frames="-1"', script)
        self.assertIn('data-step-frames="1"', script)
        self.assertIn("recorded-frame-review", script)
        self.assertIn('sessionPath(`/frame/', script)
        self.assertIn('"/api/recordings/<subject_id>/<session_id>/handoffs"', backend)


if __name__ == "__main__":
    unittest.main()
