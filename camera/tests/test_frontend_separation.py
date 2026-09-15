import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendSeparationTests(unittest.TestCase):
    def test_frontend_server_serves_static_surface_and_local_hardware_bridge(self):
        source = (ROOT / "frontend_server.py").read_text(encoding="utf-8")
        self.assertIn("class FrontendHandler", source)
        self.assertIn("/static/", source)
        self.assertIn("/annotate", source)
        self.assertIn("/faces", source)
        self.assertIn("/recordings", source)
        self.assertIn('route == "/api/environment"', source)
        self.assertNotIn("VideoCapture", source)

    def test_annotation_surface_exists(self):
        template = (ROOT / "templates" / "annotate.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "annotate.js").read_text(encoding="utf-8")
        self.assertIn('id="annotationCanvas"', template)
        self.assertIn("/api/annotation/config", script)
        self.assertIn("/api/annotation/save", script)
        self.assertIn("completePoints(camera.image_points)", script)
        self.assertIn("calibration_points_changed: state.referenceDirty", script)
        self.assertIn("base_revision: state.revision", script)

    def test_main_aisle_uses_adjustable_parallelogram(self):
        template = (ROOT / "templates" / "annotate.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "annotate.js").read_text(encoding="utf-8")
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("主通道（平行四边形）", template)
        self.assertIn("drawPolygonRegion", script)
        self.assertIn("moveParallelogramHandle", script)
        self.assertIn('if "points" in region', app_source)
        self.assertIn('if region_type == "main_aisle"', app_source)
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

    def test_provisional_tracks_are_visible_but_excluded_from_people_count(self):
        source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("floorMap.counted_people", source)
        self.assertIn('person.counted === false', source)
        self.assertIn("待确认（不计数）", source)
        self.assertIn("临时候选 · 不计数", source)
        self.assertIn("跨摄疑似重复 · 不计数", source)
        self.assertNotIn("目标 ${match[1]}（未注册）", source)

    def test_monitoring_page_exposes_hardware_environment_state(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        backend = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn('id="environmentStatus"', template)
        self.assertIn('id="temperatureValue"', template)
        self.assertIn('id="humidityValue"', template)
        self.assertIn('id="flameValue"', template)
        self.assertIn("renderEnvironment(state.environment)", script)
        self.assertIn("__LAB_HARDWARE_URL__", script)
        self.assertIn('"environment": self.environment.status()', backend)
        self.assertIn('@app.route("/api/environment/ingest", methods=["POST"])', backend)

    def test_frontend_server_injects_configured_backend(self):
        source = (ROOT / "frontend_server.py").read_text(encoding="utf-8")
        self.assertIn("window.__LAB_API_BASE__", source)
        self.assertIn("window.__LAB_HARDWARE_URL__", source)
        self.assertIn("server.api_base = args.api", source)

    def test_cluster_service_uses_persistent_floor_map_config(self):
        service = (ROOT.parent / "cluster" / "camera_service.slurm").read_text(encoding="utf-8")

        self.assertIn('export LAB_FLOOR_MAP_PATH="${LAB_FLOOR_MAP_PATH:-$PROJECT_ROOT/camera/config/floor_map.json}"', service)
        self.assertIn('export LAB_MTMC_EVENT_PATH="${LAB_MTMC_EVENT_PATH:-$PROJECT_ROOT/camera/runtime/mtmc_events.jsonl}"', service)

    def test_rtsp_push_watchdog_detects_a_stuck_publisher(self):
        script = (ROOT / "scripts" / "start_rtsp_push.ps1").read_text(encoding="utf-8")

        self.assertIn("function Test-PublisherConnection", script)
        self.assertIn("alive but no longer publishing; forcing restart", script)
        self.assertIn("Get-NetTCPConnection -OwningProcess", script)

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

    def test_cam2_detection_rate_is_sufficient_for_main_aisle_handoffs(self):
        cameras = json.loads((ROOT / "config" / "cameras.json").read_text(encoding="utf-8"))
        cam2 = next(camera for camera in cameras if camera["id"] == "cam_2")

        self.assertLessEqual(cam2["yolo_frame_stride"], 3)

    def test_roi_filter_runs_before_short_track_hold_decision(self):
        backend = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertLess(
            backend.index("unfiltered_track_count = len(tracks)"),
            backend.index("hold_cached_tracks = bool("),
        )

    def test_approximate_map_only_enables_observed_entrance_cam2_overlap(self):
        floor_map = json.loads((ROOT / "config" / "floor_map.json").read_text(encoding="utf-8"))
        transitions = floor_map["camera_transitions"]

        self.assertEqual(floor_map["calibration"]["status"], "approximate")
        self.assertEqual(len(transitions), 3)
        self.assertTrue(all(item["bidirectional"] for item in transitions))
        validated = [
            item for item in transitions if item["simultaneous_overlap_validated"]
        ]
        self.assertEqual(len(validated), 1)
        self.assertEqual(
            {validated[0]["from"], validated[0]["to"]},
            {"cam_entrance", "cam_2"},
        )
        self.assertEqual(validated[0]["simultaneous_reid_threshold"], 0.95)

    def test_state_reports_arcface_gallery_completion_counts(self):
        backend = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn('"arcface_gallery_required_features"', backend)
        self.assertIn('"arcface_gallery_feature_counts"', backend)

    def test_map_trail_stays_continuous_during_short_localization_hold(self):
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("floorTrailLastSeen", script)
        self.assertIn("now - lastSeenAt > 4500", script)
        self.assertNotIn("last.source !== person.position_source_camera", script)
        self.assertIn("person.position_estimated", script)
        self.assertIn("定位保持", script)
        self.assertIn("breakBefore", script)

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
