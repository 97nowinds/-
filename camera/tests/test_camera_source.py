import unittest

from camera_source import resolve_camera_source


class CameraSourceTests(unittest.TestCase):
    def test_rtsp_credentials_are_not_exposed_in_display_value(self):
        result = resolve_camera_source(
            {"source": "rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/102"}
        )

        self.assertEqual(result["source_type"], "rtsp")
        self.assertEqual(result["source_display"], "rtsp://192.168.1.64:554/Streaming/Channels/102")
        self.assertNotIn("secret", result["source_display"])

    def test_reads_rtsp_url_from_environment_variable(self):
        result = resolve_camera_source(
            {"source_env": "LAB_CAM_1_RTSP"},
            environ={"LAB_CAM_1_RTSP": "rtsp://operator:pw@10.0.0.8:554/Streaming/Channels/101"},
        )

        self.assertEqual(result["source_type"], "rtsp")
        self.assertEqual(result["source"], "rtsp://operator:pw@10.0.0.8:554/Streaming/Channels/101")

    def test_reports_missing_rtsp_environment_variable(self):
        result = resolve_camera_source({"source_env": "LAB_CAM_1_RTSP"}, environ={})

        self.assertEqual(result["source_type"], "unconfigured")
        self.assertIn("LAB_CAM_1_RTSP", result["configuration_error"])


if __name__ == "__main__":
    unittest.main()
