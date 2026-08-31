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
        self.assertNotIn("/api/", source)
        self.assertNotIn("VideoCapture", source)

    def test_annotation_surface_exists(self):
        template = (ROOT / "templates" / "annotate.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "annotate.js").read_text(encoding="utf-8")
        self.assertIn('id="annotationCanvas"', template)
        self.assertIn("/api/annotation/config", script)
        self.assertIn("/api/annotation/save", script)

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


if __name__ == "__main__":
    unittest.main()
