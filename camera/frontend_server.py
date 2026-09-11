import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "index.html"
ANNOTATE_TEMPLATE = ROOT / "templates" / "annotate.html"
FACES_TEMPLATE = ROOT / "templates" / "faces.html"
STATIC = ROOT / "static"


class FrontendHandler(SimpleHTTPRequestHandler):
    server_version = "LabFrontend/1.0"

    def _send_html(self, path):
        html = path.read_text(encoding="utf-8")
        api_base = self.server.api_base.replace("\\", "\\\\").replace('"', '\\"')
        bootstrap = f'<script>window.__LAB_API_BASE__ = "{api_base}";</script>'
        html = html.replace("</head>", f"  {bootstrap}\n  </head>", 1)
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/":
            self._send_html(TEMPLATE)
            return
        if route == "/annotate":
            self._send_html(ANNOTATE_TEMPLATE)
            return
        if route == "/faces":
            self._send_html(FACES_TEMPLATE)
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

    def _send_file(self, path, content_type):
        self._send_bytes(path.read_bytes(), content_type)

    def _send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
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
    server.serve_forever()


if __name__ == "__main__":
    main()
