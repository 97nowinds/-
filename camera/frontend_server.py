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

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/":
            self._send_file(TEMPLATE, "text/html; charset=utf-8")
            return
        if route == "/annotate":
            self._send_file(ANNOTATE_TEMPLATE, "text/html; charset=utf-8")
            return
        if route == "/faces":
            self._send_file(FACES_TEMPLATE, "text/html; charset=utf-8")
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
        data = path.read_bytes()
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
    # The static bundle reads this value from the query string. Keeping the
    # server free of API proxy logic makes the frontend/backend boundary clear.
    print(f"Frontend: http://127.0.0.1:{args.port}/?api={args.api}")
    ThreadingHTTPServer((args.host, args.port), FrontendHandler).serve_forever()


if __name__ == "__main__":
    main()
