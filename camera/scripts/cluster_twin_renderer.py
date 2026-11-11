"""Loopback-only frame renderer for the dashboard's Three.js scene."""
import argparse
import base64
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import websocket


ORIGINS = {"http://127.0.0.1:5173", "http://localhost:5173"}


class Browser:
    def __init__(self, chrome, scene):
        self.profile = tempfile.TemporaryDirectory(prefix="lab-twin-")
        self.process = subprocess.Popen(
            [str(chrome), "--headless", "--no-first-run", "--disable-background-networking",
             "--disable-dev-shm-usage", "--enable-gpu", "--disable-software-rasterizer",
             "--use-gl=angle", "--use-angle=vulkan", "--disable-vulkan-surface",
             "--remote-debugging-port=0", "--user-data-dir=" + self.profile.name,
             "--window-size=1000,640", scene.as_uri()],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 15
            port_file = Path(self.profile.name) / "DevToolsActivePort"
            while not port_file.exists() and time.monotonic() < deadline and self.process.poll() is None:
                time.sleep(.1)
            port = int(port_file.read_text().splitlines()[0])
            with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=3) as response:
                page = next(tab for tab in json.load(response) if tab["type"] == "page")
            self.socket = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=8,
                                                      suppress_origin=True)
            self.sequence = 0
            self.command("Emulation.setDeviceMetricsOverride", width=1000, height=640,
                         deviceScaleFactor=1, mobile=False)
            while time.monotonic() < deadline:
                self.gpu = self.evaluate("window.labRendererGPU")
                if self.gpu:
                    break
                time.sleep(.1)
            if not self.gpu or "NVIDIA" not in self.gpu:
                raise RuntimeError("NVIDIA hardware WebGL unavailable")
        except Exception:
            self.close()
            raise

    def command(self, method, **params):
        self.sequence += 1
        self.socket.send(json.dumps({"id": self.sequence, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self.sequence:
                if "error" in message:
                    raise RuntimeError(method)
                return message["result"]

    def evaluate(self, expression):
        result = self.command("Runtime.evaluate", expression=expression, returnByValue=True,
                              awaitPromise=True)
        if "exceptionDetails" in result:
            raise RuntimeError("Scene render failed")
        return result["result"].get("value")

    def render(self, data):
        bounds = self.evaluate("window.renderLabFrame(" + json.dumps(data, ensure_ascii=True) + ")")
        image = base64.b64decode(self.command("Page.captureScreenshot", format="jpeg", quality=78)["data"])
        return image, json.dumps(bounds, ensure_ascii=True, separators=(",", ":"))

    def close(self):
        if hasattr(self, "socket"):
            self.socket.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.profile.cleanup()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, data=b"", content_type="application/json", bounds=None):
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if bounds is not None:
            self.send_header("X-Instrument-Bounds", bounds)
            self.send_header("Access-Control-Expose-Headers", "X-Instrument-Bounds")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        if self.path != "/frame" or self.headers.get("Origin") not in ORIGINS:
            return self.reply(403)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Vary", "Origin")
        self.end_headers()

    def do_GET(self):
        if self.path != "/health":
            return self.reply(404)
        self.reply(200, json.dumps({"gpu": self.server.browser.gpu}).encode())

    def do_POST(self):
        if self.path != "/frame" or self.headers.get("Origin") not in ORIGINS:
            return self.reply(403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 256 * 1024:
                return self.reply(413)
            self.connection.settimeout(3)
            data = json.loads(self.rfile.read(length))
            map_data, view = data["map"], data["view"]
            for key in ("width_m", "height_m"):
                if not 0 < float(map_data[key]) <= 100:
                    return self.reply(400)
            if not all(math.isfinite(view[k]) for k in ("yaw", "pitch", "zoom")):
                return self.reply(400)
            if abs(view["yaw"]) > 360 or not .18 <= view["pitch"] <= 1.48 or not .16 <= view["zoom"] <= 2.2:
                return self.reply(400)
            if any(not isinstance(rows, list) or len(rows) > limit for rows, limit in
                   ((map_data["people"], 80), (map_data["fixtures"], 60), (data["equipment"], 100),
                    (data["trails"], 80))):
                return self.reply(400)
        except (ValueError, KeyError, TypeError, OSError):
            return self.reply(400)
        if not self.server.render_lock.acquire(blocking=False):
            return self.reply(503)
        try:
            image, bounds = self.server.browser.render(data)
        except (OSError, websocket.WebSocketException):
            os._exit(1)
        except RuntimeError:
            return self.reply(502)
        finally:
            self.server.render_lock.release()
        self.reply(200, image, "image/jpeg", bounds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--port", type=int, default=25182)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.render_lock = threading.Lock()
        server.browser = Browser(args.chrome, args.scene.resolve())
        print(server.browser.gpu, flush=True)
        try:
            server.serve_forever()
        finally:
            server.browser.close()


if __name__ == "__main__":
    main()
