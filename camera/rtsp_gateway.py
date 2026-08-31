"""Low-CPU RTSP copy gateway for the two indoor Hikvision cameras."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass

import imageio_ffmpeg


RESTART_SECONDS = 3


@dataclass(frozen=True)
class Relay:
    camera_id: str
    source: str
    destination: str


def _redacted_endpoint(endpoint: str) -> str:
    """Keep logs free of credentials while retaining a useful endpoint label."""
    if "://" not in endpoint:
        return endpoint
    scheme, remainder = endpoint.split("://", 1)
    host = remainder.rsplit("@", 1)[-1].split("/", 1)[0]
    return f"{scheme}://{host}"


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _relay_process(relay: Relay, ffmpeg: str) -> subprocess.Popen:
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-rtsp_transport",
        os.environ.get("LAB_RTSP_TRANSPORT", "tcp"),
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-i",
        relay.source,
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        relay.destination,
    ]
    print(
        f"Starting {relay.camera_id}: {_redacted_endpoint(relay.source)} -> "
        f"{_redacted_endpoint(relay.destination)}",
        flush=True,
    )
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=None)


def main() -> int:
    relays = [
        Relay("cam_1", _required("LAB_CAM_1_RTSP"), _required("LAB_REMOTE_CAM_1_RTSP")),
        Relay("cam_2", _required("LAB_CAM_2_RTSP"), _required("LAB_REMOTE_CAM_2_RTSP")),
    ]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print("RTSP gateway mode: stream copy only; no AI inference or local recording", flush=True)
    print(f"FFmpeg: {ffmpeg}", flush=True)
    processes: dict[str, subprocess.Popen] = {}
    stopping = False

    def stop_all(*_args):
        nonlocal stopping
        stopping = True
        for process in processes.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    try:
        while not stopping:
            for relay in relays:
                process = processes.get(relay.camera_id)
                if process is None or process.poll() is not None:
                    if process is not None:
                        print(
                            f"{relay.camera_id} relay exited with code {process.returncode}; "
                            f"retrying in {RESTART_SECONDS}s",
                            flush=True,
                        )
                    processes[relay.camera_id] = _relay_process(relay, ffmpeg)
            time.sleep(RESTART_SECONDS)
    finally:
        stop_all()
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
