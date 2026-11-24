"""Run the visual observer on the lab cluster and publish its heartbeat locally."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from queue import Empty, Queue
import shlex
import subprocess
import sys
import tarfile
import threading
import time

REMOTE = """
import os, subprocess, sys, tarfile, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory(prefix='aims-hazard-live-') as temp:
    root=Path(temp)
    with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as archive:
        for member in archive:
            path=root/member.name
            if not member.isfile() or not path.resolve().is_relative_to(root):
                raise ValueError('unexpected bundle member')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    env=dict(os.environ,PYTHONPATH='/cluster/dataX/xqyi/lab-camera-python')
    sys.exit(subprocess.call([sys.executable,'-u',str(root/'hazard_live_worker.py')],env=env))
"""


def bundle(root):
    config = root / "config/fire_model.json"
    model = json.loads(config.read_text(encoding="utf-8"))["path"]
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name in ("fire_detector.py", "config/fire_model.json", model, "visual_hazard_observer.py",
                     'models/face_detection_yunet_2023mar.onnx'):
            archive.add(root / name, arcname=name)
        archive.add(Path(__file__).with_name("hazard_live_worker.py"), arcname="hazard_live_worker.py")
    return payload.getvalue()


def publish(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    for attempt in range(4):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(.10)


def watch(ssh, payload, path):
    from alert_evidence import AlertEvidence
    from management_notifications import iso

    evidence = AlertEvidence(path.parent / 'alert-evidence', ())
    saved = {}
    python = "/cluster/home/opt/anaconda3/envs/omat25/bin/python"
    command = f"{python} -c {shlex.quote(REMOTE)}"
    child = subprocess.Popen(ssh + [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
    queue = Queue(maxsize=8)

    def read():
        for line in iter(child.stdout.readline, b""):
            queue.put(line)
        queue.put(None)

    try:
        threading.Thread(target=read, daemon=True).start()
        child.stdin.write(payload)
        child.stdin.close()
        while True:
            line = queue.get(timeout=40)
            if line is None:
                raise RuntimeError("cluster observer closed")
            data = json.loads(line)
            producer_time = float(data["observed_at_unix"])
            current = time.time()
            for camera_id, camera in data["cameras"].items():
                for event in camera.get("recent", ()):
                    event["source_at"] = event["at"]
                    event["at"] = current - max(0, producer_time - event["source_at"])
                    snapshot = event.pop("snapshot", None)
                    kind = event.get('kind')
                    if not snapshot or kind not in {'smoke', 'flame'}:
                        continue
                    source = f'{camera_id}/{kind}/{event["source_at"]:.6f}'
                    try:
                        if source not in saved:
                            if len(snapshot['jpeg']) > 4 * 1024 * 1024:
                                continue
                            jpeg = base64.b64decode(snapshot['jpeg'], validate=True)
                            row = {'camera_id': camera_id, 'snapshot_id': hashlib.sha256(source.encode()).hexdigest(),
                                   'source_frame_id': str(snapshot['frame_id']), 'captured_at': iso(event['at']),
                                   'fire_bbox': snapshot['bbox'], 'kind': 'smoke_scene' if kind == 'smoke' else 'fire_scene'}
                            saved[source] = evidence.store_scene_jpeg(row, jpeg, snapshot['width'], snapshot['height'])
                            if len(saved) > 256:
                                saved.pop(next(iter(saved)))
                        event['evidence_id'] = saved[source]
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        print(f'Visual {kind} snapshot unavailable: {type(error).__name__}', flush=True)
            data["observed_at_unix"] = current
            data["event_identity"] = "source_at"
            publish(path, data)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera", type=Path)
    args = parser.parse_args()
    root = args.camera.resolve()
    sys.path.insert(0, str(root / "runtime"))
    from remote_identity_ops import SSH

    path = root / "runtime/visual_hazard_observations.json"
    payload = bundle(root)
    while True:
        try:
            watch(SSH, payload, path)
        except (Empty, OSError, ValueError, RuntimeError, KeyError) as problem:
            print(f"reconnecting after {type(problem).__name__}", file=sys.stderr, flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
