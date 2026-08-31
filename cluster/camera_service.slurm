#!/usr/bin/env bash
#SBATCH --job-name=lab-camera
#SBATCH --partition=gpu
#SBATCH --nodelist=lnx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -Eeuo pipefail

if ! type module >/dev/null 2>&1; then
    source /usr/share/lmod/lmod/init/bash
fi
export MODULEPATH="${MODULEPATH:-/cluster/home/opt/modulefiles:/etc/lmod/modules:/usr/share/lmod/lmod/modulefiles}"
module load anaconda

PROJECT_ROOT="${LAB_PROJECT_ROOT:-$PWD}"
WORK_ROOT="${SLURM_TMPDIR:-/tmp}/lab-camera-${SLURM_JOB_ID}"
PYTHON_BIN="${PYTHON_BIN:-/cluster/home/opt/anaconda3/envs/omat25/bin/python}"
PY_ROOT="${LAB_PYTHON_ROOT:-/tmp/lab-camera-python}"
CAM1_RTSP="${LAB_CAM_1_RTSP:?LAB_CAM_1_RTSP is required}"
CAM2_RTSP="${LAB_CAM_2_RTSP:?LAB_CAM_2_RTSP is required}"

echo "node=$(hostname) job=${SLURM_JOB_ID}"
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PYTHON_BIN" - <<'PY'
import os
from urllib.parse import urlsplit

for name in ("LAB_CAM_1_RTSP", "LAB_CAM_2_RTSP"):
    value = os.environ[name]
    parsed = urlsplit(value)
    if parsed.scheme not in ("rtsp", "rtsps") or not parsed.hostname or not parsed.path:
        raise SystemExit(f"Invalid RTSP URL in {name}")
    print(f"{name}=configured host={parsed.hostname} port={parsed.port or (322 if parsed.scheme == 'rtsps' else 554)} path={parsed.path}")
PY

rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"
cp -a "$PROJECT_ROOT/camera" "$WORK_ROOT/camera"
cp -a "$PROJECT_ROOT/person_track" "$WORK_ROOT/person_track"

cd "$WORK_ROOT/camera"
if [[ ! -d "$PY_ROOT" ]]; then
    echo "Prepared Python package root not found: $PY_ROOT" >&2
    exit 2
fi
export PYTHONPATH="$PY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import torch
import onnxruntime as ort

print("torch=", torch.__version__)
print("torch.cuda.is_available=", torch.cuda.is_available())
print("onnxruntime_providers=", ort.get_available_providers())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm allocation")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("ONNX Runtime CUDAExecutionProvider is not available")
PY

export LAB_CAM_1_RTSP="$CAM1_RTSP"
export LAB_CAM_2_RTSP="$CAM2_RTSP"
if [[ -n "${LAB_CAM_ENTRANCE_RTSP:-}" ]]; then
    export LAB_CAM_ENTRANCE_RTSP
fi
export LAB_RTSP_TRANSPORT="${LAB_RTSP_TRANSPORT:-tcp}"
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;${LAB_RTSP_TRANSPORT}|stimeout;5000000|fflags;nobuffer|flags;low_delay|max_delay;0|analyzeduration;0|probesize;32768"
export LAB_FACE_ENGINE="arcface"
export LAB_FACE_DATA_DIR="${LAB_FACE_DATA_DIR:-$PROJECT_ROOT/camera/data}"
mkdir -p "$LAB_FACE_DATA_DIR/known_faces"
export LAB_APP_HOST="0.0.0.0"
export LAB_APP_PORT="${LAB_APP_PORT:-5000}"
export LAB_API_ONLY="${LAB_API_ONLY:-0}"
export LAB_FRONTEND_ORIGIN="${LAB_FRONTEND_ORIGIN:-*}"

"$PYTHON_BIN" - <<'PY'
import os
import time
import cv2

urls = [os.environ["LAB_CAM_1_RTSP"], os.environ["LAB_CAM_2_RTSP"]]
if os.environ.get("LAB_CAM_ENTRANCE_RTSP"):
    urls.append(os.environ["LAB_CAM_ENTRANCE_RTSP"])
for index, url in enumerate(urls, 1):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"RTSP frame check failed for camera {index}")
    print(f"rtsp_camera_{index}=ok shape={frame.shape}")
PY

"$PYTHON_BIN" app.py &
APP_PID=$!
cleanup() { kill "$APP_PID" 2>/dev/null || true; wait "$APP_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
"$PYTHON_BIN" - <<'PY'
import json
import time
import urllib.request

for _ in range(30):
    try:
        with urllib.request.urlopen("http://127.0.0.1:5000/api/health", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("status") == "ok":
                print("api_health=ok", payload)
                break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("/api/health did not become ready")
PY
wait "$APP_PID"
