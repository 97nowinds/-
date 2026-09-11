#!/usr/bin/env bash
# Build the isolated project environment on lnx local disk.
#SBATCH --job-name=lab-env-install
#SBATCH --partition=gpu
#SBATCH --nodelist=lnx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -Eeuo pipefail

if ! type module >/dev/null 2>&1; then
    source /usr/share/lmod/lmod/init/bash
fi
export MODULEPATH="${MODULEPATH:-/cluster/home/opt/modulefiles:/etc/lmod/modules:/usr/share/lmod/lmod/modulefiles}"
module load anaconda
PYTHON_BIN="/cluster/home/opt/anaconda3/envs/omat25/bin/python"
NODE_DATA_ROOT="${LAB_NODE_DATA_ROOT:-/cluster/dataX/$USER}"
PY_ROOT="${LAB_PYTHON_ROOT:-$NODE_DATA_ROOT/lab-camera-python}"
WHEELHOUSE="${LAB_WHEELHOUSE:-$HOME/lab-safety-wheels}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

echo "node=$(hostname) job=${SLURM_JOB_ID}"
echo "python_bin=${PYTHON_BIN}"
echo "python_root=${PY_ROOT}"
echo "wheelhouse=${WHEELHOUSE}"
df -h /tmp

rm -rf "$PY_ROOT"
mkdir -p "$NODE_DATA_ROOT" "$PY_ROOT"
if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "Offline wheelhouse not found: $WHEELHOUSE" >&2
    exit 2
fi
# PyTorch 2.8.0+cu128 is supplied by the cluster environment. Install every
# other package from the uploaded Linux wheelhouse without contacting PyPI.
"$PYTHON_BIN" -m pip install --no-index --find-links "$WHEELHOUSE" \
    --target "$PY_ROOT" --no-cache-dir --upgrade --no-deps "$WHEELHOUSE"/*.whl

export PYTHONPATH="$PY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - <<'PY'
import cv2
import flask
import insightface
import numpy
import onnxruntime as ort
import torch
import torchvision
import ultralytics

print("python_environment=ok")
print("torch=", torch.__version__)
print("torch_cuda=", torch.cuda.is_available())
print("torchvision=", torchvision.__version__)
print("opencv=", cv2.__version__)
print("numpy=", numpy.__version__)
print("onnxruntime=", ort.__version__)
print("onnxruntime_providers=", ort.get_available_providers())
print("ultralytics=", ultralytics.__version__)
print("insightface=", insightface.__version__)

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("ONNX Runtime CUDAExecutionProvider is unavailable")
PY

echo "lnx_environment_install=ok"
echo "Environment is on lnx local disk and no AI service was started."
