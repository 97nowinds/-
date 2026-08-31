#!/usr/bin/env bash
# Check the shared lnx Python environment. This job does not start the app.
#SBATCH --job-name=lab-env-check
#SBATCH --partition=gpu
#SBATCH --nodelist=lnx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -Eeuo pipefail

module load anaconda
echo "node=$(hostname) job=${SLURM_JOB_ID}"
echo "python=$(command -v python)"
python --version
python -m pip --version
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python - <<'PY'
import importlib

packages = (
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "onnxruntime",
    "flask",
    "ultralytics",
    "insightface",
)
failed = []
for name in packages:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "available")
        print(f"{name}={version}")
    except Exception as error:
        failed.append(f"{name}: {error}")

torch = importlib.import_module("torch")
print(f"torch_cuda={torch.cuda.is_available()}")
ort = None
try:
    ort = importlib.import_module("onnxruntime")
    print(f"onnxruntime_providers={ort.get_available_providers()}")
except Exception as error:
    print(f"onnxruntime_runtime_error={error}")
if failed:
    raise SystemExit("Missing or unusable packages:\n" + "\n".join(failed))
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the allocated CUDA device")
if ort is None or "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("ONNX Runtime CUDAExecutionProvider is unavailable")
PY

echo "lnx_environment=ok"
echo "No AI inference or Flask service was started."
