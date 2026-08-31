#!/usr/bin/env bash
# GPU and RTSP preflight only. This job does not start Python AI or Flask.
#SBATCH --job-name=lab-gpu-preflight
#SBATCH --partition=gpu
#SBATCH --nodelist=lnx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -Eeuo pipefail

CAM1_RTSP="${LAB_CAM_1_RTSP:?LAB_CAM_1_RTSP is required}"
CAM2_RTSP="${LAB_CAM_2_RTSP:?LAB_CAM_2_RTSP is required}"
ENTRANCE_RTSP="${LAB_CAM_ENTRANCE_RTSP:-}"

echo "node=$(hostname) job=${SLURM_JOB_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

command -v python3 >/dev/null
python3 --version

RTSP_VALUES=("$CAM1_RTSP" "$CAM2_RTSP")
if [[ -n "$ENTRANCE_RTSP" ]]; then
    RTSP_VALUES+=("$ENTRANCE_RTSP")
fi
for value in "${RTSP_VALUES[@]}"; do
    case "$value" in
        rtsp://127.0.0.1:8554/cam_1|rtsp://127.0.0.1:8554/cam_2|rtsp://127.0.0.1:8554/cam_entrance) ;;
        *) echo "Unexpected local RTSP endpoint: $value" >&2; exit 2 ;;
    esac
done

# MediaMTX creates the HLS manifest only after a publisher has supplied frames.
CAMERAS=(cam_1 cam_2)
if [[ -n "$ENTRANCE_RTSP" ]]; then
    CAMERAS+=(cam_entrance)
fi
for camera in "${CAMERAS[@]}"; do
    manifest="$(curl -fsSL --max-time 10 "http://127.0.0.1:8888/${camera}/index.m3u8")"
    grep -q '^#EXTM3U' <<<"$manifest"
    echo "${camera}_hls=ok"
done

echo "gpu_preflight=ok"
echo "No AI inference or Flask service was started."
