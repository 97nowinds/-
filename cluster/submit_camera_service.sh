#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${LAB_CAM_1_RTSP:-}" || -z "${LAB_CAM_2_RTSP:-}" ]]; then
    echo "Set LAB_CAM_1_RTSP and LAB_CAM_2_RTSP to the administrator-provided RTSP ingest URLs." >&2
    exit 2
fi

CAMERA_EXPORTS="ALL,LAB_CAM_1_RTSP=${LAB_CAM_1_RTSP},LAB_CAM_2_RTSP=${LAB_CAM_2_RTSP}"
if [[ -n "${LAB_CAM_ENTRANCE_RTSP:-}" ]]; then
    CAMERA_EXPORTS+=",LAB_CAM_ENTRANCE_RTSP=${LAB_CAM_ENTRANCE_RTSP}"
fi

CHECK_JOB="$(sbatch --parsable --export="$CAMERA_EXPORTS" cluster/test_lnx_camera.slurm)"
echo "preflight_job=${CHECK_JOB}"
echo "RTSP-only preflight submitted. No GPU or AI service was started."
echo "Inspect it with: squeue -j ${CHECK_JOB}; cat lab-camera-check-${CHECK_JOB}.out"

if [[ "${1:-}" != "--start-service" ]]; then
    exit 0
fi

echo "Submitting camera service to lnx..."
sbatch --export="$CAMERA_EXPORTS" cluster/camera_service.slurm
