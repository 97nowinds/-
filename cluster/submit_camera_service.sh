#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${LAB_CAM_1_RTSP:-}" || -z "${LAB_CAM_2_RTSP:-}" ]]; then
    echo "Set LAB_CAM_1_RTSP and LAB_CAM_2_RTSP to the administrator-provided RTSP ingest URLs." >&2
    exit 2
fi

CHECK_JOB="$(sbatch --parsable --export="ALL,LAB_CAM_1_RTSP=${LAB_CAM_1_RTSP},LAB_CAM_2_RTSP=${LAB_CAM_2_RTSP}" cluster/test_lnx_camera.slurm)"
echo "preflight_job=${CHECK_JOB}"
echo "RTSP-only preflight submitted. No GPU or AI service was started."
echo "Inspect it with: squeue -j ${CHECK_JOB}; cat lab-camera-check-${CHECK_JOB}.out"

if [[ "${1:-}" != "--start-service" ]]; then
    exit 0
fi

echo "Submitting camera service to lnx..."
sbatch --export="ALL,LAB_CAM_1_RTSP=${LAB_CAM_1_RTSP},LAB_CAM_2_RTSP=${LAB_CAM_2_RTSP}" cluster/camera_service.slurm
