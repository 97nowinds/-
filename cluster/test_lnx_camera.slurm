#!/usr/bin/env bash
#SBATCH --job-name=lab-camera-check
#SBATCH --partition=gpu
#SBATCH --nodelist=lnx
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=lab-camera-check-%j.out
#SBATCH --error=lab-camera-check-%j.err

set -Eeuo pipefail
CAM1_RTSP="${LAB_CAM_1_RTSP:?LAB_CAM_1_RTSP is required}"
CAM2_RTSP="${LAB_CAM_2_RTSP:?LAB_CAM_2_RTSP is required}"
echo "node=$(hostname)"
command -v ffmpeg >/dev/null
ffmpeg -hide_banner -loglevel error -rtsp_transport tcp \
  -i "$CAM1_RTSP" -map 0:v:0 -frames:v 1 -f null -
echo "cam_1_frame=ok"
ffmpeg -hide_banner -loglevel error -rtsp_transport tcp \
  -i "$CAM2_RTSP" -map 0:v:0 -frames:v 1 -f null -
echo "cam_2_frame=ok"
echo "RTSP push relay read test passed from lnx."
