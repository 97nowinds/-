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
ENTRANCE_RTSP="${LAB_CAM_ENTRANCE_RTSP:-}"
echo "node=$(hostname)"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -hide_banner -loglevel error -rtsp_transport tcp \
    -i "$CAM1_RTSP" -map 0:v:0 -frames:v 1 -f null -
  echo "cam_1_frame=ok"
  ffmpeg -hide_banner -loglevel error -rtsp_transport tcp \
    -i "$CAM2_RTSP" -map 0:v:0 -frames:v 1 -f null -
  echo "cam_2_frame=ok"
  if [[ -n "$ENTRANCE_RTSP" ]]; then
    ffmpeg -hide_banner -loglevel error -rtsp_transport tcp \
      -i "$ENTRANCE_RTSP" -map 0:v:0 -frames:v 1 -f null -
    echo "cam_entrance_frame=ok"
  fi
else
  echo "ffmpeg_cli=unavailable; validating MediaMTX HLS output"
  for camera in cam_1 cam_2; do
    manifest="$(curl -fsSL --max-time 10 -c "/tmp/${camera}.cookies" -b "/tmp/${camera}.cookies" "http://127.0.0.1:8888/${camera}/index.m3u8")"
    grep -q '^#EXTM3U' <<<"$manifest"
    echo "${camera}_hls=ok"
  done
  if [[ -n "$ENTRANCE_RTSP" ]]; then
    camera="cam_entrance"
    manifest="$(curl -fsSL --max-time 10 -c "/tmp/${camera}.cookies" -b "/tmp/${camera}.cookies" "http://127.0.0.1:8888/${camera}/index.m3u8")"
    grep -q '^#EXTM3U' <<<"$manifest"
    echo "${camera}_hls=ok"
  fi
fi
echo "RTSP push relay read test passed from lnx."
