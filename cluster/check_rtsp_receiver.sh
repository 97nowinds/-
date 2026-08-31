#!/usr/bin/env bash
set -Eeuo pipefail

echo "node=$(hostname)"
ss -lnt | grep -E ':(8554|8888)[[:space:]]'
for camera in cam_1 cam_2 cam_entrance; do
    manifest="$(curl -fsSL --max-time 10 -c "/tmp/${camera}.cookies" -b "/tmp/${camera}.cookies" "http://127.0.0.1:8888/${camera}/index.m3u8")"
    grep -q '^#EXTM3U' <<<"$manifest"
    echo "${camera}_hls=ok"
done
