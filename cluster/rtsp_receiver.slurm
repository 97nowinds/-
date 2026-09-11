#!/usr/bin/env bash
#SBATCH --job-name=lab-rtsp-receiver
#SBATCH --partition=gpu
#SBATCH --nodelist=lnx
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
NODE_DATA_ROOT="${LAB_NODE_DATA_ROOT:-/cluster/dataX/$USER}"
WORK_ROOT="$NODE_DATA_ROOT/lab-rtsp-${SLURM_JOB_ID}"
ARCHIVE="$PROJECT_ROOT/tools/mediamtx/mediamtx_v1.18.2_linux_amd64.tar.gz"

if [[ ! -f "$ARCHIVE" ]]; then
    echo "MediaMTX archive not found: $ARCHIVE" >&2
    exit 2
fi
if ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:)8554$'; then
    echo "TCP 8554 is already in use on $(hostname). Stop the manual MediaMTX process first." >&2
    exit 3
fi

rm -rf "$WORK_ROOT"
mkdir -p "$NODE_DATA_ROOT" "$WORK_ROOT"
tar -xzf "$ARCHIVE" -C "$WORK_ROOT"
cat > "$WORK_ROOT/mediamtx.yml" <<'YAML'
logLevel: info
rtsp: true
rtspAddress: :8554
rtspTransports: [tcp]
hls: true
hlsAddress: :8888
webrtc: false
rtmp: false
srt: false
paths:
  cam_1:
    source: publisher
  cam_2:
    source: publisher
  cam_entrance:
    source: publisher
YAML

cleanup() { rm -rf "$WORK_ROOT"; }
trap cleanup EXIT

echo "node=$(hostname) job=${SLURM_JOB_ID}"
echo "RTSP receiver: rtsp://127.0.0.1:8554/cam_1, /cam_2, and /cam_entrance"
cd "$WORK_ROOT"
exec ./mediamtx mediamtx.yml
