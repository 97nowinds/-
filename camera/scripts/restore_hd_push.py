"""Supervise original main-stream copy using an existing protected relay config."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--remote-base', default='rtsp://127.0.0.1:18554')
    args = parser.parse_args()
    config = json.loads(args.source_config.read_text(encoding='utf-8-sig'))
    processes, misses = {}, {}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        while True:
            for camera in ('cam_1', 'cam_2'):
                process = processes.get(camera)
                if process is None or process.poll() is not None:
                    source = config['paths'][camera + '_hd']['source']
                    processes[camera] = subprocess.Popen([ffmpeg, '-hide_banner', '-nostdin',
                        '-loglevel', 'error', '-rtsp_transport', 'tcp', '-i', source,
                        '-map', '0:v:0', '-an', '-c:v', 'copy', '-f', 'rtsp', '-rtsp_transport',
                        'tcp', args.remote_base.rstrip('/') + '/' + camera],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    misses[camera] = 0
                    print(camera, 'copy publisher started', flush=True)
                    continue
                # Frame copying has no consumer queue. Verify the publisher TCP
                # connection so a disconnected, still-alive FFmpeg is restarted.
                import psutil
                from urllib.parse import urlsplit
                try:
                    connected = any(row.status == psutil.CONN_ESTABLISHED and row.raddr
                        and row.raddr.port == urlsplit(args.remote_base).port
                        for row in psutil.Process(process.pid).net_connections('tcp'))
                except psutil.NoSuchProcess:
                    continue
                misses[camera] = 0 if connected else misses[camera] + 1
                if misses[camera] >= 4:
                    process.terminate()
                    process.wait(timeout=5)
                    print(camera, 'publisher disconnected; restarting', flush=True)
            time.sleep(5)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()


if __name__ == '__main__':
    main()
