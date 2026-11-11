import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


def run_cluster(root, job, scene, log):
    config_path = (root / 'cluster_connection.txt').read_text(encoding='utf-8-sig').strip()
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    key = str(Path(config['ssh_identity']).expanduser())
    ssh = str(Path(os.environ['WINDIR']) / 'System32/OpenSSH/ssh.exe')
    proxy = f'"{ssh}" -i "{key}" -o BatchMode=yes -o ConnectTimeout=8 -W %h:%p {config["ssh_jump"]}'
    runtime = '/cluster/home/opt/anaconda3/envs/omat25/bin/python'
    worker = '/cluster/dataX/xqyi/labsuite-20261110/wizard_live_worker.py'
    command = [ssh, '-T', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
               '-o', 'ProxyCommand=' + proxy, config['ssh_target'], runtime, worker, job, scene]
    run = root / 'runs' / job
    with tempfile.TemporaryDirectory(dir=run) as work:
        work = Path(work)
        with tarfile.open(work / 'input.tar', 'w') as archive:
            archive.add(root.parent / '仪器资料库/data', arcname='library/data')
            archive.add(run / 'request.json', arcname=f'run/{job}/request.json')
            if scene.startswith('capture_'):
                capture = scene[8:]
                archive.add(root / 'captures' / capture, arcname=f'capture/{capture}')
        with (work / 'input.tar').open('rb') as source, (work / 'output.tar').open('wb') as output:
            code = subprocess.run(command, stdin=source, stdout=output, stderr=log, timeout=900,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).returncode
        if not (work / 'output.tar').stat().st_size:
            raise RuntimeError('集群识别服务未返回结果，请查看任务日志后重试。')
        with tarfile.open(work / 'output.tar') as archive:
            archive.extractall(work / 'result', filter='data')
        files = sorted((work / 'result').iterdir(), key=lambda path: path.name == 'status.json')
        for path in files:
            if path.name == 'request.json':
                raise ValueError('Cluster response included request.json')
            os.replace(path, run / path.name)
    return code
