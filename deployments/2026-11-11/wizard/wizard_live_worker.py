import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

base = Path('/cluster/dataX/xqyi/labsuite-20261110')
source = base / 'wizard-revision27-final' / '产品化'
job, scene = sys.argv[1:]
if not re.fullmatch(r'[a-f0-9]{32}', job) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', scene):
    raise ValueError('Invalid wizard job')
with (base / 'wizard-live.lock').open('a') as lock, tempfile.TemporaryDirectory(dir=base, prefix='wizard-live-') as work:
    fcntl.flock(lock, fcntl.LOCK_EX)
    root = Path(work) / '产品化'

    def ignore(path, names):
        excluded = {'runs', 'verification', '__pycache__', 'reference_snapshots'}
        if Path(path).name == '仪器资料库':
            excluded.add('data')
        return excluded.intersection(names)

    shutil.copytree(source, root, copy_function=os.link, ignore=ignore)
    with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as archive:
        for member in archive:
            path = Path(member.name)
            allowed = (
                path.parts[:2] == ('library', 'data')
                or path.parts[:2] == ('run', job)
                or (scene.startswith('capture_') and path.parts[:2] == ('capture', scene[8:]))
            )
            if not allowed or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Invalid wizard input archive')
            archive.extract(member, work)
    wizard = root / 'P3_MaskSetupWizard_20261022'
    shutil.move(str(Path(work) / 'library/data'), str(root / '仪器资料库/data'))
    run = wizard / 'runs' / job
    run.parent.mkdir(exist_ok=True)
    shutil.move(str(Path(work) / 'run' / job), str(run))
    if scene.startswith('capture_'):
        capture = wizard / 'captures' / scene[8:]
        capture.parent.mkdir(exist_ok=True)
        if capture.exists():
            shutil.rmtree(capture)
        shutil.move(str(Path(work) / 'capture' / scene[8:]), str(capture))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', LAB_MASK_PRECISION='bfloat16',
               YOLO_CONFIG_DIR=str(Path(work) / 'ultralytics-config'),
               PYTHONPATH=f'/cluster/dataX/xqyi/lab-camera-python:{base}/wizard-deps:{base}/python-deps')
    code = subprocess.call([sys.executable, '-B', str(wizard / 'inference_runtime.py'),
                            '--job', job, '--scene', scene], cwd=wizard, env=env,
                           stdin=subprocess.DEVNULL, stdout=sys.stderr)
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        for path in run.iterdir():
            if path.name != 'request.json':
                archive.add(path, arcname=path.name)
    sys.exit(code)
