"""Package the working files and model weights without local runtime data."""
from pathlib import Path
import os
import re
import json
import hashlib
import zipfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / ('project-package-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
OUT.mkdir(parents=True, exist_ok=False)
SKIP = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.cache',
        'node_modules', 'runtime', 'data', 'tmp', 'camera_probe', 'runs',
        'known_faces', 'recordings', 'output', 'outputs', '.vscode', '.idea',
        '开发测试结果', '演示视频'}
TEXT = {'.py', '.ps1', '.sh', '.bat', '.json', '.yaml', '.yml', '.md',
        '.txt', '.html', '.js', '.css', '.slurm', '.example'}
excluded, redacted, sources, models = [], [], [], []
for current, dirs, files in os.walk(ROOT):
    for name in list(dirs):
        if name in SKIP or name.startswith(('.chart-data-', '.ppt-build')):
            excluded.append((Path(current) / name).relative_to(ROOT).as_posix() + '/')
            dirs.remove(name)
    for name in files:
        path = Path(current) / name
        rel = path.relative_to(ROOT)
        if ((name.startswith('.env') and name != '.env.example')
            or path.suffix.lower() in {'.zip', '.gz', '.7z', '.pyc', '.pyo', '.log', '.bak', '.tmp', '.dpapi', '.pem', '.key', '.incomplete'}
            or name in {'Thumbs.db', '.DS_Store'}):
            excluded.append(rel.as_posix())
            continue
        (models if 'models' in rel.parts or 'weights' in rel.parts else sources).append(path)

readme = '''# 项目移交说明

本包包含当前工作目录的源代码（含尚未提交的修改）、展示页面、配置、测试和文档。
project-models.zip 是本机已有模型文件，解压到与 project-source.zip 相同的目录，保留原目录结构。
camera/ 是当前网页服务；camera_track/ 是项目内已有另一份代码；最新版/ 是仪器交互模块移交包。

## 仅在另一台电脑浏览展示页
不需要解压项目或安装模型。保持原机器服务在线，使用已配置的跨网络访问链接即可。
当前原机器前端端口 5173，后端本地转发端口 15001。
两台电脑连接到同一个 Tailscale 网络后，在原机器运行：
```powershell
tailscale serve --bg --http=18080 http://127.0.0.1:5173
tailscale serve --bg --http=18081 http://127.0.0.1:15001
```
然后在展示机器浏览器打开：
http://100.106.153.101:18080/display?api=http://100.106.153.101:18081
此地址依赖原机器的 Tailscale IP 和网络权限，隧道未包含在压缩包中。

## 把前端迁移到另一台 Windows 电脑
安装 Python，解压源码，在解压后的项目根目录执行：
```powershell
py -m venv camera/.venv
camera/.venv/Scripts/python.exe -m pip install pyserial
powershell -ExecutionPolicy Bypass -File camera/scripts/start_frontend.ps1 -BackendUrl http://100.106.153.101:18081
```
浏览器打开 http://127.0.0.1:5173/display。
仅运行前端不需要模型。环境串口和仪器交互等本机集成需要单独重新配置，迁移前端不会自动复制原机器的运行状态。

## 完整迁移后端和仪器模块
依照 camera/README.md、cluster/ 下的部署脚本以及 最新版/README_移交包_先看.md 安装依赖和配置。
camera/requirements.txt 是后端依赖清单。后端使用集群时，需要重新建立 SSH 连接和端口转发；15001 不是永久可用的公共地址。
配置文件中的原机器网络地址需要按新环境调整。

## 打包范围
未打包虚拟环境、Git 历史、缓存、运行日志、录像、人脸样本、运行数据库、本机环境文件、密钥、旧压缩包和生成输出。
源码中 RTSP URL 的用户名密码已替换为占位符；重新使用摄像头时需要配置自己的凭据。
模型包仅包含本机已经存在的文件，不保证每个模型下载完整。依照相应模块文档检查环境。
这是一份迁移材料包，不能直接替代已经运行的后端服务。
'''
(OUT / '打包说明.md').write_text(readme, encoding='utf-8')
for filename, paths in [('project-source.zip', sources), ('project-models.zip', models)]:
    print(f'Writing {filename}: {len(paths)} files, {sum(p.stat().st_size for p in paths)/1024**2:.1f} MiB', flush=True)
    with zipfile.ZipFile(OUT / filename, 'w', allowZip64=True) as archive:
        for path in sorted(paths):
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() in TEXT or path.name == '.env.example':
                try:
                    content = path.read_text(encoding='utf-8-sig')
                except UnicodeError:
                    archive.write(path, rel, compress_type=zipfile.ZIP_STORED)
                    continue
                content, count = re.subn(r'(rtsp[s]?://)[^\s\"\'/]+:[^\s\"\'/@]+@', r'\1YOUR_USERNAME:YOUR_PASSWORD@', content)
                if count:
                    redacted.append(rel)
                archive.writestr(rel, content.encode('utf-8'), compress_type=zipfile.ZIP_DEFLATED)
            else:
                archive.write(path, rel, compress_type=zipfile.ZIP_STORED)
        if filename == 'project-source.zip':
            archive.writestr('打包说明.md', readme.encode('utf-8'), compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(OUT / filename) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'ZIP CRC failure: {bad}')
        names = set(archive.namelist())
        if filename == 'project-source.zip':
            assert {'camera/frontend_server.py', 'camera/templates/display.html', 'camera/static/display.js', 'camera/config/floor_map.json'} <= names
    print(f'Validated {filename}', flush=True)
manifest = {'source_files': len(sources), 'model_files': len(models), 'excluded': sorted(excluded), 'redacted_rtsp_files': sorted(set(redacted))}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
checksums = []
for path in sorted(OUT.glob('*.zip')):
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    checksums.append(f'{digest}  {path.name}')
(OUT / 'SHA256SUMS.txt').write_text('\n'.join(checksums) + '\n', encoding='utf-8')
print(f'OUTPUT={OUT}', flush=True)
