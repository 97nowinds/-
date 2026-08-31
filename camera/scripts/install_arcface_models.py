"""Install the official InsightFace buffalo_l ONNX models for this project."""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_MD5 = "6c0e929fd3b6ab517170b732ced18c68"
REQUIRED_MODELS = {"det_10g.onnx", "w600k_r50.onnx"}


def download(url, destination):
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "lab-safety-model-installer/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        append = downloaded > 0 and response.status == 206
        mode = "ab" if append else "wb"
        if not append:
            downloaded = 0
        total = response.headers.get("Content-Length")
        total = downloaded + int(total) if total else None
        with partial.open(mode) as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if total:
                    print(f"\rDownloading buffalo_l: {downloaded / total:.1%}", end="", flush=True)
    print()
    partial.replace(destination)


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    installed = set()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = Path(member.filename).name
            if not name.endswith(".onnx"):
                continue
            with bundle.open(member) as source, (target / name).open("wb") as output:
                shutil.copyfileobj(source, output)
            installed.add(name)
    missing = REQUIRED_MODELS - installed
    if missing:
        raise RuntimeError(f"model archive is missing: {sorted(missing)}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=project_root / "models" / "insightface",
    )
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    archive = args.model_root / "buffalo_l.zip"
    target = args.model_root / "models" / "buffalo_l"
    args.model_root.mkdir(parents=True, exist_ok=True)

    if not archive.exists() or file_md5(archive) != MODEL_MD5:
        download(MODEL_URL, archive)
    if file_md5(archive) != MODEL_MD5:
        raise RuntimeError("buffalo_l archive integrity check failed")
    install(archive, target)
    if not args.keep_archive:
        archive.unlink()
    print(f"ArcFace models installed in {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ArcFace model installation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
