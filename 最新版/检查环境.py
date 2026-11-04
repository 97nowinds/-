from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
MODULE_DIR = PACKAGE_ROOT / "模块"
sys.path.insert(0, str(MODULE_DIR))


EXPECTED = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "transformers": "5.16.1",
    "ultralytics": "8.4.135",
    "opencv-python": "5.0.0.93",
    "numpy": "2.4.6",
    "Pillow": "12.3.0",
    "av": "18.1.0",
}


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="检查仪器交互模块运行环境")
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="实际加载MiniCPM与YOLO并检查显存环境",
    )
    args = parser.parse_args()

    versions = {name: installed_version(name) for name in EXPECTED}
    missing = [name for name, value in versions.items() if value is None]
    mismatched = {
        name: {"expected": EXPECTED[name], "actual": value}
        for name, value in versions.items()
        if value is not None and value != EXPECTED[name]
    }

    try:
        from lab_instrument_interaction import InstrumentInteractionModule

        module = InstrumentInteractionModule(preload_models=args.load_models)
        try:
            health = module.health_check(require_loaded=args.load_models)
        finally:
            module.close()
    except Exception as exc:
        health = {
            "ok": False,
            "import_or_load_error": f"{type(exc).__name__}: {exc}",
        }

    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "package_root": str(PACKAGE_ROOT),
        "versions": versions,
        "missing_packages": missing,
        "version_mismatches": mismatched,
        "model_load_requested": args.load_models,
        "health": health,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if missing or not health.get("ok"):
        return 1
    if mismatched:
        print("警告：存在与已验证环境不同的版本，请先做固定视频回归测试。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

