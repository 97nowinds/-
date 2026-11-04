from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TESTS_DIR = PROJECT_ROOT / "tests"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "default_prompt.txt"

DEFAULT_MODEL_ID = os.getenv("LAB_VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
DEFAULT_MODEL_LOCAL_DIR = MODELS_DIR / DEFAULT_MODEL_ID.replace("/", "__")
DEFAULT_QUANTIZATION = os.getenv("LAB_VLM_QUANTIZATION", "bnb4").lower()

MAX_VIDEO_MB = int(os.getenv("LAB_VLM_MAX_VIDEO_MB", "512"))
MAX_CLIP_SECONDS = float(os.getenv("LAB_VLM_MAX_CLIP_SECONDS", "30"))
DEFAULT_FRAME_COUNT = int(os.getenv("LAB_VLM_DEFAULT_FRAME_COUNT", "6"))
DEFAULT_MAX_IMAGE_SIDE = int(os.getenv("LAB_VLM_DEFAULT_MAX_IMAGE_SIDE", "960"))
MAX_NEW_TOKENS = int(os.getenv("LAB_VLM_MAX_NEW_TOKENS", "768"))


def configure_local_cache() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "huggingface" / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


configure_local_cache()
