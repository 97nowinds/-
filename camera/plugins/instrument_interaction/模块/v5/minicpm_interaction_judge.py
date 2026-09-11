from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODULE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MODULE_ROOT.parent
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "models" / "MiniCPM-V-4.6"
DEFAULT_PROMPT_PATH = (
    MODULE_ROOT / "prompts" / "minicpm_v46_interaction_system_prompt.txt"
)


@dataclass
class MiniCPMJudgeResult:
    is_interacting: bool
    confidence: float
    evidence: str
    raw_response: str
    valid_response: bool
    validation_error: str | None
    latency_seconds: float
    peak_gpu_memory_mb: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_response(text: str) -> tuple[bool, float, str, bool, str | None]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    # Normalize punctuation before locating the object so a truncated closing
    # brace can still be handled conservatively below.
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace('"is_interacting"：', '"is_interacting":')
    cleaned = cleaned.replace('"confidence"：', '"confidence":')
    cleaned = cleaned.replace('"evidence"：', '"evidence":')
    cleaned = cleaned.replace('"evidence："', '"evidence":"')

    def recover_complete_fields(
        source: str, error: str
    ) -> tuple[bool, float, str, bool, str | None]:
        bool_match = re.search(
            r'"is_interacting"\s*:\s*(true|false)', source, flags=re.IGNORECASE
        )
        confidence_match = re.search(
            r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', source
        )
        evidence_match = re.search(r'"evidence"\s*:\s*"([^"\r\n]+)"', source)
        if not bool_match or not confidence_match:
            return False, 0.0, "模型没有返回可恢复的完整字段", False, error
        interacting = bool_match.group(1).lower() == "true"
        confidence = max(0.0, min(1.0, float(confidence_match.group(1))))
        evidence = evidence_match.group(1).strip() if evidence_match else ""
        if interacting and (confidence < 0.75 or not evidence):
            return False, min(confidence, 0.4), "截断输出缺少完整正向证据", False, error
        if not evidence:
            evidence = "模型判定无交互（输出结尾被截断）"
        return interacting, confidence, evidence, True, "recovered_truncated_json"

    match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if not match:
        return recover_complete_fields(cleaned, "missing_json")
    candidate = match.group(0)
    # Small VLMs occasionally mix Chinese punctuation into an otherwise valid
    # JSON object. Repair punctuation only; never alter semantic field values.
    try:
        payload = json.loads(candidate)
    except Exception as exc:
        return recover_complete_fields(candidate, f"invalid_json: {exc}")
    if not isinstance(payload, dict):
        return False, 0.0, "模型输出不是JSON对象", False, "json_not_object"
    if not isinstance(payload.get("is_interacting"), bool):
        return False, 0.0, "缺少布尔型is_interacting", False, "invalid_is_interacting"
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence"))))
    except Exception:
        return False, 0.0, "缺少数值型confidence", False, "invalid_confidence"
    evidence = str(payload.get("evidence") or "").strip()
    if not evidence:
        if not bool(payload["is_interacting"]):
            return False, confidence, "模型判定无交互（未补充说明）", True, None
        return False, min(confidence, 0.4), "模型判真但没有提供可见证据", False, "missing_positive_evidence"
    interacting = bool(payload["is_interacting"])
    if interacting and confidence < 0.75:
        return (
            False,
            confidence,
            f"模型判真但置信度低于0.75：{evidence}",
            False,
            "positive_below_confidence_threshold",
        )
    return interacting, confidence, evidence, True, None


class MiniCPMInteractionJudge:
    """Binary VLM judge. It does not consume wrist or old interaction scores."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        downsample_mode: str = "4x",
        max_new_tokens: int = 96,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.prompt_path = Path(prompt_path).resolve()
        if not (self.model_path / "model.safetensors").is_file():
            raise FileNotFoundError(self.model_path / "model.safetensors")
        if not self.prompt_path.is_file():
            raise FileNotFoundError(self.prompt_path)
        if not torch.cuda.is_available():
            raise RuntimeError("MiniCPM-V 4.6 backend requires CUDA")
        if downsample_mode not in {"4x", "16x"}:
            raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")
        if max_new_tokens < 8:
            raise ValueError("max_new_tokens must be at least 8")
        self.downsample_mode = downsample_mode
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_text = self.prompt_path.read_text(encoding="utf-8").strip()
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
            attn_implementation="sdpa",
        ).eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.model_gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2

    def judge(
        self,
        frames: list[Image.Image],
        *,
        instrument_name: str,
        person_description: str,
        instrument_hint: str | None = None,
    ) -> MiniCPMJudgeResult:
        if len(frames) < 2:
            raise ValueError("MiniCPM interaction judging requires at least two frames")
        hint_text = (
            f"\n目标仪器可见部件说明：{instrument_hint}"
            if instrument_hint
            else ""
        )
        task_text = (
            f"{self.prompt_text}\n\n"
            f"指定仪器：{instrument_name}。"
            f"指定人员：{person_description}。"
            f"{hint_text}"
            f"输入包含{len(frames)}帧，已按时间先后排列。请执行二分类审核。"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": task_text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "downsample_mode": self.downsample_mode,
                "max_num_frames": len(frames),
                "stack_frames": 1,
                "max_slice_nums": 1,
                "use_image_id": False,
                "do_sample_frames": False,
            },
        ).to(self.model.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                downsample_mode=self.downsample_mode,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        interacting, confidence, evidence, valid, error = _parse_response(response)
        return MiniCPMJudgeResult(
            is_interacting=interacting,
            confidence=confidence,
            evidence=evidence,
            raw_response=response,
            valid_response=valid,
            validation_error=error,
            latency_seconds=round(latency, 4),
            peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "prompt_path": str(self.prompt_path),
            "device": f"cuda:0 - {torch.cuda.get_device_name(0)}",
            "load_seconds": round(self.load_seconds, 4),
            "model_gpu_memory_mb": round(self.model_gpu_memory_mb, 2),
            "downsample_mode": self.downsample_mode,
            "max_new_tokens": self.max_new_tokens,
        }
