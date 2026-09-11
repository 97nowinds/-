from __future__ import annotations

import gc
import json
import threading
from pathlib import Path
from typing import Any

import torch

from settings import PROJECT_ROOT
from v5.minicpm_interaction_judge import MiniCPMInteractionJudge
from v5.minicpm_multi_instrument_realtime import (
    MultiInstrumentPersonDetector,
    _decode_video,
    _render_multi_result_video,
)
from v5.minicpm_realtime_video import FAST_PERSON_WEIGHTS
from v5.minicpm_vrm_hybrid import analyze_video_hybrid

from .identity import attach_external_identities


INTERFACE_VERSION = "1.1"
PLUGIN_VERSION = "0.2.0"
DEFAULT_CALIBRATION = (
    PROJECT_ROOT / "configs" / "fixed_instrument_calibration_v0_1.json"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class InstrumentInteractionModule:
    """Stable public facade for the replaceable interaction backend.

    The teammate application should import only this class.  Everything under
    v4/ and v5/ is private implementation and may change between releases.
    """

    interface_version = INTERFACE_VERSION
    plugin_version = PLUGIN_VERSION

    def __init__(
        self,
        *,
        default_camera_id: str = "lab_camera_view_2",
        preload_models: bool = False,
    ) -> None:
        self.default_camera_id = str(default_camera_id)
        self._judge: MiniCPMInteractionJudge | None = None
        self._person_detector: MultiInstrumentPersonDetector | None = None
        self._streams: set[Any] = set()
        self._closed = False
        self._lock = threading.RLock()
        self._validate_camera(self.default_camera_id)
        if preload_models:
            self.load()

    @staticmethod
    def supported_cameras() -> dict[str, list[dict[str, str]]]:
        payload = _read_json(DEFAULT_CALIBRATION)
        return {
            camera_id: [
                {
                    "instrument_id": str(item["id"]),
                    "instrument_name": str(item["display_name"]),
                }
                for item in camera.get("instruments") or []
            ]
            for camera_id, camera in payload.get("cameras", {}).items()
        }

    @classmethod
    def plugin_info(cls) -> dict[str, Any]:
        return {
            "interface_version": cls.interface_version,
            "plugin_version": cls.plugin_version,
            "backend": "MiniCPM-V 4.6 + constrained wrist/forearm VRM",
            "input_modes": [
                "video_file_with_optional_external_identity_timeline",
                "non_blocking_live_frame_stream",
            ],
            "gpu_required": True,
        }

    def _validate_camera(self, camera_id: str) -> None:
        if camera_id not in self.supported_cameras():
            supported = ", ".join(sorted(self.supported_cameras()))
            raise ValueError(
                f"未知 camera_id={camera_id!r}；当前支持：{supported}"
            )

    def load(self) -> dict[str, Any]:
        """Load MiniCPM and YOLO once and keep them for subsequent calls."""
        with self._lock:
            if self._closed:
                raise RuntimeError("模块已经 close，不能再次使用")
            if self._judge is None:
                self._judge = MiniCPMInteractionJudge(max_new_tokens=64)
            if self._person_detector is None:
                self._person_detector = MultiInstrumentPersonDetector(
                    weights=FAST_PERSON_WEIGHTS,
                    image_size=768,
                )
            return {
                **self.plugin_info(),
                "loaded": True,
                "model": self._judge.metadata(),
            }

    def health_check(self, *, require_loaded: bool = False) -> dict[str, Any]:
        model_path = PROJECT_ROOT.parent / "models" / "MiniCPM-V-4.6"
        checks = {
            "calibration_exists": DEFAULT_CALIBRATION.is_file(),
            "minicpm_weights_exist": (model_path / "model.safetensors").is_file(),
            "person_detector_exists": FAST_PERSON_WEIGHTS.is_file(),
            "pose_detector_exists": (
                PROJECT_ROOT / "models" / "detectors" / "yolo11n-pose.pt"
            ).is_file(),
            "cuda_available": torch.cuda.is_available(),
            "models_loaded": self._judge is not None
            and self._person_detector is not None,
        }
        ok = all(
            value
            for key, value in checks.items()
            if key != "models_loaded" or require_loaded
        )
        return {
            **self.plugin_info(),
            "ok": ok,
            "checks": checks,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "supported_cameras": self.supported_cameras(),
        }

    def analyze_video(
        self,
        video_path: str | Path,
        *,
        camera_id: str | None = None,
        identity_timeline: Any = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Analyze one video and optionally attach externally recognized IDs.

        Reuse the same object for multiple calls.  The method is intentionally
        serialized because one MiniCPM instance must not receive concurrent
        generate calls from several application threads.
        """
        camera = str(camera_id or self.default_camera_id)
        self._validate_camera(camera)
        source = Path(video_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with self._lock:
            self.load()
            result = analyze_video_hybrid(
                source,
                camera_id=camera,
                output_dir=output_dir,
                judge_instance=self._judge,
                person_detector_instance=self._person_detector,
            )
            rule_result = _read_json(result["components"]["vrm"]["result_json"])
            enriched, identity_metadata = attach_external_identities(
                list(result.get("interactions") or []),
                identity_timeline=identity_timeline,
                rule_result=rule_result,
                video_fps=float(rule_result["video"]["fps"]),
            )
            for event in enriched:
                event["is_interacting"] = True
                event["state"] = "interaction_interval"
            result["interface_version"] = self.interface_version
            result["plugin_version"] = self.plugin_version
            result["identity_association"] = identity_metadata
            result["interactions"] = enriched
            result["summary"]["person_ids"] = sorted(
                {str(item["person_id"]) for item in enriched}
            )
            result["summary"]["person_names"] = sorted(
                {
                    str(item["person_name"])
                    for item in enriched
                    if item.get("person_name")
                }
            )
            result_path = Path(result["artifacts"]["result_json"])
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if identity_timeline is not None and enriched:
                _, fps, width, height = _decode_video(source)
                _render_multi_result_video(
                    source,
                    Path(result["artifacts"]["annotated_video"]),
                    fps=fps,
                    width=width,
                    height=height,
                    events=enriched,
                )
            return result

    def create_stream(
        self,
        *,
        camera_id: str | None = None,
        frame_size: tuple[int, int] | None = None,
        pose_fps: float = 5.0,
        window_seconds: float = 2.0,
        frame_count: int = 8,
        normal_review_interval: float = 2.0,
        active_review_interval: float = 1.0,
        idle_review_interval: float = 6.0,
        activity_threshold: float = 0.04,
    ) -> Any:
        """Create one long-running camera session with ``process_frame``."""
        from .stream import InstrumentInteractionStream

        camera = str(camera_id or self.default_camera_id)
        self._validate_camera(camera)
        with self._lock:
            self.load()
            stream = InstrumentInteractionStream(
                camera_id=camera,
                judge=self._judge,
                person_detector=self._person_detector,
                inference_lock=self._lock,
                frame_size=frame_size,
                pose_fps=pose_fps,
                window_seconds=window_seconds,
                frame_count=frame_count,
                normal_review_interval=normal_review_interval,
                active_review_interval=active_review_interval,
                idle_review_interval=idle_review_interval,
                activity_threshold=activity_threshold,
                on_close=self._remove_stream,
            )
            self._streams.add(stream)
            return stream

    def _remove_stream(self, stream: Any) -> None:
        with self._lock:
            self._streams.discard(stream)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            streams = list(self._streams)
        for stream in streams:
            stream.close()
        with self._lock:
            if self._closed:
                return
            self._streams.clear()
            self._judge = None
            self._person_detector = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._closed = True

    def __enter__(self) -> "InstrumentInteractionModule":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def analyze_video(
    video_path: str | Path,
    *,
    camera_id: str = "lab_camera_view_2",
    identity_timeline: Any = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convenient one-shot entry; persistent applications should use the class."""
    with InstrumentInteractionModule(
        default_camera_id=camera_id,
        preload_models=True,
    ) as module:
        return module.analyze_video(
            video_path,
            camera_id=camera_id,
            identity_timeline=identity_timeline,
            output_dir=output_dir,
        )
