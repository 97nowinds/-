from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from settings import PROJECT_ROOT
from v4.fixed_instrument_interaction import (
    DEFAULT_CALIBRATION_PATH,
    FixedInstrumentInteractionEngine,
    FixedInstrumentRegions,
)
from v4.fixed_instrument_video import (
    aggregate_person_instrument_interactions,
    monitoring_pose_detections,
)
from v4.personal_activity import (
    build_personal_object_intervals,
    suppress_personal_object_false_interactions,
)
from v4.pose import HandPoseEstimator, get_hand_pose_estimator
from v4.tracking import StableTracker
from v5.minicpm_interaction_judge import MiniCPMInteractionJudge
from v5.minicpm_multi_instrument_realtime import MultiInstrumentPersonDetector
from v5.minicpm_realtime_video import _motion_region, _motion_score
from v5.minicpm_video_interaction import (
    _mask_bounds,
    _select_near_person,
)
from v5.minicpm_vrm_hybrid import _strong_rule_event

from .identity import _box_match_score, normalize_identity_timeline


STREAM_SCHEMA_VERSION = "1.0"
INSTRUMENT_HINTS_PATH = (
    PROJECT_ROOT / "prompts" / "minicpm_v46_instrument_hints.json"
)


@dataclass
class _FramePacket:
    sequence: int
    timestamp: float
    frame: np.ndarray
    persons: list[dict[str, Any]]
    persons_provided: bool


@dataclass
class _PairState:
    person_id: str
    person_name: str | None
    instrument_id: str
    instrument_name: str
    reviews: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=12)
    )
    active: bool = False
    event_start: float | None = None
    last_review_time: float = float("-inf")
    confidence: float = 0.0
    evidence: set[str] = field(default_factory=set)
    decision_sources: set[str] = field(default_factory=set)
    vrm_expire_time: float | None = None


def _crop_for_stream(
    frames: list[np.ndarray],
    *,
    instrument_id: str,
    instrument_mask: np.ndarray,
    person: dict[str, Any],
) -> tuple[list[Image.Image], tuple[int, int, int, int]]:
    height, width = frames[0].shape[:2]
    ix1, iy1, ix2, iy2 = _mask_bounds(instrument_mask)
    px1, py1, px2, py2 = [float(value) for value in person["bbox"]]
    if instrument_id == "instrument_006":
        x1, y1 = ix1 - 90, iy1 - 125
        x2, y2 = ix2 + 165, iy2 + 155
        person_margin = 20
        text_scale = 0.52
    else:
        x1, y1 = ix1 - 75, iy1 - 115
        x2, y2 = ix2 + 60, iy2 + 85
        person_margin = 12
        text_scale = 0.48
    x1, y1 = min(x1, px1 - person_margin), min(y1, py1 - person_margin)
    x2, y2 = max(x2, px2 + person_margin), max(y2, py2 + person_margin)
    roi = (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )
    contours, _ = cv2.findContours(
        instrument_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rx1, ry1, rx2, ry2 = roi
    output: list[Image.Image] = []
    for index, frame in enumerate(frames):
        marked = frame.copy()
        if index == 0:
            cv2.drawContours(marked, contours, -1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.rectangle(
                marked,
                (int(round(px1)), int(round(py1))),
                (int(round(px2)), int(round(py2))),
                (255, 120, 20),
                3,
            )
            cv2.putText(
                marked,
                "TARGET INSTRUMENT / PERSON",
                (max(4, rx1 + 6), max(24, ry1 + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        crop = marked[ry1:ry2, rx1:rx2]
        output.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    return output, roi


def _nearest_packets(
    packets: list[_FramePacket],
    *,
    start_time: float,
    end_time: float,
    count: int,
) -> list[_FramePacket]:
    targets = np.linspace(start_time, end_time, count)
    return [
        min(packets, key=lambda packet: abs(packet.timestamp - float(target)))
        for target in targets
    ]


class InstrumentInteractionStream:
    """One non-blocking, bounded-memory session for a live camera stream."""

    def __init__(
        self,
        *,
        camera_id: str,
        judge: MiniCPMInteractionJudge,
        person_detector: MultiInstrumentPersonDetector,
        inference_lock: threading.RLock,
        frame_size: tuple[int, int] | None = None,
        pose_fps: float = 5.0,
        window_seconds: float = 2.0,
        frame_count: int = 8,
        normal_review_interval: float = 2.0,
        active_review_interval: float = 1.0,
        idle_review_interval: float = 6.0,
        activity_threshold: float = 0.04,
        max_queue_frames: int = 300,
        event_history_limit: int = 500,
        on_close: Callable[["InstrumentInteractionStream"], None] | None = None,
    ) -> None:
        if pose_fps <= 0 or window_seconds <= 0 or frame_count < 2:
            raise ValueError("流式采样参数无效")
        self.session_id = f"stream_{uuid.uuid4().hex[:12]}"
        self.camera_id = str(camera_id)
        self._judge = judge
        self._person_detector = person_detector
        self._pose: HandPoseEstimator = get_hand_pose_estimator()
        self._inference_lock = inference_lock
        self._configured_frame_size = frame_size
        self._pose_fps = float(pose_fps)
        self._window_seconds = float(window_seconds)
        self._frame_count = int(frame_count)
        self._normal_review_interval = float(normal_review_interval)
        self._active_review_interval = float(active_review_interval)
        self._idle_review_interval = float(idle_review_interval)
        self._activity_threshold = float(activity_threshold)
        self._max_queue_frames = int(max_queue_frames)
        self._on_close = on_close

        self._condition = threading.Condition(threading.RLock())
        self._incoming: deque[_FramePacket] = deque()
        self._window_buffer: deque[_FramePacket] = deque()
        self._pose_records: deque[dict[str, Any]] = deque()
        self._raw_hand_events: deque[dict[str, Any]] = deque()
        self._updates: deque[dict[str, Any]] = deque(maxlen=event_history_limit)
        self._pair_states: dict[tuple[str, str], _PairState] = {}
        self._instrument_last_review: dict[str, float] = {}
        self._last_vrm_evidence_end: dict[tuple[str, str], float] = {}
        self._known_names: dict[str, str | None] = {}
        self._regions: FixedInstrumentRegions | None = None
        self._vrm: FixedInstrumentInteractionEngine | None = None
        self._fallback_tracker: StableTracker | None = None
        self._instrument_masks: dict[str, np.ndarray] = {}
        self._instrument_motion_regions: dict[str, tuple[int, int, int, int]] = {}
        self._instrument_hints = (
            json.loads(INSTRUMENT_HINTS_PATH.read_text(encoding="utf-8"))
            if INSTRUMENT_HINTS_PATH.is_file()
            else {}
        )

        self._received_frames = 0
        self._dropped_frames = 0
        self._processed_batches = 0
        self._pose_samples = 0
        self._vlm_calls = 0
        self._frame_sequence = 0
        self._update_sequence = 0
        self._delivered_update_sequence = 0
        self._last_input_timestamp: float | None = None
        self._last_processed_timestamp: float | None = None
        self._last_window_end: float | None = None
        self._next_pose_time: float | None = None
        self._status = "warming_up"
        self._error: str | None = None
        self._stop_requested = False
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_main,
            name=f"instrument-interaction-{self.session_id}",
            daemon=True,
        )
        self._worker.start()

    def _initialize_for_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        actual_size = (width, height)
        if self._configured_frame_size and tuple(self._configured_frame_size) != actual_size:
            raise ValueError(
                f"首帧尺寸 {actual_size} 与 create_stream(frame_size="
                f"{self._configured_frame_size}) 不一致"
            )
        self._configured_frame_size = actual_size
        self._regions = FixedInstrumentRegions.from_file(
            camera_id=self.camera_id,
            frame_size=actual_size,
            calibration_path=DEFAULT_CALIBRATION_PATH,
        )
        self._vrm = FixedInstrumentInteractionEngine(self._regions)
        self._fallback_tracker = StableTracker(
            detection_fps=self._pose_fps,
            hold_seconds=0.7,
            archive_seconds=2.0,
        )
        self._instrument_masks = dict(self._regions.core_masks)
        self._instrument_motion_regions = {
            instrument.instrument_id: _motion_region(
                self._instrument_masks[instrument.instrument_id],
                frame_size=actual_size,
            )
            for instrument in self._regions.instruments
        }

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        timestamp: float,
        persons: list[dict[str, Any]] | None,
        after_event_sequence: int | None = None,
    ) -> dict[str, Any]:
        """Queue one BGR frame and immediately return the latest known state."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame 必须是 OpenCV BGR 三通道 numpy.ndarray")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp 必须是非负有限秒数")
        normalized_people = normalize_identity_timeline(
            [{"timestamp": timestamp, "persons": persons or []}]
        )[0]["persons"]
        with self._condition:
            if self._closed or self._stop_requested:
                raise RuntimeError("流式会话已经关闭")
            if self._error is not None:
                return self._snapshot_locked(
                    accepted_frame=False,
                    after_event_sequence=after_event_sequence,
                )
            if (
                self._last_input_timestamp is not None
                and timestamp <= self._last_input_timestamp
            ):
                raise ValueError("同一会话的 timestamp 必须严格递增")
            if self._configured_frame_size:
                expected = tuple(self._configured_frame_size)
                actual = (int(frame.shape[1]), int(frame.shape[0]))
                if expected != actual:
                    raise ValueError(
                        f"帧尺寸在会话中发生变化：期望 {expected}，实际 {actual}"
                    )
            self._frame_sequence += 1
            self._received_frames += 1
            self._last_input_timestamp = timestamp
            for person in normalized_people:
                self._known_names[str(person["person_id"])] = person.get(
                    "person_name"
                )
            if len(self._incoming) >= self._max_queue_frames:
                self._incoming.popleft()
                self._dropped_frames += 1
            self._incoming.append(
                _FramePacket(
                    sequence=self._frame_sequence,
                    timestamp=timestamp,
                    frame=frame.copy(),
                    persons=normalized_people,
                    persons_provided=persons is not None,
                )
            )
            self._condition.notify()
            return self._snapshot_locked(
                accepted_frame=True,
                after_event_sequence=after_event_sequence,
            )

    def poll(self, *, after_event_sequence: int | None = None) -> dict[str, Any]:
        with self._condition:
            return self._snapshot_locked(
                accepted_frame=None,
                after_event_sequence=after_event_sequence,
            )

    def _snapshot_locked(
        self,
        *,
        accepted_frame: bool | None,
        after_event_sequence: int | None,
    ) -> dict[str, Any]:
        cursor = (
            self._delivered_update_sequence
            if after_event_sequence is None
            else int(after_event_sequence)
        )
        updates = [
            dict(item)
            for item in self._updates
            if int(item["event_sequence"]) > cursor
        ]
        oldest_available = (
            int(self._updates[0]["event_sequence"])
            if self._updates
            else self._update_sequence + 1
        )
        event_gap_detected = bool(cursor < oldest_available - 1)
        if after_event_sequence is None and updates:
            self._delivered_update_sequence = int(updates[-1]["event_sequence"])
        current = [
            self._state_payload(pair, event_state="ongoing")
            for pair in self._pair_states.values()
            if pair.active
        ]
        input_timestamp = self._last_input_timestamp
        processed_timestamp = self._last_processed_timestamp
        lag = (
            max(0.0, input_timestamp - processed_timestamp)
            if input_timestamp is not None and processed_timestamp is not None
            else None
        )
        buffered = (
            max(
                0.0,
                self._window_buffer[-1].timestamp
                - self._window_buffer[0].timestamp,
            )
            if len(self._window_buffer) >= 2
            else 0.0
        )
        return {
            "stream_schema_version": STREAM_SCHEMA_VERSION,
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "accepted_frame": accepted_frame,
            "timestamp": input_timestamp,
            "status": self._status,
            "analysis_pending": bool(self._incoming) or self._status == "analyzing",
            "analysis_lag_seconds": round(lag, 4) if lag is not None else None,
            "buffered_seconds": round(buffered, 4),
            "current_interactions": current,
            "events": updates,
            "last_event_sequence": self._update_sequence,
            "oldest_available_event_sequence": oldest_available,
            "event_gap_detected": event_gap_detected,
            "error": self._error,
            "metrics": {
                "received_frames": self._received_frames,
                "dropped_frames": self._dropped_frames,
                "processed_batches": self._processed_batches,
                "pose_samples": self._pose_samples,
                "vlm_calls": self._vlm_calls,
                "queued_frames": len(self._incoming),
            },
        }

    def _worker_main(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._incoming and not self._stop_requested:
                        self._condition.wait(timeout=0.5)
                    if self._stop_requested and not self._incoming:
                        break
                    batch = list(self._incoming)
                    self._incoming.clear()
                    self._status = "analyzing"
                self._process_batch(batch)
                with self._condition:
                    self._processed_batches += 1
                    self._last_processed_timestamp = batch[-1].timestamp
                    self._status = (
                        "running"
                        if self._window_duration() >= self._window_seconds
                        else "warming_up"
                    )
        except Exception as exc:
            with self._condition:
                self._error = f"{type(exc).__name__}: {exc}"
                self._status = "error"
                self._emit_update_locked(
                    {
                        "type": "analysis_error",
                        "state": "error",
                        "is_interacting": False,
                        "timestamp": self._last_input_timestamp,
                        "error": self._error,
                    }
                )

    def _window_duration(self) -> float:
        if len(self._window_buffer) < 2:
            return 0.0
        return max(
            0.0,
            self._window_buffer[-1].timestamp
            - self._window_buffer[0].timestamp,
        )

    def _process_batch(self, batch: list[_FramePacket]) -> None:
        if not batch:
            return
        if self._regions is None:
            self._initialize_for_frame(batch[0].frame)
        for packet in batch:
            self._window_buffer.append(packet)
        latest_time = batch[-1].timestamp
        while (
            self._window_buffer
            and self._window_buffer[0].timestamp
            < latest_time - max(4.0, self._window_seconds + 1.0)
        ):
            self._window_buffer.popleft()

        self._process_pose_samples(batch)
        self._expire_states(latest_time)
        if self._window_duration() + 1e-6 < self._window_seconds:
            return
        if (
            self._last_window_end is not None
            and latest_time - self._last_window_end < 0.80
        ):
            return
        packets = list(self._window_buffer)
        sampled = _nearest_packets(
            packets,
            start_time=latest_time - self._window_seconds,
            end_time=latest_time,
            count=self._frame_count,
        )
        self._review_window(sampled)
        self._last_window_end = latest_time

    def _process_pose_samples(self, batch: list[_FramePacket]) -> None:
        assert self._vrm is not None
        if self._next_pose_time is None:
            self._next_pose_time = batch[0].timestamp
        for packet in batch:
            if packet.timestamp + 1e-7 < self._next_pose_time:
                continue
            with self._inference_lock:
                poses, _, _ = self._pose.detect(
                    packet.frame,
                    imgsz=640,
                    suppress_split_people=False,
                )
            pose_people, _ = monitoring_pose_detections(
                poses,
                frame_size=(packet.frame.shape[1], packet.frame.shape[0]),
                min_person_area_ratio=0.018,
            )
            if packet.persons_provided:
                people = self._attach_pose_to_external_people(
                    pose_people,
                    packet.persons,
                )
            else:
                assert self._fallback_tracker is not None
                people = self._fallback_tracker.update(
                    pose_people,
                    timestamp=packet.timestamp,
                    frame_index=packet.sequence,
                    frame_size=(packet.frame.shape[1], packet.frame.shape[0]),
                )
            interaction_frame = self._vrm.update(
                timestamp=packet.timestamp,
                people=people,
            )
            self._pose_samples += 1
            self._pose_records.append(
                {
                    "frame_index": packet.sequence,
                    "timestamp": packet.timestamp,
                    "people": people,
                    "interaction": interaction_frame,
                }
            )
            self._consume_vrm_events(packet.timestamp)
            while self._next_pose_time <= packet.timestamp + 1e-7:
                self._next_pose_time += 1.0 / self._pose_fps
        cutoff = batch[-1].timestamp - 15.0
        while self._pose_records and self._pose_records[0]["timestamp"] < cutoff:
            self._pose_records.popleft()
        if self._fallback_tracker is not None:
            for track in [
                *self._fallback_tracker.active.values(),
                *self._fallback_tracker.archived.values(),
            ]:
                if len(track.full_history) > 240:
                    track.full_history = track.full_history[-120:]
            self._fallback_tracker.archived = {
                key: track
                for key, track in self._fallback_tracker.archived.items()
                if batch[-1].timestamp - track.last_seen <= 10.0
            }
            if len(self._fallback_tracker.events) > 200:
                self._fallback_tracker.events = self._fallback_tracker.events[-100:]

    def _attach_pose_to_external_people(
        self,
        pose_people: list[dict[str, Any]],
        external_people: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[tuple[float, int, int]] = []
        for external_index, external in enumerate(external_people):
            for pose_index, pose in enumerate(pose_people):
                score = _box_match_score(
                    list(external["bbox"]), list(pose["bbox"])
                )
                if score >= 0.20:
                    candidates.append((score, external_index, pose_index))
        used_external: set[int] = set()
        used_pose: set[int] = set()
        matches: dict[int, int] = {}
        for _, external_index, pose_index in sorted(candidates, reverse=True):
            if external_index in used_external or pose_index in used_pose:
                continue
            used_external.add(external_index)
            used_pose.add(pose_index)
            matches[external_index] = pose_index
        output: list[dict[str, Any]] = []
        for external_index, external in enumerate(external_people):
            pose_index = matches.get(external_index)
            if pose_index is None:
                continue
            pose = pose_people[pose_index]
            output.append(
                {
                    **pose,
                    "track_id": str(external["person_id"]),
                    "person_id": str(external["person_id"]),
                    "person_name": external.get("person_name"),
                    "bbox": list(external["bbox"]),
                    "external_identity": True,
                }
            )
        return output

    def _consume_vrm_events(self, timestamp: float) -> None:
        assert self._vrm is not None
        completed = list(self._vrm.completed_interactions)
        if not completed:
            return
        self._vrm.completed_interactions.clear()
        self._vrm.transitions.clear()
        for event in completed:
            self._raw_hand_events.append(dict(event))
        cutoff = timestamp - 15.0
        while self._raw_hand_events and float(
            self._raw_hand_events[0]["end_time"]
        ) < cutoff:
            self._raw_hand_events.popleft()
        aggregated = aggregate_person_instrument_interactions(
            list(self._raw_hand_events)
        )
        personal = build_personal_object_intervals(
            list(self._pose_records),
            rules=self._regions.temporal_rules if self._regions else None,
        )
        kept, _ = suppress_personal_object_false_interactions(
            aggregated,
            personal,
            rules=self._regions.temporal_rules if self._regions else None,
        )
        for event in kept:
            strong, _ = _strong_rule_event(event)
            if not strong:
                continue
            person_id = str(event["person_id"])
            instrument_id = str(event["instrument_id"])
            pair_key = (person_id, instrument_id)
            evidence_end = float(event["end_time"])
            if evidence_end <= self._last_vrm_evidence_end.get(
                pair_key, float("-inf")
            ) + 1e-4:
                continue
            self._last_vrm_evidence_end[pair_key] = evidence_end
            state = self._get_pair_state(
                person_id=person_id,
                person_name=self._known_names.get(person_id),
                instrument_id=instrument_id,
            )
            if not state.active:
                state.confidence = float(event.get("confidence") or 0.0)
                state.evidence = set(event.get("evidence") or [])
                state.decision_sources = {"strong_vrm_recovery"}
                state.active = True
                state.event_start = float(event["start_time"])
                self._emit_state_update(state, "started", timestamp)
            else:
                state.confidence = max(
                    state.confidence, float(event.get("confidence") or 0.0)
                )
                state.evidence.update(event.get("evidence") or [])
                state.decision_sources.add("strong_vrm_recovery")
            state.last_review_time = timestamp
            # A hand-level episode may close while the other hand is still
            # operating. Keep the public state alive long enough for the next
            # strong episode to join it; an explicit MiniCPM negative can end
            # it sooner.
            state.vrm_expire_time = timestamp + 3.0

    def _review_window(self, sampled_packets: list[_FramePacket]) -> None:
        assert self._regions is not None
        frames = [packet.frame for packet in sampled_packets]
        anchor = sampled_packets[len(sampled_packets) // 2]
        if anchor.persons_provided:
            detected_people = [
                {
                    **person,
                    "confidence": float(person.get("confidence") or 1.0),
                }
                for person in anchor.persons
            ]
        else:
            with self._inference_lock:
                detected_people = self._person_detector.detect(anchor.frame)
            detected_people = self._attach_internal_ids_to_boxes(
                detected_people,
                anchor.timestamp,
            )
        if not detected_people:
            return

        attributed_candidates: list[dict[str, Any]] = []
        for instrument in self._regions.instruments:
            instrument_id = instrument.instrument_id
            mask = self._instrument_masks[instrument_id]
            person = _select_near_person(
                detected_people,
                instrument_bounds=_mask_bounds(mask),
                frame_size=self._regions.frame_size,
            )
            if person is None:
                continue
            motion_score, _ = _motion_score(
                frames,
                self._instrument_motion_regions[instrument_id],
            )
            active = any(
                state.active and state.instrument_id == instrument_id
                for state in self._pair_states.values()
            )
            last_review = self._instrument_last_review.get(
                instrument_id, float("-inf")
            )
            interval = (
                self._active_review_interval
                if active or motion_score >= self._activity_threshold
                else self._normal_review_interval
            )
            periodic_due = anchor.timestamp - last_review >= interval
            idle_due = anchor.timestamp - last_review >= self._idle_review_interval
            bbox = list(person["bbox"])
            ix1, iy1, ix2, iy2 = _mask_bounds(mask)
            person_center = (
                (bbox[0] + bbox[2]) / 2.0,
                (bbox[1] + bbox[3]) / 2.0,
            )
            instrument_center = ((ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0)
            proximity = max(
                0.0,
                1.0
                - math.dist(person_center, instrument_center)
                / max(1.0, math.hypot(*self._regions.frame_size)),
            )
            attributed_candidates.append(
                {
                    "instrument": instrument,
                    "person": person,
                    "mask": mask,
                    "motion_score": motion_score,
                    "priority": 5.0 * motion_score + proximity + (3.0 if active else 0.0),
                    "due": bool(periodic_due or idle_due),
                    "person_key": str(
                        person.get("person_id")
                        or person.get("track_id")
                        or tuple(int(round(value / 8.0)) for value in bbox)
                    ),
                }
            )
        # Attribute a person to the most likely instrument before checking
        # review timers. Otherwise overdue neighboring instruments would take
        # turns receiving the same person's crop.
        best_by_person: dict[str, dict[str, Any]] = {}
        for candidate in attributed_candidates:
            key = str(candidate["person_key"])
            current = best_by_person.get(key)
            if current is None or float(candidate["priority"]) > float(
                current["priority"]
            ):
                best_by_person[key] = candidate
        candidates = [item for item in best_by_person.values() if item["due"]]
        if not candidates:
            return
        selected = max(candidates, key=lambda item: float(item["priority"]))
        instrument = selected["instrument"]
        person = selected["person"]
        target_frames, _ = _crop_for_stream(
            frames,
            instrument_id=instrument.instrument_id,
            instrument_mask=selected["mask"],
            person=person,
        )
        with self._inference_lock:
            verdict = self._judge.judge(
                target_frames,
                instrument_name=instrument.display_name,
                person_description="第一帧蓝色矩形指定的人员",
                instrument_hint=str(
                    self._instrument_hints.get(instrument.instrument_id) or ""
                ),
            )
        self._vlm_calls += 1
        self._instrument_last_review[instrument.instrument_id] = anchor.timestamp
        person_id = str(person.get("person_id") or person.get("track_id") or "person_unknown")
        person_name = person.get("person_name") or self._known_names.get(person_id)
        state = self._get_pair_state(
            person_id=person_id,
            person_name=person_name,
            instrument_id=instrument.instrument_id,
        )
        review = {
            "timestamp": anchor.timestamp,
            "window_start": sampled_packets[0].timestamp,
            "window_end": sampled_packets[-1].timestamp,
            "is_interacting": verdict.is_interacting,
            "confidence": verdict.confidence,
            "evidence": verdict.evidence,
        }
        state.reviews.append(review)
        state.last_review_time = anchor.timestamp
        state.person_name = person_name
        recent = list(state.reviews)[-3:]
        positive = [item for item in recent if item["is_interacting"]]
        if not state.active and len(positive) >= 2:
            state.active = True
            state.event_start = min(float(item["window_start"]) for item in positive)
            state.confidence = max(float(item["confidence"]) for item in positive)
            state.evidence = {str(item["evidence"]) for item in positive}
            state.decision_sources = {"minicpm_v46_positive"}
            state.vrm_expire_time = None
            self._emit_state_update(state, "started", anchor.timestamp)
        elif state.active and verdict.is_interacting:
            state.confidence = max(state.confidence, verdict.confidence)
            state.evidence.add(verdict.evidence)
            state.decision_sources.add("minicpm_v46_positive")
            state.vrm_expire_time = None
        elif state.active and not verdict.is_interacting:
            if "strong_vrm_recovery" in state.decision_sources:
                # The validated hybrid policy allows strong wrist/forearm
                # evidence to recover a MiniCPM miss. A single VLM negative
                # therefore must not make a strong live state flicker off.
                state.evidence.add("minicpm_negative_overridden_by_strong_vrm")
            else:
                self._end_state(
                    state, anchor.timestamp, "explicit_minicpm_negative"
                )

    def _attach_internal_ids_to_boxes(
        self,
        boxes: list[dict[str, Any]],
        timestamp: float,
    ) -> list[dict[str, Any]]:
        recent_people: list[dict[str, Any]] = []
        if self._pose_records:
            recent = min(
                self._pose_records,
                key=lambda item: abs(float(item["timestamp"]) - timestamp),
            )
            if abs(float(recent["timestamp"]) - timestamp) <= 0.6:
                recent_people = list(recent.get("people") or [])
        output: list[dict[str, Any]] = []
        for index, box in enumerate(boxes, start=1):
            matched = None
            ranked = [
                (_box_match_score(list(box["bbox"]), list(person["bbox"])), person)
                for person in recent_people
            ]
            if ranked:
                score, candidate = max(ranked, key=lambda item: item[0])
                if score >= 0.20:
                    matched = candidate
            person_id = str((matched or {}).get("track_id") or f"person_gate_{index:02d}")
            output.append(
                {
                    **box,
                    "person_id": person_id,
                    "track_id": person_id,
                    "person_name": (matched or {}).get("person_name"),
                }
            )
        return output

    def _get_pair_state(
        self,
        *,
        person_id: str,
        person_name: str | None,
        instrument_id: str,
    ) -> _PairState:
        assert self._regions is not None
        key = (str(person_id), str(instrument_id))
        if key not in self._pair_states:
            instrument = self._regions.instrument_lookup[instrument_id]
            self._pair_states[key] = _PairState(
                person_id=str(person_id),
                person_name=person_name,
                instrument_id=instrument_id,
                instrument_name=instrument.display_name,
            )
        elif person_name:
            self._pair_states[key].person_name = person_name
        return self._pair_states[key]

    def _expire_states(self, timestamp: float) -> None:
        stale_keys: list[tuple[str, str]] = []
        for key, state in self._pair_states.items():
            if not state.active:
                if timestamp - state.last_review_time > 600.0:
                    stale_keys.append(key)
                continue
            if state.vrm_expire_time is not None and timestamp >= state.vrm_expire_time:
                vrm_still_active = bool(
                    self._vrm is not None
                    and any(
                        memory.state == "interacting"
                        and str(person_id) == state.person_id
                        and str(memory.target_id) == state.instrument_id
                        for (person_id, _), memory in self._vrm.memory.items()
                    )
                )
                if vrm_still_active:
                    state.vrm_expire_time = timestamp + 1.0
                else:
                    self._end_state(
                        state, timestamp, "strong_vrm_recovery_completed"
                    )
            elif timestamp - state.last_review_time > max(
                4.0, self._idle_review_interval + 0.5
            ):
                self._end_state(state, timestamp, "review_timeout")
        for key in stale_keys:
            self._pair_states.pop(key, None)
            self._last_vrm_evidence_end.pop(key, None)
        if self._vrm is not None:
            if len(self._vrm.memory) > 1000:
                self._vrm.memory = {
                    key: memory
                    for key, memory in self._vrm.memory.items()
                    if memory.state != "idle"
                }
            if len(self._vrm.distance_history) > 5000:
                self._vrm.distance_history.clear()

    def _state_payload(
        self, pair: _PairState, *, event_state: str
    ) -> dict[str, Any]:
        return {
            "person_id": pair.person_id,
            "person_name": pair.person_name,
            "instrument_id": pair.instrument_id,
            "instrument_name": pair.instrument_name,
            "is_interacting": event_state != "ended",
            "state": event_state,
            "start_time": pair.event_start,
            "confidence": round(pair.confidence, 4),
            "decision_sources": sorted(pair.decision_sources),
            "evidence": sorted(pair.evidence),
        }

    def _emit_state_update(
        self, pair: _PairState, event_state: str, timestamp: float
    ) -> None:
        with self._condition:
            payload = self._state_payload(pair, event_state=event_state)
            payload["type"] = f"interaction_{event_state}"
            payload["timestamp"] = round(float(timestamp), 6)
            if event_state == "ended":
                payload["end_time"] = round(float(timestamp), 6)
                payload["duration_seconds"] = round(
                    max(0.0, float(timestamp) - float(pair.event_start or timestamp)),
                    4,
                )
            self._emit_update_locked(payload)

    def _emit_update_locked(self, payload: dict[str, Any]) -> None:
        self._update_sequence += 1
        self._updates.append(
            {"event_sequence": self._update_sequence, **payload}
        )

    def _end_state(self, state: _PairState, timestamp: float, reason: str) -> None:
        if not state.active:
            return
        state.evidence.add(reason)
        self._emit_state_update(state, "ended", timestamp)
        state.active = False
        state.event_start = None
        state.vrm_expire_time = None
        state.reviews.clear()

    def close(self, *, timeout_seconds: float = 60.0) -> dict[str, Any]:
        with self._condition:
            if self._closed:
                return self._snapshot_locked(
                    accepted_frame=None, after_event_sequence=None
                )
            self._stop_requested = True
            self._condition.notify_all()
        self._worker.join(timeout=max(0.0, float(timeout_seconds)))
        if self._worker.is_alive():
            raise RuntimeError(
                "流式后台线程未能在超时时间内停止；为避免损坏模型，暂不能释放模块"
            )
        with self._condition:
            timestamp = float(self._last_input_timestamp or 0.0)
            for state in self._pair_states.values():
                if state.active:
                    self._end_state(state, timestamp, "stream_closed")
            self._closed = True
            if self._status != "error":
                self._status = "closed"
            snapshot = self._snapshot_locked(
                accepted_frame=None, after_event_sequence=None
            )
        if self._on_close is not None:
            self._on_close(self)
        return snapshot

    def __enter__(self) -> "InstrumentInteractionStream":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
