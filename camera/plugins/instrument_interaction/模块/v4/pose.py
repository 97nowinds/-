from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from settings import PROJECT_ROOT


POSE_WEIGHTS = PROJECT_ROOT / "models" / "detectors" / "yolo11n-pose.pt"
KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
}

MIN_OPERATOR_AREA_RATIO = 0.06
MIN_OPERATOR_WIDTH_RATIO = 0.15
MIN_OPERATOR_TALL_HEIGHT_RATIO = 0.55


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_over_smaller(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(1.0, min(_area(a), _area(b)))


def _center_distance(a: list[float], b: list[float], diagonal: float) -> float:
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1]) / max(1.0, diagonal)


def _is_vrm_operator_bbox(bbox: list[float], frame_size: tuple[int, int]) -> bool:
    """Keep people large enough to provide reliable wrist/object relationships."""
    width, height = frame_size
    box_width = max(0.0, bbox[2] - bbox[0])
    box_height = max(0.0, bbox[3] - bbox[1])
    area_ratio = _area(bbox) / max(1.0, float(width * height))
    width_ratio = box_width / max(1.0, float(width))
    height_ratio = box_height / max(1.0, float(height))
    return area_ratio >= MIN_OPERATOR_AREA_RATIO and (
        width_ratio >= MIN_OPERATOR_WIDTH_RATIO
        or height_ratio >= MIN_OPERATOR_TALL_HEIGHT_RATIO
    )


def _suppress_split_pose_people(
    poses: list[dict[str, Any]], frame_size: tuple[int, int]
) -> tuple[list[dict[str, Any]], int]:
    diagonal = math.hypot(frame_size[0], frame_size[1])
    selected: list[dict[str, Any]] = []
    removed = 0
    for pose in sorted(poses, key=lambda item: float(item["confidence"]), reverse=True):
        duplicate = any(
            _intersection_over_smaller(pose["bbox"], kept["bbox"]) >= 0.12
            or _center_distance(pose["bbox"], kept["bbox"], diagonal) <= 0.17
            for kept in selected
        )
        if duplicate:
            removed += 1
            continue
        selected.append(pose)
    strong = [item for item in selected if float(item["confidence"]) >= 0.50]
    if strong:
        kept: list[dict[str, Any]] = []
        for pose in selected:
            if float(pose["confidence"]) >= 0.40:
                kept.append(pose)
                continue
            px1, py1, px2, py2 = pose["bbox"]
            ph = max(1.0, py2 - py1)
            split_fragment = False
            for anchor in strong:
                ax1, ay1, ax2, ay2 = anchor["bbox"]
                ah = max(1.0, ay2 - ay1)
                vertical_overlap = max(0.0, min(py2, ay2) - max(py1, ay1)) / min(ph, ah)
                horizontal_gap = max(0.0, max(px1, ax1) - min(px2, ax2))
                if vertical_overlap >= 0.70 and horizontal_gap <= frame_size[0] * 0.20:
                    split_fragment = True
                    break
            if split_fragment:
                removed += 1
            else:
                kept.append(pose)
        selected = kept
    return selected, removed


class HandPoseEstimator:
    """Small pose model used only for left/right wrist evidence."""

    def __init__(self, weights: Path = POSE_WEIGHTS) -> None:
        if not weights.exists():
            raise FileNotFoundError(f"缺少手部关键点模型：{weights}")
        self.weights = weights.resolve()
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(self.weights))
        self.load_count = 1

    def detect(
        self,
        frame_bgr,
        *,
        imgsz: int = 640,
        suppress_split_people: bool = True,
    ) -> tuple[list[dict[str, Any]], float, int]:
        started = time.perf_counter()
        result = self.model.predict(
            frame_bgr,
            conf=0.10,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )[0]
        elapsed = time.perf_counter() - started
        poses: list[dict[str, Any]] = []
        if result.boxes is None or result.keypoints is None or result.keypoints.conf is None:
            return poses, elapsed, 0
        xy = result.keypoints.xy.detach().cpu().tolist()
        confidence = result.keypoints.conf.detach().cpu().tolist()
        for index, box in enumerate(result.boxes):
            joints: dict[str, dict[str, Any]] = {}
            for name, keypoint_index in KEYPOINTS.items():
                score = float(confidence[index][keypoint_index])
                point = xy[index][keypoint_index]
                if score < 0.12 or not point or (float(point[0]) == 0.0 and float(point[1]) == 0.0):
                    continue
                joints[name] = {
                    "point": [round(float(point[0]), 3), round(float(point[1]), 3)],
                    "confidence": round(score, 6),
                }
            poses.append(
                {
                    "bbox": [float(value) for value in box.xyxy[0].detach().cpu().tolist()],
                    "confidence": float(box.conf[0].item()),
                    "joints": joints,
                    "hands": {
                        side: joints[f"{side}_wrist"]
                        for side in ("left", "right")
                        if f"{side}_wrist" in joints
                    },
                }
            )
        height, width = frame_bgr.shape[:2]
        if suppress_split_people:
            poses, removed = _suppress_split_pose_people(poses, (width, height))
        else:
            removed = 0
        return poses, elapsed, removed

    def metadata(self) -> dict[str, Any]:
        return {
            "weights": str(self.weights),
            "purpose": "left/right wrist keypoints only",
            "device": f"cuda:0 - {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu",
            "load_count": self.load_count,
        }


def fuse_pose_people(
    detections: list[dict[str, Any]],
    poses: list[dict[str, Any]],
    *,
    frame_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Attach pose evidence to General-YOLO people without creating pose-only identities."""
    detected_people = [dict(item) for item in detections if item.get("class_name") == "person"]
    others = [item for item in detections if item.get("class_name") != "person"]
    people = [
        item
        for item in detected_people
        if _is_vrm_operator_bbox(list(item["bbox"]), frame_size)
    ]
    small_general_people_ignored = len(detected_people) - len(people)
    if not poses:
        return [*others, *people], {
            "pose_person_count": 0,
            "pose_people_matched": 0,
            "pose_people_seeded": 0,
            "split_person_boxes_merged": 0,
            "unmatched_pose_people_ignored": 0,
            "low_quality_pose_people_ignored": 0,
            "small_general_people_ignored": small_general_people_ignored,
        }

    width, height = frame_size
    diagonal = math.hypot(width, height)
    available = set(range(len(people)))
    fused: list[dict[str, Any]] = []
    merged_count = 0
    matched_pose_count = 0
    seeded_pose_count = 0
    low_quality_pose_count = 0
    operator_poses: list[dict[str, Any]] = []
    for pose in poses:
        if _is_vrm_operator_bbox(list(pose["bbox"]), frame_size):
            operator_poses.append(pose)
        else:
            low_quality_pose_count += 1
    for pose in sorted(operator_poses, key=lambda item: float(item["confidence"]), reverse=True):
        pose_bbox = list(pose["bbox"])
        matched = [
            index
            for index in available
            if _intersection_over_smaller(pose_bbox, people[index]["bbox"]) >= 0.18
            or _center_distance(pose_bbox, people[index]["bbox"], diagonal) <= 0.10
        ]
        if matched:
            for index in matched:
                available.discard(index)
            matched_boxes = [people[index]["bbox"] for index in matched]
            bbox = [
                max(0.0, min([pose_bbox[0], *[box[0] for box in matched_boxes]])),
                max(0.0, min([pose_bbox[1], *[box[1] for box in matched_boxes]])),
                min(float(width), max([pose_bbox[2], *[box[2] for box in matched_boxes]])),
                min(float(height), max([pose_bbox[3], *[box[3] for box in matched_boxes]])),
            ]
            confidence = max([float(pose["confidence"]), *[float(people[index]["confidence"]) for index in matched]])
            merged_count += max(0, len(matched) - 1)
        else:
            # A large, wrist-bearing pose may recover an operator when the
            # general detector misses a cropped torso. Tiny pose fragments are
            # rejected above and never establish a person identity.
            bbox = pose_bbox
            confidence = float(pose["confidence"])
            seeded_pose_count += 1
        if matched:
            matched_pose_count += 1
        fused.append(
            {
                "bbox": bbox,
                "class_id": 0,
                "class_name": "person",
                "raw_class_name": "person",
                "display_name": "人员",
                "confidence": confidence,
                "detector_source": "general_yolo11n+pose_fused",
                "tracking_enabled": True,
                "hands": pose.get("hands") or {},
                "pose_joints": pose.get("joints") or {},
                "pose_confidence": float(pose["confidence"]),
            }
        )
    fused.extend(people[index] for index in sorted(available))
    return [*others, *fused], {
        "pose_person_count": len(poses),
        "pose_people_matched": matched_pose_count,
        "pose_people_seeded": seeded_pose_count,
        "split_person_boxes_merged": merged_count,
        "unmatched_pose_people_ignored": 0,
        "low_quality_pose_people_ignored": low_quality_pose_count,
        "small_general_people_ignored": small_general_people_ignored,
    }


_POSE_INSTANCE: HandPoseEstimator | None = None


def get_hand_pose_estimator() -> HandPoseEstimator:
    global _POSE_INSTANCE
    if _POSE_INSTANCE is None:
        _POSE_INSTANCE = HandPoseEstimator()
    return _POSE_INSTANCE
