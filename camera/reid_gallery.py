"""Bounded, quality-aware track-level ReID feature galleries."""

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class ReIDSample:
    feature: np.ndarray
    monotonic_time: float
    unix_time: float
    quality: float
    camera_id: str
    quality_details: dict


class BodyQualityScorer:
    """Estimate whether a body crop is useful for appearance matching."""

    @staticmethod
    def assess(frame, box):
        height, width = frame.shape[:2]
        x, y, box_width, box_height = [int(value) for value in box]
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + box_width), min(height, y + box_height)
        crop = frame[top:bottom, left:right]
        if crop.size == 0 or width <= 0 or height <= 0:
            return {"score": 0.0, "reason": "empty_crop"}

        area_ratio = ((right - left) * (bottom - top)) / max(width * height, 1)
        size_score = float(np.clip(area_ratio / 0.08, 0.0, 1.0))
        requested_area = max(box_width * box_height, 1)
        visible_ratio = ((right - left) * (bottom - top)) / requested_area
        edge_score = float(np.clip(visible_ratio, 0.0, 1.0))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = float(np.clip(sharpness / 120.0, 0.0, 1.0))
        brightness = float(np.mean(gray))
        brightness_score = float(np.clip(1.0 - abs(brightness - 128.0) / 128.0, 0.0, 1.0))
        aspect = (right - left) / max(bottom - top, 1)
        aspect_score = float(np.clip(1.0 - abs(aspect - 0.45) / 0.45, 0.0, 1.0))
        score = (
            size_score * 0.30
            + edge_score * 0.20
            + sharpness_score * 0.25
            + brightness_score * 0.15
            + aspect_score * 0.10
        )
        return {
            "score": round(float(score), 4),
            "area_ratio": round(float(area_ratio), 5),
            "visible_ratio": round(float(visible_ratio), 4),
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "aspect_ratio": round(float(aspect), 4),
        }


class TrackFeatureGallery:
    def __init__(
        self,
        capacity=12,
        min_quality=0.45,
        duplicate_similarity=0.985,
        top_k=3,
    ):
        self.capacity = int(capacity)
        self.min_quality = float(min_quality)
        self.duplicate_similarity = float(duplicate_similarity)
        self.top_k = int(top_k)
        self.samples = []
        self.rejected_samples = 0
        self.duplicate_samples = 0
        self.last_updated_monotonic = None
        self.last_updated_unix = None

    @staticmethod
    def _normalize(feature):
        if feature is None:
            return None
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else None

    def add(
        self,
        feature,
        *,
        monotonic_time=None,
        unix_time=None,
        quality,
        camera_id,
        quality_details=None,
    ):
        vector = self._normalize(feature)
        quality = float(quality)
        if vector is None or quality < self.min_quality:
            self.rejected_samples += 1
            return False
        monotonic_time = time.monotonic() if monotonic_time is None else float(monotonic_time)
        unix_time = time.time() if unix_time is None else float(unix_time)
        sample = ReIDSample(
            vector,
            monotonic_time,
            unix_time,
            quality,
            str(camera_id),
            dict(quality_details or {}),
        )
        if self.samples:
            similarities = [float(np.dot(vector, item.feature)) for item in self.samples]
            duplicate_index = int(np.argmax(similarities))
            if similarities[duplicate_index] >= self.duplicate_similarity:
                self.duplicate_samples += 1
                if quality > self.samples[duplicate_index].quality:
                    self.samples[duplicate_index] = sample
                    self._touch(sample)
                    return True
                return False
        self.samples.append(sample)
        if len(self.samples) > self.capacity:
            self._prune_one()
        self._touch(sample)
        return True

    def _touch(self, sample):
        self.last_updated_monotonic = sample.monotonic_time
        self.last_updated_unix = sample.unix_time

    def _prune_one(self):
        utilities = []
        for index, sample in enumerate(self.samples):
            others = [
                float(np.dot(sample.feature, other.feature))
                for other_index, other in enumerate(self.samples)
                if other_index != index
            ]
            diversity = 1.0 - max(others) if others else 1.0
            utilities.append(sample.quality * 0.8 + diversity * 0.2)
        del self.samples[int(np.argmin(utilities))]

    @property
    def representative(self):
        if not self.samples:
            return None
        weights = np.asarray([sample.quality for sample in self.samples], dtype=np.float32)
        vectors = np.asarray([sample.feature for sample in self.samples], dtype=np.float32)
        vector = np.average(vectors, axis=0, weights=weights)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else None

    def compare(self, other):
        if other is None or not self.samples or not other.samples:
            return {
                "score": None,
                "best_score": None,
                "pair_count": 0,
                "top_k": 0,
                "quality": 0.0,
            }
        pairs = []
        for first in self.samples:
            for second in other.samples:
                similarity = float(np.dot(first.feature, second.feature))
                quality = float(np.sqrt(first.quality * second.quality))
                pairs.append((similarity, quality))
        pairs.sort(key=lambda item: item[0], reverse=True)
        selected = pairs[: min(self.top_k, len(pairs))]
        total_quality = sum(quality for _, quality in selected)
        score = (
            sum(similarity * quality for similarity, quality in selected) / total_quality
            if total_quality > 1e-8
            else float(np.mean([similarity for similarity, _ in selected]))
        )
        return {
            "score": round(float(score), 6),
            "best_score": round(float(selected[0][0]), 6),
            "pair_count": len(pairs),
            "top_k": len(selected),
            "quality": round(float(total_quality / len(selected)), 4),
        }

    def merge(self, other):
        if other is None:
            return
        for sample in other.samples:
            self.add(
                sample.feature,
                monotonic_time=sample.monotonic_time,
                unix_time=sample.unix_time,
                quality=sample.quality,
                camera_id=sample.camera_id,
                quality_details=sample.quality_details,
            )

    def snapshot(self):
        return {
            "gallery_size": len(self.samples),
            "valid_samples": len(self.samples),
            "rejected_samples": self.rejected_samples,
            "duplicate_samples": self.duplicate_samples,
            "last_updated_at": self.last_updated_unix,
            "camera_ids": sorted({sample.camera_id for sample in self.samples}),
        }
