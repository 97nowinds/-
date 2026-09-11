import json
import shutil
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


class ModernFaceEngine:
    """Optional SCRFD + ArcFace/ResNet50 backend from local ONNX models.

    The engine is intentionally opt-in: it never downloads weights while the
    service is starting. This keeps the RTSP service reliable on offline lab
    networks and lets the existing YuNet/SFace path remain a safe fallback.
    """

    def __init__(self, model_root=None, *, det_size=(640, 640)):
        self.model_root = Path(model_root) if model_root else None
        self.det_size = tuple(det_size)
        self.analysis = None
        self.error = None
        self.engine = "unavailable"
        self.lock = threading.RLock()
        self.providers = []
        if self.model_root is None:
            return
        model_dir = self.model_root / "models" / "buffalo_l"
        required = (model_dir / "det_10g.onnx", model_dir / "w600k_r50.onnx")
        if not all(path.exists() and path.stat().st_size > 1024 for path in required):
            self.error = "SCRFD/ArcFace model files are not installed"
            return
        try:
            from insightface.app import FaceAnalysis
            import onnxruntime

            available_providers = onnxruntime.get_available_providers()
            selected_providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available_providers
                else ["CPUExecutionProvider"]
            )
            self.analysis = FaceAnalysis(
                name="buffalo_l",
                root=str(self.model_root),
                allowed_modules=["detection", "recognition"],
                providers=selected_providers,
            )
            self.providers = list(selected_providers)
            ctx_id = 0 if "CUDAExecutionProvider" in selected_providers else -1
            self.analysis.prepare(ctx_id=ctx_id, det_thresh=0.35, det_size=self.det_size)
            self.engine = "SCRFD + ArcFace/ResNet50"
        except Exception as error:
            self.error = str(error)
            self.analysis = None

    @property
    def available(self):
        return self.analysis is not None

    def detect(self, frame):
        if self.analysis is None:
            return []
        records = []
        with self.lock:
            faces = self.analysis.get(frame, max_num=0)
        for face in faces:
            left, top, right, bottom = [int(round(value)) for value in face.bbox]
            box = (left, top, max(1, right - left), max(1, bottom - top))
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is not None:
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(embedding))
                embedding = embedding / norm if norm > 1e-8 else None
            records.append(
                {
                    "box": box,
                    "detection": None,
                    "aligned_face": None,
                    "embedding": embedding,
                    "score": float(getattr(face, "det_score", 0.0)),
                }
            )
        return records

    def embedding(self, image):
        if self.analysis is None or image is None or getattr(image, "size", 0) == 0:
            return None
        with self.lock:
            faces = self.analysis.get(image, max_num=1)
        if not faces:
            return None
        embedding = getattr(faces[0], "embedding", None)
        if embedding is None:
            return None
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else None


FACE_SIZE = (160, 160)
REGISTRATION_TARGET_SAMPLES = 60
REGISTRATION_MIN_SAMPLES = 50
REGISTRATION_MAX_SAMPLES = 80
ARCFACE_MIN_GALLERY_FEATURES = 5

REGISTRATION_POSES = (
    {"id": "front", "label": "正脸", "weight": 15, "instruction": "正对摄像头，保持自然表情"},
    {"id": "left", "label": "向左转头", "weight": 10, "instruction": "头部缓慢向左转约 20～35 度"},
    {"id": "right", "label": "向右转头", "weight": 10, "instruction": "头部缓慢向右转约 20～35 度"},
    {"id": "down", "label": "轻微低头", "weight": 8, "instruction": "轻微低头，双眼仍尽量可见"},
    {"id": "near", "label": "近距离", "weight": 8, "instruction": "向前一步，让脸部清晰但不要超出画面"},
    {"id": "far", "label": "较远距离", "weight": 9, "instruction": "向后一步，保持脸部宽度至少约 100 像素"},
)


def registration_pose_plan(target_samples):
    """Allocate the requested sample total across the fixed pose sequence."""
    target_samples = max(REGISTRATION_MIN_SAMPLES, min(REGISTRATION_MAX_SAMPLES, int(target_samples)))
    total_weight = sum(item["weight"] for item in REGISTRATION_POSES)
    counts = [max(1, round(target_samples * item["weight"] / total_weight)) for item in REGISTRATION_POSES]
    difference = target_samples - sum(counts)
    index = 0
    while difference:
        position = index % len(counts)
        if difference > 0:
            counts[position] += 1
            difference -= 1
        elif counts[position] > 1:
            counts[position] -= 1
            difference += 1
        index += 1
    return [
        {**pose, "target": counts[index]}
        for index, pose in enumerate(REGISTRATION_POSES)
    ]


class FaceCapturePolicy:
    """Simple face-quality gate for capture and registration."""

    def __init__(
        self,
        *,
        min_side=48,
        min_brightness=38.0,
        max_brightness=225.0,
        min_contrast=10.0,
        min_sharpness=22.0,
    ):
        self.min_side = int(min_side)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)
        self.min_contrast = float(min_contrast)
        self.min_sharpness = float(min_sharpness)

    @staticmethod
    def _safe_gray(face_gray):
        if face_gray is None or face_gray.size == 0:
            return None
        if face_gray.ndim == 3:
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
        return np.asarray(face_gray, dtype=np.uint8)

    def assess(self, face_gray):
        face_gray = self._safe_gray(face_gray)
        if face_gray is None:
            return {"accepted": False, "reason": "empty_face"}

        height, width = face_gray.shape[:2]
        if min(height, width) < self.min_side:
            return {
                "accepted": False,
                "reason": "face_too_small",
                "width": int(width),
                "height": int(height),
            }

        brightness = float(face_gray.mean())
        contrast = float(face_gray.std())
        sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        reason = None
        if brightness < self.min_brightness:
            reason = "face_too_dark"
        elif brightness > self.max_brightness:
            reason = "face_too_bright"
        elif contrast < self.min_contrast:
            reason = "face_low_contrast"
        elif sharpness < self.min_sharpness:
            reason = "face_blurry"
        accepted = reason is None
        return {
            "accepted": accepted,
            "reason": reason,
            "width": int(width),
            "height": int(height),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "sharpness": round(sharpness, 2),
        }


class FaceRegistrationSession:
    """Collect a set of quality-filtered face samples for one person."""

    def __init__(
        self,
        store,
        person_id,
        name,
        label,
        *,
        target_samples=REGISTRATION_TARGET_SAMPLES,
        min_samples=REGISTRATION_MIN_SAMPLES,
        max_samples=REGISTRATION_MAX_SAMPLES,
        session_id=None,
    ):
        self.store = store
        self.person_id = str(person_id)
        self.name = str(name)
        self.label = int(label)
        self.target_samples = int(target_samples)
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.directory = self.store.faces_dir / f"{self.person_id}_{self.session_id}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.samples = 0
        self.rejections = []
        self.started_at = time.time()

    def add_sample(self, face_gray, metadata=None, embedding=None):
        if self.samples >= self.max_samples:
            return {
                "accepted": False,
                "reason": "sample_limit_reached",
                "sample_count": self.samples,
                "directory": str(self.directory),
            }

        assessment = self.store.capture_policy.assess(face_gray)
        if not assessment["accepted"]:
            self.rejections.append(assessment)
            return {
                "accepted": False,
                "reason": assessment["reason"],
                "quality": assessment,
                "sample_count": self.samples,
                "directory": str(self.directory),
            }

        sample_index = self.samples + 1
        path = self.store.save_face_sample(
            self.person_id,
            face_gray,
            directory=self.directory,
            sample_index=sample_index,
        )
        if embedding is not None:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                np.save(path.with_suffix(".npy"), vector / norm)
        self.samples += 1
        return {
            "accepted": True,
            "quality": assessment,
            "sample_count": self.samples,
            "target_samples": self.target_samples,
            "minimum_samples": self.min_samples,
            "ready": self.samples >= self.target_samples,
            "completed": self.samples >= self.max_samples,
            "path": str(path),
            "directory": str(self.directory),
            "metadata": metadata or {},
        }

    @property
    def can_finish(self):
        return self.samples >= self.min_samples

    def finalize(self, *, replace_existing=False):
        if not self.can_finish:
            raise ValueError(
                f"至少需要 {self.min_samples} 张有效人脸样本，目前只有 {self.samples} 张"
            )
        if replace_existing:
            for directory in self.store.faces_dir.glob(f"{self.person_id}_*"):
                if directory != self.directory:
                    shutil.rmtree(directory, ignore_errors=True)
        self.store.register_person(
            self.person_id,
            self.name,
            self.label,
            sample_count=self.samples,
            created_at=int(self.started_at),
        )
        self.store.reload()
        return {
            "person_id": self.person_id,
            "name": self.name,
            "label": self.label,
            "sample_count": self.samples,
            "directory": str(self.directory),
        }

    def summary(self):
        return {
            "person_id": self.person_id,
            "name": self.name,
            "label": self.label,
            "sample_count": self.samples,
            "target_samples": self.target_samples,
            "minimum_samples": self.min_samples,
            "maximum_samples": self.max_samples,
            "can_finish": self.can_finish,
            "directory": str(self.directory),
            "rejections": len(self.rejections),
        }


class FaceIdentityStore:
    """LBPH verifier with one model per registered person and margin rejection."""

    def __init__(
        self,
        people_path,
        faces_dir,
        threshold=60.0,
        min_margin=8.0,
        capture_policy=None,
        recognition_model_path=None,
        modern_model_root=None,
        required_engine=None,
    ):
        self.people_path = Path(people_path)
        self.faces_dir = Path(faces_dir)
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self.capture_policy = capture_policy or FaceCapturePolicy()
        self.required_engine = str(required_engine or "").strip().lower() or None
        self.lock = threading.RLock()
        self.people = {}
        self.models = {}
        self.sface = None
        self.sface_features = {}
        self.modern = ModernFaceEngine(modern_model_root)
        self.modern_features = {}
        self.modern_feature_counts = {}
        self.query_features = defaultdict(lambda: deque(maxlen=5))
        if recognition_model_path and Path(recognition_model_path).exists() and hasattr(cv2, "FaceRecognizerSF"):
            try:
                self.sface = cv2.FaceRecognizerSF.create(str(recognition_model_path), "")
            except cv2.error:
                self.sface = None
        self.reload()

    def _write_people(self, people):
        self.people_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.people_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(people, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.people_path)

    def reload(self):
        with self.lock:
            self.people = (
                json.loads(self.people_path.read_text(encoding="utf-8"))
                if self.people_path.exists()
                else {}
            )
            self.models = {}
            self.sface_features = {}
            self.modern_features = {}
            self.modern_feature_counts = {}
            for person_id, person in self.people.items():
                images = []
                modern_features = []
                sface_features = []
                for directory in sorted(self.faces_dir.glob(f"{person_id}_*")):
                    for image_path in sorted(directory.glob("*.png")):
                        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            images.append(cv2.resize(image, FACE_SIZE))
                            if self.modern.available:
                                feature_path = image_path.with_suffix(".npy")
                                embedding = (
                                    np.asarray(
                                        np.load(feature_path, allow_pickle=False),
                                        dtype=np.float32,
                                    )
                                    if feature_path.exists()
                                    else self.modern.embedding(
                                        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                                    )
                                )
                                if embedding is not None:
                                    if not feature_path.exists():
                                        np.save(feature_path, embedding)
                                    modern_features.append(embedding)
                            if self.sface is not None:
                                color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                                sface_feature = self.sface.feature(cv2.resize(color, (112, 112)))
                                if sface_feature is not None:
                                    sface_features.append(sface_feature)
                if not images:
                    continue
                self.modern_feature_counts[person_id] = len(modern_features)
                model = cv2.face.LBPHFaceRecognizer_create()
                model.train(images, np.zeros(len(images), dtype=np.int32))
                self.models[person_id] = (dict(person), model)
                if (
                    self.modern.available
                    and len(modern_features) >= ARCFACE_MIN_GALLERY_FEATURES
                ):
                    mean = np.mean(np.asarray(modern_features, dtype=np.float32), axis=0)
                    norm = float(np.linalg.norm(mean))
                    if norm > 1e-8:
                        self.modern_features[person_id] = mean / norm
                if sface_features:
                    self.sface_features[person_id] = sface_features

    @staticmethod
    def normalize(face_gray):
        face = cv2.equalizeHist(cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY) if face_gray.ndim == 3 else face_gray)
        return cv2.resize(face, FACE_SIZE)

    def save_face_sample(self, person_id, face_gray, *, directory=None, sample_index=None):
        normalized = self.normalize(face_gray)
        directory = Path(directory) if directory is not None else self.faces_dir / f"{person_id}_{int(time.time())}"
        directory.mkdir(parents=True, exist_ok=True)
        if sample_index is None:
            sample_index = 1 + sum(1 for _ in directory.glob("*.png"))
        path = directory / f"{int(sample_index):04d}.png"
        cv2.imwrite(str(path), normalized)
        return path

    def register_person(self, person_id, name, label, *, sample_count, created_at=None):
        with self.lock:
            people = dict(self.people)
            now = int(time.time())
            entry = dict(people.get(person_id, {}))
            entry.update(
                {
                    "id": person_id,
                    "label": int(label),
                    "name": name,
                    "sample_count": int(sample_count),
                    "created_at": int(created_at if created_at is not None else entry.get("created_at", now)),
                    "updated_at": now,
                }
            )
            people[person_id] = entry
            self._write_people(people)
            self.people = people

    def start_registration(
        self,
        person_id,
        name,
        label,
        *,
        target_samples=REGISTRATION_TARGET_SAMPLES,
        min_samples=REGISTRATION_MIN_SAMPLES,
        max_samples=REGISTRATION_MAX_SAMPLES,
        session_id=None,
    ):
        return FaceRegistrationSession(
            self,
            person_id,
            name,
            label,
            target_samples=target_samples,
            min_samples=min_samples,
            max_samples=max_samples,
            session_id=session_id,
        )

    @staticmethod
    def normalize_face_box(face_gray):
        return cv2.resize(face_gray, FACE_SIZE)

    def align_face(self, frame, detection):
        if self.sface is None:
            return None
        try:
            return self.sface.alignCrop(frame, np.asarray(detection, dtype=np.float32))
        except cv2.error:
            return None

    def _sface_feature(self, face_gray, aligned_face=None):
        if self.sface is None:
            return None
        image = aligned_face
        if image is None:
            if face_gray is None or face_gray.size == 0:
                return None
            image = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR) if face_gray.ndim == 2 else face_gray
            image = cv2.resize(image, (112, 112))
        try:
            return self.sface.feature(image)
        except cv2.error:
            return None

    def predict(
        self,
        face_gray,
        *,
        aligned_face=None,
        track_key=None,
        query_embedding=None,
    ):
        with self.lock:
            if not self.models:
                return self.unknown("未注册")
            quality = self.capture_policy.assess(face_gray)
            if not quality["accepted"]:
                return self.unknown("未注册", reason=quality["reason"], quality=quality)
            if self.modern.available and self.modern_features:
                query = query_embedding
                if query is None:
                    query = self.modern.embedding(
                        cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
                        if face_gray.ndim == 2
                        else face_gray
                    )
                if query is not None:
                    query = np.asarray(query, dtype=np.float32).reshape(-1)
                    norm = float(np.linalg.norm(query))
                    query = query / norm if norm > 1e-8 else None
                if query is not None:
                    if track_key is not None:
                        history = self.query_features[track_key]
                        history.append(query)
                        query = np.mean(np.asarray(history), axis=0)
                        norm = float(np.linalg.norm(query))
                        if norm > 1e-8:
                            query = query / norm
                    candidates = sorted(
                        (
                            float(np.dot(query, reference)),
                            person_id,
                            self.people[person_id],
                        )
                        for person_id, reference in self.modern_features.items()
                    )[::-1]
                    score, person_id, person = candidates[0]
                    second = candidates[1][0] if len(candidates) > 1 else 0.0
                    margin = score - second
                    # Open-set gate: low-quality/ambiguous faces remain
                    # unregistered instead of being forced to the top match.
                    if score < 0.45 or margin < 0.08:
                        return self.unknown("未注册", 1.0 - score, margin, quality=quality)
                    return {
                        "known": True,
                        "person_id": person_id,
                        "person_number": int(person.get("label", 0)),
                        "name": person.get("name", person_id),
                        "distance": round(1.0 - score, 4),
                        "margin": round(margin, 4),
                        "similarity": round(score, 4),
                        "recognition_engine": self.modern.engine,
                        "quality": quality,
                        "sample_count": int(person.get("sample_count", 0)),
                        "feature_frames": len(self.query_features.get(track_key, ())) if track_key is not None else 1,
                    }
            if self.required_engine == "arcface":
                reason = (
                    "arcface_unavailable"
                    if not self.modern.available
                    else "arcface_gallery_empty"
                    if not self.modern_features
                    else "arcface_embedding_unavailable"
                )
                return self.unknown("未注册", reason=reason, quality=quality)
            query_feature = self._sface_feature(face_gray, aligned_face)
            if query_feature is not None and self.sface_features:
                candidates = []
                for person_id, features in self.sface_features.items():
                    score = max(
                        float(self.sface.match(query_feature, reference, cv2.FaceRecognizerSF_FR_COSINE))
                        for reference in features
                    )
                    candidates.append((score, person_id, self.people[person_id]))
                candidates.sort(key=lambda item: item[0], reverse=True)
                score, person_id, person = candidates[0]
                second = candidates[1][0] if len(candidates) > 1 else 0.0
                margin = score - second
                # OpenCV's SFace cosine threshold is 0.363. Requiring a
                # margin prevents a low-resolution stranger from matching the
                # closest registered sample by accident.
                if score < 0.363 or margin < 0.035:
                    return self.unknown("未注册", 1.0 - score, margin, quality=quality)
                return {
                    "known": True,
                    "person_id": person_id,
                    "person_number": int(person.get("label", 0)),
                    "name": person.get("name", person_id),
                    "distance": round(1.0 - score, 4),
                    "margin": round(margin, 4),
                    "similarity": round(score, 4),
                    "recognition_engine": "SFace",
                    "quality": quality,
                    "sample_count": int(person.get("sample_count", 0)),
                }
            normalized = self.normalize(face_gray)
            candidates = []
            for person_id, (person, model) in self.models.items():
                _, distance = model.predict(normalized)
                candidates.append((float(distance), person_id, person))
            candidates.sort(key=lambda item: item[0])
            distance, person_id, person = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else float("inf")
            margin = second - distance
            if distance > self.threshold or margin < self.min_margin:
                return self.unknown("未注册", distance, margin, quality=quality)
            return {
                "known": True,
                "person_id": person_id,
                "person_number": int(person.get("label", 0)),
                "name": person.get("name", person_id),
                "distance": distance,
                "margin": margin,
                "quality": quality,
                "sample_count": int(person.get("sample_count", 0)),
            }

    @staticmethod
    def unknown(name="未注册", distance=None, margin=None, reason=None, quality=None):
        result = {
            "known": False,
            "person_id": None,
            "person_number": None,
            "name": name,
            "distance": distance,
            "margin": margin,
        }
        if reason is not None:
            result["reason"] = reason
        if quality is not None:
            result["quality"] = quality
        return result
