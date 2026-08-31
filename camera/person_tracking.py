import time

import cv2

from camshift_tracker import CamShiftTracker


class PersonTrack:
    """One-person local track seeded by a face/body box."""

    def __init__(self, lost_seconds=2.0, evidence_seconds=5.0):
        self.tracker = CamShiftTracker(max_failures=8)
        self.spatial_tracker = None
        self.spatial_valid = False
        self.spatial_failures = 0
        self.lost_seconds = float(lost_seconds)
        self.evidence_seconds = float(evidence_seconds)
        self.box = None
        self.last_seen_at = None
        self.last_evidence_at = None

    def _start_spatial_tracker(self, frame, box):
        self.spatial_tracker = None
        self.spatial_valid = False
        self.spatial_failures = 0
        try:
            tracker = cv2.TrackerCSRT_create()
            initialized = tracker.init(frame, tuple(int(value) for value in box))
            if initialized is False:
                return
            self.spatial_tracker = tracker
            self.spatial_valid = True
        except (AttributeError, cv2.error):
            self.spatial_tracker = None

    @staticmethod
    def tracker_box(box, frame_shape):
        return CamShiftTracker.clamp_box(box, frame_shape)

    @staticmethod
    def smooth_box(previous, current, position_alpha=0.32, size_alpha=0.18):
        if previous is None:
            return tuple(int(value) for value in current)
        px, py, pw, ph = previous
        cx, cy, cw, ch = current
        previous_center_x = px + pw / 2.0
        previous_center_y = py + ph / 2.0
        current_center_x = cx + cw / 2.0
        current_center_y = cy + ch / 2.0
        center_x = previous_center_x + (current_center_x - previous_center_x) * position_alpha
        center_y = previous_center_y + (current_center_y - previous_center_y) * position_alpha
        width = pw + (cw - pw) * size_alpha
        height = ph + (ch - ph) * size_alpha
        return (
            int(round(center_x - width / 2.0)),
            int(round(center_y - height / 2.0)),
            int(round(width)),
            int(round(height)),
        )

    @staticmethod
    def box_iou(first, second):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / max(union, 1)

    @staticmethod
    def face_to_person(face_box, frame_shape):
        x, y, width, height = face_box
        person_width = int(width * 1.8)
        person_height = int(height * 2.5)
        person_x = int(x + width / 2 - person_width / 2)
        person_y = int(y - height * 0.12)
        return CamShiftTracker.clamp_box(
            (person_x, person_y, person_width, person_height), frame_shape
        )

    def seed(self, frame, box, target_id="local", validated=True):
        seeded_box = CamShiftTracker.clamp_box(box, frame.shape)
        try:
            state = self.tracker.start(frame, box, target_id)
        except ValueError:
            state = None
        if state and state.get("box"):
            seeded_box = tuple(state["box"])
        self.box = self.smooth_box(self.box, seeded_box)
        self._start_spatial_tracker(frame, self.box)
        self.last_seen_at = time.time()
        if validated:
            self.last_evidence_at = self.last_seen_at
        return self.box

    def mark_evidence(self):
        self.last_evidence_at = time.time()

    def observe(self, frame, box, target_id="local"):
        """Confirm a detector observation without restarting a healthy tracker."""
        self.mark_evidence()
        state = self.tracker.snapshot()
        if (
            self.box is None
            or (self.spatial_tracker is None and state["status"] != "tracking")
            or self.box_iou(self.box, box) < 0.12
        ):
            return self.seed(frame, box, target_id, validated=True)
        return self.box

    def evidence_expired(self, now=None):
        if self.box is None or self.last_evidence_at is None:
            return self.box is not None
        return (now or time.time()) - self.last_evidence_at > self.evidence_seconds

    def visual_track_healthy(self, minimum_confidence=0.018):
        """Return true while CamShift still has a valid response on the target."""
        state = self.tracker.snapshot()
        return (
            self.box is not None
            and (
                self.spatial_valid
                or (
                    state["status"] == "tracking"
                    and state["valid"]
                    and state["confidence"] >= float(minimum_confidence)
                )
            )
        )

    def observation_is_distinct(self, box, maximum_iou=0.04):
        return self.box is not None and self.box_iou(self.box, box) < maximum_iou

    def status_payload(self):
        state = self.tracker.snapshot()
        camshift_valid = bool(state["status"] == "tracking" and state["valid"])
        if self.spatial_tracker is not None or camshift_valid:
            valid = bool(self.spatial_valid or camshift_valid)
            status = "tracking" if valid else "recovering"
            confidence = 1.0 if self.spatial_valid else state["confidence"]
            failures = min(self.spatial_failures, state["failures"])
        else:
            status = state["status"]
            confidence = state["confidence"]
            failures = state["failures"]
            valid = state["valid"]
        return {
            "status": status,
            "confidence": confidence,
            "failures": failures,
            "valid": valid,
        }

    def update(self, frame):
        if self.spatial_tracker is not None:
            try:
                ok, next_box = self.spatial_tracker.update(frame)
            except cv2.error:
                ok, next_box = False, None
            if ok and next_box is not None:
                next_box = CamShiftTracker.clamp_box(next_box, frame.shape)
                _, _, width, height = next_box
                frame_area = frame.shape[0] * frame.shape[1]
                area = width * height
                previous_area = max(1, self.box[2] * self.box[3]) if self.box else area
                if (
                    width >= 12
                    and height >= 20
                    and area <= frame_area * 0.42
                    and 0.45 <= area / previous_area <= 2.2
                ):
                    self.box = self.smooth_box(self.box, next_box, 0.58, 0.35)
                    self.last_seen_at = time.time()
                    self.spatial_valid = True
                    self.spatial_failures = 0
                    return self.box
            self.spatial_valid = False
            self.spatial_failures += 1

        state = self.tracker.update(frame)
        if state["status"] == "tracking" and state.get("box") and state.get("valid"):
            self.box = self.smooth_box(self.box, tuple(state["box"]))
            self.last_seen_at = time.time()
            self._start_spatial_tracker(frame, self.box)
            return self.box
        elif self.last_seen_at and time.time() - self.last_seen_at > self.lost_seconds:
            self.box = None
        return None

    def stop(self):
        self.tracker.stop()
        self.spatial_tracker = None
        self.spatial_valid = False
        self.spatial_failures = 0
        self.box = None
        self.last_seen_at = None
        self.last_evidence_at = None
