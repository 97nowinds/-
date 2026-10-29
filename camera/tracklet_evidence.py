"""Bounded evidence from fresh observations; cached frames never add votes."""
from collections import deque

import numpy as np


class TrackletEvidence:
    def __init__(self, capacity=12, window_seconds=3.0, max_gap_seconds=1.0):
        self.samples = deque(maxlen=capacity)
        self.window_seconds = float(window_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.candidate = None
        self.votes = deque(maxlen=capacity)
        self.last_confirmation_reason = None

    def add(self, feature, observed_at, kind='full', quality=1.0, unix_time=None,
            support_feature=None, support_quality=None, box=None):
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not vector.size or not np.isfinite(vector).all() or norm < 1e-8:
            return False
        now = float(observed_at)
        if self.samples and now <= self.samples[-1][0] + 1e-6:
            return False
        if self.samples and now - self.samples[-1][0] > self.max_gap_seconds:
            self.samples.clear()
            self.reset_candidate()
        while self.samples and now - self.samples[0][0] > self.window_seconds:
            self.samples.popleft()
        support = None
        if support_feature is not None:
            support = np.asarray(support_feature, dtype=np.float32).reshape(-1)
            support_norm = float(np.linalg.norm(support))
            support = support / support_norm if support_norm > 1e-8 and np.isfinite(support).all() else None
        self.samples.append((now, vector / norm, kind, {'quality': quality, 'unix_time': unix_time,
                              'support_feature': support, 'support_quality': support_quality,
                              'box': tuple(box) if box is not None else None}))
        return True

    def query_gallery(self, minimum_quality, allowed_kinds=None):
        from reid_gallery import TrackFeatureGallery
        gallery = TrackFeatureGallery(capacity=len(self.samples) * 2 + 1, min_quality=minimum_quality,
                                      duplicate_similarity=1.01, robust=True)
        for observed_at, feature, kind, metadata in self.samples:
            if allowed_kinds is None or kind in allowed_kinds:
                gallery.add(feature, monotonic_time=observed_at, unix_time=metadata['unix_time'],
                            quality=metadata['quality'], camera_id='tracklet', quality_details={'body_kind': kind})
            if ((allowed_kinds is None or 'upper' in allowed_kinds)
                    and metadata['support_feature'] is not None and metadata['support_quality'] is not None):
                gallery.add(metadata['support_feature'], monotonic_time=observed_at, unix_time=metadata['unix_time'],
                            quality=metadata['support_quality'], camera_id='tracklet', quality_details={'body_kind': 'upper'})
        return gallery

    def reset_candidate(self):
        self.candidate = None
        self.votes.clear()

    def confirm(self, candidate, gallery, now, *, similarity, minimum=3,
                duration=0.2, max_gap=0.5, body_kind=None, full_min_similarity=.52):
        if candidate != self.candidate:
            self.reset_candidate()
            self.candidate = candidate
        fresh = self.samples[-1] if self.samples else None
        if not fresh or abs(float(now) - fresh[0]) > 1e-6 or gallery is None:
            self.last_confirmation_reason = 'no_fresh_observation'
            return False
        if self.votes and fresh[0] <= self.votes[-1][0] + 1e-6:
            self.last_confirmation_reason = 'already_counted_observation'
            return False
        kind = body_kind or fresh[2]
        references = [s.feature for s in gallery.samples
                      if s.quality_details.get('body_kind', 'full') == kind]
        if not references and fresh[2] == 'full' and fresh[3]['support_feature'] is not None:
            kind = 'upper'
            references = [s.feature for s in gallery.samples
                          if s.quality_details.get('body_kind', 'full') == 'upper']
        if not references:
            self.last_confirmation_reason = 'reference_kind_unavailable'
            self.votes.clear()
            return False
        full_refs = [s.feature for s in gallery.samples
                     if s.quality_details.get('body_kind', 'full') == 'full']
        # Re-score the real fragment, including clear views before the route
        # gate first yielded a candidate. No cached image gets a new timestamp.
        self.votes.clear()
        previous = None
        latest_usable = False
        for sample in self.samples:
            at, vector, sample_kind, metadata = sample
            if now - at > self.window_seconds:
                continue
            if previous is not None:
                dt = at - previous[0]
                discontinuous = dt > max_gap
                first, second = previous[3]['box'], metadata['box']
                if first is not None and second is not None and dt > 1e-6:
                    distance = np.hypot(second[0] + second[2]/2 - first[0] - first[2]/2,
                                        second[1] + second[3]/2 - first[1] - first[3]/2)
                    discontinuous |= distance / max(first[3], second[3], 1) / dt > 5.0
                if discontinuous:
                    self.votes.clear()
            previous = sample
            if kind != sample_kind:
                if kind != 'upper' or metadata['support_feature'] is None:
                    continue
                if full_refs and max(float(np.dot(vector, ref)) for ref in full_refs) < full_min_similarity:
                    self.votes.clear()
                    continue
                vector = metadata['support_feature']
            score = max(float(np.dot(vector, ref)) for ref in references)
            if score < min(similarity, full_min_similarity):
                self.votes.clear()
                continue
            self.votes.append((at, score))
            if abs(at - now) <= 1e-6:
                latest_usable = True
        consistent = bool(self.votes) and float(np.median([vote[1] for vote in self.votes])) >= similarity
        confirmed = (len(self.votes) >= minimum
                     and self.votes[-1][0] - self.votes[0][0] >= duration - 1e-6 and consistent and latest_usable)
        self.last_confirmation_reason = ('confirmed' if confirmed else 'awaiting_independent_observations'
                                         if consistent else 'awaiting_consistent_tracklet')
        return confirmed

    def snapshot(self):
        return {'fresh_samples': len(self.samples), 'candidate': self.candidate,
                'candidate_votes': len(self.votes),
                'confirmation_reason': self.last_confirmation_reason,
                'first_observed_at': self.samples[0][0] if self.samples else None,
                'last_observed_at': self.samples[-1][0] if self.samples else None}
