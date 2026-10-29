import unittest
from unittest.mock import patch

import numpy as np

from mtmc_config import load_mtmc_config
from reid_gallery import TrackFeatureGallery
from tracklet_evidence import TrackletEvidence
from yolo_person_tracker import CrossCameraTrackCoordinator


class MeanExtractor:
    def extract(self, frame, box):
        x, y, w, h = map(int, box)
        value = frame[y:y+h, x:x+w].mean(axis=(0, 1))
        return value / max(np.linalg.norm(value), 1e-8)


class TrackletRecoveryTests(unittest.TestCase):
    def setUp(self):
        config = load_mtmc_config()
        config['reid']['tracklet_recovery_enabled'] = True
        self.coordinator = CrossCameraTrackCoordinator(
            MeanExtractor(), mtmc_config=config, feature_refresh_seconds=0)
        self.quality = patch('yolo_person_tracker.BodyQualityScorer.assess', return_value={'score': .9})
        self.quality.start()
        self.addCleanup(self.quality.stop)
        self.red = np.full((140, 220, 3), (0, 0, 180), dtype=np.uint8)
        self.blue = np.full_like(self.red, (180, 0, 0))

    def update(self, local, now, frame=None, box=(10, 20, 40, 80), partial=False):
        track = {'track_id': local, 'box': box, 'partial_body': partial}
        return self.coordinator.update_batch('cam_1', [track], self.red if frame is None else frame,
                                              monotonic_time=now, unix_time=100+now)[local]

    def known(self, gid, name='registered_a'):
        self.coordinator.set_identity(gid, {'known': True, 'person_id': name, 'name': name}, 'cam_1', now=1)

    def test_new_local_waits_for_three_fresh_views_then_recovers_original(self):
        old = self.update(1, 1)
        self.known(old)
        self.coordinator.forget_local('cam_1', 1)
        first = self.update(9, 1.2)
        self.assertNotEqual(first, old)
        self.assertIsNone(self.coordinator.identity_for(first, 'cam_1', now=1.2))
        self.assertNotEqual(self.update(9, 1.35), old)
        self.assertEqual(self.update(9, 1.5), old)

    def test_cached_features_do_not_create_votes(self):
        old = self.update(1, 1)
        self.coordinator.forget_local('cam_1', 1)
        self.update(9, 1.2)
        with patch.object(self.coordinator.feature_extractor, 'extract', return_value=None):
            for now in (1.3, 1.4, 1.5, 1.6, 1.7):
                self.assertNotEqual(self.update(9, now), old)
        self.assertEqual(len(self.coordinator.tracklet_evidence[('cam_1', 9)].samples), 1)

    def test_equal_candidates_remain_unknown(self):
        tracks = [{'track_id': 1, 'box': (10, 20, 40, 80)},
                  {'track_id': 2, 'box': (100, 20, 40, 80)}]
        originals = self.coordinator.update_batch('cam_1', tracks, self.red, monotonic_time=1)
        for local in (1, 2):
            self.coordinator.forget_local('cam_1', local)
        for now in (1.2, 1.4, 1.6, 1.8):
            self.assertNotIn(self.update(9, now, box=(55, 20, 40, 80)), originals.values())

    def test_matching_appearance_cannot_override_impossible_motion(self):
        old = self.update(1, 1)
        self.coordinator.forget_local('cam_1', 1)
        for now in (1.2, 1.4, 1.6):
            self.assertNotEqual(self.update(9, now, box=(170, 20, 40, 80)), old)

    def test_same_local_id_changing_person_withdraws_name_and_freezes_old_memory(self):
        old = self.update(1, 1)
        self.known(old)
        clean = self.coordinator.global_galleries[old].representative.copy()
        suspect = self.update(1, 1.2, self.blue)
        self.assertEqual(suspect, old)
        self.assertIsNone(self.coordinator.identity_for(suspect, 'cam_1', now=1.2))
        self.assertFalse(self.coordinator.identity_binding_is_current(
            self.coordinator.global_identities[old]['identity'], suspect))
        np.testing.assert_allclose(self.coordinator.global_galleries[old].representative, clean)
        self.assertEqual(self.coordinator.global_identities[old]['identity']['person_id'], 'registered_a')
        self.update(1, 1.4, self.blue)
        self.assertNotEqual(self.update(1, 1.6, self.blue), old)

    def test_two_people_swapping_local_ids_recover_separately(self):
        boxes = [(10, 20, 40, 80), (65, 20, 40, 80)]
        tracks = [{'track_id': i+1, 'box': box} for i, box in enumerate(boxes)]
        frame = self.red.copy(); frame[20:100, 65:105] = (180, 0, 0)
        original = self.coordinator.update_batch('cam_1', tracks, frame, monotonic_time=1)
        self.known(original[1], 'a'); self.known(original[2], 'b')
        swapped = self.blue.copy(); swapped[20:100, 65:105] = (0, 0, 180)
        results = []
        for now in (1.2, 1.4, 1.6, 1.8, 2.0):
            results.append(self.coordinator.update_batch('cam_1', tracks, swapped, monotonic_time=now))
        self.assertNotIn(results[2][1], original.values())
        self.assertNotIn(results[2][2], original.values())
        self.assertEqual(results[-1], {1: original[2], 2: original[1]})

    def test_one_bad_view_does_not_split_identity_or_update_memory(self):
        old = self.update(1, 1)
        self.known(old)
        self.assertEqual(self.update(1, 1.2, self.blue), old)
        self.assertIsNone(self.coordinator.identity_for(old, 'cam_1', now=1.2))
        self.assertEqual(self.update(1, 1.4), old)
        self.assertEqual(self.coordinator.identity_for(old, 'cam_1', now=1.4)['person_id'], 'registered_a')
        self.assertFalse(self.coordinator.identity_suspects)

    def test_partial_recovery_uses_upper_memory_and_preserves_full_memory(self):
        old = self.update(1, 1)
        memory = self.coordinator.global_galleries[old]
        clean = memory.representative.copy()
        self.assertIsNotNone(memory.representative_for('upper'))
        self.coordinator.forget_local('cam_1', 1)
        results = [self.update(9, now, box=(10, 20, 40, 44), partial=True) for now in (1.2, 1.4, 1.6)]
        self.assertNotEqual(results[0], old)
        self.assertEqual(results[-1], old)
        np.testing.assert_allclose(self.coordinator.global_galleries[old].representative, clean)

    def test_occluded_crops_never_refresh_clean_memory(self):
        self.update(1, 1)
        before = self.coordinator.local_galleries[('cam_1', 1)].last_updated_monotonic
        overlapping = [{'track_id': 1, 'box': (10, 20, 40, 80)},
                       {'track_id': 2, 'box': (12, 20, 40, 80)}]
        self.coordinator.update_batch('cam_1', overlapping, self.blue, monotonic_time=1.2)
        self.assertEqual(self.coordinator.local_galleries[('cam_1', 1)].last_updated_monotonic, before)

    def test_reset_and_forget_clear_evidence(self):
        self.update(1, 1)
        self.coordinator.forget_camera('cam_1')
        self.assertFalse(self.coordinator.tracklet_evidence)
        self.update(1, 2)
        self.coordinator.reset_replay_cycle()
        self.assertFalse(self.coordinator.tracklet_evidence)
        self.assertFalse(self.coordinator.tracklet_occluded)

    def test_cross_camera_high_score_also_requires_fresh_tracklet(self):
        class Floor:
            config = {'calibrated': False, 'zones': [], 'camera_transitions': [
                {'from': 'cam_entrance', 'to': 'cam_2', 'max_gap_seconds': 20}]}
        config = self.coordinator.mtmc_config
        c = CrossCameraTrackCoordinator(MeanExtractor(), Floor(), mtmc_config=config, feature_refresh_seconds=0)
        track = [{'track_id': 1, 'box': (10, 20, 40, 80)}]
        old = c.update_batch('cam_entrance', track, self.red, monotonic_time=1)[1]
        c.set_identity(old, {'known': True, 'person_id': 'a', 'name': 'a'}, 'cam_entrance', now=1)
        c.update_batch('cam_entrance', [], self.red, active_local_ids=[], monotonic_time=1.1)
        values = [c.update_batch('cam_2', track, self.red, monotonic_time=t)[1] for t in (1.2, 1.4, 1.6)]
        self.assertNotEqual(values[0], old)
        self.assertNotEqual(values[1], old)
        self.assertEqual(values[2], old)

    def test_contradiction_cannot_reuse_earlier_candidate_votes(self):
        old = self.update(1, 1)
        self.coordinator.forget_local('cam_1', 1)
        self.update(9, 1.2)
        self.update(9, 1.4)
        self.assertNotEqual(self.update(9, 1.6, self.blue), old)
        self.assertNotEqual(self.update(9, 1.8), old)

    def test_contained_fragment_does_not_pollute_large_body_crop(self):
        self.update(1, 1)
        memory = self.coordinator.local_galleries[('cam_1', 1)].representative.copy()
        tracks = [{'track_id': 1, 'box': (10, 20, 40, 80)},
                  {'track_id': 2, 'box': (18, 25, 12, 20)}]
        self.coordinator.update_batch('cam_1', tracks, self.blue, monotonic_time=1.2)
        np.testing.assert_allclose(self.coordinator.local_galleries[('cam_1', 1)].representative, memory)

    def test_head_only_edge_detection_waits_for_usable_body(self):
        first = self.update(9, 1, box=(10, 115, 40, 25))
        self.assertFalse(self.coordinator.local_galleries[('cam_1', 9)].samples)
        self.assertNotIn(('cam_1', 9), self.coordinator.tracklet_evidence)
        self.assertEqual(self.update(9, 1.2), first)
        self.assertIsNotNone(self.coordinator.local_galleries[('cam_1', 9)].representative)

    def test_edge_clipped_body_is_not_added_as_full_body(self):
        self.update(9, 1, box=(0, 20, 40, 80))
        memory = self.coordinator.local_galleries[('cam_1', 9)]
        self.assertIsNone(memory.representative)
        self.assertIsNotNone(memory.representative_for('upper'))


class MemoryTests(unittest.TestCase):
    def test_contaminated_and_nonfinite_quality_are_rejected(self):
        gallery = TrackFeatureGallery(robust=True)
        for details, quality in [({'contaminated': True}, .9), ({'occluded': True}, .9), ({}, float('nan'))]:
            self.assertFalse(gallery.add([1, 0], quality=quality, camera_id='a', quality_details=details))
        self.assertFalse(gallery.samples)

    def test_full_and_upper_vectors_are_not_compared_or_averaged(self):
        full, upper = TrackFeatureGallery(robust=True), TrackFeatureGallery(robust=True)
        full.add([1, 0], quality=.9, camera_id='a')
        upper.add([1, 0], quality=.9, camera_id='b', quality_details={'body_kind': 'upper'})
        self.assertIsNone(full.compare(upper)['score'])
        self.assertIsNone(upper.representative)

    def test_one_lucky_crop_cannot_fill_multiple_pair_votes(self):
        query, memory = TrackFeatureGallery(robust=True), TrackFeatureGallery(robust=True)
        for v in ([1, 0, 0], [0, 1, 0], [0, .9, .4]):
            query.add(v, quality=.9, camera_id='a')
        for v in ([1, 0, 0], [.9, 0, .4], [.9, 0, -.4]):
            memory.add(v, quality=.9, camera_id='b')
        self.assertLess(query.compare(memory)['score'], .5)

    def test_repeated_timestamp_and_expired_votes_do_not_confirm(self):
        evidence, gallery = TrackletEvidence(), TrackFeatureGallery(robust=True)
        gallery.add([1, 0], quality=.9, camera_id='a')
        evidence.add([1, 0], 1)
        for _ in range(5):
            self.assertFalse(evidence.confirm('p', gallery, 1, similarity=.8))
        self.assertEqual(len(evidence.votes), 1)
        evidence.add([1, 0], 2)
        self.assertFalse(evidence.confirm('p', gallery, 2, similarity=.8))
        self.assertEqual(len(evidence.votes), 1)

    def test_recent_fragment_does_not_use_expired_old_view_as_query(self):
        evidence, target = TrackletEvidence(), TrackFeatureGallery(robust=True)
        target.add([1, 0], quality=.9, camera_id='a')
        evidence.add([0, 1], 1, quality=.8, unix_time=101)
        evidence.add([0, 1], 1.2, quality=.8, unix_time=101.2)
        evidence.add([1, 0], 3, quality=.8, unix_time=103)
        query = evidence.query_gallery(.45)
        self.assertEqual(len(query.samples), 1)
        self.assertGreater(query.compare(target)['score'], .99)
        self.assertEqual(query.samples[0].unix_time, 103)

    def test_confirmation_uses_the_same_body_kind_as_candidate_scoring(self):
        evidence, memory = TrackletEvidence(), TrackFeatureGallery(robust=True)
        for kind in ('full', 'upper'):
            memory.add([1, 0], quality=.9, camera_id='a', quality_details={'body_kind': kind})
        accepted = []
        for now in (1, 1.2, 1.4):
            evidence.add([.6, .8], now, support_feature=[1, 0], support_quality=.9)
            accepted.append(evidence.confirm('p', memory, now, similarity=.8, body_kind='upper'))
        self.assertEqual(accepted, [False, False, True])

    def test_current_upper_crop_does_not_query_with_stale_full_body(self):
        evidence = TrackletEvidence()
        evidence.add([1, 0], 1, support_feature=[0, 1], support_quality=.9)
        evidence.add([0, 1], 1.2, kind='upper')
        query = evidence.query_gallery(.45, {'upper'})
        self.assertIsNone(query.representative)
        self.assertIsNotNone(query.representative_for('upper'))

    def test_fragment_can_tolerate_one_moderate_clean_outlier(self):
        evidence, memory = TrackletEvidence(), TrackFeatureGallery(robust=True)
        memory.add([1, 0], quality=.9, camera_id='a')
        accepted = []
        for now, score in ((1, .85), (1.2, .6), (1.4, .86)):
            evidence.add([score, (1-score*score)**.5], now)
            accepted.append(evidence.confirm('p', memory, now, similarity=.8))
        self.assertEqual(accepted, [False, False, True])

    def test_one_high_crop_does_not_confirm_a_weak_fragment(self):
        evidence, memory = TrackletEvidence(), TrackFeatureGallery(robust=True)
        memory.add([1, 0], quality=.9, camera_id='a')
        for now, score in ((1, .98), (1.2, .55), (1.4, .56), (1.6, .55)):
            evidence.add([score, (1-score*score)**.5], now)
            self.assertFalse(evidence.confirm('p', memory, now, similarity=.8))

    def test_first_eligible_candidate_can_use_existing_real_fragment(self):
        evidence, memory = TrackletEvidence(), TrackFeatureGallery(robust=True)
        memory.add([1, 0], quality=.9, camera_id='a')
        for now in (1, 1.2, 1.4):
            evidence.add([1, 0], now, box=(10, 20, 40, 80))
        self.assertTrue(evidence.confirm('p', memory, 1.4, similarity=.8))
        self.assertEqual(len(evidence.votes), 3)

    def test_motion_jump_breaks_otherwise_matching_fragment(self):
        evidence, memory = TrackletEvidence(), TrackFeatureGallery(robust=True)
        memory.add([1, 0], quality=.9, camera_id='a')
        for now, x in ((1, 10), (1.1, 200), (1.2, 10)):
            evidence.add([1, 0], now, box=(x, 20, 40, 80))
        self.assertFalse(evidence.confirm('p', memory, 1.2, similarity=.8))

    def test_torso_match_cannot_override_contradictory_clean_full_body(self):
        evidence, memory = TrackletEvidence(), TrackFeatureGallery(robust=True)
        for kind in ('full', 'upper'):
            memory.add([1, 0], quality=.9, camera_id='a', quality_details={'body_kind': kind})
        for now in (1, 1.2, 1.4, 1.6):
            evidence.add([0, 1], now, support_feature=[1, 0], support_quality=.9)
            self.assertFalse(evidence.confirm('p', memory, now, similarity=.8, body_kind='upper'))
