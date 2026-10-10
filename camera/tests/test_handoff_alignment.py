import unittest
from frontend_server import LiveSnapshotPublisher
from handoff_result_index import SnapshotTrackIndex
from member_integration import build_monitoring_snapshot_frame_event


def event(**kwargs):
    return build_monitoring_snapshot_frame_event(event_id='snapshot.frame:synthetic-1',
        snapshot_id='synthetic-1', camera_id='cam_2',
        captured_at='1970-01-01T00:01:40.000+00:00',
        image_url='http://192.168.31.127:5174/api/handoff/test.jpg',
        frame_width=64, frame_height=36, tracks=[], **kwargs)


class AlignmentTests(unittest.TestCase):
    def test_map_reference_requires_explicit_opt_in(self):
        base = event(map_id='lab-floor', map_revision='revision-test')
        self.assertNotIn('map_id', base)
        self.assertNotIn('map_revision', base)
        enabled = event(map_id='lab-floor', map_revision='revision-test', include_map_reference=True)
        self.assertEqual(enabled['map_id'], 'lab-floor')
        with self.assertRaises(ValueError):
            event(include_map_reference=True)

    def test_location_uses_actual_position_time_not_person_update_time(self):
        camera = {'yolo_tracks': [{'track_id': 'g1', 'local_id': 7, 'box': [1, 2, 3, 4]}]}
        people = [{'track_id': 'g1', 'x': 2, 'y': 3, 'observed_at': 100,
                   'position_observed_at': 90, 'localization_state': 'held'}]
        tracks = LiveSnapshotPublisher._tracks(camera, '1970-01-01T00:01:40+00:00', people)
        self.assertNotIn('location', tracks[0])
        people[0].update(position_observed_at=99.8, localization_state='measured')
        tracks = LiveSnapshotPublisher._tracks(camera, '1970-01-01T00:01:40+00:00', people)
        self.assertEqual(tracks[0]['location']['observed_at'], '1970-01-01T00:01:39.800+00:00')

    def test_join_is_by_global_id_not_row_order_and_identity_is_track_specific(self):
        camera = {'yolo_tracks': [
            {'track_id': 'g2', 'local_id': 8, 'box': [20, 2, 3, 4]},
            {'track_id': 'g1', 'local_id': 7, 'box': [1, 2, 3, 4]}],
            'identities': [{'local_id': 7, 'global_track_id': 'g1',
                'person_id': 'p1', 'name': 'synthetic-person', 'identity_lock_status': 'locked'}]}
        people = [{'track_id': 'g1', 'x': 1, 'y': 3, 'observed_at': 100},
                  {'track_id': 'g2', 'x': 2, 'y': 3, 'observed_at': 100}]
        tracks = LiveSnapshotPublisher._tracks(camera, '1970-01-01T00:01:40+00:00', people)
        self.assertEqual([row['location']['x_m'] for row in tracks], [2, 1])
        self.assertIsNone(tracks[0]['person_id'])
        self.assertEqual(tracks[1]['person_id'], 'p1')

    def test_ambiguous_map_alias_does_not_assign_location(self):
        camera = {'yolo_tracks': [{'track_id': 'g1', 'box': [1, 2, 3, 4]}]}
        people = [{'track_id': key, 'global_track_id': 'g1', 'x': x, 'y': 1, 'observed_at': 100}
                  for key, x in [('a', 1), ('b', 8)]]
        self.assertNotIn('location', LiveSnapshotPublisher._tracks(
            camera, '1970-01-01T00:01:40+00:00', people)[0])

    def test_results_match_camera_snapshot_track_without_trusting_peer_names(self):
        index = SnapshotTrackIndex(max_snapshots=1)
        original = event()
        original['camera']['tracks'] = [
            {'track_id': '7', 'global_id': 'g1', 'person_id': 'p1', 'display_name': 'synthetic-a'},
            {'track_id': '8', 'global_id': 'g2', 'person_id': None, 'display_name': None}]
        index.remember(original)
        result = index.resolve({'camera_id': 'cam_2', 'snapshot_id': 'synthetic-1',
            'track_id': '8', 'display_name': 'incorrect-peer-name', 'instrument_id': 'i1'})
        self.assertEqual(result['global_id'], 'g2')
        self.assertIsNone(result['display_name'])
        wrong_camera = index.resolve({'camera_id': 'cam_1', 'snapshot_id': 'synthetic-1', 'track_id': '7'})
        self.assertIsNone(wrong_camera['person_id'])
        original['snapshot_id'] = 'synthetic-2'
        index.remember(original)
        self.assertIsNone(index.resolve({'camera_id': 'cam_2', 'snapshot_id': 'synthetic-1', 'track_id': '7'})['person_id'])


if __name__ == '__main__':
    unittest.main()
