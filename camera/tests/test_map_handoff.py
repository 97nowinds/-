import unittest
from map_handoff import map_snapshot


class MapHandoffTests(unittest.TestCase):
    def test_coordinates_and_uncertainty_survive_export(self):
        state = {'floor_map': {'width_m': 12, 'height_m': 7, 'people': [
            {'track_id': 'person_1', 'x': 2, 'y': 3, 'position_estimated': True, 'position_age_seconds': 4}
        ]}}
        payload = map_snapshot(state)
        self.assertEqual(payload['people'][0]['x_m'], 2)
        self.assertEqual(payload['people'][0]['y_m'], 3)
        self.assertTrue(payload['people'][0]['position_estimated'])
        self.assertEqual(payload['people'][0]['position_age_seconds'], 4)
        state['floor_map']['people'][0]['x'] = 5
        self.assertEqual(map_snapshot(state)['map_revision'], payload['map_revision'])
        state['floor_map']['width_m'] = 13
        self.assertNotEqual(map_snapshot(state)['map_revision'], payload['map_revision'])

    def test_unmapped_track_is_not_given_fabricated_position(self):
        payload = map_snapshot({'cameras': [{'id': 'cam_2', 'yolo_tracks': [
            {'track_id': 'person_2', 'localization_reason': 'outside calibrated aisle', 'counted': False}
        ]}]})
        self.assertEqual(payload['people'], [])
        self.assertEqual(payload['camera_tracks'][0]['track_id'], 'person_2')
        self.assertNotIn('x_m', payload['camera_tracks'][0])


if __name__ == '__main__':
    unittest.main()
