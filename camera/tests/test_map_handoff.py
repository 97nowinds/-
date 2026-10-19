import unittest
import time
from unittest.mock import Mock
from map_handoff import map_snapshot, MapHandler


class TransferAccountingTests(unittest.TestCase):
    def handler(self, sends):
        handler = object.__new__(MapHandler)
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = Mock()
        handler.connection = Mock()
        handler.connection.send.side_effect = sends
        handler._transfer_log = Mock()
        return handler

    def test_partial_write_disconnect_is_not_full_transfer(self):
        handler = self.handler([3, BrokenPipeError()])
        handler._send_jpeg('s1', b'12345678', (time.time(), time.monotonic()), .1)
        row = handler._transfer_log.call_args.args[0]
        self.assertEqual(row['body_bytes_accepted_by_socket'], 3)
        self.assertFalse(row['complete_write'])
        self.assertEqual(row['error'], 'BrokenPipeError')

    def test_partial_sends_sum_to_complete_original(self):
        handler = self.handler([3, 5])
        handler._send_jpeg('s1', b'12345678', (time.time(), time.monotonic()), .1)
        row = handler._transfer_log.call_args.args[0]
        self.assertEqual(row['body_bytes_accepted_by_socket'], 8)
        self.assertTrue(row['complete_write'])
        self.assertIn('not remote', row['evidence_scope'])


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
