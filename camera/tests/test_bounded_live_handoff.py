import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_live_handoff import encode, saved_confirmation, submit


def response(status, payload, headers=None):
    r = Mock(status_code=status, headers=headers or {})
    r.json.return_value = payload
    return r


class TransportTests(unittest.TestCase):
    def test_unknown_receipt_retries_identical_request(self):
        s = Mock()
        s.post.side_effect = [requests.ReadTimeout(), response(200, {'receipt_id': 'r1'})]
        raw = encode({'event_id': 'same', 'captured_at': 'original', 'camera': {'image_sha256': 'original'}})
        rows = []
        result = submit(s, 'http://receiver/api/handoffs', raw, time.time()+60, time.monotonic()+60, rows.append, lambda _: None)
        self.assertEqual(result['receipt_id'], 'r1')
        self.assertEqual([c.kwargs['data'] for c in s.post.call_args_list], [raw, raw])
        self.assertTrue(rows[0]['receipt_unknown'])

    def test_busy_honors_retry_after(self):
        s = Mock()
        s.post.side_effect = [response(503, {'error': 'receiver_busy'}, {'Retry-After': '2'}), response(200, {'receipt_id': 'r2'})]
        waits = []
        result = submit(s, 'http://receiver/api/handoffs', b'{}', time.time()+60, time.monotonic()+60, lambda _: None, waits.append)
        self.assertEqual(waits, [2.])
        self.assertEqual(result['state'], 'accepted')

    def test_expired_original_is_not_republished(self):
        s = Mock()
        result = submit(s, 'http://receiver/api/handoffs', b'{}', time.time()-1, time.monotonic()+60, lambda _: None)
        s.post.assert_not_called()
        self.assertEqual(result['attempts'], 0)

    def test_processing_is_not_save_confirmation(self):
        for data in ({'status': 'processed'}, {'receipt_id': 'r'}, {'source_may_delete': 'true'}, {'image_cached': True}):
            self.assertFalse(saved_confirmation(data))
        self.assertTrue(saved_confirmation({'receipt': {'source_may_delete': True}}))


if __name__ == '__main__':
    unittest.main()
