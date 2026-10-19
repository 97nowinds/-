import json
import unittest
from contextlib import contextmanager
from io import BytesIO
from unittest.mock import patch

from handoff_transport import AuthoritySync, iso_time


class AuthoritySyncValidityTests(unittest.TestCase):
    def make_sync(self):
        return AuthoritySync('source', dict(image_base='http://authority',
            sync_valid_seconds=10, sync_max_uncertainty_ms=150,
            sync_jump_limit_ms=250), 'test-token')

    @contextmanager
    def reply(self, request, timeout):
        body = json.loads(request.data)
        yield BytesIO(json.dumps(dict(**body, authority_clock='authority',
            authority_received_at=iso_time(102.02),
            authority_sent_at=iso_time(102.02))).encode())

    def test_advertised_window_includes_probe_and_is_only_ten_seconds(self):
        sync = self.make_sync()
        with patch('handoff_transport.time.time', side_effect=[100,100.04]), \
             patch('handoff_transport.time.monotonic', side_effect=[200,200.04]), \
             patch('handoff_transport.urlopen', side_effect=self.reply):
            sync.probe()
        start = sync.mapping['sync_valid_from']
        self.assertAlmostEqual(sync.mapping['sync_valid_until']-start, 10)
        self.assertEqual(sync.mapping['deadline'], 210)

    def test_failed_explicit_probe_revokes_previous_mapping(self):
        sync = self.make_sync()
        sync.mapping = {'old': True}
        with patch('handoff_transport.urlopen', side_effect=TimeoutError), \
             self.assertRaises(TimeoutError):
            sync.probe()
        self.assertIsNone(sync.mapping)

    def test_over_budget_probe_revokes_previous_mapping(self):
        sync = self.make_sync()
        sync.mapping = {'old': True}
        with patch('handoff_transport.time.time', side_effect=[100,100.4]), \
             patch('handoff_transport.time.monotonic', side_effect=[200,200.4]), \
             patch('handoff_transport.urlopen', side_effect=self.reply), \
             self.assertRaises(ValueError):
            sync.probe()
        self.assertIsNone(sync.mapping)

    def test_expiration_and_clock_jump_revoke_mapping(self):
        sync = self.make_sync()
        mapping = dict(deadline=210, received_wall=100, received_monotonic=200)
        sync.mapping = mapping.copy()
        with patch('handoff_transport.time.monotonic', return_value=210):
            self.assertIsNone(sync.current())
        self.assertIsNone(sync.mapping)
        sync.mapping = mapping.copy()
        with patch('handoff_transport.time.monotonic', return_value=201), \
             patch('handoff_transport.time.time', return_value=102):
            self.assertIsNone(sync.current())
        self.assertIsNone(sync.mapping)


if __name__ == '__main__':
    unittest.main()
