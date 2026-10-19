import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feishu_app import FeishuAppClient, FeishuError, protect, save_credentials, load_credentials


class FeishuAppTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.now = 1000
        self.client = FeishuAppClient({'app_id':'cli_synthetic', 'app_secret':'synthetic'},
                                     request=self.request, wall=lambda: self.now)

    def request(self, method, path, payload, token=None):
        self.calls.append((method, path, payload, token))
        if '/auth/' in path:
            return {'code':0, 'expire':7200, 'tenant_access_token':'synthetic-token'}
        return {'code':0, 'data':{'message_id':'om_synthetic'}}

    def test_token_cached_and_refreshed(self):
        self.client.token()
        self.client.token()
        self.assertEqual(len(self.calls), 1)
        self.now += 7141
        self.client.token()
        self.assertEqual(len(self.calls), 2)

    def test_only_group_transport_is_accepted(self):
        from management_notifications import ManagementNotifications
        with self.assertRaises(ValueError):
            ManagementNotifications._validate_config({'lab_id':'lab','routes':[{
                'id':'obsolete','transport':'app','labs':['lab'],'event_types':['safety.no_mask']}]})




    def test_windows_credentials_encrypted_round_trip(self):
        import os
        if os.name != 'nt':
            self.skipTest('Windows only')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'credentials.dpapi'
            saved = {'app_id':'cli_synthetic', 'app_secret':'never-plaintext-synthetic'}
            save_credentials(saved,path)
            self.assertNotIn(b'never-plaintext-synthetic',path.read_bytes())
            self.assertEqual(load_credentials(path),saved)

    def test_webhook_failure_is_specific_without_echoing_secret(self):
        from management_notifications import send_feishu
        with self.assertRaises(FeishuError) as error:
            send_feishu('cli_synthetic',{})
        self.assertIn('App ID',str(error.exception))
        self.assertNotIn('cli_synthetic',str(error.exception))
        from io import BytesIO
        class Response(BytesIO):
            def __enter__(self):
                return self
            def __exit__(self,*args):
                self.close()
        with patch('management_notifications.urlopen',return_value=Response(b'{"code":19021}')):
            with self.assertRaises(FeishuError) as error:
                send_feishu('https://open.feishu.cn/open-apis/bot/v2/hook/synthetic-secret',{})
        self.assertIn('19021',str(error.exception))
        self.assertNotIn('synthetic-secret',str(error.exception))



if __name__ == '__main__':
    unittest.main()
