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
        if '/contact/' in path:
            return {'code':0, 'data':{'user_list':[{'user_id':'ou_synthetic'}]}}
        if '/urgent_phone' in path:
            return {'code':0, 'data':{'invalid_user_id_list':[]}}
        return {'code':0, 'data':{'message_id':'om_synthetic'}}

    def test_token_cached_and_refreshed(self):
        self.client.token()
        self.client.token()
        self.assertEqual(len(self.calls), 1)
        self.now += 7141
        self.client.token()
        self.assertEqual(len(self.calls), 2)

    def test_contact_and_message_binding_and_uuid(self):
        recipient = self.client.resolve_recipient('synthetic@example.invalid')
        self.assertEqual(recipient, 'ou_synthetic')
        self.assertEqual(self.calls[-1][2], {'emails':['synthetic@example.invalid']})
        self.assertEqual(self.client.send_text(recipient, '合成数据测试', 'event1'), 'om_synthetic')
        first = self.calls[-1][2]
        self.client.send_text(recipient, '合成数据测试', 'event1')
        self.assertEqual(first['uuid'], self.calls[-1][2]['uuid'])
        self.client.send_text('ou_second', '合成数据测试', 'event1')
        self.assertNotEqual(first['uuid'], self.calls[-1][2]['uuid'])
        self.assertEqual(json.loads(first['content'])['text'], '合成数据测试')

    def test_unknown_or_ambiguous_recipient_fails_closed(self):
        original = self.request
        for users in [[], [{'user_id':'ou_a'},{'user_id':'ou_b'}]]:
            self.client.request = lambda method,path,payload,token=None: (
                {'code':0,'data':{'user_list':users}} if '/contact/' in path else original(method,path,payload,token))
            with self.assertRaises(FeishuError):
                self.client.resolve_recipient('synthetic@example.invalid')

    def test_phone_requires_explicit_enable_and_valid_recipient(self):
        with self.assertRaises(FeishuError):
            self.client.urgent_phone('om_synthetic','ou_synthetic')
        self.assertEqual(self.calls, [])
        self.client.urgent_phone('om_synthetic','ou_synthetic', enabled=True)
        self.assertEqual(self.calls[-1][0], 'PATCH')
        self.assertEqual(self.calls[-1][2], {'user_id_list':['ou_synthetic']})
        with self.assertRaises(FeishuError):
            self.client.urgent_phone('om_../../bad','ou_synthetic', enabled=True)

    def test_windows_credentials_encrypted_round_trip(self):
        import os
        if os.name != 'nt':
            self.skipTest('Windows only')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'credentials.dpapi'
            saved = {'app_id':'cli_synthetic', 'app_secret':'never-plaintext-synthetic', 'primary_open_id':'ou_synthetic'}
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

    def test_application_outbox_uses_same_event_id(self):
        from management_notifications import ManagementNotifications, iso
        with tempfile.TemporaryDirectory() as directory:
            config = {'lab_id':'lab','principals':[], 'routes':[{'id':'direct','transport':'app','recipient_key':'primary',
                      'labs':['lab'],'event_types':['person.entered']}]}
            service = ManagementNotifications(Path(directory)/'db',config,wall=lambda:2000000000.)
            event={'event_id':'entry1','event_type':'person.entered','lab_id':'lab','person_id':'p1','display_name':'合成人员',
                   'observed_at':iso(2000000000.),'evidence_source':'synthetic'}
            service.record(event)
            fake_path=Path(directory)/'configured.dpapi'
            fake_path.write_bytes(b'synthetic')
            self.client.credentials['primary_open_id']='ou_synthetic'
            with patch('feishu_app.DEFAULT_CREDENTIALS',fake_path), patch('feishu_app.load_credentials',return_value={
                        'app_id':'cli_synthetic','app_secret':'synthetic','primary_open_id':'ou_synthetic'}), \
                    patch('feishu_app.FeishuAppClient',return_value=self.client):
                service.deliver_once()
            delivery_state=service.events({'role':'admin'})[0]['deliveries'][0]['state']
            service.db.close()
            self.assertEqual(delivery_state,'sent')
            self.assertEqual(self.calls[-1][2]['receive_id'],'ou_synthetic')


if __name__ == '__main__':
    unittest.main()
