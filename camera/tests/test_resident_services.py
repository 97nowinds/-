"""Transport export contracts; no camera credentials or network required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from resident_services import publisher_command,source_url


class ResidentContracts(unittest.TestCase):
    def setUp(self):
        self.camera=dict(id='cam_1',ip='192.0.2.1',username='demo@user',
                         password='synthetic:p@ss word',stream='Streaming/Channels/101')

    def test_special_characters_are_url_escaped(self):
        url=source_url(self.camera)
        self.assertIn('demo%40user:synthetic%3Ap%40ss%20word@192.0.2.1',url)

    def test_probe_uses_input_without_unsupported_option(self):
        command=publisher_command(self.camera,18554,probe=True)
        self.assertNotIn('-rw_timeout',command)
        self.assertEqual(command[-5:],['-t','1','-f','null','-'])

    def test_publication_is_streamcopy_to_loopback(self):
        command=publisher_command(self.camera,18554)
        self.assertEqual(command[command.index('-c:v')+1],'copy')
        self.assertEqual(command[-1],'rtsp://127.0.0.1:18554/cam_1')

    def test_unexpected_source_and_destination_rejected(self):
        for field,value in [('ip','example.invalid'),('stream','arbitrary'),('id','other')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                publisher_command({**self.camera,field:value},18554)


if __name__=='__main__':unittest.main()
