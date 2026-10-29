import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from cluster_joint_bridge import receiver_healthy, recover_receiver


class BridgeHealthTests(unittest.TestCase):
    def test_no_events_is_healthy(self):
        opener=Mock()
        opener.open.return_value=io.BytesIO(json.dumps({'sessions':[]}).encode())
        self.assertTrue(receiver_healthy(opener))

    def test_listener_or_invalid_json_is_not_sufficient(self):
        opener=Mock()
        opener.open.return_value=io.BytesIO(b'{"status":"ok"}')
        self.assertFalse(receiver_healthy(opener))

    def test_timeout_is_failure(self):
        opener=Mock()
        opener.open.side_effect=TimeoutError()
        self.assertFalse(receiver_healthy(opener))

    def test_remote_check_can_only_restart_the_owned_joint_unit(self):
        config={'ssh_identity':'~/.ssh/test','ssh_jump':'synthetic@jump','ssh_target':'synthetic@target'}
        with patch('cluster_joint_bridge.subprocess.run') as run:
            run.return_value=Mock(returncode=0,stdout='remote_healthy')
            self.assertEqual(recover_receiver(config),'remote_healthy')
            command=run.call_args.args[0][-1]
            self.assertIn('curl -fsS --max-time 5',command)
            self.assertIn('else systemctl --user restart lab-joint-8765.service',command)
            self.assertNotIn('lab-safety-api',command)


if __name__=='__main__': unittest.main()
