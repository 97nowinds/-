import unittest
from types import SimpleNamespace

from hardware_serial import EnvironmentSerialMonitor, select_hardware_port


class EnvironmentSerialMonitorTests(unittest.TestCase):
    def test_selects_cp210x_before_other_supported_adapters(self):
        ports = [
            SimpleNamespace(device="COM8", vid=0x1A86, pid=0x7523),
            SimpleNamespace(device="COM3", vid=0x10C4, pid=0xEA60),
        ]

        self.assertEqual(select_hardware_port(ports), "COM3")

    def test_does_not_claim_an_unrelated_serial_device(self):
        ports = [SimpleNamespace(device="COM9", vid=0x1234, pid=0x5678)]

        self.assertIsNone(select_hardware_port(ports))

    def test_parses_current_hafs_text_output(self):
        monitor = EnvironmentSerialMonitor()
        # Firmware's current 16-bit / 10 decoding: humidity 55.0, temperature 26.4.
        self.assertTrue(monitor.feed_line("RAW: 02 26 01 08 31"))
        self.assertTrue(monitor.feed_line("[SAFE] System Normal"))

        state = monitor.status()
        self.assertEqual(state["humidity"], 55.0)
        self.assertEqual(state["temperature"], 26.4)
        self.assertTrue(state["dht_valid"])
        self.assertFalse(state["flame_alarm"])
        self.assertEqual(state["protocol"], "hafs-text")

    def test_parses_json_lines_protocol(self):
        monitor = EnvironmentSerialMonitor()
        monitor.feed_line(
            '{"device_id":"ENV01","temperature":26.4,"humidity":51.2,'
            '"flame_alarm":true,"dht_valid":true}'
        )

        state = monitor.status()
        self.assertEqual(state["device_id"], "ENV01")
        self.assertEqual(state["status"], "alarm")
        self.assertTrue(state["flame_alarm"])

    def test_rejects_out_of_range_telemetry(self):
        monitor = EnvironmentSerialMonitor()
        with self.assertRaises(ValueError):
            monitor.ingest({"humidity": 120})

    def test_ignores_malformed_json_line(self):
        monitor = EnvironmentSerialMonitor()
        self.assertFalse(monitor.feed_line('{"temperature":'))
        self.assertIsNone(monitor.status()["last_seen_at"])

    def test_out_of_range_firmware_raw_data_marks_dht_invalid(self):
        monitor = EnvironmentSerialMonitor()
        # Standard DHT11-like bytes would be mis-decoded by the supplied
        # firmware's DHT22-style 16-bit formula.
        monitor.feed_line("RAW: 37 00 1A 00 51")

        state = monitor.status()
        self.assertFalse(state["dht_valid"])
        self.assertIn("DHT11/DHT22", state["error"])


if __name__ == "__main__":
    unittest.main()
