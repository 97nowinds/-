"""STM32 environmental telemetry bridge.

The current HAFS firmware prints diagnostic text (``RAW: ...`` followed by a
``[SAFE]``/``[ALARM]`` line).  The intended firmware protocol is JSON Lines.
This module accepts both so the web application can be integrated before the
firmware protocol is upgraded.
"""

import json
import re
import threading
import time


RAW_PATTERN = re.compile(
    r"^RAW:\s*([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})\s+"
    r"([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})\s*$"
)

# Common USB-to-UART adapters used by the HAFS board.  Explicit configuration
# always wins; this list is only used to make a connected board work by default.
KNOWN_USB_UARTS = (
    (0x10C4, 0xEA60),  # Silicon Labs CP2102/CP210x
    (0x1A86, 0x7523),  # WCH CH340/CH341
    (0x1A86, 0x5523),
)


def select_hardware_port(ports):
    """Return the best matching USB-UART device name from port records."""
    candidates = list(ports)
    for vid, pid in KNOWN_USB_UARTS:
        for port in candidates:
            if getattr(port, "vid", None) == vid and getattr(port, "pid", None) == pid:
                return str(getattr(port, "device", "") or "").strip() or None
    return None


def discover_hardware_port():
    """Discover a supported USB-UART without making pyserial mandatory."""
    try:
        from serial.tools import list_ports
    except (ImportError, ModuleNotFoundError):
        return None
    try:
        return select_hardware_port(list_ports.comports())
    except Exception:
        return None


class EnvironmentSerialMonitor:
    def __init__(self, port=None, baudrate=115200, stale_seconds=30.0, device_id="HAFS-STM32"):
        self.port = str(port or "").strip()
        self.baudrate = int(baudrate)
        self.stale_seconds = float(stale_seconds)
        self.device_id = device_id
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._serial = None
        self._state = {
            "enabled": bool(self.port),
            "device_id": self.device_id,
            "port": self.port or None,
            "connected": False,
            "status": "disconnected" if self.port else "disabled",
            "temperature": None,
            "humidity": None,
            "flame_alarm": False,
            "dht_valid": False,
            "last_seen_at": None,
            "protocol": None,
            "error": None,
        }

    def start(self):
        if not self.port or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="environment-serial", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        serial_port = self._serial
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def ingest(self, payload, protocol="jsonl"):
        if not isinstance(payload, dict):
            raise ValueError("environment telemetry must be a JSON object")
        known_fields = {"device_id", "sequence", "temperature", "humidity", "flame_alarm", "dht_valid"}
        if not known_fields.intersection(payload):
            raise ValueError("environment telemetry contains no supported fields")
        updates = {}
        for field in ("temperature", "humidity"):
            if field in payload and payload[field] is not None:
                value = float(payload[field])
                if field == "temperature" and not -80.0 <= value <= 125.0:
                    raise ValueError("temperature is outside the supported range")
                if field == "humidity" and not 0.0 <= value <= 100.0:
                    raise ValueError("humidity is outside the supported range")
                updates[field] = round(value, 1)
        for field in ("flame_alarm", "dht_valid"):
            if field in payload:
                updates[field] = bool(payload[field])
        if "device_id" in payload:
            updates["device_id"] = str(payload["device_id"])[:64]
        now = time.time()
        with self._lock:
            self._state.update(updates)
            self._state.update(
                enabled=True,
                connected=True,
                status="alarm" if self._state["flame_alarm"] else "online",
                last_seen_at=now,
                protocol=protocol,
                error=None,
            )
        return self.status()

    def feed_line(self, line):
        line = str(line).strip()
        if not line:
            return False
        if line.startswith("{"):
            try:
                self.ingest(json.loads(line), protocol="jsonl")
                return True
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
        match = RAW_PATTERN.match(line)
        if match:
            raw = [int(value, 16) for value in match.groups()]
            checksum_valid = (sum(raw[:4]) & 0xFF) == raw[4]
            raw_humidity = (raw[0] << 8) | raw[1]
            raw_temperature = (raw[2] << 8) | raw[3]
            negative = bool(raw_temperature & 0x8000)
            temperature = (raw_temperature & 0x7FFF) / 10.0
            if negative:
                temperature = -temperature
            values_valid = -80.0 <= temperature <= 125.0 and 0.0 <= raw_humidity / 10.0 <= 100.0
            # A real DHT11 normally uses integer/decimal bytes rather than the
            # DHT22-style 16-bit values used by the supplied firmware.  Treat
            # impossible decoded values as invalid telemetry instead of
            # repeatedly tearing down the serial connection.
            if not values_valid:
                with self._lock:
                    self._state.update(
                        enabled=True,
                        connected=True,
                        status="online",
                        dht_valid=False,
                        last_seen_at=time.time(),
                        protocol="hafs-text",
                        error="DHT value is outside range; verify DHT11/DHT22 decoding",
                    )
                return True
            self.ingest(
                {
                    "temperature": temperature,
                    "humidity": raw_humidity / 10.0,
                    "dht_valid": checksum_valid,
                },
                protocol="hafs-text",
            )
            return True
        if line.startswith("[ALARM]"):
            self.ingest({"flame_alarm": True}, protocol="hafs-text")
            return True
        if line.startswith("[SAFE]"):
            self.ingest({"flame_alarm": False}, protocol="hafs-text")
            return True
        return False

    def status(self):
        with self._lock:
            result = dict(self._state)
        last_seen = result["last_seen_at"]
        result["age_seconds"] = None if last_seen is None else round(max(0.0, time.time() - last_seen), 1)
        if result["connected"] and result["age_seconds"] is not None and result["age_seconds"] > self.stale_seconds:
            result["connected"] = False
            result["status"] = "stale"
        return result

    def _run(self):
        try:
            import serial
        except ModuleNotFoundError:
            with self._lock:
                self._state.update(status="error", error="pyserial is not installed")
            return
        while not self._stop.is_set():
            try:
                with serial.Serial(self.port, self.baudrate, timeout=1.0) as serial_port:
                    self._serial = serial_port
                    with self._lock:
                        self._state.update(status="connected", connected=True, error=None)
                    while not self._stop.is_set():
                        data = serial_port.readline()
                        if data:
                            self.feed_line(data.decode("utf-8", errors="replace"))
            except Exception as error:
                with self._lock:
                    self._state.update(status="error", connected=False, error=str(error)[:200])
                self._stop.wait(2.0)
            finally:
                self._serial = None
