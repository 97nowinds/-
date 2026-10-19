"""Append-only MTMC association history without biometric feature payloads."""

from collections import Counter, deque
import json
import atexit
import os
import queue
from pathlib import Path
import threading
import time


EVENT_TYPES = {"create", "associate", "merge", "expire", "redirect"}
FORBIDDEN_KEYS = {"feature", "features", "embedding", "embeddings", "raw_feature"}


def _sanitize(value):
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if str(key).lower() not in FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class AssociationEventStore:
    STARTUP_ROTATE_BYTES = 64 * 1024 * 1024
    def __init__(self, path, memory_limit=300, retention_days=30, now_fn=time.time):
        self.path = Path(path)
        self.memory_limit = int(memory_limit)
        self.retention_seconds = float(retention_days) * 86400.0
        self.now_fn = now_fn
        self.lock = threading.RLock()
        self.events = deque(maxlen=self.memory_limit)
        self.next_sequence = 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.startup_archive = None
        self._load_history()
        self.write_queue = None
        self.write_error = None
        if os.environ.get("LAB_ASYNC_EVENT_WRITES") == "1":
            self.write_queue = queue.Queue()
            self.writer = threading.Thread(target=self._write_loop, daemon=True, name="mtmc-event-writer")
            self.writer.start()
            atexit.register(self.close)

    def _write_loop(self):
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                while True:
                    line = self.write_queue.get()
                    if line is None:
                        handle.flush()
                        self.write_queue.task_done()
                        return
                    handle.write(line)
                    # Batch queued events into one flush on the network filesystem.
                    if self.write_queue.empty():
                        handle.flush()
                    self.write_queue.task_done()
        except Exception as error:
            self.write_error = str(error)

    def close(self):
        if self.write_queue is not None and self.writer.is_alive():
            self.write_queue.put(None)
            self.writer.join(timeout=10)

    def _load_history(self):
        if not self.path.exists():
            return
        cutoff = self.now_fn() - self.retention_seconds
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        if self.path.stat().st_size > self.STARTUP_ROTATE_BYTES:
            # Only the bounded API history is needed to start the tracker.
            # Preserve the entire original file; do not read gigabytes into RAM.
            tail = deque(maxlen=self.memory_limit)
            with self.path.open("rb") as handle:
                handle.seek(0, 2)
                position = handle.tell()
                pending = b""
                valid = 0
                while position and valid < self.memory_limit:
                    size = min(position, 65536)
                    position -= size
                    handle.seek(position)
                    pending = handle.read(size) + pending
                    lines = pending.split(b"\n")
                    pending = lines[0]
                    for line in reversed(lines[1:]):
                        try:
                            event = json.loads(line)
                            sequence = int(event.get("sequence", 0))
                        except (ValueError, TypeError, AttributeError):
                            continue
                        self.next_sequence = max(self.next_sequence, sequence + 1)
                        valid += 1
                        if float(event.get("unix_time", 0.0)) >= cutoff:
                            tail.appendleft(event)
                        if valid >= self.memory_limit:
                            break
                if position == 0 and pending and valid < self.memory_limit:
                    try:
                        event = json.loads(pending)
                        self.next_sequence = max(self.next_sequence, int(event.get("sequence", 0)) + 1)
                        if float(event.get("unix_time", 0.0)) >= cutoff:
                            tail.appendleft(event)
                    except (ValueError, TypeError, AttributeError):
                        pass
            self.events.extend(tail)
            temporary.write_text("".join(json.dumps(e, ensure_ascii=False, separators=(",", ":"))+"\n" for e in tail), encoding="utf-8")
            archive = self.path.with_name(self.path.stem + ".archive-" + str(time.time_ns()) + self.path.suffix)
            self.path.replace(archive)
            temporary.replace(self.path)
            self.startup_archive = str(archive)
            return
        # Small histories keep the original retention behavior, with bounded RAM.
        with self.path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
            for line in source:
                try:
                    event = json.loads(line)
                    self.next_sequence = max(self.next_sequence, int(event.get("sequence", 0)) + 1)
                    if float(event.get("unix_time", 0.0)) >= cutoff:
                        self.events.append(event)
                        output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"))+"\n")
                except (ValueError, TypeError, AttributeError):
                    continue
        temporary.replace(self.path)

    def record(self, event_type, **payload):
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported MTMC event type: {event_type}")
        with self.lock:
            event = _sanitize(
                {
                    "sequence": self.next_sequence,
                    "event": event_type,
                    "unix_time": float(payload.pop("unix_time", self.now_fn())),
                    **payload,
                }
            )
            self.next_sequence += 1
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            if self.write_queue is not None:
                if self.write_error:
                    raise OSError(self.write_error)
                self.write_queue.put(line)
            else:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            self.events.append(event)
            return dict(event)

    def recent(self, limit=50):
        with self.lock:
            limit = max(0, min(int(limit), self.memory_limit))
            return list(self.events)[-limit:]

    def summary(self):
        with self.lock:
            counts = Counter(event.get("event") for event in self.events)
            confidence = Counter(
                event.get("confidence")
                for event in self.events
                if event.get("confidence")
            )
            rejection_reasons = Counter(
                event.get("reason")
                for event in self.events
                if event.get("accepted") is False and event.get("reason")
            )
            return {
                "history_events": len(self.events),
                "latest_sequence": self.next_sequence - 1,
                "event_counts": dict(counts),
                "confidence_counts": dict(confidence),
                "top_rejection_reasons": dict(rejection_reasons.most_common(8)),
                "persistence": "jsonl_history_only",
                "startup_archive": self.startup_archive,
                "live_tracks_restored": False,
                "write_pending": self.write_queue.qsize() if self.write_queue is not None else 0,
                "write_error": self.write_error,
            }
