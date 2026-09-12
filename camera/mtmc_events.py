"""Append-only MTMC association history without biometric feature payloads."""

from collections import Counter, deque
import json
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
    def __init__(self, path, memory_limit=300, retention_days=30, now_fn=time.time):
        self.path = Path(path)
        self.memory_limit = int(memory_limit)
        self.retention_seconds = float(retention_days) * 86400.0
        self.now_fn = now_fn
        self.lock = threading.RLock()
        self.events = deque(maxlen=self.memory_limit)
        self.next_sequence = 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        if not self.path.exists():
            return
        cutoff = self.now_fn() - self.retention_seconds
        retained = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            self.next_sequence = max(self.next_sequence, int(event.get("sequence", 0)) + 1)
            if float(event.get("unix_time", 0.0)) >= cutoff:
                self.events.append(event)
                retained.append(event)
        # Apply the retention policy to disk as well as the in-memory query
        # window. Live tracker state is intentionally never reconstructed.
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
                for event in retained
            ),
            encoding="utf-8",
        )
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
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
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
                "live_tracks_restored": False,
            }
