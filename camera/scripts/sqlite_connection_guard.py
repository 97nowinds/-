"""Gate WAL connection opens AND closes for SQLite 3.51.0/3.51.1.

Ordinary statements and transactions retain their existing concurrency.
See https://sqlite.org/releaselog/3_51_2.html (WAL-mode mutex deadlock fix).
"""
import os
import sqlite3
import threading

_gate = threading.RLock()
_original_connect = sqlite3.connect


def _after_fork():
    global _gate
    _gate = threading.RLock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


class GuardedConnection(sqlite3.Connection):
    def close(self):
        with _gate:
            super().close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def guarded_connect(*args, **kwargs):
    if len(args) > 5 or ('factory' in kwargs and kwargs['factory'] is not GuardedConnection):
        raise ValueError('Custom SQLite connection factory requires the open/close guard')
    kwargs['factory'] = GuardedConnection
    with _gate:
        return _original_connect(*args, **kwargs)


def install():
    sqlite3.connect = guarded_connect
