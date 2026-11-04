"""Public, versioned entry points for the lab interaction plug-in."""

from pathlib import Path
import sys


# The transferred bundle currently keeps private ``v4``/``v5`` packages next
# to this public package.  Put that exact directory before the host project's
# working directory so unrelated folders with the same legacy names cannot be
# imported accidentally during plug-in startup.
_BUNDLE_MODULE_ROOT = str(Path(__file__).resolve().parents[1])
if not sys.path or sys.path[0] != _BUNDLE_MODULE_ROOT:
    sys.path.insert(0, _BUNDLE_MODULE_ROOT)

from .api import (
    INTERFACE_VERSION,
    PLUGIN_VERSION,
    InstrumentInteractionModule,
    analyze_video,
)
from .stream import InstrumentInteractionStream

__all__ = [
    "INTERFACE_VERSION",
    "PLUGIN_VERSION",
    "InstrumentInteractionModule",
    "InstrumentInteractionStream",
    "analyze_video",
]
