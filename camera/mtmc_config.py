"""Validated configuration for the incremental MTMC pipeline."""

import json
from copy import deepcopy
from pathlib import Path


class MTMCConfigError(ValueError):
    pass


DEFAULT_CONFIG = {
    "algorithm_version": "mtmc-2.0.6",
    "calibration": {
        "formal_required_for_geometry": True,
        "max_stream_skew_seconds": 0.5,
        "fusion_max_distance_m": 2.0,
        "observation_max_age_seconds": 2.0,
        "max_map_speed_mps": 2.2,
        "max_motion_gap_seconds": 0.25,
        "map_position_hold_seconds": 8.0,
    },
    "reid": {
        "gallery_capacity": 12,
        "min_sample_quality": 0.45,
        "duplicate_similarity": 0.985,
        "top_k": 3,
        "feature_refresh_seconds": 0.8,
        "high_similarity": 0.82,
        "medium_similarity": 0.70,
        "match_margin": 0.08,
    },
    "association": {
        "window_seconds": 0.75,
        "global_track_ttl_seconds": 8.0,
        "identity_retention_seconds": 90.0,
        "pending_ttl_seconds": 3.0,
        "high_confidence_score": 0.78,
        "medium_confidence_score": 0.62,
        "weights": {"reid": 0.65, "time": 0.20, "geometry": 0.10, "motion": 0.05},
    },
    "events": {
        "path": "runtime/mtmc_events.jsonl",
        "memory_limit": 300,
        "retention_days": 30,
    },
}


def _merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _number(config, path, minimum, maximum, *, integer=False):
    value = config
    for key in path.split("."):
        if key not in value:
            raise MTMCConfigError(f"Missing MTMC setting: {path}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MTMCConfigError(f"MTMC setting {path} must be numeric")
    if integer and not isinstance(value, int):
        raise MTMCConfigError(f"MTMC setting {path} must be an integer")
    if not minimum <= float(value) <= maximum:
        raise MTMCConfigError(
            f"MTMC setting {path} must be in [{minimum}, {maximum}], got {value}"
        )


def validate_config(config):
    version = config.get("algorithm_version")
    if not isinstance(version, str) or not version.strip():
        raise MTMCConfigError("algorithm_version must be a non-empty string")
    ranges = {
        "calibration.max_stream_skew_seconds": (0.01, 10.0, False),
        "calibration.fusion_max_distance_m": (0.1, 50.0, False),
        "calibration.observation_max_age_seconds": (0.1, 30.0, False),
        "calibration.max_map_speed_mps": (0.2, 10.0, False),
        "calibration.max_motion_gap_seconds": (0.05, 2.0, False),
        "calibration.map_position_hold_seconds": (0.0, 30.0, False),
        "reid.gallery_capacity": (1, 100, True),
        "reid.min_sample_quality": (0.0, 1.0, False),
        "reid.duplicate_similarity": (0.0, 1.0, False),
        "reid.top_k": (1, 20, True),
        "reid.feature_refresh_seconds": (0.0, 60.0, False),
        "reid.high_similarity": (-1.0, 1.0, False),
        "reid.medium_similarity": (-1.0, 1.0, False),
        "reid.match_margin": (0.0, 1.0, False),
        "association.window_seconds": (0.01, 10.0, False),
        "association.global_track_ttl_seconds": (0.1, 300.0, False),
        "association.identity_retention_seconds": (1.0, 86400.0, False),
        "association.pending_ttl_seconds": (0.1, 60.0, False),
        "association.high_confidence_score": (0.0, 1.0, False),
        "association.medium_confidence_score": (0.0, 1.0, False),
        "events.memory_limit": (10, 10000, True),
        "events.retention_days": (1, 3650, True),
    }
    for path, (minimum, maximum, integer) in ranges.items():
        _number(config, path, minimum, maximum, integer=integer)
    reid = config["reid"]
    if reid["medium_similarity"] > reid["high_similarity"]:
        raise MTMCConfigError("reid.medium_similarity cannot exceed high_similarity")
    association = config["association"]
    if association["medium_confidence_score"] > association["high_confidence_score"]:
        raise MTMCConfigError(
            "association.medium_confidence_score cannot exceed high_confidence_score"
        )
    weights = association.get("weights", {})
    if set(weights) != {"reid", "time", "geometry", "motion"}:
        raise MTMCConfigError("association.weights must define reid/time/geometry/motion")
    if any(not isinstance(value, (int, float)) or value < 0 for value in weights.values()):
        raise MTMCConfigError("association weights must be non-negative numbers")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-6:
        raise MTMCConfigError("association weights must sum to 1.0")
    path = config["events"].get("path")
    if not isinstance(path, str) or not path.strip():
        raise MTMCConfigError("events.path must be a non-empty relative path")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise MTMCConfigError("events.path must stay inside the camera project")
    return config


def load_mtmc_config(path=None):
    path = Path(path) if path is not None else Path(__file__).parent / "config" / "mtmc.json"
    override = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return validate_config(_merge(DEFAULT_CONFIG, override))


def public_config(config):
    """Return a JSON-safe copy; configuration contains no model features."""
    return deepcopy(config)
