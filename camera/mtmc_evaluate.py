"""Offline evaluator for manually annotated multi-camera recordings.

Ground truth and predictions are deliberately plain JSON so a lab operator can
edit them without a special annotation database. This module computes identity
and handoff diagnostics; it does not claim to implement HOTA.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from association import hungarian_maximize


FORMAT_VERSION = "mtmc-ground-truth-v1"


def _floor_position(item, label):
    position = item.get("floor_position_m")
    if position is None:
        return None
    if not isinstance(position, dict) or not all(axis in position for axis in ("x", "y")):
        raise ValueError(f"{label}.floor_position_m requires x and y")
    try:
        point = (float(position["x"]), float(position["y"]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.floor_position_m must contain finite numbers") from error
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{label}.floor_position_m must contain finite numbers")
    return point


def _percentile_nearest_rank(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_diagnostic_predictions(path):
    """Convert the offline replay's per-frame JSONL into evaluator predictions."""
    predictions = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for track in record.get("tracks", []):
                item = {
                    "camera_id": record["camera_id"],
                    "frame_index": record["frame_index"],
                    "local_id": track["track_id"],
                    "global_id": track.get("global_id"),
                }
                position = track.get("position")
                if position is not None:
                    item["floor_position_m"] = position
                    _floor_position(item, f"line {line_number} track {track['track_id']}")
                predictions.append(item)
    return {"predictions": predictions}


def validate_ground_truth(payload):
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"ground truth format_version must be {FORMAT_VERSION}")
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("ground truth annotations must be a list")
    for index, item in enumerate(annotations):
        if not item.get("camera_id") or not item.get("person_id"):
            raise ValueError(f"annotation {index} requires camera_id and person_id")
        if "frame_index" not in item and "unix_time" not in item:
            raise ValueError(f"annotation {index} requires frame_index or unix_time")
        detection = item.get("local_detection")
        if not isinstance(detection, dict) or "local_id" not in detection:
            raise ValueError(f"annotation {index} requires local_detection.local_id")
        _floor_position(item, f"annotation {index}")
    for index, item in enumerate(payload.get("handoffs", [])):
        required = {"person_id", "from_camera", "to_camera", "start_time", "end_time"}
        if not required.issubset(item):
            raise ValueError(f"handoff {index} is missing required fields")
    return payload


def load_session_timestamps(session_directory):
    """Read DatasetRecorder timestamp files keyed by camera and frame."""
    session_directory = Path(session_directory)
    info_path = session_directory / "info.json"
    if not info_path.exists():
        raise ValueError(f"recording metadata not found: {info_path}")
    info = _load(info_path)
    timestamps = {}
    for camera_id, stream in info.get("streams", {}).items():
        filename = stream.get("timestamps_file")
        if not filename:
            continue
        path = session_directory / filename
        with path.open("r", encoding="utf-8", newline="") as handle:
            timestamps[camera_id] = {
                int(row["frame_index"]): float(row["unix_time"])
                for row in csv.DictReader(handle)
            }
    return timestamps


def apply_session_timestamps(ground_truth, timestamps):
    payload = json.loads(json.dumps(ground_truth))
    for index, annotation in enumerate(payload.get("annotations", [])):
        if "frame_index" not in annotation:
            continue
        camera_id = annotation.get("camera_id")
        frame_index = int(annotation["frame_index"])
        try:
            recorded_time = timestamps[camera_id][frame_index]
        except KeyError as error:
            raise ValueError(
                f"annotation {index} references missing recorded frame {camera_id}:{frame_index}"
            ) from error
        annotation.setdefault("unix_time", recorded_time)
    return payload


def _key(item):
    position = (
        ("frame", int(item["frame_index"]))
        if "frame_index" in item
        else ("time", round(float(item["unix_time"]), 3))
    )
    local = item.get("local_detection", {}).get("local_id", item.get("local_id"))
    return str(item["camera_id"]), position, str(local)


def _identity_assignment(rows):
    people = sorted({row["person_id"] for row in rows})
    globals_ = sorted({row["predicted_global_id"] for row in rows if row.get("predicted_global_id")})
    if not people or not globals_:
        return {}, 0
    matrix = np.zeros((len(people), len(globals_)), dtype=np.float64)
    for row in rows:
        predicted = row.get("predicted_global_id")
        if predicted:
            matrix[people.index(row["person_id"]), globals_.index(predicted)] += 1
    pairs = hungarian_maximize(matrix)
    mapping = {globals_[column]: people[row] for row, column in pairs if matrix[row, column] > 0}
    return mapping, int(sum(matrix[row, column] for row, column in pairs))


def _metrics(rows, handoffs=None):
    mapping, idtp = _identity_assignment(rows)
    total_gt = len(rows)
    total_pred = sum(bool(row.get("predicted_global_id")) for row in rows)
    idfn = total_gt - idtp
    idfp = total_pred - idtp
    denominator = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / denominator if denominator else None

    switches = 0
    by_person = defaultdict(list)
    for row in rows:
        order = float(row.get("unix_time", row.get("frame_index", 0)))
        by_person[row["person_id"]].append((order, row.get("predicted_global_id")))
    for sequence in by_person.values():
        previous = None
        for _, current in sorted(sequence):
            if not current:
                continue
            if previous is not None and current != previous:
                switches += 1
            previous = current

    predicted_owners = defaultdict(set)
    for row in rows:
        if row.get("predicted_global_id"):
            predicted_owners[row["predicted_global_id"]].add(row["person_id"])
    wrong_rows = sum(
        bool(row.get("predicted_global_id"))
        and mapping.get(row["predicted_global_id"]) != row["person_id"]
        for row in rows
    )

    position_annotated = sum(row.get("floor_position_m") is not None for row in rows)
    position_errors = [row["position_error_m"] for row in rows if row.get("position_error_m") is not None]
    position_metrics = {
        "annotated": position_annotated,
        "evaluated": len(position_errors),
        "coverage": round(len(position_errors) / position_annotated, 6) if position_annotated else None,
        "median_error_m": round(_percentile_nearest_rank(position_errors, 0.5), 3) if position_errors else None,
        "p90_error_m": round(_percentile_nearest_rank(position_errors, 0.9), 3) if position_errors else None,
        "max_error_m": round(max(position_errors), 3) if position_errors else None,
    }

    handoff_results = []
    for handoff in handoffs or []:
        person = handoff["person_id"]
        start, end = float(handoff["start_time"]), float(handoff["end_time"])
        source = [
            row.get("predicted_global_id")
            for row in rows
            if row["person_id"] == person
            and row["camera_id"] == handoff["from_camera"]
            and float(row.get("unix_time", row.get("frame_index", 0))) <= end
        ]
        target = [
            row.get("predicted_global_id")
            for row in rows
            if row["person_id"] == person
            and row["camera_id"] == handoff["to_camera"]
            and float(row.get("unix_time", row.get("frame_index", 0))) >= start
        ]
        source_id = next((item for item in reversed(source) if item), None)
        target_id = next((item for item in target if item), None)
        handoff_results.append(bool(source_id and source_id == target_id))

    handoff_total = len(handoff_results)
    successes = sum(handoff_results)
    return {
        "annotations": total_gt,
        "identity_true_positives": idtp,
        "identity_false_positives": idfp,
        "identity_false_negatives": idfn,
        "idf1": round(idf1, 6) if idf1 is not None else None,
        "id_switches": switches,
        "handoff_total": handoff_total,
        "handoff_successes": successes,
        "cross_camera_handoff_success_rate": round(successes / handoff_total, 6) if handoff_total else None,
        "missed_handoff_rate": round((handoff_total - successes) / handoff_total, 6) if handoff_total else None,
        "wrong_identity_inheritance_rate": round(wrong_rows / total_pred, 6) if total_pred else None,
        "ambiguous_global_ids": sum(len(owners) > 1 for owners in predicted_owners.values()),
        "position": position_metrics,
    }


def evaluate(ground_truth, predictions):
    ground_truth = validate_ground_truth(ground_truth)
    prediction_items = predictions.get("predictions", [])
    for index, item in enumerate(prediction_items):
        _floor_position(item, f"prediction {index}")
    prediction_index = {_key(item): item for item in prediction_items}
    rows = []
    for annotation in ground_truth["annotations"]:
        prediction = prediction_index.get(_key(annotation), {})
        actual_position = _floor_position(annotation, "annotation")
        predicted_position = _floor_position(prediction, "prediction")
        rows.append(
            {
                **annotation,
                "predicted_global_id": prediction.get("global_id"),
                "position_error_m": (
                    math.dist(actual_position, predicted_position)
                    if actual_position is not None and predicted_position is not None
                    else None
                ),
            }
        )
    if not rows:
        return {
            "status": "no_real_evaluation_data",
            "message": "尚无真实评测数据",
            "metrics": _metrics([]),
            "groups": {},
        }

    groups = {}
    group_specs = {
        "camera": lambda row: row["camera_id"],
        "crowd_size": lambda row: str(row.get("attributes", {}).get("crowd_size", "unknown")),
        "occlusion": lambda row: str(bool(row.get("attributes", {}).get("occluded", False))).lower(),
        "registered": lambda row: str(bool(row.get("attributes", {}).get("registered", False))).lower(),
    }
    for group_name, selector in group_specs.items():
        buckets = defaultdict(list)
        for row in rows:
            buckets[selector(row)].append(row)
        groups[group_name] = {key: _metrics(value) for key, value in sorted(buckets.items())}
    handoff_pairs = defaultdict(list)
    for item in ground_truth.get("handoffs", []):
        handoff_pairs[f"{item['from_camera']}->{item['to_camera']}"] .append(item)
    groups["camera_pair"] = {
        key: _metrics(
            [
                row
                for row in rows
                if row["camera_id"] in set(key.split("->", 1))
            ],
            handoffs=value,
        )
        for key, value in sorted(handoff_pairs.items())
    }
    return {
        "status": "evaluated",
        "format_version": FORMAT_VERSION,
        "metrics": _metrics(rows, ground_truth.get("handoffs", [])),
        "groups": groups,
        "notes": [
            "Position error compares paired floor coordinates in meters; missing predictions reduce coverage.",
            "Position P90 uses nearest-rank percentile.",
            "IDF1 uses optimal one-to-one identity assignment.",
            "HOTA is not computed.",
        ],
    }


def markdown_report(result):
    if result["status"] != "evaluated":
        return "# MTMC 评测报告\n\n尚无真实评测数据。\n"
    metrics = result["metrics"]
    lines = [
        "# MTMC 评测报告",
        "",
        f"- 标注数：{metrics['annotations']}",
        f"- 跨摄交接成功率：{metrics['cross_camera_handoff_success_rate']}",
        f"- 错误身份继承率：{metrics['wrong_identity_inheritance_rate']}",
        f"- 漏交接率：{metrics['missed_handoff_rate']}",
        f"- ID switches：{metrics['id_switches']}",
        f"- IDF1：{metrics['idf1']}",
        f"- 地图位置真值标注：{metrics['position']['annotated']}，成功匹配：{metrics['position']['evaluated']}，覆盖率：{metrics['position']['coverage']}",
        f"- 地图位置误差（米）：中位数 {metrics['position']['median_error_m']}，P90 {metrics['position']['p90_error_m']}，最大 {metrics['position']['max_error_m']}",
        "",
        "> 本报告不计算 HOTA，也不把自定义指标称为标准 HOTA。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate recorded MTMC predictions")
    parser.add_argument("--ground-truth", required=True)
    predictions_source = parser.add_mutually_exclusive_group(required=True)
    predictions_source.add_argument("--predictions", help="Evaluator predictions JSON")
    predictions_source.add_argument("--diagnostic-jsonl", help="Output of scripts/diagnose_recorded_tracking.py")
    parser.add_argument("--output", required=True, help="Output path without extension")
    parser.add_argument(
        "--session",
        help="DatasetRecorder session directory; validates frames and supplies recorded timestamps",
    )
    args = parser.parse_args(argv)
    ground_truth = _load(args.ground_truth)
    if args.session:
        ground_truth = apply_session_timestamps(
            ground_truth, load_session_timestamps(args.session)
        )
    predictions = (
        _load(args.predictions)
        if args.predictions
        else load_diagnostic_predictions(args.diagnostic_jsonl)
    )
    result = evaluate(ground_truth, predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.with_suffix(".md").write_text(markdown_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
