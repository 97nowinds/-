"""Offline evaluator for manually annotated multi-camera recordings.

Ground truth and predictions are deliberately plain JSON so a lab operator can
edit them without a special annotation database. This module computes identity
and handoff diagnostics; it does not claim to implement HOTA.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from association import hungarian_maximize


FORMAT_VERSION = "mtmc-ground-truth-v1"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    }


def evaluate(ground_truth, predictions):
    ground_truth = validate_ground_truth(ground_truth)
    prediction_index = {_key(item): item for item in predictions.get("predictions", [])}
    rows = []
    for annotation in ground_truth["annotations"]:
        prediction = prediction_index.get(_key(annotation), {})
        rows.append(
            {
                **annotation,
                "predicted_global_id": prediction.get("global_id"),
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
        "notes": ["IDF1 uses optimal one-to-one identity assignment.", "HOTA is not computed."],
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
        "",
        "> 本报告不计算 HOTA，也不把自定义指标称为标准 HOTA。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate recorded MTMC predictions")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
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
    result = evaluate(ground_truth, _load(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.with_suffix(".md").write_text(markdown_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
