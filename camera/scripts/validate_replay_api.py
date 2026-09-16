#!/usr/bin/env python3
"""Sample a running replay API for localization and tracking regressions."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from urllib.error import URLError
from urllib.request import urlopen


def get_json(url, timeout=3.0):
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:15001")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()

    started = time.monotonic()
    cameras = defaultdict(
        lambda: {
            "samples": 0,
            "track_samples": 0,
            "estimated_track_samples": 0,
            "unavailable_track_samples": 0,
            "retained_without_position_samples": 0,
            "first_frame": None,
            "last_frame": None,
            "start_cycle": None,
            "max_cycle": None,
        }
    )
    floor_samples = 0
    floor_estimated_samples = 0
    maximum_floor_people = 0
    maximum_counted_people = 0
    maximum_counted_samples = []
    interesting = []
    complete_cycle = False
    errors = []

    while time.monotonic() - started < args.timeout_seconds:
        try:
            state = get_json(f"{args.base_url.rstrip('/')}/api/state")
        except (OSError, URLError, ValueError) as error:
            errors.append(str(error))
            time.sleep(args.poll_seconds)
            continue
        floor = state.get("floor_map") or {}
        people = floor.get("people") or []
        floor_samples += 1
        maximum_floor_people = max(maximum_floor_people, len(people))
        counted_people = int(floor.get("counted_people") or 0)
        if counted_people > maximum_counted_people:
            maximum_counted_people = counted_people
            maximum_counted_samples = [
                {
                    "counted_people": counted_people,
                    "people": people,
                }
            ]
        elif counted_people == maximum_counted_people and counted_people > 1:
            if len(maximum_counted_samples) < 3:
                maximum_counted_samples.append(
                    {"counted_people": counted_people, "people": people}
                )
        if any(person.get("localization_state") == "estimated" for person in people):
            floor_estimated_samples += 1

        all_looped = True
        for camera in state.get("cameras") or []:
            camera_id = camera.get("id") or "unknown"
            replay = camera.get("replay") or {}
            stats = cameras[camera_id]
            stats["samples"] += 1
            frame = int(replay.get("frame_index") or 0)
            cycle = int(replay.get("cycle") or 0)
            stats["first_frame"] = frame if stats["first_frame"] is None else stats["first_frame"]
            stats["last_frame"] = frame
            if stats["start_cycle"] is None:
                stats["start_cycle"] = cycle
            stats["max_cycle"] = (
                cycle if stats["max_cycle"] is None else max(stats["max_cycle"], cycle)
            )
            tracks = int(camera.get("tracked_people") or 0)
            estimated = int(camera.get("localization_estimated_tracks") or 0)
            unavailable = int(camera.get("localization_unavailable_tracks") or 0)
            if tracks:
                stats["track_samples"] += 1
            if estimated:
                stats["estimated_track_samples"] += 1
            if unavailable:
                stats["unavailable_track_samples"] += 1
            if tracks and unavailable:
                stats["retained_without_position_samples"] += 1
            if (estimated or (tracks and unavailable)) and len(interesting) < 80:
                interesting.append(
                    {
                        "camera_id": camera_id,
                        "cycle": cycle,
                        "frame_index": frame,
                        "tracked_people": tracks,
                        "estimated": estimated,
                        "unavailable": unavailable,
                        "roi_rejected_tracks": int(camera.get("roi_rejected_tracks") or 0),
                    }
                )
            # A cycle increment alone only proves that playback crossed EOF.
            # Continue until each camera reaches its starting frame again so an
            # invocation made halfway through playback still samples one full
            # recording duration.
            all_looped = all_looped and (
                cycle > stats["start_cycle"]
                and frame >= int(stats["first_frame"] or 0)
            )
        if len(cameras) == 3 and all_looped:
            complete_cycle = True
            break
        time.sleep(args.poll_seconds)

    report = {
        "base_url": args.base_url,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "complete_cycle": complete_cycle,
        "floor_samples": floor_samples,
        "floor_estimated_samples": floor_estimated_samples,
        "maximum_floor_people": maximum_floor_people,
        "maximum_counted_people": maximum_counted_people,
        "maximum_counted_samples": maximum_counted_samples,
        "cameras": dict(cameras),
        "interesting_samples": interesting,
        "request_errors": errors[-10:],
    }
    report["checks"] = {
        "all_cameras_completed_cycle": complete_cycle and len(cameras) == 3,
        "person_retained_without_footpoint": any(
            item["retained_without_position_samples"] > 0 for item in cameras.values()
        ),
        "estimated_position_observed": any(
            item["estimated_track_samples"] > 0 for item in cameras.values()
        ),
        "legacy_roi_rejection_absent": all(
            item["roi_rejected_tracks"] == 0 for item in interesting
        ),
        "localization_counts_match_published_tracks": all(
            item["tracked_people"] >= item["estimated"] + item["unavailable"]
            for item in interesting
        ),
    }
    report["success"] = all(report["checks"].values())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
