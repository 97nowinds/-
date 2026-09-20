#!/usr/bin/env python3
"""Run the authorized round-two synthetic member-service contract check."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from member_integration import (  # noqa: E402
    MemberIntegrationError,
    MemberServiceClient,
    build_connection_test_event,
    build_synthetic_snapshot_ready_event,
    extract_receipt_id,
    extract_results,
)


def _step(response):
    return response.public_dict()


def _acknowledged(response):
    return response.ok and isinstance(response.body, dict) and response.body.get("status") == "acknowledged"


def run(
    client: MemberServiceClient,
    event_id: str,
    snapshot_event_id: str = "snapshot.ready:synthetic-round2-001:cam_2",
) -> tuple[dict, bool]:
    event = build_connection_test_event(event_id)
    snapshot_event = build_synthetic_snapshot_ready_event(snapshot_event_id)
    health = client.health()
    first = client.send_handoff(event)
    second = client.send_handoff(event)
    snapshot_first = client.send_handoff(snapshot_event)
    snapshot_second = client.send_handoff(snapshot_event)
    latest = client.latest_interactions()
    cam_2 = client.latest_interactions("cam_2")
    first_receipt = extract_receipt_id(first.body)
    second_receipt = extract_receipt_id(second.body)
    snapshot_first_receipt = extract_receipt_id(snapshot_first.body)
    snapshot_second_receipt = extract_receipt_id(snapshot_second.body)
    report = {
        "test_type": "connection.test",
        "contains_real_person_data": False,
        "event_id": event_id,
        "health": _step(health),
        "handoff_first": _step(first),
        "handoff_duplicate": _step(second),
        "idempotency": {
            "first_receipt_id": first_receipt,
            "second_receipt_id": second_receipt,
            "same_receipt_id": bool(first_receipt and first_receipt == second_receipt),
        },
        "snapshot_ready_first": _step(snapshot_first),
        "snapshot_ready_duplicate": _step(snapshot_second),
        "snapshot_ready_idempotency": {
            "first_receipt_id": snapshot_first_receipt,
            "second_receipt_id": snapshot_second_receipt,
            "same_receipt_id": bool(
                snapshot_first_receipt
                and snapshot_first_receipt == snapshot_second_receipt
            ),
        },
        "interaction_results_latest": _step(latest),
        "interaction_results_cam_2": _step(cam_2),
    }
    success = all(
        response.ok
        for response in (
            health,
            first,
            second,
            snapshot_first,
            snapshot_second,
            latest,
            cam_2,
        )
    ) and _acknowledged(first) and _acknowledged(second) and _acknowledged(
        snapshot_first
    ) and _acknowledged(snapshot_second) and report["idempotency"][
        "same_receipt_id"
    ] and report["snapshot_ready_idempotency"]["same_receipt_id"] and all(
        extract_results(response.body) == [] for response in (latest, cam_2)
    )
    report["success"] = success
    return report, success


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Send a connection.test and synthetic snapshot.ready twice, then verify "
            "acknowledgement, idempotency, and result reads"
        )
    )
    parser.add_argument(
        "--event-id",
        default="connection.test:synthetic-round2-001",
        help="Synthetic connection.test event ID sent twice",
    )
    parser.add_argument(
        "--snapshot-event-id",
        default="snapshot.ready:synthetic-round2-001:cam_2",
        help="Synthetic snapshot.ready event ID sent twice",
    )
    args = parser.parse_args()
    try:
        client = MemberServiceClient.from_environment(os.environ)
        report, success = run(client, args.event_id, args.snapshot_event_id)
    except (MemberIntegrationError, ValueError) as error:
        report = {"success": False, "error": str(error), "contains_real_person_data": False}
        success = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
