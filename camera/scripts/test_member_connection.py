#!/usr/bin/env python3
"""Run the authorized no-person member-service connectivity check."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from member_integration import (  # noqa: E402
    MemberIntegrationError,
    MemberServiceClient,
    build_connection_test_event,
    extract_receipt_id,
    extract_results,
)


def _step(response):
    return response.public_dict()


def run(client: MemberServiceClient, event_id: str) -> tuple[dict, bool]:
    event = build_connection_test_event(event_id)
    health = client.health()
    first = client.send_handoff(event)
    second = client.send_handoff(event)
    latest = client.latest_interactions()
    cam_2 = client.latest_interactions("cam_2")
    first_receipt = extract_receipt_id(first.body)
    second_receipt = extract_receipt_id(second.body)
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
        "interaction_results_latest": _step(latest),
        "interaction_results_cam_2": _step(cam_2),
    }
    success = all(
        response.ok for response in (health, first, second, latest, cam_2)
    ) and report["idempotency"]["same_receipt_id"] and all(
        extract_results(response.body) == [] for response in (latest, cam_2)
    )
    report["success"] = success
    return report, success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send only a no-person connection.test and verify peer idempotency"
    )
    parser.add_argument(
        "--event-id",
        default=f"connection.test:{uuid.uuid4()}",
        help="Stable event ID sent twice (default: generated UUID)",
    )
    args = parser.parse_args()
    try:
        client = MemberServiceClient.from_environment(os.environ)
        report, success = run(client, args.event_id)
    except (MemberIntegrationError, ValueError) as error:
        report = {"success": False, "error": str(error), "contains_real_person_data": False}
        success = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
