"""Authenticated client for the team member handoff service.

The browser must never receive the bearer token.  This module is deliberately
small and uses the Python standard library so it can run in the lightweight
frontend process without adding another deployment dependency.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 3.0


class MemberIntegrationError(RuntimeError):
    """Base error that is safe to show without leaking credentials."""


class MemberIntegrationConfigError(MemberIntegrationError):
    pass


class MemberIntegrationTransportError(MemberIntegrationError):
    pass


@dataclass(frozen=True)
class MemberServiceResponse:
    status: int
    body: Any
    content_type: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def public_dict(self) -> dict[str, Any]:
        return {
            "http_status": self.status,
            "ok": self.ok,
            "content_type": self.content_type,
            "response": self.body,
        }


def _validate_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if not base_url:
        raise MemberIntegrationConfigError("member service URL is not configured")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MemberIntegrationConfigError("member service URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise MemberIntegrationConfigError(
            "member service URL must not contain credentials, a query, or a fragment"
        )
    return base_url


class MemberServiceClient:
    """Server-side adapter for handoffs and interaction result reads."""

    def __init__(self, base_url: str, token: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        self.base_url = _validate_base_url(base_url)
        self._token = token.strip()
        if not self._token:
            raise MemberIntegrationConfigError("member service bearer token is not configured")
        self.timeout_seconds = float(timeout_seconds)
        if not 0.1 <= self.timeout_seconds <= 30.0:
            raise MemberIntegrationConfigError("member service timeout must be between 0.1 and 30 seconds")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "MemberServiceClient":
        values = os.environ if environ is None else environ
        return cls(
            values.get("LAB_MEMBER_SERVICE_URL", ""),
            values.get("LAB_MEMBER_SERVICE_TOKEN", ""),
            float(values.get("LAB_MEMBER_SERVICE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
        )

    @property
    def public_config(self) -> dict[str, Any]:
        return {
            "configured": True,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "authentication": "bearer (server-side, redacted)",
        }

    def _request(self, method: str, path: str, payload: Any | None = None) -> MemberServiceResponse:
        data = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
            "User-Agent": "lab-camera-member-integration/1.0",
        }
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return self._decode_response(response.status, response.headers, response.read())
        except HTTPError as error:
            return self._decode_response(error.code, error.headers, error.read())
        except (URLError, TimeoutError, socket.timeout, OSError) as error:
            reason = getattr(error, "reason", error)
            raise MemberIntegrationTransportError(
                f"member service request failed: {type(reason).__name__}: {reason}"
            ) from error

    @staticmethod
    def _decode_response(status: int, headers: Any, raw: bytes) -> MemberServiceResponse:
        content_type = (headers.get("Content-Type", "") if headers else "").split(";", 1)[0]
        text = raw.decode("utf-8", errors="replace")
        if content_type == "application/json" or text.lstrip().startswith(("{", "[")):
            try:
                body: Any = json.loads(text)
            except json.JSONDecodeError:
                body = text
        else:
            body = text
        return MemberServiceResponse(int(status), body, content_type or "unknown")

    def health(self) -> MemberServiceResponse:
        return self._request("GET", "/health")

    def send_handoff(self, event: Mapping[str, Any]) -> MemberServiceResponse:
        if not isinstance(event, Mapping) or not str(event.get("event_id", "")).strip():
            raise ValueError("handoff event requires a non-empty event_id")
        return self._request("POST", "/api/handoffs", dict(event))

    def latest_interactions(self, camera_id: str | None = None) -> MemberServiceResponse:
        path = "/api/interaction-results/latest"
        if camera_id:
            path = f"{path}?{urlencode({'camera_id': camera_id})}"
        return self._request("GET", path)


def build_connection_test_event(event_id: str, producer: str = "lab-camera-host") -> dict[str, Any]:
    """Build the only event allowed by the one-time connectivity workflow.

    The fixed schema intentionally has no person, face, location, camera frame,
    image URL, or biometric field.
    """

    normalized_id = event_id.strip()
    if not normalized_id:
        raise ValueError("connection test event_id must not be empty")
    return {
        "schema_version": "lab-handoff-1.1",
        "event_id": normalized_id,
        "event_type": "connection.test",
        "occurred_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "producer": producer,
        "test": {
            "purpose": "no-person connectivity and idempotency verification",
            "contains_real_data": False,
        },
    }


def extract_results(body: Any) -> list[Any] | None:
    """Return a result list when the peer uses either supported response shape."""

    if isinstance(body, list):
        return body
    if isinstance(body, Mapping):
        results = body.get("results")
        return results if isinstance(results, list) else None
    return None


def extract_receipt_id(body: Any) -> str | None:
    if not isinstance(body, Mapping):
        return None
    direct = body.get("receipt_id")
    if direct is not None:
        return str(direct)
    receipt = body.get("receipt")
    if isinstance(receipt, Mapping) and receipt.get("receipt_id") is not None:
        return str(receipt["receipt_id"])
    return None
