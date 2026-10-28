from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import deque

# Any keeps nested JSON fixtures readable without changing the payloads under test.
from typing import Any

import pytest

from scripts.galam_l4_quality8 import publisher


def test_update_formats_only_mask_verified_metrics() -> None:
    receipt, summary = _snapshot()
    body = publisher.format_update(receipt, summary, "hourly")
    assert (
        "Official validation: full `is_scored`; train-eval: diagnostic `all_after_warmup`." in body
    )
    assert "0.520000 (0.510000 / 0.530000)" in body
    assert "0.410000 (0.400000 / 0.420000)" in body
    assert "3/20" in body

    cell = summary["cells"]["g0-event211"]
    cell["latest_official_validation"]["context"]["valid"]["coverage_complete"] = False
    body = publisher.format_update(receipt, summary, "hourly", "g0-event211")
    assert "0.520000" not in body
    assert "0.410000" not in body


def test_update_rejects_summary_from_different_source() -> None:
    receipt, summary = _snapshot()
    summary["source_sha256"] = "f" * 64
    with pytest.raises(publisher.PublisherError, match="source_sha256"):
        publisher.format_update(receipt, summary, "hourly")


def test_github_account_guard_fails_closed_without_auth_retry() -> None:
    responses = deque([_Response(401, {}, b"{}")])
    calls: list[urllib.request.Request] = []
    sleeps: list[float] = []

    def opener(request: urllib.request.Request, **_kwargs: object) -> _Response:
        calls.append(request)
        return responses.popleft()

    client = publisher.GitHubClient("secret-must-not-be-logged", opener, sleeps.append)
    with pytest.raises(publisher.AuthenticationError, match="HTTP 401"):
        client.verify_account(10_000)
    assert len(calls) == 1
    assert sleeps == []


def test_github_account_guard_rejects_a_different_login() -> None:
    calls: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, **_kwargs: object) -> _Response:
        calls.append(request)
        return _Response(200, {}, b'{"login":"someone-else"}')

    client = publisher.GitHubClient("private", opener, lambda _seconds: None)
    with pytest.raises(publisher.AuthenticationError, match="SemyonEpanov"):
        client.verify_account(10_000)
    assert len(calls) == 1


def test_github_transient_retry_respects_retry_after_and_is_bounded() -> None:
    responses = deque(
        [
            _Response(503, {"Retry-After": "4"}, b"{}"),
            _Response(503, {"Retry-After": "7"}, b"{}"),
            _Response(503, {"Retry-After": "9"}, b"{}"),
        ]
    )
    calls: list[urllib.request.Request] = []
    sleeps: list[float] = []

    def opener(request: urllib.request.Request, **_kwargs: object) -> _Response:
        calls.append(request)
        return responses.popleft()

    client = publisher.GitHubClient("private", opener, sleeps.append, clock=lambda: 100.0)
    with pytest.raises(publisher.TransientGitHubError) as raised:
        client.verify_account(10_000)
    assert len(calls) == publisher.RETRIES
    assert sleeps == [4.0, 7.0]
    assert raised.value.retry_after == 9.0


def test_post_recovery_reuses_persisted_comment_marker() -> None:
    marker = "<!-- galam-l4-quality8:launch:abc -->"
    comment_url = "https://github.com/VladKochetov007/WunderFunder/issues/46#issuecomment-123"
    responses = deque(
        [_Response(200, {}, json.dumps([{"body": marker, "html_url": comment_url}]).encode())]
    )
    calls: list[str] = []
    sleeps: list[float] = []

    def opener(request: urllib.request.Request, **_kwargs: object) -> _Response:
        calls.append(request.method)
        return responses.popleft()

    client = publisher.GitHubClient("private", opener, sleeps.append, clock=lambda: 1.0)
    url = client.post(f"{marker}\nstatus", marker, 10_000, pending=True)
    assert url == comment_url
    assert calls == ["GET"]
    assert sleeps == []


def test_publisher_host_guard_blocks_non_linux() -> None:
    with pytest.raises(publisher.PublisherError, match="Linux receiving host"):
        publisher.require_linux_host("win32")


class _Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        return self._body[:amount]


def _snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    source = "c" * 64
    receipt: dict[str, Any] = {
        "schema": "wnn.galam-l4-quality8-launch.v1",
        "launch_id": "a" * 32,
        "panel_sha256": "b" * 64,
        "source_sha256": source,
        "split_sha256": "d" * 64,
        "cells": {},
    }
    cells: dict[str, Any] = {}
    for index in range(8):
        cell_id = f"g{index}-event211"
        identity = {"source_sha256": source, "config_sha256": f"{index:x}" * 64}
        identity_hash = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        receipt["cells"][cell_id] = {"gpu": index, "identity_sha256": identity_hash}
        cells[cell_id] = {
            "state": "running",
            "process_state": "running",
            "terminal_status": {"identity": identity},
            "checkpoint_pointers": {
                "run_identity_sha256": identity_hash,
                "best_raw_wp": 0.55,
                "best": {"epoch": 2, "valid_wp": 0.55},
            },
            "latest_official_validation": {
                "schema": "wnn.monitor.v1",
                "kind": "evaluation",
                "epoch": 3,
                "context": {
                    "valid": {
                        "row_mask": "is_scored",
                        "coverage_complete": True,
                        "split_sha256": receipt["split_sha256"],
                        "member_count": 2,
                        "selected_rows": 10,
                    },
                    "train_eval": {
                        "row_mask": "all_after_warmup",
                        "coverage_complete": True,
                        "member_count": 2,
                    },
                },
                "metrics": {
                    "valid/sequence_count": 2,
                    "valid/selected_rows": 10,
                    "valid/wp": 0.52,
                    "valid/wp/t0": 0.51,
                    "valid/wp/t1": 0.53,
                    "train_eval/sequence_count": 2,
                    "train_eval/wp": 0.41,
                    "train_eval/wp/t0": 0.40,
                    "train_eval/wp/t1": 0.42,
                },
            },
        }
    summary: dict[str, Any] = {
        "schema": "wnn.galam-l4-quality8-summary.v1",
        "launch_id": receipt["launch_id"],
        "panel_sha256": receipt["panel_sha256"],
        "source_sha256": source,
        "split_sha256": receipt["split_sha256"],
        "status": "running",
        "train_eval_mask": "all_after_warmup",
        "official_valid_mask": "is_scored",
        "cells": cells,
    }
    return receipt, summary
