from __future__ import annotations

import hashlib
import http.client
import io
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest

from scripts import kaggle_vast_prefetch as prefetch
from scripts import vast_campaign as vast

ATTEMPT = "vast-attempt-001"
LABEL = "wnn-e529-001-run"
REMOTE_PATH = "datasets/valid.parquet"
URL = "https://storage.googleapis.com/bucket/object?X-Goog-Signature=private-test-token"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object, mode: int) -> bytes:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return raw


def _documents(root: Path, *, url: str = URL, source_hash: str | None = None):
    runtime = root / ".runtime"
    runtime.mkdir(mode=0o700)
    contents = b"frozen parquet bytes"
    source_hash = _sha256(contents) if source_hash is None else source_hash
    private_path = runtime / "signed-plan.private.json"
    private_value = {
        "schema_version": 1,
        "inputs": [
            {
                "remote_path": REMOTE_PATH,
                "sha256": source_hash,
                "size_bytes": len(contents),
                "url": url,
            }
        ],
    }
    private_raw = _write_json(private_path, private_value, 0o600)
    admission_path = runtime / "admission.json"
    admission_value = {
        "schema_version": 1,
        "attempt_id": ATTEMPT,
        "campaign_label": LABEL,
        "private_plan_sha256": _sha256(private_raw),
        "source_binding": [
            {
                "remote_path": REMOTE_PATH,
                "sha256": source_hash,
                "size_bytes": len(contents),
            }
        ],
    }
    _write_json(admission_path, admission_value, 0o600)
    frozen = [
        {
            "remote_path": REMOTE_PATH,
            "source_sha256": source_hash,
            "size_bytes": len(contents),
        }
    ]
    return runtime, private_path, admission_path, frozen, contents


def _remote_payload(root: Path, contents: bytes, *, url: str = URL) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "attempt_id": ATTEMPT,
            "campaign_label": LABEL,
            "output_root": str(root / "results"),
            "budget_seconds": 60,
            "inputs": [
                {
                    "remote_path": REMOTE_PATH,
                    "sha256": _sha256(contents),
                    "size_bytes": len(contents),
                    "url": url,
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _run_remote(payload: bytes, monkeypatch: pytest.MonkeyPatch, opener: object) -> tuple[int, str]:
    stdin = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8")
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(prefetch.urllib.request, "build_opener", lambda *_args: opener)
    namespace = {"__name__": "__main__"}
    with pytest.raises(SystemExit) as exit_info:
        exec(compile(prefetch.REMOTE_PROGRAM, "<remote-prefetch>", "exec"), namespace)
    return int(exit_info.value.code), stdout.getvalue()


class _Response(io.BytesIO):
    status = 200

    def __init__(self, body: bytes, url: str) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}
        self._url = url
        self.read_sizes: list[int] = []
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=_FakeSocket()))

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)


class _Opener:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.timeouts: list[float] = []

    def open(self, request: Request, timeout: float) -> _Response:
        self.timeouts.append(timeout)
        assert request.full_url == URL
        return self.response


def test_public_binding_validates_private_plan_and_hides_url(tmp_path: Path) -> None:
    runtime, private, admission, frozen, _ = _documents(tmp_path)
    prepared = prefetch.prepare_prefetch(
        private,
        admission,
        attempt_id=ATTEMPT,
        campaign_label=LABEL,
        frozen_inputs=frozen,
        runtime_root=runtime,
    )
    assert prepared.items[0].remote_path == REMOTE_PATH
    assert prepared.items[0].sha256 == frozen[0]["source_sha256"]
    assert prepared.private_plan_sha256 == _sha256(private.read_bytes())
    assert URL not in repr(prepared)


@pytest.mark.parametrize("remote_path", ["cache/train_targets.npz", "cache/valid_targets.npz"])
def test_target_cache_prefetch_is_hash_bound_and_downloadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_path: str
) -> None:
    monkeypatch.setitem(globals(), "REMOTE_PATH", remote_path)
    runtime, private, admission, frozen, contents = _documents(tmp_path)
    prepared = prefetch.prepare_prefetch(
        private,
        admission,
        attempt_id=ATTEMPT,
        campaign_label=LABEL,
        frozen_inputs=frozen,
        runtime_root=runtime,
    )
    payload = prefetch.make_remote_payload(
        prepared,
        attempt_id=ATTEMPT,
        campaign_label=LABEL,
        output_root=str(tmp_path / "results"),
        budget_seconds=60,
    )
    code, output = _run_remote(payload, monkeypatch, _Opener(_Response(contents, URL)))
    assert code == 0
    assert URL not in output
    assert (tmp_path / "results/.worker" / LABEL / "inputs" / remote_path).read_bytes() == contents
    receipt = prefetch.validate_remote_result(output, prepared)
    assert receipt["status"] == "complete"
    frozen[0]["source_sha256"] = "0" * 64
    with pytest.raises(prefetch.PrefetchPlanError, match="differ"):
        prefetch.prepare_prefetch(
            private,
            admission,
            attempt_id=ATTEMPT,
            campaign_label=LABEL,
            frozen_inputs=frozen,
            runtime_root=runtime,
        )


@pytest.mark.parametrize(
    ("url", "source_hash", "mode"),
    [
        ("https://attacker.example/object?secret=private-test-token", None, 0o600),
        (URL, "0" * 64, 0o600),
        (URL, None, 0o644),
    ],
)
def test_plan_host_binding_and_private_mode_fail_closed(
    tmp_path: Path, url: str, source_hash: str | None, mode: int
) -> None:
    runtime, private, admission, frozen, _ = _documents(tmp_path, url=url, source_hash=source_hash)
    private.chmod(mode)
    if source_hash == "0" * 64:
        # The source receipt must still bind to the independently frozen source.
        frozen[0]["source_sha256"] = "f" * 64
    with pytest.raises(prefetch.PrefetchPlanError):
        prefetch.prepare_prefetch(
            private,
            admission,
            attempt_id=ATTEMPT,
            campaign_label=LABEL,
            frozen_inputs=frozen,
            runtime_root=runtime,
        )


def test_plan_refuses_symlink_and_wrong_attempt(tmp_path: Path) -> None:
    runtime, private, admission, frozen, _ = _documents(tmp_path)
    linked = runtime / "linked.private.json"
    linked.symlink_to(private)
    with pytest.raises(prefetch.PrefetchPlanError):
        prefetch.prepare_prefetch(
            linked,
            admission,
            attempt_id=ATTEMPT,
            campaign_label=LABEL,
            frozen_inputs=frozen,
            runtime_root=runtime,
        )
    with pytest.raises(prefetch.PrefetchPlanError):
        prefetch.prepare_prefetch(
            private,
            admission,
            attempt_id="another-attempt",
            campaign_label=LABEL,
            frozen_inputs=frozen,
            runtime_root=runtime,
        )


def test_remote_fetch_streams_hash_bound_bytes_without_url_in_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = b"streamed bytes from signed object"
    response = _Response(contents, URL)
    opener = _Opener(response)
    code, output = _run_remote(_remote_payload(tmp_path, contents), monkeypatch, opener)
    receipt = json.loads(output)
    destination = tmp_path / "results" / ".worker" / LABEL / "inputs" / REMOTE_PATH
    assert code == 0
    assert destination.read_bytes() == contents
    assert receipt["status"] == "complete"
    assert receipt["completed"][0]["sha256"] == _sha256(contents)
    assert receipt["completed"][0]["size_bytes"] == len(contents)
    assert max(response.read_sizes) <= 1024 * 1024
    assert opener.timeouts and max(opener.timeouts) <= 30
    assert URL not in output


def test_remote_hash_failure_cleans_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = b"expected bytes"
    response = _Response(b"different bytes", URL)
    opener = _Opener(response)
    code, output = _run_remote(_remote_payload(tmp_path, expected), monkeypatch, opener)
    receipt = json.loads(output)
    destination = tmp_path / "results" / ".worker" / LABEL / "inputs" / REMOTE_PATH
    assert code == 1
    assert receipt["error_code"] == "size_mismatch"
    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))
    assert URL not in output


def test_remote_refuses_symlink_parent_without_writing_through_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    base = tmp_path / "results" / ".worker" / LABEL / "inputs"
    base.mkdir(parents=True)
    (base / "datasets").symlink_to(outside, target_is_directory=True)
    body = b"should not be written"
    code, output = _run_remote(
        _remote_payload(tmp_path, body), monkeypatch, _Opener(_Response(body, URL))
    )
    receipt = json.loads(output)
    assert code == 1
    assert receipt["error_code"] == "unsafe_destination"
    assert not list(outside.iterdir())
    assert URL not in output


def test_remote_redirect_cannot_leave_approved_storage_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"redirected bytes"
    attacker_url = "https://attacker.example/object?signature=private"
    code, output = _run_remote(
        _remote_payload(tmp_path, body),
        monkeypatch,
        _Opener(_Response(body, attacker_url)),
    )
    receipt = json.loads(output)
    destination = tmp_path / "results" / ".worker" / LABEL / "inputs" / REMOTE_PATH
    assert code == 1
    assert receipt["error_code"] == "invalid_url"
    assert not destination.exists()
    assert attacker_url not in output
    assert URL not in output


def test_remote_result_parser_rejects_secret_bearing_or_unbound_receipts(tmp_path: Path) -> None:
    runtime, private, admission, frozen, _ = _documents(tmp_path)
    prepared = prefetch.prepare_prefetch(
        private,
        admission,
        attempt_id=ATTEMPT,
        campaign_label=LABEL,
        frozen_inputs=frozen,
        runtime_root=runtime,
    )
    bad = json.dumps(
        {
            "schema_version": 1,
            "status": "complete",
            "completed": [
                {
                    "remote_path": REMOTE_PATH,
                    "sha256": frozen[0]["source_sha256"],
                    "size_bytes": frozen[0]["size_bytes"],
                    "elapsed_ms": 1,
                    "status": "downloaded",
                    "url": URL,
                }
            ],
            "error_code": None,
        }
    )
    with pytest.raises(prefetch.PrefetchPlanError):
        prefetch.validate_remote_result(bad, prepared)


def test_pinned_ssh_stdin_keeps_private_url_out_of_argv_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b'{"safe":true}')

    monkeypatch.setattr(vast.subprocess, "run", fake_run)
    monkeypatch.setattr(vast, "DEFAULT_SSH_PUBLIC_KEY", tmp_path / ".ssh/id_ed25519.pub")
    result = vast._ssh_stdin(
        "ssh-proxy.example",
        2200,
        "python3 -c 'print(1)'",
        1200,
        known_hosts_path=tmp_path / "known_hosts",
        input_bytes=json.dumps({"url": URL}).encode("utf-8"),
        allowed_exit_codes=(1,),
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert URL not in " ".join(command)
    assert URL.encode() in captured["input"]
    assert "StrictHostKeyChecking=yes" in command
    assert f"UserKnownHostsFile={tmp_path / 'known_hosts'}" in command
    assert result == '{"safe":true}'


def test_remote_accepts_http_response_that_closes_at_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = b"complete real HTTP response"
    reader, writer = socket.socketpair()
    try:
        writer.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: "
            + str(len(contents)).encode()
            + b"\r\n\r\n"
            + contents
        )
        writer.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(reader)
        response.begin()
        monkeypatch.setattr(response, "geturl", lambda: URL, raising=False)
        code, output = _run_remote(
            _remote_payload(tmp_path, contents), monkeypatch, _Opener(response)
        )
        receipt = json.loads(output)
        destination = tmp_path / "results" / ".worker" / LABEL / "inputs" / REMOTE_PATH
        assert code == 0, receipt
        assert receipt["status"] == "complete"
        assert destination.read_bytes() == contents
        assert receipt["completed"][0]["sha256"] == _sha256(contents)
    finally:
        reader.close()
        writer.close()


def test_remote_accepts_complete_four_input_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = b"four frozen inputs"
    payload = json.loads(_remote_payload(tmp_path, contents))
    payload["inputs"] = [
        {
            "remote_path": path,
            "sha256": _sha256(contents),
            "size_bytes": len(contents),
            "url": URL,
        }
        for path in sorted(prefetch.ALLOWED_REMOTE_PATHS)
    ]

    class FreshResponseOpener:
        def open(self, request: Request, timeout: float) -> _Response:
            assert request.full_url == URL
            return _Response(contents, URL)

    code, output = _run_remote(json.dumps(payload).encode(), monkeypatch, FreshResponseOpener())
    assert code == 0
    receipt = json.loads(output)
    assert len(receipt["completed"]) == 4
    assert URL not in output
    for path in prefetch.ALLOWED_REMOTE_PATHS:
        assert (tmp_path / "results/.worker" / LABEL / "inputs" / path).read_bytes() == contents
