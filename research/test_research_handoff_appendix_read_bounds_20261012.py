#!/usr/bin/env python3
"""Focused stdlib checks for bounded handoff and JSONL input reads.

Run under scripts/limited with a source root and receipt path. The test writes
temporary oversized files under the source repository's research directory,
observes actual ``readline(size)``/``read(size)`` requests, streams the three
small source ledgers, then removes every temporary file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("build_research_complete_handoff_appendix_20261012.py")
MODULE_SPEC = importlib.util.spec_from_file_location("handoff_appendix_builder", SCRIPT)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("cannot load appendix builder")
BUILDER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(BUILDER)


class ReadProbe:
    """Transparent binary stream proxy that records bounded read calls."""

    def __init__(self, stream: Any, events: list[dict[str, int]]) -> None:
        self._stream = stream
        self._events = events

    def __enter__(self) -> "ReadProbe":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stream.close()

    def fileno(self) -> int:
        return self._stream.fileno()

    def readline(self, size: int = -1) -> bytes:
        data = self._stream.readline(size)
        self._events.append({"requested_bytes": size, "returned_bytes": len(data)})
        return data

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self._events.append({"requested_bytes": size, "returned_bytes": len(data)})
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _load_open_probe(target: Path, events: list[dict[str, int]]) -> Any:
    original_open = Path.open
    target_real = target.resolve()

    def probed_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream = original_open(path, *args, **kwargs)
        if path.resolve() == target_real:
            return ReadProbe(stream, events)
        return stream

    return probed_open


def _stream_ledgers(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for _, relative_path, label in BUILDER.LEDGERS:
        path = root / relative_path
        count = 0
        ids: Counter[str] = Counter()
        record_digest = hashlib.sha256()
        last_line = 0
        for line_number, raw, record in BUILDER._iter_ledger(path):
            if line_number <= last_line:
                raise AssertionError(f"non-monotonic source line numbers in {relative_path}")
            last_line = line_number
            count += 1
            ids[BUILDER._record_id(record, line_number)] += 1
            record_digest.update(raw)
            record_digest.update(b"\n")
        source_hash = BUILDER._sha256_file(path)
        stream_hash = record_digest.hexdigest()
        if stream_hash != source_hash:
            raise AssertionError(f"bounded row stream did not preserve source bytes: {relative_path}")
        results.append(
            {
                "label": label,
                "path": relative_path,
                "rows": count,
                "distinct_ids": len(ids),
                "source_sha256": source_hash,
                "normalized_record_stream_sha256": stream_hash,
                "record_bytes_preserved": True,
            }
        )
    return results


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=Path(__file__).with_name("LEDGER_READ_BOUNDS_VALIDATION_R1.json"))
    args = parser.parse_args()
    root = args.source_root.resolve()
    temp_dir = root / "research/campaign_2026-10-12/research_archive_inventory_r1"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    receipt: dict[str, Any] = {"schema_version": 1, "status": "FAIL"}
    try:
        ledger_fd, ledger_name = tempfile.mkstemp(prefix=".ledger-line-bound-", suffix=".tmp", dir=temp_dir)
        os.close(ledger_fd)
        oversized_ledger = Path(ledger_name)
        temporary_paths.append(oversized_ledger)
        with oversized_ledger.open("wb") as stream:
            remaining = BUILDER.MAX_LEDGER_LINE_BYTES + 4096
            chunk = b"x" * (1024 * 1024)
            while remaining:
                piece = chunk[: min(len(chunk), remaining)]
                stream.write(piece)
                remaining -= len(piece)
        physical_ledger_size = oversized_ledger.stat().st_size
        ledger_events: list[dict[str, int]] = []
        patched_open = _load_open_probe(oversized_ledger, ledger_events)
        oversized_ledger_rejected = False
        try:
            with patch.object(Path, "open", new=patched_open):
                list(BUILDER._iter_ledger(oversized_ledger))
        except BUILDER.ArchiveError as exc:
            oversized_ledger_rejected = "fixed bound" in str(exc)
        if not oversized_ledger_rejected:
            raise AssertionError("oversized JSONL line was not rejected by the fixed bound")
        if ledger_events != [
            {
                "requested_bytes": BUILDER.MAX_LEDGER_LINE_BYTES + 1,
                "returned_bytes": BUILDER.MAX_LEDGER_LINE_BYTES + 1,
            }
        ]:
            raise AssertionError(f"unexpected oversized-line reads: {ledger_events!r}")

        handoff_path = root / "research/RESEARCH_COMPLETE_HANDOFF_20261012.md"
        handoff_events: list[dict[str, int]] = []
        patched_open = _load_open_probe(handoff_path, handoff_events)
        with patch.object(Path, "open", new=patched_open):
            handoff_text = BUILDER._read_handoff(handoff_path)
        handoff_size = handoff_path.stat().st_size
        if len(handoff_text.encode("utf-8")) != handoff_size:
            raise AssertionError("bounded handoff read changed the existing file bytes")
        if handoff_events != [
            {
                "requested_bytes": BUILDER.MAX_HANDOFF_BYTES + 1,
                "returned_bytes": handoff_size,
            }
        ]:
            raise AssertionError(f"unexpected existing-handoff read: {handoff_events!r}")

        handoff_fd, handoff_name = tempfile.mkstemp(prefix=".handoff-prefix-bound-", suffix=".tmp", dir=temp_dir)
        os.close(handoff_fd)
        oversized_handoff = Path(handoff_name)
        temporary_paths.append(oversized_handoff)
        with oversized_handoff.open("wb") as stream:
            stream.truncate(BUILDER.MAX_HANDOFF_BYTES + 1)
        opened_oversized_handoff = False
        original_open = Path.open

        def reject_if_opened(path: Path, *open_args: Any, **open_kwargs: Any) -> Any:
            nonlocal opened_oversized_handoff
            if path.resolve() == oversized_handoff.resolve():
                opened_oversized_handoff = True
                raise AssertionError("size-rejected handoff was opened")
            return original_open(path, *open_args, **open_kwargs)

        oversized_handoff_rejected = False
        try:
            with patch.object(Path, "open", new=reject_if_opened):
                BUILDER._read_handoff(oversized_handoff)
        except BUILDER.ArchiveError as exc:
            oversized_handoff_rejected = "fixed" in str(exc)
        if not oversized_handoff_rejected or opened_oversized_handoff:
            raise AssertionError("oversized handoff was not rejected by stat before open")

        ledger_results = _stream_ledgers(root)
        expected_counts = (479, 796, 134)
        actual_counts = tuple(item["rows"] for item in ledger_results)
        if actual_counts != expected_counts:
            raise AssertionError(f"source ledger row counts changed: {actual_counts!r}")

        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "source_main_head": _git_head(root),
            "builder": {
                "path": "research/build_research_complete_handoff_appendix_20261012.py",
                "sha256": BUILDER._sha256_file(SCRIPT),
            },
            "ledger_line_read": {
                "maximum_record_bytes": BUILDER.MAX_LEDGER_LINE_BYTES,
                "synthetic_physical_file_bytes": physical_ledger_size,
                "requested_read_bytes": ledger_events[0]["requested_bytes"],
                "returned_read_bytes": ledger_events[0]["returned_bytes"],
                "rejected_before_json_parse": oversized_ledger_rejected,
            },
            "handoff_input_read": {
                "maximum_file_bytes": BUILDER.MAX_HANDOFF_BYTES,
                "existing_handoff_bytes": handoff_size,
                "existing_handoff_sha256": BUILDER._sha256_file(handoff_path),
                "requested_read_bytes": handoff_events[0]["requested_bytes"],
                "returned_read_bytes": handoff_events[0]["returned_bytes"],
                "oversized_sparse_file_bytes": BUILDER.MAX_HANDOFF_BYTES + 1,
                "oversized_file_rejected_by_stat_before_open": oversized_handoff_rejected and not opened_oversized_handoff,
            },
            "physical_temp_directory": "research/campaign_2026-10-12/research_archive_inventory_r1",
            "temporary_files_removed_in_finally": True,
            "actual_source_ledgers": ledger_results,
            "execution_scope": {
                "requested_cpu_cores": 1,
                "requested_memory_max_bytes": 700000000,
                "requested_runtime_max_seconds": 120,
                "swap": "disabled by scripts/limited campaign slice policy",
                "model_or_provider_work": False,
            },
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BUILDER.ArchiveError as exc:
        print(f"validation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
