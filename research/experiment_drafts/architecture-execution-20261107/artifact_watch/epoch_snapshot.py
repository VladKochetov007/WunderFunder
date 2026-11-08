"""Immutable native-byte prefixes for one actually mirrored full-valid epoch."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from wnn.multistage_search import atomic_write_json, canonical_json_sha256, file_sha256

SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
JSON_FILES = (
    "run_metadata.json",
    "monitor_progress.json",
    "epoch_index.json",
    "terminal_status.json",
    ".sequence_mirror_owner.json",
)
JOURNALS = (
    "monitor_events.jsonl",
    "monitor_evaluations.jsonl",
    "monitor_observer.jsonl",
    "monitor_rank0.jsonl",
    "rank0.jsonl",
)


def read_json(path: Path) -> dict[str, object]:
    """Read a direct bounded JSON object from a local producer-owned mirror."""
    if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("snapshot JSON is a link or exceeds16MiB")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("snapshot JSON must be an object")
    return value


def complete_events(path: Path) -> list[tuple[dict[str, object], bytes]]:
    """Validate every complete full-valid event; a trailing partial mirror line waits."""
    if not path.is_file():
        return []
    if path.is_symlink():
        raise ValueError("evaluation journal cannot be a link")
    events = []
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > 32768:
                raise ValueError("evaluation line exceeds32KiB")
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("metrics"), dict):
                raise ValueError("evaluation event and metrics must be JSON objects")
            metrics = event.get("metrics", {})
            if (
                event.get("schema") != "wnn.monitor.v1"
                or event.get("kind") != "evaluation"
                or type(event.get("epoch")) is not int
                or event["epoch"] != len(events) + 1
            ):
                raise ValueError("evaluation journal has a missing/repeated/malformed epoch")
            if (
                metrics.get("valid/sequence_count") != 1873
                or metrics.get("valid/selected_rows") != 3_528_273
            ):
                raise ValueError("evaluation must cover exactly1873sequences/3528273selectedrows")
            for name in ("valid/wp", "valid/wp/t0", "valid/wp/t1"):
                if (
                    isinstance(metrics.get(name), bool)
                    or not isinstance(metrics.get(name), (int, float))
                    or not math.isfinite(metrics[name])
                ):
                    raise ValueError("evaluation is missing a finite full-valid target metric")
            events.append((event, line))
    return events


def journal_prefix(path: Path, target: dict[str, object], mode: str) -> bytes | None:
    """Preserve exact native bytes through an evaluation, update, or observed wall clock."""
    if path.is_symlink():
        raise ValueError("native journal cannot be a link")
    target_time = datetime.fromisoformat(str(target["wall_time"]).replace("Z", "+00:00"))
    chunks = []
    found = mode != "evaluation"
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > 131072:
                raise ValueError("native journal line exceeds128KiB")
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if mode == "step" and int(event["step"]) > int(target["step"]):
                break
            if (
                mode == "wall"
                and datetime.fromisoformat(str(event["wall_time"]).replace("Z", "+00:00"))
                > target_time
            ):
                break
            chunks.append(line)
            if (
                mode == "evaluation"
                and event.get("kind") == "evaluation"
                and event.get("epoch") == target["epoch"]
            ):
                if event != target:
                    raise ValueError("native/archive evaluation records disagree")
                found = True
                break
    return b"".join(chunks) if found else None


def verify_snapshot(path: Path, event_sha256: str) -> dict[str, object]:
    """Verify a previously published snapshot without replacing a single byte."""
    manifest = read_json(path / "snapshot_manifest.json")
    marker = path.parent / f"epoch{manifest['epoch']}.json"
    if marker.exists() and read_json(marker)["snapshot_manifest_sha256"] != file_sha256(
        path / "snapshot_manifest.json"
    ):
        raise ValueError("immutable snapshot manifest changed")
    if manifest["evaluation_event_sha256"] != event_sha256:
        raise ValueError("already snapshotted epoch changed its native evaluation bytes")
    for name, digest in manifest["files_sha256"].items():
        if file_sha256(path / name) != digest:
            raise ValueError("immutable snapshot bytes changed")
    return manifest


def capture(
    run: Path, native: Path, root: Path, target: dict[str, object], raw_event: bytes
) -> Path | None:
    """Publish one replay-safe immutable per-epoch snapshot, never touching the live mirror.

    Evaluation/telemetry are exact prefixes through this epoch, rank rows use the
    actual update clock, and observer rows use the recorded evaluation wall clock.
    Latest progress/checkpoint metadata are labelled capture-time observations.
    """
    event_sha = hashlib.sha256(raw_event).hexdigest()
    run_root = root / "snapshots" / run.name
    run_root.mkdir(parents=True, exist_ok=True)
    marker = run_root / f"epoch{target['epoch']}.json"
    if marker.exists():
        reference = read_json(marker)
        snapshot = Path(reference["snapshot"])
        if not snapshot.resolve().is_relative_to(run_root.resolve()):
            raise ValueError("snapshot marker escapes its declared run root")
        verify_snapshot(snapshot, event_sha)
        return snapshot
    if (
        not (native / "collection_receipt.json").is_file()
        or not (run / "run_metadata.json").is_file()
    ):
        return None
    owner = read_json(run / ".sequence_mirror_owner.json")
    if (
        owner != read_json(native / ".sequence_mirror_owner.json")
        or not run.name.startswith(f"{owner['host']}-")
        or owner["host"] not in ("l4", "b300", "b300-opt")
    ):
        raise ValueError("run is not an actual host-qualified owned bridge mirror")
    metadata_payload = (run / "run_metadata.json").read_bytes()
    metadata = json.loads(metadata_payload)
    source = metadata["source"]
    data = metadata["provenance"]["data"]
    if (
        canonical_json_sha256(source["files"]) != source["sha256"]
        or canonical_json_sha256(data) != metadata["provenance"]["data_sha256"]
    ):
        raise ValueError("recorded training source/data identity is inconsistent")
    if (
        data["split_sha256"] != SPLIT_SHA
        or len(data["fit_members"]) != 10325
        or any(member[0] != "train" for member in data["fit_members"])
        or len(data["evaluation_members"]) != 1873
        or any(member[0] != "valid" for member in data["evaluation_members"])
    ):
        raise ValueError(
            "recorded model ancestry does not use exact train-only/full-valid membership"
        )
    payloads: dict[str, bytes] = {}
    for name in JOURNALS:
        path = run / name
        if not path.is_file():
            continue
        mode = (
            "evaluation"
            if name in ("monitor_events.jsonl", "monitor_evaluations.jsonl")
            else "step"
            if name in ("rank0.jsonl", "monitor_rank0.jsonl")
            else "wall"
        )
        payload = journal_prefix(path, target, mode)
        if payload is None:
            return None
        if name != "monitor_rank0.jsonl":
            if (native / name).is_symlink():
                raise ValueError("native mirror journal cannot be a link")
            with (native / name).open("rb") as incoming:
                if incoming.read(len(payload)) != payload:
                    return None
        payloads[name] = payload
    if not {"monitor_events.jsonl", "monitor_evaluations.jsonl"}.issubset(payloads):
        return None
    for name in JSON_FILES:
        path = run / name
        if path.is_file():
            read_json(path)
            payloads[name] = path.read_bytes()
    payloads["run_metadata.json"] = metadata_payload
    payloads["collection_receipt.json"] = (native / "collection_receipt.json").read_bytes()
    timestamp = (
        datetime.fromisoformat(str(target["wall_time"]).replace("Z", "+00:00"))
        .astimezone(UTC)
        .strftime("%Y%m%dT%H%M%S%fZ")
    )
    snapshot = run_root / f"epoch{target['epoch']}-{timestamp}-{event_sha[:12]}"
    if snapshot.exists():
        verify_snapshot(snapshot, event_sha)
    else:
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=run_root))
        for name, payload in payloads.items():
            with (staging / name).open("xb") as stream:
                stream.write(payload)
        manifest = {
            "schema": "wnn.sequence-epoch-snapshot.v1",
            "run": run.name,
            "owner": owner,
            "epoch": target["epoch"],
            "step": target["step"],
            "evaluation_wall_time": target["wall_time"],
            "captured_at": datetime.now(UTC).isoformat(),
            "evaluation_event_sha256": event_sha,
            "training_source": source,
            "data_sha256": metadata["provenance"]["data_sha256"],
            "files_sha256": {name: file_sha256(staging / name) for name in payloads},
            "journal_scope": "unchanged native byte prefixes through this evaluation; rank rows use update clock; observer rows use recorded wall time",
            "metadata_scope": "latest mirrored JSON metadata at capture time; may describe later producer progress",
            "derived_stream": "monitor_rank0.jsonl is the existing bridge view; no wall-time inference or densification",
            "model_loads": 0,
            "model_evaluations": 0,
        }
        atomic_write_json(staging / "snapshot_manifest.json", manifest, replace=False)
        os.rename(staging, snapshot)
    atomic_write_json(
        marker,
        {
            "snapshot": str(snapshot.resolve()),
            "snapshot_manifest_sha256": file_sha256(snapshot / "snapshot_manifest.json"),
            "evaluation_event_sha256": event_sha,
        },
        replace=False,
    )
    return snapshot
