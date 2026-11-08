"""Shared snapshot encoding, validation, and bounded local mirror helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zlib
from datetime import datetime
from pathlib import Path

REPOSITORY = "VladKochetov007/WunderFunder"
RESULTS_BRANCH = "results/tabdata-multistage-b300-20261031"
PARENT_CODE_BRANCH = "experiment/multistage-optuna-b300-20261031"
SNAPSHOT_SCHEMA = "wnn.multistage-github-snapshot.v1"
MANIFEST_SCHEMA = "wnn.multistage-github-snapshot-manifest.v1"
SNAPSHOT_PATH = "results/tabdata-multistage-b300-20261031/snapshot.json.zlib"
MANIFEST_PATH = "results/tabdata-multistage-b300-20261031/snapshot.manifest.json"
MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 256 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
METRIC_KEY = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")


def canonical_json(value: object) -> bytes:
    """Encode stable finite JSON for hashes, compressed payloads, and receipts."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""
    return hashlib.sha256(value).hexdigest()


def is_sha256(value: object) -> bool:
    """Accept only lowercase SHA-256 values."""
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def safe_run_id(value: object) -> bool:
    """Reject traversal, separators, and dashboard-incompatible directory names."""
    return (
        isinstance(value, str)
        and RUN_ID.fullmatch(value) is not None
        and value not in {".", ".."}
        and ".." not in value
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
    )


def make_snapshot(runs: list[dict[str, object]]) -> tuple[bytes, bytes, dict[str, object]]:
    """Build canonical snapshot bytes and a deterministic compressed payload."""
    body: dict[str, object] = {
        "schema": SNAPSHOT_SCHEMA,
        "repository": REPOSITORY,
        "branch": RESULTS_BRANCH,
        "parent_code_branch": PARENT_CODE_BRANCH,
        "runs": sorted(runs, key=lambda run: str(run["run_id"])),
    }
    validate_snapshot(body)
    payload = canonical_json(body)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("snapshot exceeds the uncompressed size limit")
    compressed = zlib.compress(payload, level=9)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ValueError("snapshot exceeds the compressed GitHub blob limit")
    return payload, compressed, body


def make_manifest(payload: bytes, compressed: bytes, created_at_utc: str) -> dict[str, object]:
    """Describe the exact two immutable snapshot payloads stored on the results branch."""
    body = json.loads(payload)
    return {
        "schema": MANIFEST_SCHEMA,
        "repository": REPOSITORY,
        "branch": RESULTS_BRANCH,
        "parent_code_branch": PARENT_CODE_BRANCH,
        "snapshot_path": SNAPSHOT_PATH,
        "manifest_path": MANIFEST_PATH,
        "created_at_utc": created_at_utc,
        "snapshot_sha256": sha256_bytes(canonical_json(body)),
        "json_sha256": sha256_bytes(payload),
        "json_bytes": len(payload),
        "compressed_sha256": sha256_bytes(compressed),
        "compressed_bytes": len(compressed),
    }


def verify_snapshot(compressed: bytes, manifest: object) -> tuple[dict[str, object], bytes]:
    """Verify manifest path, compressed hash, bounded decompression, and body identity."""
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported snapshot manifest")
    if (
        manifest.get("repository") != REPOSITORY
        or manifest.get("branch") != RESULTS_BRANCH
        or manifest.get("parent_code_branch") != PARENT_CODE_BRANCH
        or manifest.get("snapshot_path") != SNAPSHOT_PATH
        or manifest.get("manifest_path") != MANIFEST_PATH
    ):
        raise ValueError("snapshot manifest names an unapproved repository path")
    compressed_size = manifest.get("compressed_bytes")
    json_size = manifest.get("json_bytes")
    if (
        type(compressed_size) is not int
        or type(json_size) is not int
        or compressed_size != len(compressed)
        or not 0 <= compressed_size <= MAX_COMPRESSED_BYTES
        or not 0 <= json_size <= MAX_JSON_BYTES
        or manifest.get("compressed_sha256") != sha256_bytes(compressed)
    ):
        raise ValueError("compressed snapshot size or hash differs from the manifest")
    inflater = zlib.decompressobj()
    payload = inflater.decompress(compressed, json_size + 1)
    if (
        len(payload) != json_size
        or inflater.unconsumed_tail
        or not inflater.eof
        or inflater.unused_data
    ):
        raise ValueError("snapshot does not match its bounded uncompressed size")
    payload += inflater.flush()
    if len(payload) != json_size or manifest.get("json_sha256") != sha256_bytes(payload):
        raise ValueError("uncompressed snapshot hash differs from the manifest")
    try:
        body = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("snapshot is not valid JSON") from None
    validate_snapshot(body)
    if manifest.get("snapshot_sha256") != sha256_bytes(canonical_json(body)):
        raise ValueError("snapshot identity differs from the manifest")
    return body, payload


def validate_snapshot(value: object) -> None:
    """Validate snapshot labels, source hashes, and dashboard-safe event journals."""
    if not isinstance(value, dict) or any(
        value.get(key) != expected
        for key, expected in (
            ("schema", SNAPSHOT_SCHEMA),
            ("repository", REPOSITORY),
            ("branch", RESULTS_BRANCH),
            ("parent_code_branch", PARENT_CODE_BRANCH),
        )
    ):
        raise ValueError("snapshot identity is unsupported")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) > 512:
        raise ValueError("snapshot run collection is malformed or too large")
    seen: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("snapshot run is not an object")
        run_id = run.get("run_id")
        if not safe_run_id(run_id) or run_id in seen:
            raise ValueError("snapshot run ID is unsafe or duplicated")
        seen.add(run_id)
        if (
            type(run.get("stage")) is not int
            or run["stage"] not in (1, 2, 3)
            or not isinstance(run.get("family"), str)
            or not run["family"].isascii()
            or not run["family"].isalnum()
            or not all(
                is_sha256(run.get(key))
                for key in (
                    "config_sha256",
                    "source_sha256",
                    "split_sha256",
                    "primary_split_sha256",
                )
            )
            or not isinstance(run.get("metadata"), dict)
        ):
            raise ValueError("snapshot run identity is incomplete")
        metadata = run["metadata"]
        source = metadata.get("source")
        config = metadata.get("config")
        provenance = metadata.get("provenance")
        canonical_metadata_sha = sha256_bytes(canonical_json(metadata))
        if (
            not isinstance(source, dict)
            or not isinstance(config, dict)
            or not isinstance(provenance, dict)
            or run.get("canonical_metadata_sha256") != canonical_metadata_sha
        ):
            raise ValueError("snapshot run metadata or canonical digest is incomplete")
        data = provenance.get("data")
        model = config.get("model")
        if (
            source.get("sha256") != run["source_sha256"]
            or sha256_bytes(canonical_json(config)) != run["config_sha256"]
            or not isinstance(data, dict)
            or data.get("evaluation_role") != "control"
            or data.get("split_sha256") != run["split_sha256"]
            or data.get("primary_split_sha256") != run["primary_split_sha256"]
            or not isinstance(model, dict)
            or model.get("architecture") != run["family"]
        ):
            raise ValueError("snapshot run metadata does not bind its inner-control identity")
        kd_weight = config.get("kd_weight", 0.0)
        if isinstance(kd_weight, bool) or not isinstance(kd_weight, (int, float)):
            raise ValueError("snapshot run has an invalid distillation weight")
        inferred_stage = (
            3
            if data.get("teacher") is not None or kd_weight > 0
            else 2
            if data.get("initializer") is not None
            else None
        )
        evidence = run.get("stage_evidence")
        binding = run.get("stage_binding")
        if binding is None:
            if (
                evidence != "checkpoint_ancestry"
                or inferred_stage not in {2, 3}
                or run["stage"] != inferred_stage
            ):
                raise ValueError("snapshot label-only stage lacks a registered metadata label")
        else:
            if not isinstance(binding, dict):
                raise ValueError("snapshot stage binding is malformed")
            request = binding.get("request")
            if (
                not isinstance(request, dict)
                or not is_sha256(binding.get("canonical_sha256"))
                or binding["canonical_sha256"] != sha256_bytes(canonical_json(request))
                or not is_sha256(binding.get("source_file_sha256"))
                or request.get("schema") != "wnn.multistage-monitor-plot-request.v1"
                or request.get("stage") != run["stage"]
                or request.get("stage_evidence") != evidence
                or request.get("run_id") != run_id
                or request.get("family") != run["family"]
                or request.get("config_sha256") != run["config_sha256"]
                or request.get("source_sha256") != run["source_sha256"]
                or request.get("auxiliary_split_sha256") != run["split_sha256"]
                or request.get("primary_split_sha256") != run["primary_split_sha256"]
                or request.get("native_metadata_sha256") != canonical_metadata_sha
                or not is_sha256(request.get("profile_identity_sha256"))
            ):
                raise ValueError("snapshot stage binding does not match native run metadata")
            if evidence not in {"trial_request", "registered_role", "checkpoint_ancestry"}:
                raise ValueError("snapshot stage binding has an unknown evidence role")
            if evidence == "checkpoint_ancestry" and inferred_stage != run["stage"]:
                raise ValueError("snapshot checkpoint stage conflicts with native ancestry")
            if inferred_stage is not None and run["stage"] < inferred_stage:
                raise ValueError("snapshot registered stage conflicts with native ancestry")
        for name in ("event_journal", "evaluation_journal"):
            journal = run.get(name)
            if (
                not isinstance(journal, dict)
                or not is_sha256(journal.get("source_sha256"))
                or type(journal.get("source_bytes")) is not int
                or type(journal.get("source_complete_lines")) is not int
                or type(journal.get("source_partial_tail_bytes")) is not int
                or type(journal.get("retained_event_count")) is not int
                or journal["source_bytes"] < 0
                or journal["source_complete_lines"] < 0
                or journal["source_partial_tail_bytes"] < 0
                or journal["retained_event_count"] < 0
            ):
                raise ValueError("snapshot journal lacks its original source digest")
            events = journal.get("events")
            if not isinstance(events, list) or journal["retained_event_count"] != len(events):
                raise ValueError("snapshot journal events are malformed")
            for event in events:
                if not _valid_event(event):
                    raise ValueError("snapshot contains a non-native event")
                if event.get("kind") not in {
                    "status",
                    "train_step",
                    "evaluation",
                    "diagnostics",
                    "resources",
                }:
                    raise ValueError("snapshot contains an unknown event kind")
                if name == "evaluation_journal" and event.get("kind") != "evaluation":
                    raise ValueError("evaluation journal contains a non-evaluation event")


def _valid_event(value: object) -> bool:
    """Check the native monitor event shape and finite dashboard metrics."""
    if not isinstance(value, dict) or value.get("schema") != "wnn.monitor.v1":
        return False
    if value.get("kind") not in {"status", "train_step", "evaluation", "diagnostics", "resources"}:
        return False
    if type(value.get("step")) is not int or value["step"] < 0:
        return False
    epoch = value.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not _finite_number(epoch):
        return False
    wall_time = value.get("wall_time")
    if not isinstance(wall_time, str) or not isinstance(value.get("context"), dict):
        return False
    try:
        parsed_time = datetime.fromisoformat(wall_time.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed_time.tzinfo is None:
        return False
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        return False
    for key, metric in metrics.items():
        if (
            not isinstance(key, str)
            or len(key) > 128
            or METRIC_KEY.fullmatch(key) is None
            or isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not _finite_number(metric)
        ):
            return False
    return True


def _finite_number(value: int | float) -> bool:
    """Check a JSON number without allowing large integer conversion to raise."""
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False
