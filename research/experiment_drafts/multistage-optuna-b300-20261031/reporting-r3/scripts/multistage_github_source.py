"""Read and sanitize source-bound native dashboard journals for publication."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from multistage_github_common import (  # noqa: E402
    canonical_json,
    is_sha256,
    safe_run_id,
    sha256_bytes,
)

from wnn.dashboard.schema import (  # noqa: E402
    BEARER_TEXT,
    SECRET_TEXT,
    SENSITIVE_KEY_PARTS,
    URL_CREDENTIALS,
    parse_event,
)

MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_JOURNAL_BYTES = 128 * 1024 * 1024
MAX_STAGE_BINDING_BYTES = 1_048_576
TRAIN_STEP_STRIDE = 10
LOG = logging.getLogger("multistage_github_source")


class UnregisteredStageError(ValueError):
    """A label-only fit is not publishable until its campaign stage is bound."""


def _read_regular(path: Path, maximum: int) -> bytes:
    """Read one bounded regular file without following its final path component."""
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError(f"unsafe or oversized dashboard input: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("dashboard input changed while opening")
        source = os.fdopen(descriptor, "rb")
        descriptor = -1
        with source:
            raw = source.read(maximum + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) != info.st_size or len(raw) > maximum:
        raise ValueError("dashboard input changed while reading")
    return raw


def _journal(path: Path, *, evaluations_only: bool) -> dict[str, object]:
    """Validate a native JSONL source and preserve exact source hash plus honest sampling."""
    raw = _read_regular(path, MAX_JOURNAL_BYTES)
    complete = raw.splitlines(keepends=True)
    tail_bytes = 0
    if complete and not complete[-1].endswith(b"\n"):
        tail_bytes = len(complete.pop())
    events: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    train_seen = 0
    train_retained = 0
    for line in complete:
        if not line.endswith(b"\n") or len(line) > 131_072:
            raise ValueError(f"native dashboard journal has an invalid line: {path.name}")
        source_event = json.loads(line)
        event = parse_event(line.removesuffix(b"\n").removesuffix(b"\r"))
        if event is None:
            raise ValueError(f"native dashboard journal contains an invalid event: {path.name}")
        event["context"] = _sanitize_event_context(source_event["context"])
        kind = str(event["kind"])
        if evaluations_only and kind != "evaluation":
            raise ValueError("evaluation journal contains a non-evaluation event")
        counts[kind] = counts.get(kind, 0) + 1
        if kind != "train_step":
            events.append(event)
        else:
            if train_seen % TRAIN_STEP_STRIDE == 0:
                events.append(event)
                train_retained += 1
            train_seen += 1
    sampling: dict[str, object] = {"rule": "all_events_preserved", "stride": 1}
    if not evaluations_only:
        sampling = {
            "rule": "retain every non-train_step event; retain train_step events at zero-based source ordinals divisible by stride",
            "stride": TRAIN_STEP_STRIDE,
            "train_step_source_count": train_seen,
            "train_step_retained_count": train_retained,
        }
    return {
        "source_sha256": sha256_bytes(raw),
        "source_bytes": len(raw),
        "source_complete_lines": len(complete),
        "source_partial_tail_bytes": tail_bytes,
        "source_event_counts": counts,
        "sampling": sampling,
        "retained_event_count": len(events),
        "events": events,
    }


def _run_snapshot(run_dir: Path) -> dict[str, object]:
    """Read one direct native monitor mirror and bind its actual fit/control metadata."""
    if not safe_run_id(run_dir.name):
        raise ValueError("dashboard run directory name is unsafe")
    info = os.lstat(run_dir)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("dashboard run must be a direct real directory")
    raw_metadata = _read_regular(run_dir / "run_metadata.json", MAX_METADATA_BYTES)
    metadata_value = json.loads(raw_metadata)
    if not isinstance(metadata_value, dict):
        raise ValueError("native run metadata is not an object")
    source = metadata_value.get("source")
    config = metadata_value.get("config")
    provenance = metadata_value.get("provenance")
    if not all(isinstance(item, dict) for item in (source, config, provenance)):
        raise ValueError("native run metadata is incomplete")
    data = provenance.get("data")
    model = config.get("model")
    if not isinstance(data, dict) or data.get("evaluation_role") != "control":
        raise ValueError("only source-bound inner-control runs can be published")
    if not isinstance(model, dict) or not isinstance(model.get("architecture"), str):
        raise ValueError("run family is missing from native config")
    family = model["architecture"]
    if not family.isascii() or not family.isalnum():
        raise ValueError("native run family is not a safe label")
    source_sha = source.get("sha256")
    split_sha = data.get("split_sha256")
    primary_sha = data.get("primary_split_sha256")
    if not all(is_sha256(value) for value in (source_sha, split_sha, primary_sha)):
        raise ValueError("native run source or split identity is incomplete")
    initializer = data.get("initializer")
    teacher = data.get("teacher")
    kd_weight = config.get("kd_weight", 0.0)
    if isinstance(kd_weight, bool) or not isinstance(kd_weight, (int, float)):
        raise ValueError("native run has an invalid distillation weight")
    inferred_stage = (
        3 if teacher is not None or kd_weight > 0 else 2 if initializer is not None else None
    )
    canonical_metadata_sha = sha256_bytes(canonical_json(metadata_value))
    stage, stage_evidence, stage_binding = _stage_binding(
        run_dir,
        family,
        sha256_bytes(canonical_json(config)),
        source_sha,
        split_sha,
        primary_sha,
        canonical_metadata_sha,
        inferred_stage,
    )
    metadata = _preserve_metadata(metadata_value)
    if not isinstance(metadata, dict):
        raise ValueError("native run metadata could not be sanitized")
    return {
        "run_id": run_dir.name,
        "stage": stage,
        "family": family,
        "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "source_sha256": source_sha,
        "split_sha256": split_sha,
        "primary_split_sha256": primary_sha,
        "metadata_sha256": sha256_bytes(raw_metadata),
        "canonical_metadata_sha256": canonical_metadata_sha,
        "stage_evidence": stage_evidence,
        "stage_binding": stage_binding,
        "metadata": metadata,
        "event_journal": _journal(run_dir / "monitor_events.jsonl", evaluations_only=False),
        "evaluation_journal": _journal(
            run_dir / "monitor_evaluations.jsonl", evaluations_only=True
        ),
    }


def _stage_binding(
    run_dir: Path,
    family: str,
    config_sha: str,
    source_sha: object,
    split_sha: object,
    primary_sha: object,
    canonical_metadata_sha: str,
    inferred_stage: int | None,
) -> tuple[int, str, dict[str, object] | None]:
    """Bind stage labels to a monitor request or to explicit checkpoint ancestry."""
    binding_path = run_dir / "multistage_plot_request.json"
    try:
        raw = _read_regular(binding_path, MAX_STAGE_BINDING_BYTES)
    except FileNotFoundError:
        if inferred_stage in {2, 3}:
            return inferred_stage, "checkpoint_ancestry", None
        raise UnregisteredStageError(
            "label-only run lacks its monitor-registered stage label"
        ) from None
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise ValueError("monitor stage binding must be a JSON object")
    stage = request.get("stage")
    evidence = request.get("stage_evidence")
    if (
        request.get("schema") != "wnn.multistage-monitor-plot-request.v1"
        or type(stage) is not int
        or stage not in {1, 2, 3}
        or evidence not in {"trial_request", "registered_role", "checkpoint_ancestry"}
        or request.get("run_id") != run_dir.name
        or request.get("family") != family
        or request.get("config_sha256") != config_sha
        or request.get("source_sha256") != source_sha
        or request.get("auxiliary_split_sha256") != split_sha
        or request.get("primary_split_sha256") != primary_sha
        or request.get("native_metadata_sha256") != canonical_metadata_sha
        or not is_sha256(request.get("profile_identity_sha256"))
    ):
        raise ValueError("monitor stage binding does not match the native run identity")
    if evidence == "checkpoint_ancestry" and inferred_stage != stage:
        raise ValueError("checkpoint stage binding conflicts with run ancestry")
    if inferred_stage is not None and stage < inferred_stage:
        raise ValueError("registered stage label conflicts with checkpoint ancestry")
    stage_binding = {
        "canonical_sha256": sha256_bytes(canonical_json(request)),
        "source_file_sha256": sha256_bytes(raw),
        "request": request,
    }
    return stage, evidence, stage_binding


def collect_snapshot(
    runs_root: Path,
    primary_prepared_root: Path | None = None,
    primary_runtime_root: Path | None = None,
) -> list[dict[str, object]]:
    """Collect direct non-symlink run directories, rejecting unsafe inputs closed."""
    root_info = os.lstat(runs_root)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("dashboard runs root must be a real directory")
    runs: list[dict[str, object]] = []
    with os.scandir(runs_root) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
            if not safe_run_id(entry.name):
                continue
            try:
                runs.append(_run_snapshot(runs_root / entry.name))
            except UnregisteredStageError as error:
                LOG.info(
                    "temporarily omitting an unlabelled fit directory=%s reason=%s",
                    entry.name,
                    error,
                )
                continue
            if len(runs) > 512:
                raise ValueError("native dashboard run count exceeds snapshot bound")
    if (primary_prepared_root is None) != (primary_runtime_root is None):
        raise ValueError("both primary prepared and runtime roots are required")
    if primary_prepared_root is not None:
        from multistage_github_primary import collect_primary_refits

        runs.extend(collect_primary_refits(primary_prepared_root, primary_runtime_root))
    return runs


def _preserve_metadata(value: object) -> object:
    """Preserve full native metadata while refusing credential-shaped content."""
    if isinstance(value, dict):
        copied: dict[str, object] = {}
        for key, item in value.items():
            lowered = key.casefold().replace(" ", "_")
            if any(part in lowered for part in SENSITIVE_KEY_PARTS):
                raise ValueError("native metadata contains a credential-shaped field")
            copied[key] = _preserve_metadata(item)
        return copied
    if isinstance(value, list):
        return [_preserve_metadata(item) for item in value]
    if isinstance(value, str):
        if (
            URL_CREDENTIALS.sub(r"\1[redacted]@", value) != value
            or SECRET_TEXT.sub(r"\1=[redacted]", value) != value
            or BEARER_TEXT.sub("Bearer [redacted]", value) != value
        ):
            raise ValueError("native metadata contains credential-shaped text")
        return value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("native metadata contains a non-finite number")
    return value


def _sanitize_event_context(value: object) -> object:
    """Redact secret event-context fields without truncating full native records."""
    if isinstance(value, dict):
        copied: dict[str, object] = {}
        for key, item in value.items():
            lowered = key.casefold().replace(" ", "_")
            copied[key] = (
                "[redacted]"
                if any(part in lowered for part in SENSITIVE_KEY_PARTS)
                else _sanitize_event_context(item)
            )
        return copied
    if isinstance(value, list):
        return [_sanitize_event_context(item) for item in value]
    if isinstance(value, str):
        value = URL_CREDENTIALS.sub(r"\1[redacted]@", value)
        value = SECRET_TEXT.sub(r"\1=[redacted]", value)
        return BEARER_TEXT.sub("Bearer [redacted]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
