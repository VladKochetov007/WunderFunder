"""Feature-arm, prior-membership and training-history validation checks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

_FEATURES = {
    "group": ("age", "group_age", "none", 323),
    "field": ("age", "field_age", "none", 323),
    "moments": ("innovation", "causal_innovations", "newest", 291),
    "innovations": ("innovation", "causal_innovations", "none", 291),
}
_COUNTERS = ("optimizer_attempts", "optimizer_completed_updates", "amp_skipped_updates")
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SEQUENCE_ROWS = 20_000
_MOMENT_SAMPLE_STRIDE = 64


def job_spec(job: dict[str, Any]) -> tuple[str, str, str, int]:
    """Validate the feature cell identity and return its frozen representation."""
    if not isinstance(job, dict) or not _RUN_NAME.fullmatch(str(job.get("id", ""))):
        raise ValueError("feature job has an invalid run id")
    if job.get("arm") not in _FEATURES or type(job.get("fold")) is not int:
        raise ValueError("feature job has an unknown arm or noninteger fold")
    if not 0 <= job["fold"] < 4 or type(job.get("seed")) is not int or job["seed"] < 0:
        raise ValueError("feature job fold or seed is outside the supported range")
    family, mode, control, width = _FEATURES[job["arm"]]
    if job.get("family") != family:
        raise ValueError("feature job family does not match its arm")
    return mode, control, str(job["id"]), width


def _argv_values(argv: list[Any]) -> dict[str, Any]:
    """Parse option names and values from the saved canonical trainer argv."""
    if not argv or any(not isinstance(item, str) for item in argv):
        raise ValueError("worker request has no string trainer argv")
    parsed: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        key, separator, value = token[2:].partition("=")
        key = key.replace("-", "_")
        if key in parsed:
            raise ValueError(f"canonical trainer argv repeats --{key.replace('_', '-')}")
        if separator:
            parsed[key] = value
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            parsed[key] = argv[index + 1]
            index += 1
        else:
            parsed[key] = True
        index += 1
    return parsed


def _coerce_cli(value: Any, template: Any, key: str) -> Any:
    """Convert one argv value using its saved argparse type."""
    if isinstance(template, bool):
        if value is not True:
            raise ValueError(f"trainer argv flag --{key.replace('_', '-')} must be a switch")
        return True
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def check_request(
    request: dict[str, Any],
    args: dict[str, Any],
    *,
    run_name: str,
    fold: int,
    seed: int,
    expected_argv: list[str],
    source_root: str,
    work_dir: str,
) -> str:
    """Bind request identity and the exact R01-builder-plus-feature argv to provenance."""
    argv = request.get("train_argv")
    if (
        request.get("schema") != "pr15-host-worker-request.v1"
        or request.get("mode") != "fit"
        or request.get("name") != run_name
        or request.get("source_root") != source_root
        or request.get("work") != work_dir
    ):
        raise ValueError("worker request identity differs from the feature fit workspace")
    if not isinstance(argv, list) or argv != expected_argv:
        raise ValueError("worker trainer argv differs from the exact R01 recipe and feature arm")
    for key, expected in (("id", run_name), ("name", run_name), ("fold", fold), ("seed", seed)):
        if key in request and request[key] != expected:
            raise ValueError(f"worker request {key} differs from the feature job")
    parsed = _argv_values(argv)
    for key, value in parsed.items():
        if key not in args:
            raise ValueError(
                f"trainer argv option --{key.replace('_', '-')} is absent from provenance"
            )
        try:
            observed = _coerce_cli(value, args[key], key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"trainer argv option --{key.replace('_', '-')} is malformed"
            ) from error
        if observed != args[key]:
            raise ValueError(
                f"trainer argv and saved provenance disagree for --{key.replace('_', '-')}"
            )
    digest = hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
    expected_hash = request.get("training_argv_sha256")
    if expected_hash is not None and expected_hash != digest:
        raise ValueError("worker request trainer argv hash is invalid")
    required = {
        "name": run_name,
        "seed": seed,
        "valid_fold": fold,
        "hidden": 512,
        "layers": 1,
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "precision": "fp32",
    }
    if any(args.get(key) != value for key, value in required.items()):
        raise ValueError("worker request does not describe the fixed R01 recipe")
    return digest


def _finite_tree(value: Any, label: str) -> None:
    """Reject nonfinite numeric values in saved fit evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"fit evidence has a nonfinite value at {label}")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")


def check_updates(
    history: list[Any], provenance: dict[str, Any], checkpoint: dict[str, Any]
) -> int:
    """Require six complete, finite, evaluation-withheld epochs and matching counters."""
    if not isinstance(history, list) or len(history) != 6:
        raise ValueError("fit history must contain exactly six epochs")
    totals = dict.fromkeys(_COUNTERS, 0)
    for epoch, record in enumerate(history):
        if not isinstance(record, dict):
            raise ValueError(f"fit history epoch {epoch} is malformed")
        _finite_tree(record, f"history[{epoch}]")
        diagnostics = record.get("update_diagnostics", {})
        current, cumulative = diagnostics.get("epoch"), diagnostics.get("cumulative")
        if record.get("epoch") != epoch or record.get("evaluation") != "withheld":
            raise ValueError("fit history is not the ordered six-epoch withheld endpoint")
        if not isinstance(current, dict) or not isinstance(cumulative, dict):
            raise ValueError("fit history lacks epoch or cumulative optimizer counters")
        for key in _COUNTERS:
            value = current.get(key)
            if type(value) is not int:
                raise ValueError(f"fit epoch {epoch} has invalid counter {key}")
            totals[key] += value
            if type(cumulative.get(key)) is not int or cumulative[key] != totals[key]:
                raise ValueError(f"fit epoch {epoch} cumulative counter {key} is inconsistent")
        if current["optimizer_attempts"] <= 0 or (
            current["optimizer_attempts"] != current["optimizer_completed_updates"]
            or current["amp_skipped_updates"] != 0
        ):
            raise ValueError(f"fit epoch {epoch} has missing, skipped or nonfinite updates")
    steps = {row["update_diagnostics"]["epoch"]["optimizer_attempts"] for row in history}
    if len(steps) != 1:
        raise ValueError("fit epochs have inconsistent attempted update totals")
    _finite_tree(provenance.get("update_diagnostics"), "provenance.update_diagnostics")
    for source in (
        provenance.get("cumulative_updates"),
        history[-1]["update_diagnostics"].get("cumulative"),
        checkpoint.get("provenance", {}).get("cumulative_updates"),
    ):
        if not isinstance(source, dict) or any(
            type(source.get(key)) is not int or source.get(key) != totals[key] for key in _COUNTERS
        ):
            raise ValueError("checkpoint, provenance and history update totals disagree")
    diagnostic_rows = provenance.get("update_diagnostics")
    if not isinstance(diagnostic_rows, list) or len(diagnostic_rows) != 6:
        raise ValueError("provenance does not contain all six update records")
    for epoch, (saved, row) in enumerate(zip(diagnostic_rows, history, strict=True)):
        expected = row["update_diagnostics"]
        if (
            saved.get("epoch_index") != epoch
            or saved.get("epoch") != expected.get("epoch")
            or saved.get("cumulative") != expected.get("cumulative")
        ):
            raise ValueError("checkpoint provenance update records differ from fit history")
    return totals["optimizer_completed_updates"]


def _prior_members(inputs: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Recreate the source-qualified deterministic 32-group moment-prior sample."""
    import numpy as np

    train, valid = inputs.get("fit_train_groups"), inputs.get("fit_valid_groups")
    if not isinstance(train, list) or not isinstance(valid, list):
        raise ValueError("feature inputs omit fitting-only source-qualified groups")
    if any(type(group) is not int or group < 0 for group in train + valid):
        raise ValueError("feature fitting membership has invalid row-group identifiers")
    members = sorted({("train", group) for group in train} | {("valid", group) for group in valid})
    if not members:
        raise ValueError("moment prior has no fitting membership")
    rng = np.random.default_rng(seed + 20261012)
    if len(members) > 32:
        choices = np.sort(rng.choice(len(members), size=32, replace=False))
        members = [members[int(index)] for index in choices]
    return [{"split": split, "row_group": group} for split, group in members]


def check_prior(
    checkpoint: dict[str, Any],
    provenance: dict[str, Any],
    inputs: dict[str, Any],
    *,
    mode: str,
    control: str,
    seed: int,
    feature_source_sha256: str | None,
) -> str | None:
    """Bind innovation prior content, sampled fit membership and source bytes."""
    prior = checkpoint.get("research_feature_prior")
    if mode != "causal_innovations":
        if (
            prior is not None
            or provenance.get("causal_feature_prior") is not None
            or "causal_feature_control" in provenance
            or "causal_feature_prior_source_sha256" in provenance
        ):
            raise ValueError("field-age checkpoint unexpectedly contains moment-prior metadata")
        return None
    if not isinstance(prior, dict) or provenance.get("causal_feature_prior") != prior:
        raise ValueError("innovation checkpoint and provenance do not share one fitted prior")
    if prior.get("schema") != "causal_moments_v1":
        raise ValueError("innovation checkpoint has the wrong causal prior schema")
    mean, scale = prior.get("mean"), prior.get("scale")
    if (
        not isinstance(mean, list)
        or not isinstance(scale, list)
        or len(mean) != 16
        or len(scale) != 16
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in mean + scale
        )
        or any(value <= 0 for value in scale)
    ):
        raise ValueError("innovation prior needs sixteen finite means and positive scales")
    expected_members = _prior_members(inputs, seed)
    encoded = json.dumps(expected_members, sort_keys=True, separators=(",", ":")).encode("ascii")
    if (
        prior.get("fit_members") != expected_members
        or prior.get("fit_members_sha256") != hashlib.sha256(encoded).hexdigest()
        or prior.get("sample_policy") != "uniform_seeded_source_qualified_32_groups_v1"
        or prior.get("sample_seed") != seed + 20261012
        or prior.get("sample_stride") != _MOMENT_SAMPLE_STRIDE
        or type(prior.get("sample_count")) is not int
        or prior["sample_count"]
        != len(expected_members) * math.ceil(_SEQUENCE_ROWS / _MOMENT_SAMPLE_STRIDE)
    ):
        raise ValueError(
            "innovation prior membership or deterministic sampling differs from fit inputs"
        )
    expected_zero = 16 if control == "newest" else 0
    expected_control = {
        "kind": control,
        "zeroed_suffix_width": expected_zero,
        "effective_capacity": "reduced" if expected_zero else "full",
    }
    if (
        provenance.get("causal_feature_control") != expected_control
        or feature_source_sha256 is None
        or provenance.get("causal_feature_prior_source_sha256") != feature_source_sha256
    ):
        raise ValueError(
            "innovation control or causal feature implementation source pin is invalid"
        )
    return hashlib.sha256(
        json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
