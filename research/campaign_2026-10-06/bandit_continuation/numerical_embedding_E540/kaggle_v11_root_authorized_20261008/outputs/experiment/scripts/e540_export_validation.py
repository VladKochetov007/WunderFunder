"""Pure-Python checks for frozen E540 fit artifacts and timing provenance."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_UPDATES = 72_996
EXPECTED_BATCHES = 924
HISTORY_ROUNDING_TOLERANCE_SECONDS = 0.11
CHECKPOINT_WRITE_ALLOWANCE_SECONDS = 30.0
POST_SNAPSHOT_PROVENANCE_FIELDS = {"checkpoint_sha256", "selected_epoch", "fitting_seconds"}


def sha256_file(path: Path) -> str:
    """Hash files with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash JSON values with a stable finite encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def finite_positive_seconds(value: object, label: str) -> float:
    """Require a finite, strictly positive duration; booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite positive number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return seconds


def validate_history_provenance_duration(
    history_seconds: object,
    provenance_seconds: object,
    *,
    checkpoint_write_allowance_seconds: float = CHECKPOINT_WRITE_ALLOWANCE_SECONDS,
) -> dict[str, float]:
    """Validate trainer's pre-save history and post-save provenance clocks.

    The trainer rounds both clocks to tenths. History is recorded before saving
    epoch checkpoints; sibling provenance is written after those saves. Permit
    only the rounding discrepancy in the reverse direction and a separately
    declared bounded checkpoint-write interval in the forward direction.
    """
    history = finite_positive_seconds(history_seconds, "history train_seconds")
    provenance = finite_positive_seconds(provenance_seconds, "provenance fitting_seconds")
    if (
        isinstance(checkpoint_write_allowance_seconds, bool)
        or not isinstance(checkpoint_write_allowance_seconds, (int, float))
        or not math.isfinite(float(checkpoint_write_allowance_seconds))
        or float(checkpoint_write_allowance_seconds) < 0
    ):
        raise ValueError("checkpoint-write allowance must be finite and nonnegative")
    delta = provenance - history
    if delta < -HISTORY_ROUNDING_TOLERANCE_SECONDS:
        raise ValueError("provenance duration precedes history beyond independent rounding")
    maximum_delta = float(checkpoint_write_allowance_seconds) + HISTORY_ROUNDING_TOLERANCE_SECONDS
    if delta > maximum_delta:
        raise ValueError("history/provenance duration gap exceeds checkpoint-write allowance")
    return {
        "history_train_seconds": history,
        "provenance_fitting_seconds": provenance,
        "observed_post_history_seconds": delta,
        "rounding_tolerance_seconds": HISTORY_ROUNDING_TOLERANCE_SECONDS,
        "checkpoint_write_allowance_seconds": float(checkpoint_write_allowance_seconds),
    }


def _same_group_membership(left: object, right: object) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and all(type(value) is int for value in (*left, *right))
        and len(left) == len(right) == len(set(left)) == len(set(right))
        and set(left) == set(right)
    )


def _validate_update_record(record: object, label: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} update diagnostics are malformed")
    epoch = record.get("epoch")
    grad = record.get("grad_stats")
    if not isinstance(epoch, Mapping) or not isinstance(grad, Mapping):
        raise ValueError(f"{label} update diagnostics are incomplete")
    if (
        epoch.get("optimizer_attempts") != EXPECTED_UPDATES
        or epoch.get("optimizer_completed_updates") != EXPECTED_UPDATES
        or epoch.get("amp_skipped_updates") != 0
        or epoch.get("completed_batches") != EXPECTED_BATCHES
        or grad.get("steps") != EXPECTED_UPDATES
        or grad.get("nonfinite_steps") != 0
    ):
        raise ValueError(f"{label} does not meet the fixed FP32 update contract")
    return record


def validate_fit_artifact_metadata(
    mode: str,
    history: object,
    provenance: Mapping[str, Any],
    subset: Mapping[str, Any],
    *,
    model_sha256: str,
    epoch0_sha256: str,
    expected_run_name: str,
    checkpoint_write_allowance_seconds: float = CHECKPOINT_WRITE_ALLOWANCE_SECONDS,
) -> dict[str, object]:
    """Validate saved fit metadata without loading tensor payloads."""
    if mode not in {"linear", "ple"}:
        raise ValueError("unsupported E540 mode")
    if not isinstance(history, list) or len(history) != 1 or not isinstance(history[0], Mapping):
        raise ValueError("fixed epoch-0 history must contain exactly one record")
    row = history[0]
    if row.get("epoch") != 0 or row.get("evaluation") != "withheld":
        raise ValueError("history is not the fixed epoch-0 no-evaluation record")
    duration = validate_history_provenance_duration(
        row.get("train_seconds"),
        provenance.get("fitting_seconds"),
        checkpoint_write_allowance_seconds=checkpoint_write_allowance_seconds,
    )

    args = provenance.get("args")
    numeric = provenance.get("numeric_preprocessing")
    if not isinstance(args, Mapping) or not isinstance(numeric, Mapping):
        raise ValueError("fit provenance omits trainer arguments or numeric preprocessing")
    if (
        provenance.get("seed") != 540
        or provenance.get("selected_epoch") != 0
        or args.get("name") != expected_run_name
        or args.get("seed") != 540
        or args.get("numeric_map") != mode
        or args.get("precision") != "fp32"
        or args.get("fit_subset_manifest_sha256") != subset.get("_sha256")
        or args.get("split_manifest_sha256") != subset.get("split_manifest_sha256")
        or args.get("epochs") != 1
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != 0
        or args.get("no_eval_during_fit") is not True
        or args.get("loss") != "wmse"
        or args.get("loss_norm") != "chunk"
        or args.get("clip") != 1.0
        or args.get("batch") != 2
        or args.get("chunk") != 256
        or args.get("train_source") != "train+valid"
        or args.get("deterministic") is not True
        or args.get("batch_order") != "seeded"
        or args.get("init") != ""
        or provenance.get("fit_subset_manifest_sha256") != subset.get("_sha256")
        or provenance.get("fit_subset_members_sha256") != subset.get("selected_members_sha256")
        or provenance.get("split_manifest_sha256") != subset.get("split_manifest_sha256")
        or provenance.get("fit_row_groups") != subset.get("selected_groups")
        or not _same_group_membership(
            provenance.get("evaluation_row_groups"), subset.get("evaluation_groups")
        )
        or numeric.get("mode") != mode
        or numeric.get("n_features") != 112
        or numeric.get("base_input_width") != 211
        or numeric.get("mapped_input_width") != 1107
        or numeric.get("n_intervals") != 8
        or numeric.get("fit_members") != subset.get("selected_groups")
        or numeric.get("fit_members_sha256") != subset.get("selected_members_sha256")
        or provenance.get("checkpoint_sha256")
        != {"model.pt": model_sha256, "model_ep0.pt": epoch0_sha256}
    ):
        raise ValueError("fit provenance does not match the frozen E540 run contract")

    updates = provenance.get("update_diagnostics")
    if not isinstance(updates, list) or len(updates) != 1:
        raise ValueError("fit provenance must contain one fixed epoch update record")
    update = _validate_update_record(updates[0], "sibling provenance")
    history_update = row.get("update_diagnostics")
    if not isinstance(history_update, Mapping):
        raise ValueError("history omits update diagnostics")
    for key in (
        "epoch",
        "cumulative",
        "sample_order_identity",
        "normalization_identity",
        "grad_stats",
    ):
        if history_update.get(key) != update.get(key):
            raise ValueError(f"history and sibling provenance differ in {key}")
    if (
        update.get("epoch_index") != 0
        or update.get("epoch", {}).get("optimizer_attempts")
        != provenance.get("cumulative_updates", {}).get("optimizer_attempts")
        or update.get("epoch", {}).get("optimizer_completed_updates")
        != provenance.get("cumulative_updates", {}).get("optimizer_completed_updates")
    ):
        raise ValueError("cumulative update counts do not match the fixed epoch")
    return {
        "run_name": expected_run_name,
        "mode": mode,
        "epoch": 0,
        "evaluation": "withheld",
        "updates": EXPECTED_UPDATES,
        "completed_batches": EXPECTED_BATCHES,
        "duration": duration,
        "sample_order_identity": update["sample_order_identity"],
        "normalization_identity": update["normalization_identity"],
    }


def validate_pair_metadata(
    linear: Mapping[str, object], ple: Mapping[str, object]
) -> dict[str, str]:
    """Require identical seed-540 sampling, normalization and non-map trainer args."""
    linear_args = linear.get("args")
    ple_args = ple.get("args")
    if not isinstance(linear_args, Mapping) or not isinstance(ple_args, Mapping):
        raise ValueError("paired runs omit trainer arguments")
    linear_common = dict(linear_args)
    ple_common = dict(ple_args)
    linear_common.pop("name", None)
    ple_common.pop("name", None)
    linear_common.pop("numeric_map", None)
    ple_common.pop("numeric_map", None)
    if linear_common != ple_common:
        raise ValueError("linear and PLE trainer arguments are not otherwise matched")
    linear_update = linear.get("update_diagnostics", [None])[0]
    ple_update = ple.get("update_diagnostics", [None])[0]
    if not isinstance(linear_update, Mapping) or not isinstance(ple_update, Mapping):
        raise ValueError("paired runs omit sampler or normalization identities")
    if linear_update.get("sample_order_identity") != ple_update.get("sample_order_identity"):
        raise ValueError("paired runs have different observed sampler identities")
    if linear_update.get("normalization_identity") != ple_update.get("normalization_identity"):
        raise ValueError("paired runs have different normalization identities")
    return {
        "sample_order_identity_sha256": canonical_json_sha256(
            linear_update["sample_order_identity"]
        ),
        "normalization_identity_sha256": canonical_json_sha256(
            linear_update["normalization_identity"]
        ),
    }


def validate_snapshot_provenance(
    checkpoint: Mapping[str, object], provenance: Mapping[str, object]
) -> None:
    """Match checkpoint args and pre-snapshot provenance exactly to sibling JSON."""
    args = checkpoint.get("args")
    embedded = checkpoint.get("provenance")
    if not isinstance(args, Mapping) or args != provenance.get("args"):
        raise ValueError("checkpoint args differ from sibling provenance")
    if not isinstance(embedded, Mapping):
        raise ValueError("checkpoint-embedded provenance is missing")
    expected = {
        key: value
        for key, value in provenance.items()
        if key not in POST_SNAPSHOT_PROVENANCE_FIELDS
    }
    if dict(embedded) != expected:
        raise ValueError("checkpoint-embedded provenance differs from sibling provenance")
