"""Stream official scores for the fixed five-arm F0/seed-1001 panel."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from first_cell_score_contract import _read_export_bundle, _validate_bound_paths
from first_cell_score_core import (
    EXPECTED_PORTABLE_REGISTRY_BYTES,
    EXPECTED_PORTABLE_REGISTRY_SHA256,
    EXPECTED_PORTABLE_SOURCE_NAMES,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    EXPECTED_SOURCE_SET_SHA256,
    EXPECTED_SPLIT_SHA256,
    MAX_EXPORT_MANIFEST_BYTES,
    N_GROUPS,
    SOURCE_COMMIT,
    _load_helpers,
    _pin_file,
    _preflight_split_inputs,
    _stable_identity,
    observe_registered_source,
    require_ordered_groups,
    score_stream,
)
from first_cell_score_predictions import _open_predictions


def _score_bundle(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate a complete export and score its fixed panel in source order."""
    binding, _source_manifest, manifest, manifest_pin, manifest_identity = _read_export_bundle(
        args.evidence_root,
        args.export_root,
        args.binding,
        args.source_manifest,
    )
    _validate_bound_paths(args, binding)
    registered_data = binding["registered_data"]
    registered_sources = registered_data["source_identities"]
    portable_registry = Path(registered_data["portable_manifest"]["absolute_path"])
    helpers = _load_helpers(args.source_root, portable_registry=portable_registry)
    portable_registry_pin, portable_registry_identity = _pin_file(portable_registry, 2_000_000)
    if (portable_registry_pin["bytes"], portable_registry_pin["sha256"]) != (
        EXPECTED_PORTABLE_REGISTRY_BYTES,
        EXPECTED_PORTABLE_REGISTRY_SHA256,
    ):
        raise ValueError("portable source registry changed after helper import")
    data_root = args.data_root
    split_path = Path(str(binding["split"]["absolute_path"]))
    if split_path.resolve(strict=True) != split_path:
        raise ValueError("campaign split path cannot traverse a symlink")
    registered_before_split = {
        relative: observe_registered_source(data_root, relative, registered_sources[relative])
        for relative in EXPECTED_PORTABLE_SOURCE_NAMES
    }
    _preflight_split_inputs(split_path, data_root, registered_sources)
    split = helpers.load_campaign_split(split_path, data_root)
    registered_after_split = {
        relative: observe_registered_source(data_root, relative, registered_sources[relative])
        for relative in EXPECTED_PORTABLE_SOURCE_NAMES
    }
    if registered_before_split != registered_after_split:
        raise ValueError("one or more registered data identities changed during split validation")
    current_registry_pin, current_registry_identity = _pin_file(portable_registry, 2_000_000)
    if (current_registry_pin, current_registry_identity) != (
        portable_registry_pin,
        portable_registry_identity,
    ):
        raise ValueError("portable source registry changed during canonical split validation")
    groups = [int(group) for group in split.evaluation_groups]
    if split.sha256 != EXPECTED_SPLIT_SHA256 or split.fold != 0 or split.n_folds != 4:
        raise ValueError("canonical split loader returned a different F0 identity")
    require_ordered_groups(groups, manifest.get("ordered_groups"))
    if len(groups) != N_GROUPS:
        raise ValueError("canonical F0 evaluation membership is not exactly 468 groups")
    valid_component_groups = [
        int(group) for source, group, _component in split.components if source == "valid"
    ]
    valid_row_group_count = binding["split"]["valid_row_group_count"]
    if (
        len(valid_component_groups) != valid_row_group_count
        or len(set(valid_component_groups)) != valid_row_group_count
        or set(valid_component_groups) != set(range(valid_row_group_count))
    ):
        raise ValueError(
            "canonical valid component inventory differs from the fixed row-group count"
        )
    component_by_group = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    predictions, prediction_pins, prediction_identities = _open_predictions(
        args.export_root, binding, manifest
    )
    registered_before_scoring = {
        relative: observe_registered_source(data_root, relative, registered_sources[relative])
        for relative in EXPECTED_PORTABLE_SOURCE_NAMES
    }
    if registered_after_split != registered_before_scoring:
        raise ValueError("registered data identity changed before scoring began")
    valid_path = Path(registered_sources["datasets/valid.parquet"]["path"])
    sequence_file = helpers.sequence_file(valid_path)
    try:
        if (
            len(sequence_file) != valid_row_group_count
            or not sequence_file.has_targets
            or not sequence_file.has_mask
        ):
            raise ValueError(
                "registered validation file lacks complete groups, targets, or scored mask"
            )
        scored = score_stream(groups, component_by_group, predictions, sequence_file, helpers)
    finally:
        sequence_file._pf.close()
    registered_after = {
        relative: observe_registered_source(data_root, relative, registered_sources[relative])
        for relative in EXPECTED_PORTABLE_SOURCE_NAMES
    }
    if registered_before_scoring != registered_after:
        raise ValueError("one or more registered data identities changed while scoring")
    final_registry_pin, final_registry_identity = _pin_file(portable_registry, 2_000_000)
    if (final_registry_pin, final_registry_identity) != (
        portable_registry_pin,
        portable_registry_identity,
    ):
        raise ValueError("portable source registry changed while scoring")
    manifest_path = args.export_root / "EXPORT_MANIFEST_V1.json"
    final_manifest_pin, final_manifest_identity = _pin_file(
        manifest_path, MAX_EXPORT_MANIFEST_BYTES
    )
    if final_manifest_pin != manifest_pin or final_manifest_identity != manifest_identity:
        raise ValueError("export completion manifest changed while being scored")
    for item in prediction_identities:
        item_path = (
            args.export_root / item["arm"] / "PREDICTION_SIDECAR.json"
            if item["kind"] == "sidecar"
            else args.export_root / item["arm"] / "predictions.npy"
        )
        linked = item_path.lstat()
        if _stable_identity(linked, linked) != item["identity"]:
            raise ValueError(f"{item['arm']} {item['kind']} changed during scoring")
    predictions.clear()
    fixed_models = [
        {
            "arm": arm["arm"],
            "run_name": arm["run_name"],
            "unit": arm["unit"],
            "invocation_id": arm["invocation_id"],
            "representation_mode": arm["representation_mode"],
            "selected_epoch": 5,
            "successful_updates": 44760,
            "checkpoint": dict(arm["model"]),
            "terminal_acceptance": dict(arm["terminal_acceptance"]),
            "exact_stop": dict(arm["exact_stop"]),
        }
        for arm in binding["arms"]
    ]
    return {
        "schema": "fg19.first-cell-five-arm-streaming-score.v1",
        "status": "FIXED_FIRST_CELL_DEVELOPMENT_EVIDENCE_SCORED",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "scope": "Only the registered F0/seed-1001 five-arm panel at fixed epoch 5; all 468 fit-excluded exposed development groups.",
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "source_set_sha256": EXPECTED_SOURCE_SET_SHA256,
        "split_manifest_sha256": split.sha256,
        "ordered_membership_sha256": manifest.get("membership_sha256"),
        "fixed_models": fixed_models,
        "export_manifest": manifest_pin,
        "exported_file_pins": prediction_pins,
        "registered_data": {
            "admission": registered_data["admission"],
            "portable_manifest": registered_data["portable_manifest"],
            "registered_source_identities": registered_sources,
            "observed_before_split": registered_before_split,
            "observed_after_split": registered_after_split,
            "observed_before_scoring": registered_before_scoring,
            "observed_after_scoring": registered_after,
        },
        "scoring": scored,
        "limitations": [
            "The 468 F0 groups are fit-excluded but repeatedly exposed development evidence, not a fresh or public holdout.",
            "Component bootstrap intervals describe only group/component resampling conditional on these fixed models; they do not estimate fit-seed or public-transfer uncertainty.",
            "No predictions were generated by this scorer; it consumes only the reviewed fixed epoch-5 export.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Validate and score the fixed first-cell export into a create-only report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = _score_bundle(args)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    descriptor = os.open(
        args.out,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_descriptor = os.open(
        args.out.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arms": list(result["scoring"]["arms"]),
                "contrasts": list(result["scoring"]["paired_contrasts_vs_control"]),
                "output": str(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
