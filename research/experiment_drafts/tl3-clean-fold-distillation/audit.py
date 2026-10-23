"""Independent artifact checks before teacher reuse and student endpoint grading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from commands import checkpoint_path, fit_args, run_name
from common import digest, output_root, preflight, read_json, write_new

# Canonical checkpoint and provenance schemas contain heterogeneous records.


def teacher_targets(assets: dict[str, Any], cell: str) -> list[dict[str, Any]]:
    """Bind both new teachers and their fitting-only, zero-padded legacy caches."""
    import numpy as np

    from wnn.splits import load_campaign_split

    root = Path(assets["project_root"])
    split = load_campaign_split(Path(assets["splits"][cell]["path"]), root)
    result = []
    for role in ("teacher_a", "teacher_b"):
        run = run_name(cell, role)
        approved = require_fit(assets, cell, role)
        record = {
            "run": run,
            "coefficient": 0.5,
            "checkpoint_sha256": approved["checkpoint_sha256"],
        }
        for source, fitting in (("train", split.train_groups), ("valid", split.valid_groups)):
            path = root / "cache" / f"pred_{run}_{source}.npy"
            meta = read_json(path.with_suffix(".json"))
            with np.load(root / "cache" / f"{source}_targets.npz", allow_pickle=False) as targets:
                order = targets["row_group"].tolist()
            if (
                meta["run"] != run
                or meta["split"] != source
                or meta["row_groups"] != order
                or meta["split_manifest_sha256"] != split.sha256
                or meta["epoch"] != 3
                or meta["checkpoint_sha256"] != approved["checkpoint_sha256"]
                or meta.get("feature_only") is not True
                or meta.get("inference_precision") != "fp32"
                or meta.get("campaign_export_scope") != "fit-only-scattered-to-full-index"
                or meta.get("zeros_outside_fit") is not True
                or set(meta["fitting_only_groups"]) != set(fitting)
                or digest(path) != meta["prediction_sha256"]
            ):
                raise ValueError("teacher target identity/coverage changed")
            incoming = root / "cache" / f"pred_{run}_fit_{source}.npy"
            raw = read_json(incoming.with_suffix(".json"))
            if (
                digest(incoming) != meta["unpadded_prediction_sha256"]
                or digest(incoming.with_suffix(".json")) != meta["unpadded_sidecar_sha256"]
                or raw.get("prediction_sha256") != meta["unpadded_prediction_sha256"]
                or raw.get("row_groups") != list(fitting)
                or raw.get("campaign_export_scope") != "fit"
                or raw.get("feature_only") is not True
                or raw.get("inference_precision") != "fp32"
                or raw.get("checkpoint_sha256") != approved["checkpoint_sha256"]
            ):
                raise ValueError(
                    "padded cache lost its original feature-only FP32 producer binding"
                )
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (len(order), 20000, 2) or values.dtype != np.float32:
                raise ValueError("teacher target shape/dtype mismatch")
            allowed = set(fitting)
            for i, group in enumerate(order):
                if not np.isfinite(values[i]).all() or (group not in allowed and np.any(values[i])):
                    raise ValueError("teacher cache has nonfinite or nonzero excluded rows")
            record[f"{source}_prediction_sha256"] = meta["prediction_sha256"]
        result.append(record)
    return result


def require_fit(assets: dict[str, Any], cell: str, role: str) -> dict[str, Any]:
    """Check a completed, independently audited endpoint has not drifted."""
    record = read_json(output_root(assets) / "audit" / f"{run_name(cell, role)}.json")
    bound = preflight(assets)
    path = checkpoint_path(assets, cell, role)
    if record.get("status") != "passed" or record["run"] != run_name(cell, role):
        raise ValueError("missing successful fit audit")
    for key, value in bound.items():
        if record[key] != value:
            raise ValueError(f"fit audit binding changed: {key}")
    if (
        digest(path) != record["checkpoint_sha256"]
        or digest(path.parent / "provenance.json") != record["provenance_sha256"]
    ):
        raise ValueError("audited model/provenance changed")
    return record


def audit_fit(assets: dict[str, Any], cell: str, role: str) -> None:
    """Validate full epochs, cold ancestry, tensors, memberships and fixed checkpoint policy."""
    import torch

    from wnn.splits import load_campaign_split
    from wnn.training.cli import build_parser

    binding = preflight(assets)
    root = Path(assets["project_root"])
    split = load_campaign_split(Path(assets["splits"][cell]["path"]), root)
    run = run_name(cell, role)
    start = read_json(output_root(assets) / "start" / f"{run}.json")
    terminal = read_json(output_root(assets) / "terminal" / f"{run}.json")
    for record in (start, terminal):
        if any(record.get(key) != value for key, value in binding.items()) or record["run"] != run:
            raise ValueError("actual training process bindings changed")
    if terminal["status"] != "completed" or terminal["profile"]:
        raise ValueError("profile/incomplete run is not a scientific endpoint")
    path = checkpoint_path(assets, cell, role)
    provenance_path = path.parent / "provenance.json"
    provenance = read_json(provenance_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    command = fit_args(assets, cell, role, False)
    explicit = {item[2:].replace("-", "_") for item in command if item.startswith("--")}
    argv = vars(build_parser().parse_args(command))
    actual = checkpoint["args"]
    for key, expected in argv.items():
        if key not in explicit:
            continue
        if key in {"split_manifest", "valid_fold", "n_valid_folds", "eval_source", "resume_state"}:
            continue  # Canonical setup resolves these from the pinned manifest/continuation.
        if actual.get(key) != expected:
            raise ValueError(f"checkpoint deviates from frozen canonical arguments: {key}")
    teacher = role.startswith("teacher_")
    epochs = 4 if teacher else 2
    if (
        checkpoint["epoch"] != epochs - 1
        or actual["split_manifest_sha256"] != split.sha256
        or actual["init"]
        or actual["propensity"]
        or actual["include_heldout_valid"]
        or checkpoint["n_in"] != (263 if teacher else 211)
    ):
        raise ValueError("invalid endpoint epoch, ancestry, feature width or fold")
    for record in (provenance, checkpoint["provenance"]):
        groups = record["fit_row_groups"]
        if set(groups["train"]) != set(split.train_groups) or set(groups["valid"]) != set(
            split.valid_groups
        ):
            raise ValueError("fitting membership differs from complete eligible components")
        split.assert_fit_groups(groups["train"], groups["valid"])
    if provenance["checkpoint_sha256"].get(path.name) != digest(path):
        raise ValueError("checkpoint bytes are not bound to their producer")
    state = checkpoint["model"]
    for name, tensor in state.items():
        if (tensor.is_floating_point() and tensor.dtype != torch.float32) or not torch.isfinite(
            tensor
        ).all():
            raise ValueError(f"invalid model tensor {name}")
    updates = provenance["update_diagnostics"]
    expected_batches = (len(split.train_groups) + len(split.valid_groups)) // 32
    if len(updates) != epochs:
        raise ValueError("scientific fitting stopped before the fixed epoch budget")
    for epoch, entry in enumerate(updates):
        counts = entry["epoch"]
        if (
            entry["epoch_index"] != epoch
            or counts["completed_batches"] != expected_batches
            or counts["optimizer_attempts"] != expected_batches * 10
            or counts["optimizer_completed_updates"] != expected_batches * 10
            or counts["amp_skipped_updates"] != 0
            or entry["grad_stats"]["nonfinite_steps"] != 0
        ):
            raise ValueError("incomplete/nonfinite scientific update budget")
    targets = [] if teacher else teacher_targets(assets, cell)
    audit = {
        **binding,
        "status": "passed",
        "run": run,
        "checkpoint_sha256": digest(path),
        "provenance_sha256": digest(provenance_path),
        "split_sha256": split.sha256,
        "initial_state_sha256": start["initial_state_sha256"],
        "runtime_sha256": start["runtime_sha256"],
        "fit_row_groups": provenance["fit_row_groups"],
        "evaluation_row_groups": list(split.evaluation_groups),
        "normalization_identity": provenance["normalization_identity"],
        "teacher_targets": targets,
        "sample_order_identities": [entry["sample_order_identity"] for entry in updates],
        "completed_epochs": epochs,
        "terminal_sha256": digest(output_root(assets) / "terminal" / f"{run}.json"),
    }
    if role == "kd":
        control = require_fit(assets, cell, "labels")
        for key in (
            "initial_state_sha256",
            "runtime_sha256",
            "fit_row_groups",
            "normalization_identity",
            "teacher_targets",
            "sample_order_identities",
        ):
            if audit[key] != control[key]:
                raise ValueError(f"paired students differ beyond KD alpha: {key}")
    write_new(output_root(assets) / "audit" / f"{run}.json", audit)
