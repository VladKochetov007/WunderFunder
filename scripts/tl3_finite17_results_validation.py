from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.tl3_finite17_dashboard_core import (
        ADOPTED_NODE,
        TRAIN_GROUPS,
        VALID_GROUPS,
        VALID_ROWS,
        BridgeError,
        _cli_args,
        _read_json,
        _same_arg,
        _sha256_file,
    )
except ModuleNotFoundError:
    from tl3_finite17_dashboard_core import (
        ADOPTED_NODE,
        TRAIN_GROUPS,
        VALID_GROUPS,
        VALID_ROWS,
        BridgeError,
        _cli_args,
        _read_json,
        _same_arg,
        _sha256_file,
    )


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _verify_run(
    root: Path,
    plan: dict[str, Any],
    node: dict[str, Any],
    state: dict[str, Any],
    train: list[int],
    valid: list[int],
) -> dict[str, Any] | None:
    """Require driver's verified receipt and independently check its source/history hashes."""
    verified = state.get("verified_run")
    if verified is None:
        if state.get("state") == "complete":
            raise BridgeError(f"{node['id']}: completed node lacks its verified-run receipt")
        return None
    if not isinstance(verified, dict):
        raise BridgeError(f"{node['id']}: verified-run receipt is malformed")
    run = str(node["run"])
    directory = root / "artifacts" / "runs" / run
    if (
        Path(str(node.get("run_directory"))).resolve(strict=False) != directory
        or directory.is_symlink()
        or not directory.is_dir()
    ):
        raise BridgeError(f"{node['id']}: run directory differs from the frozen plan")
    endpoint = int(node["endpoint"])
    paths = {
        "model.pt": directory / "model.pt",
        f"model_ep{endpoint}.pt": directory / f"model_ep{endpoint}.pt",
        "provenance.json": directory / "provenance.json",
        "history.json": directory / "history.json",
    }
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise BridgeError(f"{node['id']}: a verified fit artifact is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    source_sha = plan["source_identity"]["training_source_sha256"]
    if node["id"] == ADOPTED_NODE:
        adoption = plan.get("adopted_first_fit")
        if not isinstance(adoption, dict) or not isinstance(adoption.get("source_sha256"), str):
            raise BridgeError("adopted fin256_s3 lacks its original source identity")
        source_sha = adoption["source_sha256"]
    expected_receipt = {
        "run": run,
        "epoch": endpoint,
        "model_sha256": hashes["model.pt"],
        "endpoint_sha256": hashes[f"model_ep{endpoint}.pt"],
        "provenance_sha256": hashes["provenance.json"],
        "history_sha256": hashes["history.json"],
        "source_sha256": source_sha,
        "expected_updates": int(node["expected_updates"]),
        "split_manifest_sha256": plan["split"]["split_manifest_sha256"],
    }
    if any(verified.get(key) != value for key, value in expected_receipt.items()):
        raise BridgeError(f"{node['id']}: verified-run receipt differs from current artifacts")
    provenance = _read_json(paths["provenance.json"])
    identity = provenance.get("trainer_implementation")
    if (
        provenance.get("split_manifest_sha256") != plan["split"]["split_manifest_sha256"]
        or not isinstance(identity, dict)
        or identity.get("combined_sha256", identity.get("sha256")) != source_sha
        or provenance.get("selected_epoch") != endpoint
    ):
        raise BridgeError(f"{node['id']}: provenance source, split or endpoint differs")
    fit = provenance.get("fit_row_groups")
    evaluated = provenance.get("evaluation_row_groups")
    if (
        not isinstance(fit, dict)
        or not isinstance(fit.get("train"), list)
        or set(fit["train"]) != set(train)
        or len(fit["train"]) != len(train)
        or fit.get("valid") != []
        or not isinstance(evaluated, list)
        or set(evaluated) != set(valid)
        or len(evaluated) != len(valid)
    ):
        raise BridgeError(f"{node['id']}: provenance membership is not train-only/full-valid")
    _verify_args(plan, node, provenance.get("args"))
    checkpoint_hashes = provenance.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_hashes, dict)
        or checkpoint_hashes.get("model.pt") != hashes["model.pt"]
        or checkpoint_hashes.get(f"model_ep{endpoint}.pt") != hashes[f"model_ep{endpoint}.pt"]
    ):
        raise BridgeError(f"{node['id']}: provenance does not bind selected checkpoints")
    history = json.loads(paths["history.json"].read_text(encoding="utf-8"))
    metrics = _verify_history(history, node, train)
    return {
        **metrics,
        "fit_source_sha256": source_sha,
        "model_sha256": hashes["model.pt"],
        "history_sha256": hashes["history.json"],
        "verified_endpoint": endpoint,
    }


def _verify_args(plan: dict[str, Any], node: dict[str, Any], actual: object) -> None:
    if not isinstance(actual, dict):
        raise BridgeError(f"{node['id']}: provenance omits trainer arguments")
    expected = _cli_args(node["args"])
    expected.update({"eval_source": "vfold", "valid_fold": 0, "n_valid_folds": 1})
    expected["checkpoint_epoch"] = int(node["endpoint"])
    if node["id"] == ADOPTED_NODE:
        expected.pop("train_eval_seqs", None)
    else:
        for key, value in expected.items():
            if key not in actual or not _same_arg(value, actual[key]):
                raise BridgeError(f"{node['id']}: provenance trainer argument differs: {key}")
    train_eval_sequences = 0 if node["id"] == ADOPTED_NODE else 32
    if (
        actual.get("name") != node["run"]
        or actual.get("train_source") != "train"
        or actual.get("train_seqs") != TRAIN_GROUPS
        or actual.get("eval_source") != "vfold"
        or actual.get("full_valid") is not True
        or actual.get("no_eval_during_fit") is not False
        or actual.get("checkpoint_policy") != "fixed"
        or actual.get("precision") != "fp32"
        or actual.get("checkpoint_epoch") != node["endpoint"]
        or actual.get("epochs") != node["epochs"]
        or actual.get("train_eval_seqs", 0) != train_eval_sequences
    ):
        raise BridgeError(f"{node['id']}: provenance recipe differs from the plan")


def _verify_history(history: object, node: dict[str, Any], train: list[int]) -> dict[str, Any]:
    epochs = int(node["epochs"])
    if not isinstance(history, list) or len(history) != epochs:
        raise BridgeError(f"{node['id']}: history does not cover every frozen epoch")
    per_epoch = int(node["expected_updates"]) // epochs
    train_prefix = sorted(train)[:32]
    train_member_hash = hashlib.sha256(
        json.dumps([["train", group] for group in train_prefix], separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    records: list[dict[str, Any]] = []
    for epoch, row in enumerate(history):
        if (
            not isinstance(row, dict)
            or row.get("epoch") != epoch
            or row.get("split") != f"vfold{VALID_GROUPS}"
            or row.get("selected_rows") != VALID_ROWS
            or not all(
                _finite(row.get(key)) for key in ("weighted_pearson", "global_t0", "global_t1")
            )
        ):
            raise BridgeError(f"{node['id']}: history lacks complete official-valid metrics")
        summary = row.get("fit_summary")
        fit_counts = summary.get("optimizer_updates") if isinstance(summary, dict) else None
        diagnostics = row.get("update_diagnostics")
        epoch_counts = diagnostics.get("epoch") if isinstance(diagnostics, dict) else None
        cumulative_counts = diagnostics.get("cumulative") if isinstance(diagnostics, dict) else None
        if not all(
            isinstance(counts, dict) for counts in (fit_counts, epoch_counts, cumulative_counts)
        ):
            raise BridgeError(f"{node['id']}: history update counters differ from the plan")
        expected_counts = (
            (fit_counts, per_epoch),
            (epoch_counts, per_epoch),
            (cumulative_counts, per_epoch * (epoch + 1)),
        )
        for counters, expected in expected_counts:
            if (
                type(counters.get("optimizer_attempts")) is not int
                or counters["optimizer_attempts"] != expected
                or type(counters.get("optimizer_completed_updates")) is not int
                or counters["optimizer_completed_updates"] != expected
                or type(counters.get("amp_skipped_updates")) is not int
                or counters["amp_skipped_updates"] != 0
            ):
                raise BridgeError(f"{node['id']}: history update counters differ from the plan")
        if node["id"] != ADOPTED_NODE and (
            row.get("train_eval_source") != "train"
            or row.get("train_eval_mask") != "all_after_warmup"
            or row.get("train_eval_sequences") != 32
            or row.get("train_eval_row_groups") != train_prefix
            or row.get("train_eval_members_sha256") != train_member_hash
            or type(row.get("train_eval_selected_rows")) is not int
            or row["train_eval_selected_rows"] <= 0
            or not all(
                _finite(row.get(key))
                for key in ("train_eval_wp", "train_eval_wp_t0", "train_eval_wp_t1")
            )
        ):
            raise BridgeError(f"{node['id']}: history train-eval cohort or mask differs")
        records.append(row)
    latest = records[-1]
    best = max(records, key=lambda item: float(item["weighted_pearson"]))
    metrics: dict[str, Any] = {
        "latest_epoch": epochs,
        "latest_valid_wp": float(latest["weighted_pearson"]),
        "latest_valid_wp_t0": float(latest["global_t0"]),
        "latest_valid_wp_t1": float(latest["global_t1"]),
        "best_observed_valid_wp": float(best["weighted_pearson"]),
        "best_observed_valid_epoch": int(best["epoch"]) + 1,
    }
    if _finite(latest.get("valid_wmse")):
        metrics["latest_valid_wmse"] = float(latest["valid_wmse"])
    if node["id"] != ADOPTED_NODE:
        metrics.update(
            {
                "train_eval_wp": float(latest["train_eval_wp"]),
                "train_eval_wp_t0": float(latest["train_eval_wp_t0"]),
                "train_eval_wp_t1": float(latest["train_eval_wp_t1"]),
                "train_eval_mask": "all_after_warmup",
            }
        )
        if _finite(latest.get("train_eval_wmse")):
            metrics["train_eval_wmse"] = float(latest["train_eval_wmse"])
    return metrics
