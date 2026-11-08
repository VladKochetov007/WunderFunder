from __future__ import annotations

from pathlib import Path
from typing import Any

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, load_campaign_split
from wnn.training.sequence_data import build_data
from wnn.training.sequence_distributed import configure_strict_fp32
from wnn.training.sequence_features import sequence_input_width
from wnn.training.sequence_fit_monitor_setup import (
    prepare_monitor_event_budgets,
    record_monitoring_plan,
    select_train_eval_members,
)
from wnn.training.sequence_fit_utils import make_optimizer, total_updates
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_normalization import (
    build_normalization_identity,
    normalization_identity_sha256,
)
from wnn.training.sequence_optimizer_telemetry import optimizer_contract
from wnn.training.sequence_provenance import run_identity, source_identity

from .common import LaunchError
from .contract import CellPlan


def build_cell_identity(
    cell: CellPlan,
    data_root: Path,
    split_manifest: Path,
    repo_root: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Recreate the fit identity from official data, fit-only moments, and frozen code."""
    config = cell.config
    split = load_campaign_split(split_manifest, data_root)
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA or split.valid_groups:
        raise LaunchError("fixed fits require the official chronological manifest with no fit-valid rows")
    if split.fold != config["expected_fold"]:
        raise LaunchError(f"{cell.cell_id}: config fold differs from the supplied manifest")
    data = build_data(config, data_root, split_manifest)
    if data.provenance["teacher"] is not None or data.provenance["hint_members"] is not None:
        raise LaunchError(f"{cell.cell_id}: teacher or auxiliary hint ancestry is not allowed")
    if any(member[0] != "train" for member in data.fit.members):
        raise LaunchError(f"{cell.cell_id}: fitting members include non-train sequences")
    if data.fit.training_mask != "all_after_warmup" or data.evaluation.training_mask != "is_scored":
        raise LaunchError(f"{cell.cell_id}: train and official-valid row masks are not explicit")
    if data.provenance["normalization"]["scope"] != "all_fit":
        raise LaunchError(f"{cell.cell_id}: normalization is not restricted to all fitting members")
    if data.provenance["evaluation_members"] != [
        ["valid", group] for group in split.evaluation_groups
    ]:
        raise LaunchError(f"{cell.cell_id}: evaluation membership differs from official valid")

    _load_existing_fit_moments(data, data_root)
    updates = total_updates(config, len(data.fit), 1)
    train_eval_members = select_train_eval_members(config.get("monitoring"), data)
    event_budget, _ = prepare_monitor_event_budgets(
        config.get("monitoring"), updates, int(config["epochs"]), 0
    )
    record_monitoring_plan(
        data,
        config["monitoring"],
        train_eval_members,
        str(config["model"].get("input_feature_mode", "raw112")),
        event_budget,
    )

    source = source_identity(repo_root, repo_root / "scripts" / "train_sequence.py")
    if cell.profile["identity"].get("source_sha256") != source["sha256"]:
        raise LaunchError(f"{cell.cell_id}: profile source hash differs from this frozen checkout")
    configure_strict_fp32()
    model = build_sequence_model(config["model"])
    parameters = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    optimizer, _ = make_optimizer(model, config, updates)
    identity = run_identity(config, data.provenance, source, runtime, 1)
    identity["model_parameters"] = parameters
    identity["optimizer"] = optimizer_contract(optimizer, config, updates)
    return identity


def _load_existing_fit_moments(data: Any, data_root: Path) -> None:
    """Require a verified profile-created all-fit cache instead of rescanning source rows."""
    normalization = data.provenance["normalization"]
    mode = data.fit.input_feature_mode
    schema = data.provenance["input_schema"]
    identity = build_normalization_identity(
        normalization["source_fingerprints"],
        data.provenance["split_sha256"],
        list(data.fit.members),
        "all_fit",
        sequence_input_width(mode),
        input_schema=schema,
    )
    key = normalization_identity_sha256(identity)
    cache_dir = data_root / "cache" / "sequence_normalization"
    cache_path = cache_dir / f"{key}.json"
    if cache_dir.is_symlink() or cache_path.is_symlink() or not cache_path.is_file():
        raise LaunchError("profile-created all-fit normalization cache is missing or unsafe")
    data.normalization()



