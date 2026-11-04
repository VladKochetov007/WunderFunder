from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wnn.training.sequence_config import load_config

from .common import LaunchError, parse_cpu_list, read_json, sha256_file

EARLY_STOP = {"min_epoch": 5, "patience": 4, "min_delta": 0.0005}
PANEL_SCHEMA = "tabdata.fixed-eight-launch.v1"
CELL_IDS = (
    "g0-plain-gru256x2",
    "g1-galam-lstm64",
    "g2-galam-lstm128",
    "g3-galam-depth2",
    "g4-galam-gru72",
    "g5-galam-history256",
    "g6-galam-tbptt256",
    "g7-galam-muon",
)


@dataclass(frozen=True)
class CellPlan:
    """One immutable panel cell with its validated recipe and profile receipt."""

    cell_id: str
    gpu: int
    cpus: tuple[int, ...]
    numa_node: int
    config_path: Path
    config_file_sha256: str
    config: dict[str, Any]
    profile_path: Path
    profile_sha256: str
    profile: dict[str, Any]


def parse_profile_arguments(values: list[str]) -> dict[str, Path]:
    """Parse exactly one ``CELL=PATH`` profile argument for each fixed panel cell."""
    profiles: dict[str, Path] = {}
    for value in values:
        cell, separator, path = value.partition("=")
        if not separator or not cell or not path or cell in profiles:
            raise LaunchError("profiles must be unique CELL=PATH arguments")
        profiles[cell] = Path(path).expanduser().resolve()
    expected = {f"g{index}-" for index in range(8)}
    if len(profiles) != 8 or not all(any(cell.startswith(prefix) for prefix in expected) for cell in profiles):
        raise LaunchError("supply exactly eight profile receipts, one per frozen panel cell")
    return profiles


def load_panel_cells(panel_path: Path, profiles: dict[str, Path]) -> tuple[dict[str, Any], list[CellPlan]]:
    """Validate the approved panel, its eight config hashes, and profile receipts."""
    panel_path = panel_path.expanduser()
    if panel_path.is_symlink():
        raise LaunchError("PANEL cannot be a symlink")
    panel_path = panel_path.resolve()
    panel = read_json(panel_path)
    if panel.get("schema") != PANEL_SCHEMA:
        raise LaunchError("unsupported fixed-eight panel schema")
    if panel.get("status") != "approved_for_launch":
        raise LaunchError("PANEL status must be approved_for_launch")
    if panel.get("batch_status") != "profiled_and_frozen":
        raise LaunchError("PANEL batch_status must be profiled_and_frozen")
    if (
        panel.get("scientific_fits") != 8
        or panel.get("teacher_fits") != 0
        or panel.get("epochs_max") != 20
        or panel.get("precision") != "fp32"
        or panel.get("seed") != 1001
    ):
        raise LaunchError("PANEL does not describe the fixed eight-fit twenty-epoch recipe")
    if (
        panel.get("train_source") != "official earlier train only"
        or panel.get("validation")
        != "complete official later valid, original is_scored mask and pooled official WP"
    ):
        raise LaunchError("PANEL must preserve the official chronological train/valid protocol")
    raw_cells = panel.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 8:
        raise LaunchError("PANEL must contain exactly eight cells")
    cell_ids = [item.get("id") for item in raw_cells if isinstance(item, dict)]
    if len(cell_ids) != 8 or len(set(cell_ids)) != 8:
        raise LaunchError("PANEL cell identifiers must be unique")
    if tuple(cell_ids) != CELL_IDS:
        raise LaunchError("PANEL cell order or allocation slots differ from the frozen eight-fit set")
    if set(profiles) != set(cell_ids):
        raise LaunchError("profile receipts must match the eight PANEL cell identifiers exactly")

    plans: list[CellPlan] = []
    gpu_ids: set[int] = set()
    used_cpus: set[int] = set()
    galam_batches: set[int] = set()
    packet_dir = panel_path.parent
    for index, item in enumerate(raw_cells):
        if not isinstance(item, dict):
            raise LaunchError("PANEL cells must be JSON objects")
        cell_id = str(item["id"])
        if item.get("gpu") != index or item.get("scientific_fits") != 1:
            raise LaunchError("PANEL must map one fit to each GPU index 0 through 7")
        gpu_ids.add(int(item["gpu"]))
        cpu_list = item.get("cpu_affinity")
        numa = item.get("numa_node")
        if not isinstance(cpu_list, str) or type(numa) is not int or numa not in (0, 1):
            raise LaunchError(f"{cell_id}: missing CPU affinity or NUMA node")
        cpus = parse_cpu_list(cpu_list)
        if len(cpus) != 8 or used_cpus.intersection(cpus):
            raise LaunchError("each fit needs eight distinct, nonoverlapping panel CPUs")
        used_cpus.update(cpus)

        config_rel = item.get("config")
        expected_rel = f"configs/{cell_id}.json"
        if config_rel != expected_rel:
            raise LaunchError(f"{cell_id}: config path must be exactly {expected_rel}")
        config_candidate = packet_dir / expected_rel
        if config_candidate.is_symlink():
            raise LaunchError(f"{cell_id}: config cannot be a symlink")
        config_path = config_candidate.resolve()
        if config_path.parent != (packet_dir / "configs").resolve():
            raise LaunchError(f"{cell_id}: config path escapes the launch packet")
        config_hash = sha256_file(config_path)
        if config_hash != item.get("config_file_sha256"):
            raise LaunchError(f"{cell_id}: config file hash differs from PANEL")
        config = load_config(config_path)
        _validate_recipe(cell_id, config)
        if config["model"]["architecture"] == "galam":
            galam_batches.add(int(config["batch_size"]))

        profile_path = profiles[cell_id]
        profile = read_json(profile_path)
        if (
            profile.get("receipt_schema") != "wnn.sequence-profile.v1"
            or profile.get("status") != "profile_complete"
            or profile.get("approved") is not True
        ):
            raise LaunchError(f"{cell_id}: profile must be complete and explicitly approved")
        identity = profile.get("identity")
        if not isinstance(identity, dict):
            raise LaunchError(f"{cell_id}: profile receipt has no run identity")
        from wnn.training.sequence_provenance import canonical_sha256

        if identity.get("config_sha256") != canonical_sha256(config):
            raise LaunchError(f"{cell_id}: profile config identity differs from the pinned recipe")
        if not _sha256(identity.get("source_sha256")):
            raise LaunchError(f"{cell_id}: profile source identity is not a SHA-256 digest")
        plans.append(
            CellPlan(
                cell_id=cell_id,
                gpu=int(item["gpu"]),
                cpus=cpus,
                numa_node=numa,
                config_path=config_path,
                config_file_sha256=config_hash,
                config=config,
                profile_path=profile_path,
                profile_sha256=sha256_file(profile_path),
                profile=profile,
            )
        )
    if gpu_ids != set(range(8)) or len(used_cpus) != 64:
        raise LaunchError("PANEL must reserve eight unique GPUs and sixty-four unique CPUs")
    if len(galam_batches) != 1:
        raise LaunchError("all seven GALAM cells must use the same profile-frozen batch size")
    return panel, plans


def _validate_recipe(cell_id: str, config: dict[str, Any]) -> None:
    """Reject recipe drift from the authorized fresh-label, official-valid panel."""
    if (
        config["epochs"] != 20
        or config["seed"] != 1001
        or config["precision_mode"] != "fp32"
        or config["schedule"] != "warmup_cosine_v2"
        or config["training_mask"] != "all_after_warmup"
        or config["normalization_scope"] != "all_fit"
        or config["checkpoint_retention"] != "all_epochs"
        or config["required_world_size"] != 1
        or config["kd_weight"] != 0
        or config["hint_weight"] != 0
        or config["hint_mode"] != "aligned"
        or config["model"]["hint_dim"] != 0
    ):
        raise LaunchError(f"{cell_id}: recipe deviates from frozen twenty-epoch fresh-label policy")
    if config["adamw_beta1"] != 0.9 or config["adamw_beta2"] != 0.999:
        raise LaunchError(f"{cell_id}: AdamW beta settings differ from the frozen contract")
    model = config["model"]
    if model["dropout"] != 0 or config.get("dropout_policy") is not None:
        raise LaunchError(f"{cell_id}: unapproved dropout is present")
    expected_architecture = "gru" if cell_id == "g0-plain-gru256x2" else "galam"
    if model["architecture"] != expected_architecture:
        raise LaunchError(f"{cell_id}: architecture differs from its allocation slot")
    if cell_id == "g0-plain-gru256x2":
        if (
            model.get("input_feature_mode") != "event211"
            or (model["hidden"], model["layers"]) != (256, 2)
            or config["optimizer"] != "muon"
        ):
            raise LaunchError("g0 must be the fresh event211 GRU256x2 scoped-Muon baseline")
    else:
        if model.get("input_feature_mode", "raw112") != "raw112":
            raise LaunchError(f"{cell_id}: GALAM must use the raw112 feature schema")
        expected_optimizer = "muon" if cell_id == "g7-galam-muon" else "adamw"
        if config["optimizer"] != expected_optimizer:
            raise LaunchError(f"{cell_id}: optimizer differs from its frozen GALAM slot")
        galam = model.get("galam")
        if not isinstance(galam, dict) or galam.get("variant") not in (
            "schematic_lstm_full",
            "schematic_gru_full",
        ):
            raise LaunchError(f"{cell_id}: expected one of the six declared GALAM variants")
    monitoring = config.get("monitoring")
    if not isinstance(monitoring, dict):
        raise LaunchError(f"{cell_id}: extended monitoring policy is required")
    if (
        monitoring.get("train_eval_mask") != "all_after_warmup"
        or monitoring.get("valid_mask") is not None
        or monitoring.get("eval_batch_size", 0) < 8
        or monitoring.get("early_stopping") != EARLY_STOP
    ):
        raise LaunchError(f"{cell_id}: monitor mask, batch size, or early-stop policy differs")
    if config.get("expected_fold") is None:
        raise LaunchError(f"{cell_id}: official chronological split fold must be explicit")


def _sha256(value: object) -> bool:
    """Return whether a value is a lowercase SHA-256 string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
