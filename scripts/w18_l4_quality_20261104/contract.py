from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

PANEL_SCHEMA = "wnn.w18-l4-cohort.v1"
EXPECTED_SEEDS = (1001, 1002)
MAX_OUTPUT_BYTES = 8_000_000_000
MINIMUM_FREE_BYTES = 3_000_000_000
MAX_WAVE_SECONDS = 43_200
COLLECTION_RESERVE_SECONDS = 2_700


class ContractError(ValueError):
    """Raised when a frozen cohort panel or external approval is unsafe."""


@dataclass(frozen=True)
class CohortPanel:
    """Validated panel metadata with its immutable file identity."""

    path: Path
    sha256: str
    value: dict[str, object]


def sha256_file(path: Path) -> str:
    """Hash one file by its exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using stable key order and compact separators."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_object(path: Path) -> dict[str, object]:
    """Read a bounded JSON object while rejecting unsafe paths and malformed data."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_000_000:
        raise ContractError(f"receipt must be a regular file of at most 16 MB: {path}")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON receipt {path}: {error}") from error
    return _object(value, str(path))


def load_panel(path: Path) -> CohortPanel:
    """Load and validate the public four-configuration, two-seed panel schema."""
    resolved = path.expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise ContractError("panel must be a regular, nonsymlink JSON file")
    panel = read_object(resolved)
    validate_panel(panel)
    return CohortPanel(resolved.resolve(strict=True), sha256_file(resolved), panel)


def validate_panel(panel: dict[str, object]) -> None:
    """Reject recipe drift, invalid matched cells, and budget changes."""
    if panel.get("schema") != PANEL_SCHEMA or panel.get("status") != "recipes_frozen":
        raise ContractError("panel schema/status must declare one frozen W18 L4 cohort")
    paths = _object(panel.get("paths"), "paths")
    for name in (
        "source_root",
        "project_root",
        "python",
        "split_manifest",
        "runtime_root",
        "control_root",
        "disk_mount",
    ):
        _absolute_path(paths.get(name), f"paths.{name}")
    identity = _object(panel.get("identity"), "identity")
    for name in (
        "source_binding_sha256",
        "data_binding_sha256",
        "split_sha256",
        "fit_members_sha256",
        "valid_members_sha256",
        "baseline_ancestry_sha256",
    ):
        _sha256(identity.get(name), f"identity.{name}")
    if identity.get("valid_sequence_count") != 1873:
        raise ContractError("primary valid must contain all 1873 official sequences")
    if identity.get("valid_scored_row_count") != 3_528_273:
        raise ContractError("primary valid scoring mask must contain 3,528,273 rows")
    if type(identity.get("fit_sequence_count")) is not int or identity["fit_sequence_count"] < 1:
        raise ContractError("fit sequence count must be a positive integer")

    budget = _object(panel.get("budget"), "budget")
    expected_budgets = {
        "max_scientific_fits": 8,
        "max_wave_seconds": MAX_WAVE_SECONDS,
        "collection_reserve_seconds": COLLECTION_RESERVE_SECONDS,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "aggregate_cpu_threads": 64,
        "aggregate_memory_bytes": 400 * (1 << 30),
    }
    if any(budget.get(key) != value for key, value in expected_budgets.items()):
        raise ContractError("panel budgets differ from the authorized cohort contract")
    max_profiles = budget.get("max_profiles")
    if type(max_profiles) is not int or not 8 <= max_profiles <= 16:
        raise ContractError("profile cap must be between eight and sixteen attempts")

    profile = _object(panel.get("profile_policy"), "profile_policy")
    for name in ("warmup_batches", "timed_batches", "timeout_seconds"):
        if type(profile.get(name)) is not int or profile[name] < 1:
            raise ContractError(f"profile_policy.{name} must be a positive integer")
    if profile["timeout_seconds"] > 3600:
        raise ContractError("one engineering profile may not exceed one hour")

    cells = panel.get("cells")
    if not isinstance(cells, list) or len(cells) != 8:
        raise ContractError("the initial allocation must contain exactly eight fit cells")
    cell_ids: set[str] = set()
    gpu_uuids: set[str] = set()
    run_names: set[str] = set()
    configurations: dict[str, list[dict[str, object]]] = {}
    hypothesis_by_configuration: dict[str, str] = {}
    thread_total = 0
    roles: dict[str, str] = {}
    for index, item in enumerate(cells):
        cell = _object(item, f"cells[{index}]")
        cell_id = _text(cell.get("cell_id"), f"cells[{index}].cell_id")
        config_id = _text(cell.get("configuration_id"), f"cells[{index}].configuration_id")
        role = cell.get("role")
        if role not in ("control", "mechanism"):
            raise ContractError(f"{cell_id}: role must be control or mechanism")
        if config_id in roles and roles[config_id] != role:
            raise ContractError(f"{config_id}: role differs between its matched seeds")
        roles[config_id] = cast(str, role)
        if role == "mechanism" and not cell.get("hypothesis_id"):
            raise ContractError(f"{cell_id}: mechanism cells need a committed hypothesis id")
        if role == "mechanism":
            hypothesis_id = _text(cell.get("hypothesis_id"), f"{cell_id}.hypothesis_id")
            previous_hypothesis = hypothesis_by_configuration.setdefault(config_id, hypothesis_id)
            if previous_hypothesis != hypothesis_id:
                raise ContractError(f"{config_id}: matched seeds name different hypotheses")
        if role == "control" and cell.get("hypothesis_id") not in (None, ""):
            raise ContractError(f"{cell_id}: control must not claim a candidate hypothesis")
        if cell_id in cell_ids:
            raise ContractError(f"duplicate cell id: {cell_id}")
        cell_ids.add(cell_id)
        seed = cell.get("seed")
        if type(seed) is not int or seed not in EXPECTED_SEEDS:
            raise ContractError(f"{cell_id}: seed must be one of {EXPECTED_SEEDS}")
        gpu = _text(cell.get("gpu_uuid"), f"{cell_id}.gpu_uuid")
        if not re.fullmatch(r"GPU-[0-9A-Fa-f-]+", gpu) or gpu in gpu_uuids:
            raise ContractError(f"{cell_id}: every fit needs one unique physical GPU UUID")
        gpu_uuids.add(gpu)
        threads = cell.get("cpu_threads")
        if type(threads) is not int or not 1 <= threads <= 8:
            raise ContractError(f"{cell_id}: CPU thread allocation must be in 1..8")
        thread_total += threads
        affinity = cell.get("cpu_affinity")
        if affinity is not None:
            if (
                not isinstance(affinity, list)
                or len(affinity) != threads
                or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
                or len(set(affinity)) != len(affinity)
            ):
                raise ContractError(f"{cell_id}: cpu_affinity must contain one unique CPU per thread")
        run_name = _text(cell.get("run_name"), f"{cell_id}.run_name")
        if run_name in run_names:
            raise ContractError(f"duplicate run name: {run_name}")
        run_names.add(run_name)
        args = _string_list(cell.get("train_args"), f"{cell_id}.train_args")
        timeout = cell.get("fit_timeout_seconds")
        if type(timeout) is not int or not 1 <= timeout <= 10_800:
            raise ContractError(f"{cell_id}: fit timeout must be between 1 second and 3 hours")
        _validate_fit_args(args, cell, paths, identity)
        configurations.setdefault(config_id, []).append(cell)
    if thread_total > 64:
        raise ContractError("simultaneous fit CPU threads exceed the aggregate 64-thread cap")
    if len(configurations) != 4 or list(roles.values()).count("control") != 1:
        raise ContractError("the panel needs one control configuration and three mechanisms")
    if len(set(hypothesis_by_configuration.values())) != 3:
        raise ContractError("the three mechanism configurations need unique hypothesis ids")
    configuration_recipes: set[str] = set()
    for config_id, matched in configurations.items():
        if sorted(cast(int, cell["seed"]) for cell in matched) != list(EXPECTED_SEEDS):
            raise ContractError(f"{config_id}: each configuration needs both matched seeds")
        normalized = {_recipe_without_seed_and_name(cell["train_args"]) for cell in matched}
        if len(normalized) != 1:
            raise ContractError(f"{config_id}: matched-seed recipes differ beyond seed/name")
        configuration_recipes.update(normalized)
    if len(configuration_recipes) != 4:
        raise ContractError("the control and three mechanism recipes must be distinct")
    affinity_groups = [
        set(cast(list[int], cell["cpu_affinity"]))
        for cell in cast(list[dict[str, object]], cells)
        if cell.get("cpu_affinity") is not None
    ]
    if any(left & right for index, left in enumerate(affinity_groups) for right in affinity_groups[index + 1 :]):
        raise ContractError("per-cell CPU affinity sets must not overlap")

    references = panel.get("reference_predictions")
    if not isinstance(references, list) or not references:
        raise ContractError("fixed clean-reference prediction members are required")
    weights: list[float] = []
    for index, item in enumerate(references):
        reference = _object(item, f"reference_predictions[{index}]")
        for name in ("prediction_path", "sidecar_path"):
            _absolute_path(reference.get(name), f"reference_predictions[{index}].{name}")
        _sha256(reference.get("checkpoint_sha256"), "reference checkpoint SHA-256")
        weight = reference.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight <= 1:
            raise ContractError("reference mixture weights must be finite and positive")
        weights.append(float(weight))
    if abs(sum(weights) - 1.0) > 1e-12 or panel.get("fixed_blend_weight") != 0.1:
        raise ContractError("reference weights must sum to one and the candidate weight stay at 0.1")
    _absolute_path(panel.get("admission_receipt"), "admission_receipt")


def _validate_fit_args(
    args: list[str], cell: dict[str, object], paths: dict[str, object], identity: dict[str, object]
) -> None:
    """Require the legacy W18 CLI to fit train and report the complete official valid."""
    if "--full-valid" not in args or "--no-eval-during-fit" in args:
        raise ContractError(f"{cell['cell_id']}: full-valid reporting must remain enabled")
    if "--profile-steps" in args or "--profile-warmup" in args:
        raise ContractError(f"{cell['cell_id']}: frozen fit args may not contain profile flags")
    expected = {
        "--name": cell.get("run_name"),
        "--seed": str(cell.get("seed")),
        "--train-source": "train",
        "--train-seqs": str(identity.get("fit_sequence_count")),
        "--eval-source": "dev",
        "--split-manifest": paths.get("split_manifest"),
        "--precision": "fp32",
        "--epochs": "2",
        "--checkpoint-policy": "fixed",
    }
    for option, wanted in expected.items():
        if _option_values(args, option) != [str(wanted)]:
            raise ContractError(f"{cell['cell_id']}: {option} must equal {wanted!r}")
    if "--save-training-state" not in args:
        raise ContractError(f"{cell['cell_id']}: resume/provenance state must be retained")
    for option in ("--qat", "--activation-backend"):
        values = _option_values(args, option)
        if values not in ([], ["none"]):
            raise ContractError(f"{cell['cell_id']}: quantized training is forbidden ({option})")
    if _option_values(args, "--train-eval-seqs") != ["32"]:
        raise ContractError(f"{cell['cell_id']}: fixed 32-sequence train-eval diagnostic is required")
    checkpoint_epoch = _option_values(args, "--checkpoint-epoch")
    if checkpoint_epoch not in ([], ["1"]):
        raise ContractError(f"{cell['cell_id']}: the configured two-epoch endpoint is epoch 1")


def _recipe_without_seed_and_name(args: list[str]) -> str:
    """Return a stable recipe key after replacing only paired seed/run identity."""
    return canonical_sha256(_replace_options(args, {"--seed": "SEED", "--name": "RUN_NAME"}))


def _replace_options(args: list[str], replacements: dict[str, str]) -> list[str]:
    """Replace values for distinct `--flag VALUE` options and preserve all other argv."""
    result = args.copy()
    for flag, replacement in replacements.items():
        positions = [index for index, token in enumerate(result) if token == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(result):
            raise ContractError(f"recipe must contain exactly one {flag} VALUE pair")
        result[positions[0] + 1] = replacement
    return result


def _option_values(args: list[str], option: str) -> list[str]:
    """Read values for one `--option VALUE` or `--option=VALUE` occurrence."""
    values: list[str] = []
    for index, token in enumerate(args):
        if token == option:
            if index + 1 >= len(args):
                return []
            values.append(args[index + 1])
        elif token.startswith(option + "="):
            values.append(token.partition("=")[2])
    return values


def _object(value: object, label: str) -> dict[str, object]:
    """Require a JSON object and return its string-keyed mapping."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _text(value: object, label: str) -> str:
    """Require one nonempty string value."""
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a nonempty string")
    return value


def _string_list(value: object, label: str) -> list[str]:
    """Require a nonempty vector of nonempty command tokens."""
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ContractError(f"{label} must be a nonempty list of command strings")
    return cast(list[str], value)


def _absolute_path(value: object, label: str) -> str:
    """Require an absolute POSIX receiving-host path in the public manifest."""
    path = _text(value, label)
    if not path.startswith("/"):
        raise ContractError(f"{label} must be an absolute receiving-host path")
    return path


def _sha256(value: object, label: str) -> str:
    """Require one lowercase SHA-256 hex digest."""
    digest = _text(value, label)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return digest
