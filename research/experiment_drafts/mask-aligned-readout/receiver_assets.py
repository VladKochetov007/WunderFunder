"""Path, artifact, and campaign-split validation for the PR36 receiver."""

from __future__ import annotations

# Any is limited to canonical JSON and split payloads validated as they are loaded.
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
REPO_ROOT = PACKET.parents[2]
RECIPE_PATH = PACKET / "recipe.json"


@dataclass(frozen=True)
class Cell:
    """One fold, seed, clean parent, and its paired child run names."""

    fold: str
    fold_index: int
    seed: int
    parent_run: str
    epochs: int
    checkpoint_epoch: int

    @property
    def checkpoint_name(self) -> str:
        """Return the recipe-selected child checkpoint filename."""
        return f"model_ep{self.checkpoint_epoch}.pt"

    def run_name(self, arm: str) -> str:
        """Return the stable artifact name for one member of the pair."""
        if arm not in ("all_valid", "is_scored_weighted"):
            raise ValueError(f"unknown arm {arm!r}")
        return (
            f"mask_aligned_readout_r2_{self.fold.lower()}_s{self.seed}_"
            f"e{self.epochs}_ep{self.checkpoint_epoch}_{arm}"
        )


@dataclass(frozen=True)
class ReceiverAssets:
    """Validated paths and canonical campaign splits on the receiving host."""

    root: Path
    manifest_path: Path
    split_paths: dict[str, Path]
    parent_checkpoints: dict[str, Path]
    parent_provenance: dict[str, Path]
    splits: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    """Load an object-valued UTF-8 JSON file, rejecting other top-level shapes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def file_sha256(path: Path) -> str:
    """Hash a file in bounded-memory blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recipe_cells(recipe: dict[str, Any]) -> tuple[Cell, ...]:
    """Read R2's pinned cells and its shared fixed epoch endpoint."""
    amendment = recipe.get("amendment")
    if (
        recipe.get("id") != "E-MASK-ALIGNED-READOUT-20261022-R2"
        or recipe.get("recipe_revision") != 2
        or not isinstance(amendment, dict)
        or amendment.get("amends") != "E-MASK-ALIGNED-READOUT-20261022-R1"
    ):
        raise ValueError("recipe is not the explicit pre-outcome R2 amendment")
    frozen_recipe = PACKET / "recipe-r1.json"
    frozen_recipe_sha = (
        hashlib.sha256(frozen_recipe.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if frozen_recipe.is_file()
        else None
    )
    if amendment.get(
        "prior_recipe_path"
    ) != frozen_recipe.name or frozen_recipe_sha != amendment.get("prior_recipe_sha256"):
        raise ValueError("canonical-LF frozen R1 recipe does not match the R2 amendment binding")
    raw_cells = recipe.get("cells")
    budget = recipe.get("budget", {})
    if not isinstance(raw_cells, list) or len(raw_cells) != 2:
        raise ValueError("recipe must contain exactly F0 and F2")
    if (
        budget.get("new_parent_fits") != 0
        or budget.get("new_head_fits") != 4
        or budget.get("total_new_fits") != 4
        or budget.get("full_epochs_per_head_fit") != 3
        or budget.get("total_new_head_fit_epochs") != 12
        or budget.get("receiver_node_wall_limit_hours") != 4
    ):
        raise ValueError("recipe must preserve zero new parents and four child fits")
    fit = recipe.get("fit", {})
    epochs = fit.get("epochs")
    checkpoint_epoch = fit.get("checkpoint_epoch")
    if (
        type(epochs) is not int
        or type(checkpoint_epoch) is not int
        or epochs != 3
        or checkpoint_epoch != 2
    ):
        raise ValueError("R2 must declare three full epochs and fixed checkpoint epoch two")
    cells = tuple(
        Cell(
            fold=str(value["fold"]),
            fold_index=int(value["fold_index"]),
            seed=int(value["seed"]),
            parent_run=str(value["parent_candidate"]),
            epochs=epochs,
            checkpoint_epoch=checkpoint_epoch,
        )
        for value in raw_cells
    )
    if [(cell.fold, cell.fold_index, cell.seed) for cell in cells] != [
        ("F0", 0, 1001),
        ("F2", 2, 1002),
    ]:
        raise ValueError("recipe cell identity differs from the admitted F0/F2 panel")
    return cells


def select_cells(cells: tuple[Cell, ...], fold: str | None) -> tuple[Cell, ...]:
    """Select a complete cell pair for one isolated receiver worker."""
    if fold is None:
        return cells
    selected = tuple(cell for cell in cells if cell.fold == fold)
    if not selected:
        raise ValueError(f"fold {fold!r} is not present in the receiver recipe")
    return selected


def resolve_path(value: Any, *, label: str, base: Path) -> Path:
    """Resolve an absolute or receiver-manifest-relative path without guessing."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"receiver asset {label} must be a nonempty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def validate_parent_record(
    cell: Cell, recipe_cell: dict[str, Any], checkpoint_path: Path, provenance_path: Path
) -> None:
    """Verify the actual receiver artifacts against committed provenance pins."""
    for path in (checkpoint_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f"required parent artifact is missing: {path}")
    hashes = recipe_cell.get("artifacts", {})
    expected_checkpoint = hashes.get("checkpoint_sha256")
    expected_provenance = hashes.get("provenance_sha256")
    if not expected_checkpoint or not expected_provenance:
        raise ValueError(f"{cell.fold} parent artifact hashes are not pinned in recipe.json")
    if file_sha256(checkpoint_path) != expected_checkpoint:
        raise ValueError(f"{cell.fold} parent checkpoint differs from the pinned epoch-3 bytes")
    if file_sha256(provenance_path) != expected_provenance:
        raise ValueError(f"{cell.fold} parent provenance differs from the pinned record")
    provenance = load_json(provenance_path)
    args = provenance.get("args", {})
    expected = {
        "name": cell.parent_run,
        "train_source": "train+valid",
        "valid_fold": cell.fold_index,
        "split_manifest_sha256": recipe_cell["split_manifest_sha256"],
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "include_heldout_valid": False,
        "init": "",
        "precision": "fp16",
    }
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{cell.fold} parent provenance does not match the pinned clean parent")
    split_sha = recipe_cell.get("split_manifest_sha256")
    if provenance.get("split_manifest_sha256") != split_sha:
        raise ValueError(f"{cell.fold} parent provenance has a different split SHA")
    expected_checkpoint_name = "model_ep3.pt"
    if provenance.get("checkpoint_sha256", {}).get(expected_checkpoint_name) != expected_checkpoint:
        raise ValueError(f"{cell.fold} provenance does not bind its epoch-3 checkpoint")
    if provenance.get("selected_epoch") != 3:
        raise ValueError(f"{cell.fold} parent selected epoch is not 3")
    historical_source_sha = provenance.get("run_identity_source_sha256")
    if (
        not isinstance(historical_source_sha, str)
        or len(historical_source_sha) != 64
        or any(character not in "0123456789abcdef" for character in historical_source_sha.lower())
    ):
        raise ValueError(f"{cell.fold} parent lacks its recorded historical run-identity hash")
    if provenance.get("propensity_checkpoint_sha256") != hashes.get("propensity_checkpoint_sha256"):
        raise ValueError(f"{cell.fold} parent propensity ancestry differs from its pinned record")
    fit_groups = provenance.get("fit_row_groups", {})
    if not isinstance(fit_groups, dict) or not fit_groups.get("valid"):
        raise ValueError(f"{cell.fold} parent lacks recorded valid-side fit membership")
    if cell.fold_index == 0:
        cache = provenance.get("distillation_cache", {})
        expected_cache = hashes.get("distillation_cache_metadata_sha256")
        if (
            cache.get("metadata_sha256") != expected_cache
            or len(cache.get("source_coefficients", [])) != 8
        ):
            raise ValueError("F0 parent distillation ancestry differs from its pinned record")
    elif provenance.get("distillation_cache"):
        raise ValueError("F2 parent unexpectedly records a distillation cache")


def load_receiver_assets(path: Path) -> ReceiverAssets:
    """Validate receiver paths, hashes, fold manifests, and all parent records."""
    recipe = load_json(RECIPE_PATH)
    cells = recipe_cells(recipe)
    declared = load_json(path.resolve())
    if declared.get("schema_version") != 1:
        raise ValueError("receiver asset manifest schema_version must be 1")
    root = resolve_path(declared.get("project_root"), label="project_root", base=path.parent)
    env_root = os.environ.get("WNN_PROJECT_ROOT")
    if env_root and Path(env_root).expanduser().resolve() != root:
        raise ValueError("WNN_PROJECT_ROOT and receiver asset project_root disagree")
    if not root.is_dir():
        raise FileNotFoundError(f"receiver project root does not exist: {root}")
    split_inputs = declared.get("split_manifests", {})
    parent_inputs = declared.get("parent_artifacts", {})
    if not isinstance(split_inputs, dict) or not isinstance(parent_inputs, dict):
        raise ValueError("split_manifests and parent_artifacts must be JSON objects")
    split_paths: dict[str, Path] = {}
    parent_checkpoints: dict[str, Path] = {}
    parent_provenance: dict[str, Path] = {}
    split_objects: dict[str, Any] = {}
    for cell, recipe_cell in zip(cells, recipe["cells"], strict=True):
        split_path = resolve_path(
            split_inputs.get(cell.fold), label=f"split_manifests.{cell.fold}", base=path.parent
        )
        if (
            not split_path.is_file()
            or file_sha256(split_path) != recipe_cell["split_manifest_sha256"]
        ):
            raise ValueError(
                f"{cell.fold} split manifest is missing or differs from its pinned SHA"
            )
        parent = parent_inputs.get(cell.fold)
        if not isinstance(parent, dict):
            raise ValueError(f"receiver parent_artifacts.{cell.fold} must be an object")
        checkpoint = resolve_path(
            parent.get("checkpoint"),
            label=f"parent_artifacts.{cell.fold}.checkpoint",
            base=path.parent,
        )
        provenance = resolve_path(
            parent.get("provenance"),
            label=f"parent_artifacts.{cell.fold}.provenance",
            base=path.parent,
        )
        canonical_run = root / "artifacts" / "runs" / cell.parent_run
        if checkpoint != (canonical_run / "model_ep3.pt").resolve():
            raise ValueError(f"{cell.fold} initializer must use its canonical model_ep3.pt path")
        if provenance != (canonical_run / "provenance.json").resolve():
            raise ValueError(f"{cell.fold} provenance must use its canonical run path")
        validate_parent_record(cell, recipe_cell, checkpoint, provenance)
        split_paths[cell.fold] = split_path
        parent_checkpoints[cell.fold] = checkpoint
        parent_provenance[cell.fold] = provenance
    source_path = str(REPO_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from wnn.splits import load_campaign_split

    for cell, recipe_cell in zip(cells, recipe["cells"], strict=True):
        split = load_campaign_split(split_paths[cell.fold], root)
        if split.sha256 != recipe_cell["split_manifest_sha256"] or split.fold != cell.fold_index:
            raise ValueError(f"{cell.fold} canonical split validation returned a different fold")
        if not split.valid_groups or not split.evaluation_groups:
            raise ValueError(f"{cell.fold} split must have fitting and held-out valid groups")
        parent_record = load_json(parent_provenance[cell.fold])
        fit = parent_record["fit_row_groups"]
        if set(fit["train"]) != set(split.train_groups) or set(fit["valid"]) != set(
            split.valid_groups
        ):
            raise ValueError(f"{cell.fold} parent membership differs from the canonical split")
        if set(parent_record["evaluation_row_groups"]) != set(split.evaluation_groups):
            raise ValueError(f"{cell.fold} parent exposed a different held-out component set")
        split.assert_fit_groups(fit["train"], fit["valid"])
        split_objects[cell.fold] = split
    os.environ["WNN_PROJECT_ROOT"] = str(root)
    return ReceiverAssets(
        root,
        path.resolve(),
        split_paths,
        parent_checkpoints,
        parent_provenance,
        split_objects,
    )


def output_root(root: Path) -> Path:
    """Return the ignored evidence folder used by future receiver runs."""
    experiment_id = load_json(RECIPE_PATH).get("id")
    if not isinstance(experiment_id, str) or not experiment_id.startswith("E-"):
        raise ValueError("recipe experiment id is invalid")
    if "/" in experiment_id or "\\" in experiment_id:
        raise ValueError("recipe experiment id must be a path component")
    return root / "artifacts" / "experiment_evidence" / experiment_id


def child_checkpoint(root: Path, cell: Cell, arm: str) -> Path:
    """Return the recipe-selected fixed checkpoint path for a child fit."""
    return root / "artifacts" / "runs" / cell.run_name(arm) / cell.checkpoint_name


def prediction_name(cell: Cell, role: str) -> str:
    """Return a collision-resistant cache output name for one cell prediction."""
    names = {
        "parent": f"mask_readout_r2_e{cell.epochs}_ep{cell.checkpoint_epoch}_{cell.fold.lower()}_parent",
        "control": f"mask_readout_r2_e{cell.epochs}_ep{cell.checkpoint_epoch}_{cell.fold.lower()}_control",
        "treatment": f"mask_readout_r2_e{cell.epochs}_ep{cell.checkpoint_epoch}_{cell.fold.lower()}_treatment",
    }
    try:
        return names[role]
    except KeyError as exc:
        raise ValueError(f"unknown prediction role {role!r}") from exc


def run_preflight(assets_path: Path) -> ReceiverAssets:
    """Validate all actual campaign files before fitting or prediction export."""
    assets = load_receiver_assets(assets_path)
    required = (
        "datasets/valid.parquet",
        "cache/valid_targets.npz",
        "cache/chains.npz",
        "cache/chain_folds.npz",
    )
    missing = [relative for relative in required if not (assets.root / relative).is_file()]
    if missing:
        raise FileNotFoundError("receiver data assets missing: " + ", ".join(missing))
    return assets
