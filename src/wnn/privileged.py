from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from wnn.splits import CampaignSplit, file_sha256

PRIVILEGED_MODES = ("none", "current", "future")


def privileged_inputs(raw: torch.Tensor, mode: str, horizon: int) -> torch.Tensor:
    """Append teacher-only raw covariates at origin t or t+h; never move labels."""
    if mode not in PRIVILEGED_MODES or horizon < 1:
        raise ValueError("unknown teacher input mode or nonpositive horizon")
    if mode == "none":
        return raw[..., :0]
    if mode == "current":
        return raw
    index = (torch.arange(raw.shape[1], device=raw.device) + horizon).clamp_max(raw.shape[1] - 1)
    return raw.index_select(1, index)


def inner_manifest(outer: CampaignSplit, fold: int, folds: int, seed: int) -> dict:
    """Partition only the outer fit pool; every connected component stays intact."""
    if folds < 2 or not 0 <= fold < folds or seed < 0:
        raise ValueError("inner crossfit needs >=2 folds, valid fold index and nonnegative seed")
    component_of = {(split, group): component for split, group, component in outer.components}

    def assignment(split: str, group: int) -> int:
        value = f"origin-teacher:{seed}:{component_of[(split, group)]}"
        return int(hashlib.sha256(value.encode()).hexdigest(), 16) % folds

    fit, predict = {}, {}
    for split, groups in (("train", outer.train_groups), ("valid", outer.valid_groups)):
        fit[split] = sorted(group for group in groups if assignment(split, group) != fold)
        predict[split] = sorted(group for group in groups if assignment(split, group) == fold)
    if not fit["train"] or not any(predict.values()):
        raise ValueError("inner fold lacks training or prediction components")
    return {
        "format": "wnn.inner-component-crossfit",
        "version": 1,
        "outer_split_sha256": outer.sha256,
        "fold": fold,
        "folds": folds,
        "seed": seed,
        "fit": fit,
        "predict": predict,
    }


def validate_inner_manifest(value: dict, outer: CampaignSplit) -> dict:
    """Reject edited memberships and assignments, including any outer evaluation row."""
    expected = inner_manifest(outer, value["fold"], value["folds"], value["seed"])
    if value != expected:
        raise ValueError("inner crossfit manifest disagrees with component-disjoint outer fit pool")
    return value


def load_inner_manifest(path: Path, outer: CampaignSplit) -> dict:
    """Load a frozen inner split for fresh teacher fitting."""
    return validate_inner_manifest(json.loads(path.read_text()), outer)


def validate_teacher_checkpoint(checkpoint: dict, outer: CampaignSplit) -> dict:
    """Require fresh fixed-epoch fitting and exactly the declared inner membership."""
    args = checkpoint["args"]
    provenance = checkpoint.get("provenance", {})
    inner = validate_inner_manifest(provenance["inner_crossfit"], outer)
    if (
        args.get("split_manifest_sha256") != outer.sha256
        or args.get("init")
        or args.get("distill")
        or args.get("propensity")
        or args.get("prop_input")
        or args.get("privileged_mode") not in ("current", "future")
        or args.get("checkpoint_policy") != "fixed"
        or not args.get("no_eval_during_fit")
        or checkpoint.get("epoch") != args.get("checkpoint_epoch")
        or provenance.get("fit_row_groups") is None
    ):
        raise ValueError("teacher lacks fresh, fixed-epoch inner crossfit provenance")
    for split in ("train", "valid"):
        if sorted(provenance["fit_row_groups"][split]) != inner["fit"][split]:
            raise ValueError("teacher fitted membership differs from its inner manifest")
    return inner


def validate_crossfit_cache(sidecar: dict, outer: CampaignSplit, root: Path, split: str) -> None:
    """Verify each consumed cache row was predicted by an excluding teacher checkpoint."""
    if sidecar.get("crossfit_format") != "wnn.origin-teacher-crossfit.v1":
        raise ValueError("unsupported crossfit teacher cache")
    if sidecar.get("row_map_sha256") != file_sha256(root / "cache" / f"{split}_targets.npz"):
        raise ValueError("crossfit target row map bytes changed")
    covered: set[int] = set()
    specification = None
    for entry in sidecar["teachers"]:
        relative = Path(entry["checkpoint"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("teacher checkpoint must be a relative path inside the repository")
        checkpoint_path = root / relative
        if file_sha256(checkpoint_path) != entry["checkpoint_sha256"]:
            raise ValueError("crossfit teacher checkpoint bytes changed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        inner = validate_teacher_checkpoint(checkpoint, outer)
        args = checkpoint["args"]
        current = (
            inner["folds"],
            inner["seed"],
            args["privileged_mode"],
            args["privileged_horizon"],
        )
        if specification is not None and current != specification:
            raise ValueError("crossfit teacher arms or component assignments differ")
        specification = current
        groups = set(inner["predict"][split])
        if sorted(groups) != entry["predicted_groups"][split] or covered & groups:
            raise ValueError("crossfit cache has a duplicate or misassigned prediction row")
        covered.update(groups)
    expected = set(outer.train_groups if split == "train" else outer.valid_groups)
    if covered != expected or sidecar.get("eligible_row_groups") != sorted(expected):
        raise ValueError("crossfit cache does not cover exactly the outer fit pool")
