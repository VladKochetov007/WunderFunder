"""Receipt-bound admission of completed fixed-epoch fit cells."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    from .postprocess_common import (
        ACTOR_SHA256,
        SOURCE_MANIFEST_SHA256,
        SOURCE_TREE_SHA256,
        _read_json,
        _sha256,
    )
except ImportError:
    from postprocess_common import (
        ACTOR_SHA256,
        SOURCE_MANIFEST_SHA256,
        SOURCE_TREE_SHA256,
        _read_json,
        _sha256,
    )

_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _fit_binding(root: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Return only a completed fixed-epoch fit whose command, split and checkpoint bind."""
    run_name = str(job.get("id", ""))
    if not _RUN_NAME.fullmatch(run_name):
        raise ValueError("invalid wave run id")
    fold, seed, arm = job.get("fold"), job.get("seed"), job.get("arm")
    if (
        type(fold) is not int
        or not 0 <= fold < 4
        or type(seed) is not int
        or arm
        not in {
            "wide",
            "deep",
        }
    ):
        raise ValueError(f"invalid fold/seed/arm in job {run_name}")
    supplied_run_dir = Path(str(job["run_dir"]))
    if supplied_run_dir.is_symlink():
        raise ValueError(f"fit run directory cannot be a symlink: {supplied_run_dir}")
    run_dir = supplied_run_dir.resolve(strict=True)
    expected_run = (root / "runs" / run_name).resolve(strict=True)
    if run_dir != expected_run:
        raise ValueError(f"fit run directory is not the wave-owned path: {run_dir}")
    fit_receipt = _read_json(run_dir / "producer_receipt.json")
    if fit_receipt.get("status") != "completed_fixed_epoch5":
        raise ValueError(f"fit {run_name} is not a completed fixed-epoch-5 cell")
    expected_identity = {"run_name": run_name, "arm": arm, "fold": fold, "seed": seed}
    if any(fit_receipt.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"fit receipt identity differs from wave job {run_name}")
    if fit_receipt.get("selected_epoch") != 5:
        raise ValueError(f"fit {run_name} does not bind the fixed epoch-5 endpoint")
    bundle = fit_receipt.get("adapter_bundle_identity")
    if not isinstance(bundle, dict):
        raise ValueError(f"fit {run_name} has no adapter bundle identity")
    if (
        bundle.get("sha256") != ACTOR_SHA256
        or bundle.get("source_tree_identity_sha256") != SOURCE_TREE_SHA256
        or bundle.get("source_tree_manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError(f"fit {run_name} does not bind the deployed actor and source tree")
    command_path = run_dir / "command.json"
    command = _read_json(command_path)
    ancestry = fit_receipt.get("input_ancestry", {})
    deployed_tree = command.get("deployed_source_tree", {})
    if not isinstance(ancestry, dict) or not isinstance(deployed_tree, dict):
        raise ValueError(f"fit {run_name} has malformed source ancestry")
    if (
        fit_receipt.get("command_sha256") != _sha256(command_path)
        or command.get("mode") != "fit"
        or command.get("run_name") != run_name
        or command.get("adapter_bundle_identity") != bundle
        or deployed_tree.get("deployed_tree_identity_sha256") != SOURCE_TREE_SHA256
        or deployed_tree.get("manifest_sha256") != SOURCE_MANIFEST_SHA256
        or ancestry.get("deployed_source_tree") != deployed_tree
    ):
        raise ValueError(f"fit {run_name} command receipt does not match its producer receipt")
    expected_shape = {"wide": (512, 1), "deep": (384, 2)}[arm]
    provenance_path = run_dir / "artifacts" / "runs" / run_name / "provenance.json"
    provenance = _read_json(provenance_path)
    args = provenance.get("args", {})
    expected_args = {
        "name": run_name,
        "seed": seed,
        "valid_fold": fold,
        "hidden": expected_shape[0],
        "layers": expected_shape[1],
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "no_eval_during_fit": True,
        "precision": "fp32",
        "batch": 8,
        "chunk": 2000,
    }
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise ValueError(f"fit {run_name} provenance does not match its frozen shape and recipe")
    split_path = root / "splits" / f"fold{fold}.json"
    split_sha = _sha256(split_path)
    split = _read_json(split_path)
    groups = split.get("evaluation", {}).get("valid")
    if (
        split.get("fold") != fold
        or split.get("n_folds") != 4
        or ancestry.get("split_manifest_sha256") != split_sha
        or ancestry.get("fold") != fold
        or groups != ancestry.get("evaluation_groups")
        or not isinstance(groups, list)
        or not groups
        or len(set(groups)) != len(groups)
    ):
        raise ValueError(f"fit {run_name} split membership differs from its frozen ancestry")
    checkpoint_relative = f"artifacts/runs/{run_name}/model_ep5.pt"
    checkpoint = run_dir / checkpoint_relative
    recorded = fit_receipt.get("outputs", {}).get(checkpoint_relative, {})
    if not checkpoint.is_file() or _sha256(checkpoint) != recorded.get("sha256"):
        raise ValueError(f"fit {run_name} epoch-5 checkpoint differs from its receipt")
    return {
        "job": job,
        "run_dir": run_dir,
        "fit_receipt": fit_receipt,
        "fit_receipt_sha256": _sha256(run_dir / "producer_receipt.json"),
        "split_path": split_path,
        "split_sha256": split_sha,
        "groups": groups,
        "checkpoint_sha256": recorded["sha256"],
        "provenance_sha256": _sha256(provenance_path),
    }
