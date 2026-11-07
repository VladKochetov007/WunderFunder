from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from wnn.splits import CampaignSplit, file_sha256
from wnn.training.sequence_reference_contract import (
    canonical_sha256,
    is_sha256,
    members_sha256,
    read_object,
)
from wnn.training.sequence_reference_pins import (
    ANCESTRY_LIMITATION,
    ARITHMETIC,
    COMPOSITION,
    PR49_SOURCE_COMMIT,
    PRIMARY_SPLIT_SHA256,
    PUBLISHED_RUNS,
)
from wnn.training.sequence_source_capsule import verify_source_capsule

# Published run provenance has mixed scalar, sequence and mapping JSON fields.
MEMBER_RUNS = dict(
    zip(("w18", "tiedlead_s5", "tiedlead_s6"), (r["run"] for r in COMPOSITION), strict=True)
)


def check_source(
    root: Path, manifest: Path | None = None, manifest_sha256: str | None = None
) -> str:
    """Require the exact PR49 source commit and unmodified importable Python files."""
    if manifest is not None or manifest_sha256 is not None:
        if manifest is None or manifest_sha256 is None:
            raise ValueError("source capsule requires its manifest and trusted manifest digest")
        return verify_source_capsule(root, manifest, manifest_sha256, PR49_SOURCE_COMMIT)
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PR49_SOURCE_COMMIT:
        raise ValueError("legacy inference checkout must use the pinned PR49 commit")
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "scripts",
        ],
        text=True,
    )
    if dirty:
        raise ValueError("legacy inference source is modified or contains untracked Python files")
    return actual


def audit_ancestors(data_root: Path, split: CampaignSplit) -> dict[str, Any]:
    """Bind all17 published fixed checkpoints and original provenance before inference."""
    if split.sha256 != PRIMARY_SPLIT_SHA256 or split.valid_groups:
        raise ValueError("reference must use the exact PR49 train-only split")
    fitting = [("train", g) for g in split.train_groups]
    evaluation = [("valid", g) for g in split.evaluation_groups]
    if len(fitting) != 10325 or len(evaluation) != 1873:
        raise ValueError("reference requires complete PR49 fit and evaluation memberships")
    fitting_hash, evaluation_hash = members_sha256(fitting), members_sha256(evaluation)
    records = []
    for run, pinned in sorted(PUBLISHED_RUNS.items()):
        run_root = data_root / "artifacts" / "runs" / run
        provenance, digest = read_object(run_root / "provenance.json")
        checkpoint_name = f"model_ep{pinned['epoch']}.pt"
        checkpoint = run_root / checkpoint_name
        if (
            digest != pinned["provenance_sha256"]
            or checkpoint.is_symlink()
            or not checkpoint.is_file()
            or file_sha256(checkpoint) != pinned["checkpoint_sha256"]
        ):
            raise ValueError(f"published reference ancestor bytes differ: {run}")
        args = provenance.get("args", {})
        membership = provenance.get("fit_row_groups", {})
        if (
            provenance.get("split_manifest_sha256") != split.sha256
            or membership.get("valid") != []
            or members_sha256([("train", g) for g in membership.get("train", [])]) != fitting_hash
            or members_sha256([("valid", g) for g in provenance.get("evaluation_row_groups", [])])
            != evaluation_hash
            or provenance.get("checkpoint_sha256", {}).get(checkpoint_name)
            != pinned["checkpoint_sha256"]
            or args.get("name") != run
            or args.get("precision") != "fp32"
            or args.get("checkpoint_policy") != "fixed"
            or args.get("checkpoint_epoch") != pinned["epoch"]
            or args.get("train_source") != "train"
            or args.get("propensity") not in (None, "")
            or args.get("prop_input") not in (None, "")
        ):
            raise ValueError(f"reference ancestor has incompatible recipe or memberships: {run}")
        implementation = provenance.get("trainer_implementation", {})
        files = implementation.get("files")
        if (
            not isinstance(files, dict)
            or not files
            or not all(is_sha256(value) for value in files.values())
            or canonical_sha256(files) != implementation.get("sha256")
        ):
            raise ValueError(f"reference ancestor lacks a valid trainer-source inventory: {run}")
        records.append(
            {
                "run": run,
                **pinned,
                "fit_members_sha256": fitting_hash,
                "evaluation_members_sha256": evaluation_hash,
                "trainer_source_sha256": implementation["sha256"],
            }
        )
    return {
        "kind": "pr49-tl3-reference-v1",
        "split_sha256": split.sha256,
        "fit_members_sha256": fitting_hash,
        "evaluation_members_sha256": evaluation_hash,
        "composition": COMPOSITION,
        "arithmetic": ARITHMETIC,
        "inference_source_commit": PR49_SOURCE_COMMIT,
        "training_predictions": "in-sample",
        "ancestry_limitation": ANCESTRY_LIMITATION,
        "ancestors": records,
    }


def bank_path(data_root: Path, run: str, source: str) -> Path:
    """Name only this packet's create-only prediction bank, preserving historical arrays."""
    if run not in MEMBER_RUNS.values() or source not in ("train", "valid"):
        raise ValueError("unknown frozen reference member or source")
    return data_root / "cache" / f"pred_tl3ref_{run}_{source}.npy"


def write_new_json(path: Path, value: object) -> None:
    """Write a complete JSON receipt without replacing an existing result."""
    import json

    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
