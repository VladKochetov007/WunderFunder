"""Bind F2 prediction artifacts to a frozen, label-blind recipe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
from f2_candidates import CANDIDATES

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
ARTIFACTS_DEFAULT = Path("/home/vlad/development/wunder/research_tmp/online-features-tl3-f2")
SPLIT_REL = "research/campaign_2026-09-29/artifacts/splits/fold2.json"
TARGET_REL = "cache/valid_targets.npz"
TARGET_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
SCORED_FROM_ROW = 99
N_BOOT = 10_000
BOOT_SEED = 20261019


def sha256_file(path: Path) -> str:
    """Hash an artifact incrementally with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_campaign_scorer(root: Path) -> ModuleType:
    """Load the reviewed stored-NPZ mmap helper without executing its CLI."""
    path = root / "research/campaign_2026-09-30/audit/campaign_score.py"
    spec = importlib.util.spec_from_file_location("f2_stored_npz_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load target-cache reader {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_source_identity(root: Path, artifacts: Path) -> dict[str, object]:
    """Validate exact split/cache/ZIP and prediction identities without labels."""
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    from f2_source import validate_f2_source

    split_path = root / SPLIT_REL
    source = validate_f2_source(root, split_path, None)
    split = source.split
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    sequence_rows = {int(row["row_group"]): row for row in manifest["sequences"]["valid"]}
    groups = list(split.evaluation_groups)
    components = [int(sequence_rows[group]["component"]) for group in groups]
    if split.fold != 2 or len(groups) != 443 or len(set(components)) != 144:
        raise ValueError("F2 membership differs from the frozen 443-group/144-component recipe")

    target_path = root / TARGET_REL
    if target_path.stat().st_size != 337_186_198 or sha256_file(target_path) != TARGET_SHA256:
        raise ValueError("official valid target cache identity differs from the frozen source")

    code_paths = (
        "grade_f2.py",
        "f2_grade_preflight.py",
        "f2_candidates.py",
        "f2_source.py",
        "f2_native_runtime.py",
        "f2_replay_engine.py",
        "f2_receipts.py",
        "replay_f2.py",
    )
    code_hashes = {name: sha256_file(PACKET / name) for name in code_paths}
    code_hashes["tl3_replay/f0_replay.py"] = sha256_file(PACKET.parent / "tl3_replay/f0_replay.py")
    code_hashes["tl3_replay/native_runtime.py"] = sha256_file(
        PACKET.parent / "tl3_replay/native_runtime.py"
    )

    candidate_identities: dict[str, dict[str, object]] = {}
    for name, specification in CANDIDATES.items():
        zip_path = root / str(specification["zip"])
        if sha256_file(zip_path) != specification["zip_sha256"]:
            raise ValueError(f"{name} exact package hash differs from the frozen source")
        prediction_path = artifacts / str(specification["prediction"])
        sidecar_path = artifacts / str(specification["sidecar"])
        receipt_path = artifacts / str(specification["receipt"])
        sidecar = cast(dict[str, object], json.loads(sidecar_path.read_text(encoding="utf-8")))
        receipt = cast(dict[str, object], json.loads(receipt_path.read_text(encoding="utf-8")))
        replay_source_hashes = {
            name: value
            for name, value in code_hashes.items()
            if name not in ("grade_f2.py", "f2_grade_preflight.py", "campaign_score.py")
        }
        expected_sequences = [int(sequence_rows[group]["seq_ix"]) for group in groups]
        if (
            sidecar.get("zip_sha256") != specification["zip_sha256"]
            or sidecar.get("candidate") != name
            or sidecar.get("split_manifest_sha256") != split.sha256
            or sidecar.get("row_groups") != groups
            or sidecar.get("sequence_ids") != expected_sequences
            or sidecar.get("fold") != 2
            or sidecar.get("labels_or_scoring_mask_read") is not False
            or receipt.get("candidate") != name
            or receipt.get("fold") != 2
            or receipt.get("archive_sha256") != specification["zip_sha256"]
            or receipt.get("evaluation_groups_replayed") != len(groups)
            or receipt.get("source_sha256") != replay_source_hashes
        ):
            raise ValueError(f"{name} prediction sidecar does not match frozen F2 order")
        prediction_hash = sha256_file(prediction_path)
        prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if prediction.shape != (len(groups), 20_000, 2) or prediction.dtype != np.float32:
            raise ValueError(f"{name} prediction cache shape or dtype differs")
        for start in range(0, len(groups), 8):
            if not np.isfinite(prediction[start : start + 8]).all():
                raise ValueError(f"{name} prediction cache contains nonfinite values")
        if (
            sidecar.get("prediction_sha256") != prediction_hash
            or receipt.get("predictions_sha256") != prediction_hash
            or receipt.get("passed") is not True
            or receipt.get("labels_or_scoring_mask_read") is not False
        ):
            raise ValueError(f"{name} replay receipt does not bind its exact prediction cache")
        candidate_identities[name] = {
            "zip_path": str(specification["zip"]),
            "zip_sha256": specification["zip_sha256"],
            "prediction_path": str(prediction_path),
            "prediction_sha256": prediction_hash,
            "prediction_bytes": prediction_path.stat().st_size,
            "sidecar_sha256": sha256_file(sidecar_path),
            "receipt_sha256": sha256_file(receipt_path),
            "native_library_sha256": receipt.get("native_library_sha256"),
            "precision": receipt.get("precision"),
        }
        del prediction

    code_hashes["campaign_score.py"] = sha256_file(
        root / "research/campaign_2026-09-30/audit/campaign_score.py"
    )
    return {
        "fold": split.fold,
        "split_path": SPLIT_REL,
        "split_sha256": split.sha256,
        "evaluation_groups": len(groups),
        "evaluation_group_ids": groups,
        "evaluation_group_order_sha256": hashlib.sha256(
            json.dumps(groups, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "evaluation_components": len(set(components)),
        "component_ids": components,
        "target_cache_path": TARGET_REL,
        "target_cache_sha256": TARGET_SHA256,
        "target_cache_bytes": target_path.stat().st_size,
        "target_values_accessed": False,
        "valid_parquet_identity": source.data_identity,
        "candidate_identities": candidate_identities,
        "code_sha256": code_hashes,
    }


def grade_preflight(root: Path, artifacts: Path, recipe_path: Path) -> dict[str, object]:
    """Freeze the observed exact inputs before any target or mask is read."""
    recipe = cast(dict[str, object], json.loads(recipe_path.read_text(encoding="utf-8")))
    if recipe.get("grader_sha256") != sha256_file(PACKET / "grade_f2.py"):
        raise ValueError("grader source differs from the predeclared recipe")
    identity = current_source_identity(root, artifacts)
    if recipe.get("split_sha256") != identity["split_sha256"]:
        raise ValueError("predeclared recipe names a different split")
    if recipe.get("target_cache_sha256") != identity["target_cache_sha256"]:
        raise ValueError("predeclared recipe names a different target-cache source")
    if recipe.get("code_sha256") != identity["code_sha256"]:
        raise ValueError("predeclared recipe does not bind the current replay/grader source")
    expected_zips = {name: item["zip_sha256"] for name, item in CANDIDATES.items()}
    observed_zips = {
        name: cast(dict[str, object], item)["zip_sha256"]
        for name, item in cast(dict[str, object], identity["candidate_identities"]).items()
    }
    if recipe.get("candidate_zip_sha256") != expected_zips or observed_zips != expected_zips:
        raise ValueError("predeclared recipe names different exact candidate ZIPs")
    if (
        recipe.get("evaluation_groups") != identity["evaluation_groups"]
        or recipe.get("evaluation_components") != identity["evaluation_components"]
    ):
        raise ValueError("predeclared recipe names different F2 membership")
    expected_blends = {
        "blend_75_25": {"tl3": 0.75, "r3": 0.25},
        "blend_50_50": {"tl3": 0.5, "r3": 0.5},
    }
    expected_comparisons = [
        {"candidate": "r3", "reference": "tl3"},
        {"candidate": "blend_75_25", "reference": "tl3"},
        {"candidate": "blend_50_50", "reference": "tl3"},
    ]
    expected_bootstrap = {
        "unit": "source-qualified connected component",
        "replicates": N_BOOT,
        "seed": BOOT_SEED,
    }
    if (
        recipe.get("fixed_blends") != expected_blends
        or recipe.get("comparisons") != expected_comparisons
        or recipe.get("bootstrap") != expected_bootstrap
    ):
        raise ValueError("predeclared blends, comparisons, or bootstrap differ from the grader")
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "wnn.tl3-fixed-blend-f2-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "recipe_sha256": sha256_file(recipe_path),
        "grader_sha256": sha256_file(PACKET / "grade_f2.py"),
        "input_fingerprint": fingerprint,
        "status": "pass",
        "label_values_accessed": False,
        "identity": identity,
    }
