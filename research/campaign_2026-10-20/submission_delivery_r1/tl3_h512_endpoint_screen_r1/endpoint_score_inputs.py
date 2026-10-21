"""Load only source-bound rows and caches needed by the fixed endpoint grade."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
from endpoint_contract import SOURCE_ZIP_SHA256 as BASELINE_ZIP_SHA256

# The frozen F0 source object and versioned JSON receipts are dynamic interfaces.
from endpoint_contract import read_json, sha256_file, validate_candidate

NLL_PACKET = Path("research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation")
SPLIT_PATH = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
EVALUATION_MANIFEST_SHA256 = "94d2226849ec0822bce5cf53890bda6aa41a0677f7d77da231a9b6495f22069b"
TARGET_MAPPER_SHA256 = "b6377a0e1d67030115c6d20a64ee00dc97ef15eaad7a806408a76ef70558ab39"
TARGET_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
TARGET_INDEX_SHA256 = "ed030f767a6c650db666844ecffef21f0e96cc9fa2d7e77a0872bdf26c5bed38"
BASELINE_PREDICTION_SHA256 = "3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3"
BASELINE_SIDECAR_SHA256 = "b9d8b7371dbff016d48e155da154cd87d85c887cb08921e86caabba082059910"
EVAL_GROUPS = 468
EVAL_COMPONENTS = 141
SELECTED_ROWS = 808_863
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
VALID_FOOTER_SHA256 = "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b"


def load_campaign_split(root: Path) -> Any:
    """Validate the exact F0 source manifest and producer Parquet metadata."""
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    replay = root / "research/campaign_2026-10-18/online_features_r1/tl3_replay"
    sys.path.insert(0, str(replay))
    from f0_source import validate_f0_source

    source = validate_f0_source(root, root / SPLIT_PATH, None)
    if (
        source.split.sha256 != SPLIT_SHA256
        or source.data_identity.get("footer_sha256_from_split") != VALID_FOOTER_SHA256
    ):
        raise ValueError("campaign split hash differs from frozen F0 source")
    if len(source.groups) != EVAL_GROUPS:
        raise ValueError("campaign split does not contain the frozen 468 F0 groups")
    return source


def _load_frozen_scorer(root: Path) -> Any:
    """Reuse the campaign's stored-NPZ memory mapper without its CV gates."""
    path = root / "research/campaign_2026-09-30/audit/campaign_score.py"
    if sha256_file(path) != TARGET_MAPPER_SHA256:
        raise ValueError("frozen target-cache mapper source hash changed")
    spec = importlib.util.spec_from_file_location("campaign_score_h512_endpoint", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen target-cache mapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_targets(
    root: Path, source: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Map the exact target cache and select only the 468 scored sequences."""
    cache = root / "cache/valid_targets.npz"
    if cache.stat().st_size != 337_186_198 or sha256_file(cache) != TARGET_SHA256:
        raise ValueError("valid target cache bytes differ from the frozen evaluation plan")
    targets = _load_frozen_scorer(root).mmap_stored_npz(cache)
    row_groups = np.asarray(targets["row_group"], dtype=np.int64)
    sequence_ids = np.asarray(targets["seq_ix"], dtype=np.int64)
    if (
        targets["y"].shape != (1_873, 20_000, 2)
        or targets["y"].dtype != np.float32
        or targets["mask"].shape != (1_873, 20_000)
        or targets["mask"].dtype != np.bool_
        or row_groups.shape != (1_873,)
        or sequence_ids.shape != (1_873,)
    ):
        raise ValueError("target cache arrays differ from the frozen shape/dtype contract")
    index_sha = hashlib.sha256(
        np.asarray(row_groups, dtype="<i8").tobytes()
        + np.asarray(sequence_ids, dtype="<i8").tobytes()
    ).hexdigest()
    if index_sha != TARGET_INDEX_SHA256 or len(row_groups) != 1_873:
        raise ValueError("valid target group/sequence identity index changed")
    position = {int(group): index for index, group in enumerate(row_groups)}
    cache_sequence = dict(zip(map(int, row_groups), map(int, sequence_ids), strict=True))
    manifest_path = root / NLL_PACKET / "evaluation_manifest.json"
    if sha256_file(manifest_path) != EVALUATION_MANIFEST_SHA256:
        raise ValueError("frozen NLL F0 evaluation manifest changed")
    manifest = read_json(manifest_path)
    expected_rows = manifest.get("evaluation_groups", [])
    if len(expected_rows) != EVAL_GROUPS or any(
        item.get("source") != "valid" for item in expected_rows
    ):
        raise ValueError("evaluation manifest does not name the fixed valid-source panel")
    groups = [int(item["row_group"]) for item in expected_rows]
    components = np.asarray([int(item["component"]) for item in expected_rows], dtype=np.int64)
    if tuple(groups) != tuple(source.groups) or len(set(groups)) != EVAL_GROUPS:
        raise ValueError("evaluation manifest groups differ from validated F0 replay order")
    if len(np.unique(components)) != EVAL_COMPONENTS:
        raise ValueError("evaluation panel component count differs from frozen plan")
    for group in groups:
        if cache_sequence.get(group) != source.sequence_by_group.get(group):
            raise ValueError(f"target cache sequence identity differs at row group {group}")
    positions = [position[group] for group in groups]
    targets_selected = np.stack([targets["y"][index] for index in positions]).astype(
        np.float32, copy=False
    )
    from wnn.gmetric import WARMUP

    is_scored = np.stack([targets["mask"][index] for index in positions]).astype(bool, copy=False)
    mask = is_scored & (np.arange(20_000) >= WARMUP)[None, :]
    selected = int(mask.sum())
    if selected != SELECTED_ROWS:
        raise ValueError(f"expected {SELECTED_ROWS} scored rows, got {selected}")
    identity = {
        "cache_path": str(cache),
        "cache_sha256": TARGET_SHA256,
        "cache_bytes": cache.stat().st_size,
        "index_sha256": index_sha,
        "target_shape": list(targets["y"].shape),
        "mask_shape": list(targets["mask"].shape),
        "row_groups": groups,
        "sequence_ids": [source.sequence_by_group[group] for group in groups],
        "components": len(np.unique(components)),
        "warmup_excluded_rows_per_sequence": WARMUP,
        "selected_rows": selected,
        "evaluation_manifest_sha256": EVALUATION_MANIFEST_SHA256,
    }
    return targets_selected, mask, components, np.asarray(groups, dtype=np.int64), identity


def _load_prediction(
    path: Path, sidecar_path: Path, expected: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify a complete, aligned memmapped endpoint prediction cache."""
    path = path.resolve(strict=True)
    side = read_json(sidecar_path.resolve(strict=True))
    if (
        sha256_file(path) != expected["prediction_sha256"]
        or side.get("prediction_sha256") != expected["prediction_sha256"]
    ):
        raise ValueError(f"prediction bytes differ from sidecar: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (EVAL_GROUPS, 20_000, 2) or values.dtype != np.float32:
        raise ValueError(f"prediction shape or dtype differs: {path}")
    if list(map(int, side.get("row_groups", []))) != expected["row_groups"]:
        raise ValueError(f"prediction row-group order differs: {path}")
    if list(map(int, side.get("sequence_ids", []))) != expected["sequence_ids"]:
        raise ValueError(f"prediction sequence IDs differ: {path}")
    return values, side


def _load_candidate(
    root: Path,
    endpoint: str,
    paths: dict[str, Any],
    expected_groups: list[int],
    expected_sequence_ids: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate replay receipt, package source binding, and prediction bytes."""
    predictions_path = Path(paths["predictions"]).resolve(strict=True)
    side_path = Path(paths["sidecar"]).resolve(strict=True)
    side = read_json(side_path)
    receipt_path = Path(str(side.get("replay_receipt_path", ""))).resolve(strict=True)
    receipt_sha = sha256_file(receipt_path)
    if side.get("replay_receipt_sha256") != receipt_sha or receipt_path != Path(
        paths["replay_receipt"]
    ).resolve(strict=True):
        raise ValueError(f"replay receipt changed after endpoint replay: {endpoint}")
    replay = read_json(receipt_path)
    replay_dir = root / "research/campaign_2026-10-18/online_features_r1/tl3_replay"
    this_packet = (
        root / "research/campaign_2026-10-20/submission_delivery_r1/tl3_h512_endpoint_screen_r1"
    )
    code_hashes = {
        "f0_source_sha256": replay_dir / "f0_source.py",
        "f0_replay_sha256": replay_dir / "f0_replay.py",
        "native_runtime_sha256": replay_dir / "native_runtime.py",
        "replay_script_sha256": this_packet / "replay_h512_tl3_endpoint.py",
        "endpoint_contract_sha256": this_packet / "endpoint_contract.py",
    }
    if replay.get("schema") != "wnn.h512-tl3-f0-endpoint-replay.v1" or any(
        replay.get(key) != sha256_file(path) for key, path in code_hashes.items()
    ):
        raise ValueError(f"replay source code differs from its receipt: {endpoint}")
    package = replay.get("package_binding", {})
    zip_path = Path(str(package.get("zip_path", "")))
    build_path = Path(str(package.get("build_receipt_path", "")))
    if zip_path.resolve(strict=True) != Path(paths["zip"]).resolve(
        strict=True
    ) or build_path.resolve(strict=True) != Path(paths["build_receipt"]).resolve(strict=True):
        raise ValueError(f"replay receipt names a different frozen package path: {endpoint}")
    current = validate_candidate(root, endpoint, zip_path, build_path)
    if current != package or replay.get("status") != "REPLAY_COMPLETE":
        raise ValueError(f"replay receipt source binding no longer matches: {endpoint}")
    if (
        side.get("endpoint") != endpoint
        or side.get("schema") != "wnn.h512-tl3-f0-endpoint-predictions.v1"
        or side.get("candidate") != f"h512_{endpoint}_tl3_fixed034"
        or side.get("evidence_classification") != "full-endpoint fitted diagnostic; not clean CV"
        or side.get("zip_sha256") != current["zip_sha256"]
        or side.get("prediction_path") != str(predictions_path)
        or side.get("shape") != [EVAL_GROUPS, 20_000, 2]
        or side.get("dtype") != "float32"
        or list(map(int, side.get("row_groups", []))) != expected_groups
        or list(map(int, side.get("sequence_ids", []))) != expected_sequence_ids
        or replay.get("prediction_sha256") != side.get("prediction_sha256")
        or replay.get("split_manifest_sha256") != SPLIT_SHA256
    ):
        raise ValueError(
            f"candidate replay sidecar is not bound to this fixed endpoint: {endpoint}"
        )
    expected = {
        "prediction_sha256": side.get("prediction_sha256"),
        "row_groups": expected_groups,
        "sequence_ids": expected_sequence_ids,
    }
    values, verified_side = _load_prediction(predictions_path, side_path, expected)
    if verified_side != side or replay.get("prediction_path") != str(predictions_path.resolve()):
        raise ValueError(f"candidate prediction path differs from replay receipt: {endpoint}")
    return values, {
        "prediction_sidecar": str(side_path),
        "prediction_sidecar_sha256": sha256_file(side_path),
        "replay_receipt": str(receipt_path),
        "replay_receipt_sha256": receipt_sha,
        "package_binding": current,
    }


def load_screen_inputs(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Load and bind reference, candidate, split, and target inputs."""
    source = load_campaign_split(root)
    targets, mask, components, groups, identity = load_targets(root, source)
    baseline_zip = root / "submissions/team_tl3_original_20261018.zip"
    baseline_pred = root / "cache/pred_tl3_unknown_ancestry_f0_valid.npy"
    baseline_sidecar = root / "cache/pred_tl3_unknown_ancestry_f0_valid.json"
    if (
        sha256_file(baseline_zip) != BASELINE_ZIP_SHA256
        or sha256_file(baseline_sidecar) != BASELINE_SIDECAR_SHA256
    ):
        raise ValueError("original TL3 package or baseline sidecar changed")
    old_side = read_json(baseline_sidecar)
    if (
        old_side.get("candidate") != "team_tl3_original_20261018"
        or old_side.get("zip_sha256") != BASELINE_ZIP_SHA256
        or old_side.get("split_manifest_sha256") != source.split.sha256
    ):
        raise ValueError("baseline prediction cache is not the pinned original TL3 F0 cache")
    baseline, _ = _load_prediction(
        baseline_pred,
        baseline_sidecar,
        {
            "prediction_sha256": BASELINE_PREDICTION_SHA256,
            "row_groups": list(map(int, groups)),
            "sequence_ids": identity["sequence_ids"],
        },
    )
    candidates: dict[str, np.ndarray] = {}
    receipts: dict[str, Any] = {}
    for endpoint in ("cp5", "swa345"):
        paths = plan["runtime_paths"][endpoint]
        candidates[endpoint], receipts[endpoint] = _load_candidate(
            root,
            endpoint,
            paths,
            list(map(int, groups)),
            identity["sequence_ids"],
        )
    return {
        "targets": targets,
        "mask": mask,
        "components": components,
        "groups": groups,
        "target_identity": identity,
        "baseline": baseline,
        "candidates": candidates,
        "source_receipts": receipts,
    }
