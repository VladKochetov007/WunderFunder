"""Deterministically admit exact, bounded B300 profiles for multistage Optuna."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GIB = 1 << 30
POLICY_NAME = "PROFILE_REVIEW_POLICY_AUTOMATED_R6.json"
_ACTIVE_SOURCE_ROOT: Path | None = None
GPU_UUID_PATTERN = re.compile(
    r"^(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
WEIGHTED_POOL_TARGET_CONTRACT = {
    "formula": "sequential float32 sum(weight_i * prediction_i) in parent order",
    "parent_clipping": "none",
    "pool_clipping": "none",
    "student_target_clipping": "clamp pooled teacher target to [-2,2] in weighted MSE",
    "student_objective": "label_loss + kd_weight * teacher_loss",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical_gpu_uuid(value: object) -> str | None:
    """Normalize CUDA's optional GPU- display prefix under strict UUID syntax."""
    if not isinstance(value, str):
        return None
    match = GPU_UUID_PATTERN.fullmatch(value)
    return match.group(1).lower() if match is not None else None


def _same_gpu_uuid(left: object, right: object) -> bool:
    """Compare UUID identities while ignoring only CUDA's optional prefix/case."""
    canonical_left, canonical_right = _canonical_gpu_uuid(left), _canonical_gpu_uuid(right)
    return canonical_left is not None and canonical_left == canonical_right


def _path_equal(value: object, path: Path) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Path(value).resolve() == path.resolve()
    except OSError:
        return False


def _file_hash_equal(path: Path, expected: object) -> bool:
    try:
        return not path.is_symlink() and path.is_file() and sha256_file(path) == expected
    except OSError:
        return False


def _member_rows(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, list)
        and len(item) == 2
        and item[0] == "train"
        and type(item[1]) is int
        and item[1] >= 0
        for item in value
    )


def _canonical_member_rows(groups: object) -> list[list[object]]:
    if not isinstance(groups, list):
        return []
    return [["train", group] for group in groups if type(group) is int]


def _process_tuple(value: object) -> bool:
    row = _mapping(value)
    return (
        type(row.get("pid")) is int
        and row["pid"] > 0
        and type(row.get("pgid")) is int
        and row["pgid"] > 0
        and isinstance(row.get("birth"), (str, int))
        and str(row["birth"]) not in ("", "0")
    )


def _same_process(left: object, right: object) -> bool:
    a, b = _mapping(left), _mapping(right)
    return (
        _process_tuple(a)
        and _process_tuple(b)
        and all(a.get(key) == b.get(key) for key in ("pid", "birth", "pgid"))
    )


def _rank_logs(request: dict[str, Any], updates: int) -> tuple[list[dict[str, Any]], bool]:
    refs = request.get("rank_log_refs")
    if not isinstance(refs, list) or len(refs) != 1:
        return [], False
    ref = _mapping(refs[0])
    path_text, digest = ref.get("path"), ref.get("sha256")
    if type(ref.get("rank")) is not int or ref["rank"] != 0:
        return [], False
    if not isinstance(path_text, str) or not _sha(digest):
        return [], False
    path = Path(path_text)
    if not _file_hash_equal(path, digest):
        return [], False
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return [], False
    scalar_keys = ("total_loss", "label_loss", "gradient_norm_preclip", "learning_rate")
    ok = type(updates) is int and len(rows) == updates
    for step, row in enumerate(rows, 1):
        ok = ok and isinstance(row, dict)
        if not isinstance(row, dict):
            continue
        ok = ok and row.get("rank") == 0 and row.get("step") == step
        ok = ok and all(_finite(row.get(key)) and float(row[key]) >= 0 for key in scalar_keys)
        ok = ok and all(
            type(row.get(key)) is int and row[key] > 0
            for key in ("valid_rows", "real_rows", "forwarded_rows")
        )
        ok = ok and row.get("forwarded_rows", 0) >= row.get("real_rows", 1)
        ok = ok and _finite(row.get("label_weight")) and float(row["label_weight"]) > 0
    return [{"rank": 0, "path": str(path.resolve()), "sha256": digest, "rows": len(rows)}], bool(ok)


def _load_config(request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    path_text = request.get("config_path")
    if not isinstance(path_text, str):
        return {}, False
    try:
        path = Path(path_text)
        value = read_json(path)
        return value, _file_hash_equal(path, request.get("config_sha256"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, False


def _project_root() -> Path:
    if _ACTIVE_SOURCE_ROOT is not None:
        return _ACTIVE_SOURCE_ROOT
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "wnn" / "training").is_dir():
            return parent
    raise ValueError("could not locate the pinned project source tree")


def _activate_source_tree(
    worker: dict[str, Any], identity: dict[str, Any], request: dict[str, Any]
) -> tuple[Path | None, bool]:
    """Use only the source tree named by the owner receipt and native identity."""
    global _ACTIVE_SOURCE_ROOT
    _ACTIVE_SOURCE_ROOT = None
    root_text = worker.get("cwd") or worker.get("source_root")
    if not isinstance(root_text, str) or not root_text:
        return None, False
    root = Path(root_text)
    try:
        if root.is_symlink() or not root.is_dir() or not (root / "src/wnn/training").is_dir():
            return None, False
        files = identity.get("source_files")
        if not isinstance(files, dict) or not files:
            return root, False
        if canonical_sha256(files) != request.get("source_files_sha256"):
            return root, False
        # Verify the complete producer inventory before importing its validators.
        for relative, expected in files.items():
            if not isinstance(relative, str):
                return root, False
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts:
                return root, False
            path = root / rel
            if expected is None:
                if path.exists() or path.is_symlink():
                    return root, False
                continue
            if not _sha(expected) or not _file_hash_equal(path, expected):
                return root, False
        _ACTIVE_SOURCE_ROOT = root.resolve()
        # A reviewer is a standalone CLI, but tests may call review() repeatedly.
        for name in tuple(sys.modules):
            if name == "wnn" or name.startswith("wnn."):
                del sys.modules[name]
        source = str(_ACTIVE_SOURCE_ROOT / "src")
        sys.path[:] = [item for item in sys.path if Path(item or ".").resolve() != Path(source)]
        sys.path.insert(0, source)
        return _ACTIVE_SOURCE_ROOT, True
    except (OSError, ValueError, RuntimeError):
        _ACTIVE_SOURCE_ROOT = None
        return None, False


def _portable_registry(policy: dict[str, Any], worker: dict[str, Any]) -> tuple[Path | None, bool]:
    """Bind the footer-only source fingerprints to the sealed portable registry."""
    expected = _mapping(policy.get("portable_data_registry")).get("sha256")
    env = _mapping(worker.get("child_environment"))
    path_text = (
        os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
        or env.get("WNN_PORTABLE_DATA_MANIFEST")
    )
    if not isinstance(path_text, str) or not path_text:
        return None, False
    path = Path(path_text)
    try:
        ok = (
            not path.is_symlink()
            and path.is_file()
            and _sha(expected)
            and sha256_file(path) == expected
        )
        if ok:
            os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(path.resolve())
        return path.resolve(), bool(ok)
    except OSError:
        return None, False


def _pinned_source_file(identity: dict[str, Any], relative: str) -> Path | None:
    root = _project_root()
    path = root / Path(relative)
    expected = _mapping(identity.get("source_files")).get(relative)
    if not _sha(expected) or not _file_hash_equal(path, expected):
        return None
    return path


def _native_config_identity(
    config_path: Path, identity: dict[str, Any]
) -> str | None:
    """Recompute the producer's normalized config hash from its pinned loader."""
    required = (
        "src/wnn/training/sequence_config.py",
        "src/wnn/training/sequence_config_validation.py",
        "src/wnn/training/sequence_precision.py",
    )
    if any(_pinned_source_file(identity, relative) is None for relative in required):
        return None
    root_src = str(_project_root() / "src")
    if root_src not in sys.path:
        sys.path.insert(0, root_src)
    try:
        loader = importlib.import_module("wnn.training.sequence_config").load_config
        return canonical_sha256(loader(config_path))
    except (ImportError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _validate_split_and_members(
    request: dict[str, Any], data: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    split_path_text = request.get("auxiliary_split_manifest")
    if not isinstance(split_path_text, str):
        return {}, False
    split_path = Path(split_path_text)
    try:
        manifest = read_json(split_path)
        digest_ok = _file_hash_equal(split_path, request.get("auxiliary_split_sha256"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, False
    identity = _mapping(policy.get("identity"))
    parent = _mapping(manifest.get("parent"))
    fit, evaluation = _mapping(manifest.get("fit")), _mapping(manifest.get("evaluation"))
    expected_fit = _canonical_member_rows(fit.get("train"))
    expected_control = _canonical_member_rows(evaluation.get("train"))
    counts_ok = (
        len(expected_fit) == identity.get("auxiliary_fit_train_sequences")
        and fit.get("valid") == []
        and len(expected_control) == identity.get("auxiliary_control_train_sequences")
        and evaluation.get("source") == identity.get("auxiliary_evaluation_source")
    )
    member_ok = (
        expected_fit
        and expected_control
        and len(set(map(tuple, expected_fit))) == len(expected_fit)
        and len(set(map(tuple, expected_control))) == len(expected_control)
        and not set(map(tuple, expected_fit)).intersection(map(tuple, expected_control))
        and _member_rows(data.get("fit_members"))
        and data.get("fit_members") == expected_fit
        and _member_rows(data.get("evaluation_members"))
        and data.get("evaluation_members") == expected_control
        and data.get("evaluation_role") == identity.get("auxiliary_evaluation_role")
        and data.get("evaluation_source") == identity.get("auxiliary_evaluation_source")
        and data.get("training_mask") == "all_after_warmup"
        and data.get("evaluation_mask") == "all_after_warmup"
    )
    protocol_ok = (
        request.get("primary_split_sha256") == identity.get("primary_split_sha256")
        and request.get("auxiliary_split_sha256") == identity.get("auxiliary_split_sha256")
        and request.get("objective_scope") == identity.get("objective_scope")
        and request.get("primary_valid_used") is False
        and data.get("primary_split_sha256") == identity.get("primary_split_sha256")
        and data.get("split_sha256") == identity.get("auxiliary_split_sha256")
        and manifest.get("chronology_basis")
        == "none; deterministic whole-component auxiliary partition, not a time split"
        and parent.get("sha256") == identity.get("primary_split_sha256")
        and manifest.get("sources") == data.get("sources")
    )
    return manifest, bool(digest_ok and counts_ok and member_ok and protocol_ok)


def _validate_canonical_campaign_split(
    request: dict[str, Any], data: dict[str, Any]
) -> bool:
    """Re-open the source-pinned auxiliary split and verify its live source files."""
    manifest_text, data_root_text = request.get("auxiliary_split_manifest"), request.get("data_root")
    if not isinstance(manifest_text, str) or not isinstance(data_root_text, str):
        return False
    try:
        from wnn.splits import load_campaign_split

        split = load_campaign_split(Path(manifest_text), Path(data_root_text))
        fit = [["train", group] for group in split.train_groups]
        control = [[split.evaluation_source, group] for group in split.evaluation_groups]
        return bool(
            split.sha256 == request.get("auxiliary_split_sha256")
            and split.primary_split_sha256 == request.get("primary_split_sha256")
            and split.evaluation_source == "train"
            and split.evaluation_role == "control"
            and fit == data.get("fit_members")
            and control == data.get("evaluation_members")
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


def _validate_normalization(
    config: dict[str, Any],
    data: dict[str, Any],
    identity: dict[str, Any],
    profile_path: Path | None = None,
) -> bool:
    norm = _mapping(data.get("normalization"))
    fit = data.get("fit_members")
    sources = _mapping(data.get("sources"))
    if not isinstance(fit, list):
        return False
    scope = config.get("normalization_scope")
    if scope not in {"prefix", "all_fit"}:
        return False
    try:
        requested = int(config.get("normalization_sequences", 0))
    except (TypeError, ValueError):
        return False
    n = len(fit) if scope == "all_fit" else min(len(fit), requested)
    if n <= 0:
        return False
    selected = fit[:n]
    used_sources = sorted({str(member[0]) for member in selected if isinstance(member, list)})
    source_fingerprints = {
        f"datasets/{source}.parquet": sources.get(f"datasets/{source}.parquet")
        for source in used_sources
    }
    split_sha256 = data.get("split_sha256")
    selected_members: list[tuple[str, int]] = []
    for member in selected:
        if (
            not isinstance(member, list)
            or len(member) != 2
            or not isinstance(member[0], str)
            or type(member[1]) is not int
        ):
            return False
        selected_members.append((member[0], member[1]))
    normalization_module = "src/wnn/training/sequence_normalization.py"
    if _pinned_source_file(identity, normalization_module) is None:
        return False
    root_src = str(_project_root() / "src")
    if root_src not in sys.path:
        sys.path.insert(0, root_src)
    try:
        normalizer = importlib.import_module("wnn.training.sequence_normalization")
        computed_identity = normalizer.build_normalization_identity(
            source_fingerprints,
            split_sha256,
            selected_members,
            scope,
            112,
            input_schema=data.get("input_schema", "raw112-v1"),
        )
        identity_sha256 = normalizer.normalization_identity_sha256(computed_identity)
    except (ImportError, OSError, ValueError, TypeError, KeyError):
        return False
    moments_sha256 = norm.get("moments_sha256")
    if not _sha(moments_sha256):
        return False
    expected = {
        "scope": scope,
        "sequence_count": n,
        "source_fingerprints": source_fingerprints,
        "identity_sha256": identity_sha256,
        "moments_sha256": moments_sha256,
    }
    if scope == "prefix":
        expected["fit_prefix_sequences"] = n
    if _mapping(config.get("model")).get("input_feature_mode", "raw112") != "raw112":
        expected["input_schema"] = data.get("input_schema")
    if norm != expected:
        return False
    if profile_path is None:
        return True
    try:
        metadata_path = profile_path.parent / "run_metadata.json"
        metadata = read_json(metadata_path)
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    metadata_data = _mapping(_mapping(metadata.get("provenance")).get("data"))
    return metadata_data == data


def _campaign_run_r2_binding(
    request_path: Path, request: dict[str, Any], compatibility: dict[str, Any]
) -> dict[str, Any]:
    """Prove a missing request reviewer hash from its exact frozen campaign run."""
    try:
        request_file = request_path.resolve(strict=True)
        trial_dir = request_file.parent
        stage, trial = request.get("stage"), request.get("trial_number")
        if (
            request_file.name != "trial_request.json"
            or type(stage) is not int
            or type(trial) is not int
            or trial_dir.name != f"{trial:06d}"
            or trial_dir.parent.name != "trials"
            or trial_dir.parent.parent.name != f"stage{stage}"
        ):
            return {"valid": False}
        campaign = trial_dir.parents[2]
        run_path = campaign / "run.json"
        run = read_json(run_path)
        run_sha = sha256_file(run_path)
        policy_ref = _mapping(run.get("profile_checker_policy"))
        reviewer_ref = _mapping(run.get("profile_reviewer_script"))
        policy_path = Path(str(policy_ref.get("path")))
        reviewer_path = Path(str(reviewer_ref.get("path")))
        expected_policy_sha = compatibility.get("request_policy_sha256")
        expected_reviewer_sha = compatibility.get("request_reviewer_script_sha256")
        if (
            run.get("schema") != "wnn.multistage-run.v1"
            or run_sha != compatibility.get("campaign_run_json_sha256")
            or run.get("experiment_id") != request.get("experiment_id")
            or run.get("profile_checker_policy")
            != {"path": str(policy_ref.get("path")), "sha256": expected_policy_sha}
            or run.get("profile_reviewer_script")
            != {"path": str(reviewer_ref.get("path")), "sha256": expected_reviewer_sha}
            or request.get("profile_checker_policy_sha256") != expected_policy_sha
            or request.get("profile_policy_sha256") != expected_policy_sha
            or not _file_hash_equal(policy_path, expected_policy_sha)
            or not _file_hash_equal(reviewer_path, expected_reviewer_sha)
        ):
            return {"valid": False}
        old_policy = read_json(policy_path)
        if (
            old_policy.get("status") != "frozen"
            or old_policy.get("revision") != compatibility.get("request_revision")
            or old_policy.get("reviewer_script_sha256") != expected_reviewer_sha
        ):
            return {"valid": False}
        return {
            "valid": True,
            "run_manifest_path": str(run_path.resolve()),
            "run_manifest_sha256": run_sha,
            "profile_checker_policy_path": str(policy_path.resolve()),
            "profile_checker_policy_sha256": expected_policy_sha,
            "profile_reviewer_script_path": str(reviewer_path.resolve()),
            "profile_reviewer_script_sha256": expected_reviewer_sha,
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {"valid": False}


def _validate_initializer(
    request: dict[str, Any],
    data: dict[str, Any],
    split: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    init_path_text = request.get("initializer_path")
    init_sha = request.get("initializer_sha256")
    parent_result_path_text = request.get("parent_result_path")
    parent_result_sha = request.get("parent_result_sha256")
    parent_result_claim = request.get("initializer_parent_trial_result_sha256")
    init = _mapping(data.get("initializer"))
    try:
        if not isinstance(init_path_text, str) or not isinstance(parent_result_path_text, str):
            return False
        init_path, result_path = Path(init_path_text), Path(parent_result_path_text)
        if not _file_hash_equal(init_path, init_sha) or not _file_hash_equal(
            result_path, parent_result_sha
        ):
            return False
        result = read_json(result_path)
        cfg_path = result_path.parent / "config.json"
        if not _file_hash_equal(cfg_path, result.get("config_sha256")):
            return False
        # Parent result files retain the exact raw recipe bytes. The producer
        # identity stores load_config's normalized recipe, so compare to that
        # same Source5B-normalized representation after the byte pin above.
        loader = importlib.import_module("wnn.training.sequence_config").load_config
        parent_config = loader(cfg_path)
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    producer = _mapping(init.get("parent_producer"))
    provenance = _mapping(producer.get("provenance"))
    parent_identity = _mapping(provenance.get("identity"))
    parent_data = _mapping(parent_identity.get("data"))
    parent_cfg = _mapping(producer.get("config"))
    if (
        init.get("schema") != "wnn.sequence-initializer.v1"
        or init.get("checkpoint_sha256") != init_sha
        or init.get("parent_producer_sha256") != canonical_sha256(producer)
        or init.get("parent_identity_sha256") != canonical_sha256(parent_identity)
        or init.get("parent_config_sha256") != canonical_sha256(parent_config)
        or init.get("parent_split_sha256") != request.get("auxiliary_split_sha256")
        or init.get("primary_split_sha256") != request.get("primary_split_sha256")
        or init.get("fit_members_sha256") != canonical_sha256(data.get("fit_members"))
        or init.get("evaluation_members_sha256") != canonical_sha256(data.get("evaluation_members"))
        or init.get("evaluation_role") != "control"
        or init.get("evaluation_source") != "train"
        or init.get("training_mask") != "all_after_warmup"
        or init.get("mask") != "all_after_warmup AND row_index>=99"
        or producer.get("checkpoint_sha256") != init_sha
        or producer.get("split_sha256") != request.get("auxiliary_split_sha256")
        or producer.get("fit_members") != data.get("fit_members")
        or producer.get("teacher") is not None
        or parent_cfg != parent_config
        or parent_cfg.get("kd_weight") != 0
        or parent_cfg.get("hint_weight") != 0
        or parent_cfg.get("precision_mode") != "fp32"
        or _mapping(parent_cfg.get("model")).get("hint_dim") != 0
        or parent_data.get("teacher") is not None
        or parent_data.get("initializer") is not None
        or parent_data.get("fit_members") != data.get("fit_members")
        or parent_data.get("evaluation_members") != data.get("evaluation_members")
        or parent_data.get("split_sha256") != request.get("auxiliary_split_sha256")
        or parent_data.get("primary_split_sha256") != request.get("primary_split_sha256")
        or parent_data.get("evaluation_role") != "control"
        or parent_data.get("evaluation_source") != "train"
        or parent_data.get("training_mask") != "all_after_warmup"
        or parent_data.get("normalization") != data.get("normalization")
        or not _validate_normalization(parent_config, parent_data, parent_identity)
        or init.get("parent_producer_sha256") != canonical_sha256(producer)
        or result.get("status") != "COMPLETE"
        or result.get("stage") not in {1, 2}
        or result.get("fit_model_sha256") != init_sha
        or str((Path(str(result.get("fit_output"))) / "model.pt").resolve())
        != str(init_path.resolve())
        or _mapping(result.get("objective")).get("split_sha256")
        != request.get("auxiliary_split_sha256")
        or _mapping(result.get("objective")).get("primary_split_sha256")
        != request.get("primary_split_sha256")
        or parent_result_claim not in {parent_result_sha, request.get("parent_result_sha256")}
        or parent_result_claim != request.get("parent_result_sha256")
        or request.get("parent_identity_sha256") != canonical_sha256(parent_identity)
    ):
        return False
    # The exact split was independently hash-checked before checking this parent's cohorts.
    expected_fit = _canonical_member_rows(_mapping(split.get("fit")).get("train"))
    expected_control = _canonical_member_rows(_mapping(split.get("evaluation")).get("train"))
    metadata_ok = (
        parent_data.get("fit_members") == expected_fit
        and parent_data.get("evaluation_members") == expected_control
    )
    if not metadata_ok:
        return False
    # Let the exact Source5B loader revalidate the native parent producer,
    # footer-only source identities, portable-registry ancestry, normalizer
    # moments and state-dict compatibility. The profile's initializer record
    # must equal the record the canonical loader derives from the checkpoint.
    try:
        import torch

        from wnn.training.sequence_initializer import load_fitting_initializer
        from wnn.training.sequence_models import build_sequence_model

        checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
        normalization = _mapping(_mapping(checkpoint.get("provenance")).get("normalization"))
        if not normalization:
            return False
        child_config = importlib.import_module("wnn.training.sequence_config").load_config(
            Path(str(request["config_path"]))
        )
        if canonical_sha256(child_config) != request.get("config_identity_sha256"):
            return False
        model = build_sequence_model(_mapping(child_config.get("model")))
        canonical_initializer = load_fitting_initializer(
            init_path,
            model,
            child_config,
            data,
            normalization,
        )
        return canonical_initializer == init
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


def _validate_weighted_pool(
    request: dict[str, Any], data: dict[str, Any], split: dict[str, Any]
) -> bool:
    """Validate a Stage3 cache with the exact Source5B consumer and lineage code."""
    cache_path_text = request.get("teacher_cache_path")
    expected_manifest_sha = request.get("teacher_cache_manifest_sha256")
    data_root_text = request.get("data_root")
    split_path_text = request.get("auxiliary_split_manifest")
    if not all(isinstance(value, str) and value for value in (
        cache_path_text, data_root_text, split_path_text
    )) or not _sha(expected_manifest_sha):
        return False
    cache_path = Path(str(cache_path_text))
    manifest_path = cache_path / "manifest.json"
    try:
        if not _file_hash_equal(manifest_path, expected_manifest_sha):
            return False
        from wnn.splits import load_campaign_split
        from wnn.training.sequence_cache import TeacherCache

        campaign_split = load_campaign_split(
            Path(str(split_path_text)), Path(str(data_root_text))
        )
        fit_members = [("train", group) for group in campaign_split.train_groups]
        cache = TeacherCache(cache_path, campaign_split, fit_members)
        cache.verify_unchanged()
        teacher = _mapping(data.get("teacher"))
        producer = _mapping(cache.metadata.get("producer"))
        # TeacherCache validates every payload hash and recursively validates each
        # weighted-pool ancestor via Source5B common_provenance/validate_producer.
        return bool(
            campaign_split.sha256 == request.get("auxiliary_split_sha256")
            and campaign_split.primary_split_sha256 == request.get("primary_split_sha256")
            and campaign_split.evaluation_source == "train"
            and campaign_split.evaluation_role == "control"
            and fit_members == [tuple(row) for row in data.get("fit_members", [])]
            and cache.identity == teacher
            and cache.identity.get("manifest_sha256") == expected_manifest_sha
            and producer.get("kind") == "weighted_teacher_pool_v1"
            and producer.get("pool_id") == request.get("teacher_pool_id")
            and producer.get("membership") in (None, "fit")
            and _file_hash_equal(manifest_path, expected_manifest_sha)
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False

def _validate_stage3_pair(
    request: dict[str, Any], config: dict[str, Any], split: dict[str, Any]
) -> bool:
    pair_id, role = request.get("pair_id"), request.get("pair_role")
    mate = request.get("pair_mate_trial_number")
    if not isinstance(pair_id, str) or not pair_id or type(mate) is not int:
        return False
    trial_dir = Path(str(request.get("profile_receipt_path"))).resolve().parents[2]
    stage3_dir = trial_dir.parents[1]
    pair_path = stage3_dir / "pairs" / pair_id / "pair.json"
    try:
        pair = read_json(pair_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    variants = pair.get("variants")
    if pair.get("pair_id") != pair_id or not isinstance(variants, list) or len(variants) != 2:
        return False
    indexed: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for variant in variants:
        if not isinstance(variant, dict) or variant.get("pair_role") not in {
            "label_only_control",
            "kd_student",
        }:
            return False
        vrole = str(variant["pair_role"])
        if vrole in indexed:
            return False
        indexed[vrole] = variant
        try:
            cpath = Path(str(variant["config_path"]))
            configs[vrole] = read_json(cpath)
            if not _file_hash_equal(cpath, variant.get("config_sha256")):
                return False
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
    if set(indexed) != {"label_only_control", "kd_student"}:
        return False
    current = indexed.get(str(role))
    counterpart_role = "kd_student" if role == "label_only_control" else "label_only_control"
    counterpart = indexed[counterpart_role]
    trial_number = request.get("trial_number")
    if (
        current is None
        or current.get("trial_number") != trial_number
        or counterpart.get("trial_number") != mate
        or current.get("pair_id") != pair_id
        or counterpart.get("pair_id") != pair_id
    ):
        return False
    control_cfg, student_cfg = configs["label_only_control"], configs["kd_student"]
    c, s = dict(control_cfg), dict(student_cfg)
    cweight, sweight = c.pop("kd_weight", None), s.pop("kd_weight", None)
    return bool(
        cweight == 0
        and isinstance(sweight, (int, float))
        and not isinstance(sweight, bool)
        and sweight > 0
        and c == s
        and configs.get(str(role)) == config
        and _path_equal(current.get("config_path"), Path(str(request.get("config_path"))))
        and current.get("config_sha256") == request.get("config_sha256")
        and pair.get("teacher_pool_id") == request.get("teacher_pool_id")
        and pair.get("teacher_cache_manifest_sha256") is not None
        and isinstance(pair.get("teacher_cache_path"), str)
        and _file_hash_equal(
            Path(str(pair["teacher_cache_path"])) / "manifest.json",
            pair.get("teacher_cache_manifest_sha256"),
        )
        and (
            (
                role == "label_only_control"
                and request.get("teacher_cache_path") is None
                and request.get("teacher_cache_manifest_sha256") is None
            )
            or (
                role == "kd_student"
                and _path_equal(
                    request.get("teacher_cache_path"), Path(str(pair["teacher_cache_path"]))
                )
                and pair.get("teacher_cache_manifest_sha256")
                == request.get("teacher_cache_manifest_sha256")
            )
        )
    )


def review(
    *,
    profile_path: Path,
    request_path: Path,
    attempt_path: Path,
    worker_path: Path,
    deadline_path: Path,
    policy_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    profile, request = read_json(profile_path), read_json(request_path)
    attempt, worker = read_json(attempt_path), read_json(worker_path)
    deadline, policy = read_json(deadline_path), read_json(policy_path)
    contract = read_json(contract_path)
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def check(name: str, ok: bool, evidence: object = None) -> None:
        checks.append({"check": name, "passed": bool(ok), "evidence": evidence})
        if not ok:
            failures.append(name)

    profile_sha = sha256_file(profile_path)
    request_sha = sha256_file(request_path)
    attempt_sha = sha256_file(attempt_path)
    worker_sha = sha256_file(worker_path)
    deadline_sha = sha256_file(deadline_path)
    policy_sha = sha256_file(policy_path)
    contract_sha = sha256_file(contract_path)
    identity = _mapping(profile.get("identity"))
    data = _mapping(identity.get("data"))
    runtime = _mapping(identity.get("runtime"))
    precision = _mapping(runtime.get("precision"))
    stage, trial = request.get("stage"), request.get("trial_number")
    source_root, source_tree_ok = _activate_source_tree(worker, identity, request)
    registry_path, registry_ok = _portable_registry(policy, worker)
    config, config_bytes_ok = _load_config(request)
    normalized_config_sha = _native_config_identity(
        Path(str(request.get("config_path", ""))), identity
    )
    aux, split_ok = _validate_split_and_members(request, data, policy)
    policy_id = _mapping(policy.get("identity"))
    profile_policy = _mapping(policy.get("profile"))
    stage_rules = _mapping(policy.get("stage_rules"))
    source_policy = _mapping(policy_id.get("source_sha256_by_stage"))
    expected_source_sha = source_policy.get(str(stage))

    contract_data = _mapping(contract.get("data_policy"))
    contract_primary_valid = contract_data.get("primary_eval_valid_sequences")

    check("policy_schema", policy.get("schema") == "wnn.multistage-profile-review-policy.v7")
    check("policy_filename", policy_path.name == POLICY_NAME)
    check(
        "policy_frozen", policy.get("status") == "frozen"
    )
    current_policy_binding = (
        request.get("profile_checker_policy_sha256") == policy_sha
        and request.get("profile_policy_sha256") == policy_sha
    )
    compat_evidence: dict[str, Any] = {}
    compatible_request_binding = False
    compat_entries = policy.get("read_only_request_compatibility", [])
    if isinstance(compat_entries, list):
        for item in compat_entries:
            if not isinstance(item, dict) or stage not in item.get("stages", []):
                continue
            if (
                request.get("profile_checker_policy_sha256")
                != item.get("request_checker_policy_sha256")
                or request.get("profile_policy_sha256") != item.get("request_policy_sha256")
            ):
                continue
            evidence = _campaign_run_r2_binding(request_path, request, item)
            if evidence.get("valid") is not True:
                continue
            request_reviewer_sha = request.get("reviewer_script_sha256")
            if request_reviewer_sha is None:
                if (
                    "reviewer_script_sha256" in request
                    or item.get("allow_missing_request_reviewer_script_sha256") is not True
                ):
                    continue
            elif request_reviewer_sha != item.get("request_reviewer_script_sha256"):
                continue
            compat_evidence = evidence
            compatible_request_binding = True
            break
    policy_digest_ok = current_policy_binding or compatible_request_binding
    check("frozen_policy_digest_bound", policy_digest_ok)
    checker_sha = sha256_file(Path(__file__).resolve())
    frozen_checker_sha = policy.get("reviewer_script_sha256")
    check(
        "frozen_checker_digest",
        policy.get("status") == "frozen" and frozen_checker_sha == checker_sha,
    )
    request_checker_sha = request.get("reviewer_script_sha256")
    if request_checker_sha is None and compatible_request_binding:
        request_checker_sha = compat_evidence.get("profile_reviewer_script_sha256")
    check(
        "request_checker_digest",
        request_checker_sha == checker_sha or compatible_request_binding,
    )
    check("request_schema", request.get("schema") == "wnn.multistage-trial-plan.v1")
    check(
        "contract_identity_and_primary_boundary",
        contract.get("schema") == "wnn.multistage-optuna-contract.v1"
        and contract.get("experiment_id") == request.get("experiment_id")
        and contract_data.get("primary_split_sha256")
        == policy_id.get("primary_split_sha256")
        and contract_data.get("primary_fit_train_sequences")
        == policy_id.get("primary_fit_train_sequences")
        and contract_data.get("primary_fit_valid_sequences")
        == policy_id.get("primary_fit_valid_sequences")
        and contract_primary_valid == policy_id.get("primary_eval_valid_sequences")
        and contract_data.get("primary_valid")
        == "no learned blend weights, normalizers, teacher/initializer selection or Optuna objective fitting on primary valid",
    )
    check(
        "request_stage_trial",
        type(stage) is int
        and stage in profile_policy.get("allowed_stages", [])
        and type(trial) is int
        and trial >= 0,
    )
    check("request_anchor_is_not_unreviewed", request.get("anchor_id") is None)
    check("profile_native_schema", profile.get("receipt_schema") == "wnn.sequence-profile.v1")
    check(
        "native_profile_complete_unapproved",
        profile.get("status") == "profile_complete" and profile.get("approved") is False,
    )
    check(
        "native_profile_errors_empty",
        profile.get("errors") == [] and profile.get("error_count") == 0,
    )
    check("native_no_scientific_checkpoint", profile.get("scientific_checkpoint_written") is False)
    check(
        "native_update_count",
        profile.get("updates") == profile_policy.get("updates")
        and request.get("requested_updates") == profile_policy.get("updates")
        and request.get("requested_profile_updates") == profile_policy.get("updates"),
    )
    check(
        "native_profile_request_hash",
        request.get("profile_receipt_sha256") == profile_sha
        and _path_equal(request.get("profile_receipt_path"), profile_path),
    )
    check(
        "native_identity_digest",
        request.get("profile_identity_sha256") == canonical_sha256(identity),
    )
    check("config_exact_bytes", config_bytes_ok)
    check(
        "config_identity",
        normalized_config_sha is not None
        and identity.get("config_sha256") == normalized_config_sha
        and request.get("config_identity_sha256", normalized_config_sha)
        == normalized_config_sha,
    )
    check(
        "primary_protocol_request",
        request.get("primary_split_sha256") == policy_id.get("primary_split_sha256")
        and request.get("objective_scope") == policy_id.get("objective_scope")
        and request.get("primary_valid_used") is False,
    )
    check(
        "auxiliary_manifest_membership_protocol",
        split_ok,
        {
            "fit_count": len(_mapping(aux.get("fit")).get("train", [])),
            "control_count": len(_mapping(aux.get("evaluation")).get("train", [])),
        },
    )
    check(
        "source_exact_for_stage",
        _sha(expected_source_sha)
        and identity.get("source_sha256") == expected_source_sha
        and request.get("source_sha256") == expected_source_sha,
    )
    source_files = identity.get("source_files")
    check(
        "source_files_exact",
        isinstance(source_files, dict)
        and canonical_sha256(source_files) == request.get("source_files_sha256"),
    )
    check("source_tree_exact", source_tree_ok, str(source_root) if source_root else None)
    check(
        "portable_registry_exact",
        registry_ok,
        {
            "path": str(registry_path) if registry_path else None,
            "sha256": _mapping(policy.get("portable_data_registry")).get("sha256"),
        },
    )
    check(
        "source_script_exact",
        isinstance(source_files, dict)
        and _sha(source_files.get("scripts/train_sequence.py"))
        and request.get("source_script_sha256", expected_source_sha)
        in {expected_source_sha, source_files.get("scripts/train_sequence.py")},
    )
    check(
        "normalization_is_fit_only",
        _validate_normalization(config, data, identity, profile_path),
    )
    check(
        "canonical_campaign_split_and_live_sources",
        _validate_canonical_campaign_split(request, data),
    )
    check(
        "no_valid_selection_reporting",
        _mapping(profile.get("evaluation_annotation")).get("no_eval_during_fit") is True
        and "validation_reporting" not in _mapping(profile.get("evaluation_annotation")),
    )

    check(
        "one_process_world",
        request.get("requested_world_size") == 1
        and request.get("world_size") == 1
        and identity.get("world_size") == 1
        and runtime.get("world_size") == 1
        and profile_policy.get("world_size") == 1,
    )
    check(
        "strict_fp32_config",
        config.get("precision_mode") == "fp32"
        and _mapping(config.get("model")).get("precision_mode", config.get("precision_mode"))
        == "fp32",
    )
    check(
        "strict_fp32_runtime",
        precision.get("mode") == "fp32"
        and precision.get("parameter_dtype") == "float32"
        and precision.get("gradient_dtype") == "float32"
        and precision.get("optimizer_state_dtype") == "float32"
        and runtime.get("matmul_precision") == "highest"
        and runtime.get("matmul_tf32") is False
        and runtime.get("cudnn_tf32") is False
        and runtime.get("autocast") is False
        and runtime.get("amp_scaler") is False,
    )
    check(
        "no_hint_heads",
        config.get("hint_weight") == 0 and _mapping(config.get("model")).get("hint_dim") == 0,
    )
    losses = _mapping(profile.get("losses"))
    kd_weight, hint_weight = config.get("kd_weight"), config.get("hint_weight")
    weights_ok = (
        _finite(kd_weight)
        and _finite(hint_weight)
        and float(kd_weight) >= 0
        and float(hint_weight) >= 0
    )
    loss_ok = (
        weights_ok
        and set(losses) == {"label", "distillation", "hint", "total"}
        and all(_finite(v) and float(v) >= 0 for v in losses.values())
    )
    if loss_ok:
        expected_total = (
            float(losses["label"])
            + float(kd_weight) * float(losses["distillation"])
            + float(hint_weight) * float(losses["hint"])
        )
        loss_ok = math.isclose(float(losses["total"]), expected_total, rel_tol=1e-6, abs_tol=1e-7)
    check("profile_loss_arithmetic_finite", loss_ok)

    if stage in {1, 2}:
        rule = _mapping(stage_rules.get(f"stage{stage}"))
        check(
            f"stage{stage}_label_only",
            config.get("kd_weight") == rule.get("kd_weight")
            and config.get("hint_weight") == rule.get("hint_weight")
            and data.get("teacher") is None,
        )
        if stage == 1:
            check(
                "stage1_no_initializer",
                request.get("initializer_path") is None
                and request.get("initializer_sha256") is None
                and data.get("initializer") is None,
            )
        else:
            check(
                "stage2_exact_initializer_ancestry",
                _validate_initializer(request, data, aux, config),
            )
            check(
                "stage2_command_is_fresh_optimizer",
                "--resume" not in worker.get("command", [])
                and "--optimizer-state" not in worker.get("command", [])
                and request.get("initializer_path") is not None,
            )
    else:
        role = request.get("pair_role")
        stage3_rule = _mapping(stage_rules.get("stage3"))
        check(
            "stage3_weighted_pool_policy_frozen",
            stage3_rule.get("teacher_pool_kind") == "weighted_teacher_pool_v1"
            and stage3_rule.get("weighted_pool_method") == "ordered_fp32_weighted_sum"
            and stage3_rule.get("weighted_pool_target_contract") == WEIGHTED_POOL_TARGET_CONTRACT,
        )
        check(
            "stage3_pair_role",
            role in {"label_only_control", "kd_student"}
            and _validate_stage3_pair(request, config, aux),
        )
        check(
            "stage3_student_has_no_initializer",
            request.get("initializer_path") is None
            and request.get("initializer_sha256") is None
            and data.get("initializer") is None,
        )
        if role == "label_only_control":
            check(
                "stage3_control_has_no_cache",
                config.get("kd_weight") == 0
                and data.get("teacher") is None
                and request.get("teacher_cache_path") is None
                and request.get("teacher_cache_manifest_sha256") is None,
            )
        elif role == "kd_student":
            check(
                "stage3_student_has_weighted_fit_only_cache",
                isinstance(config.get("kd_weight"), (int, float))
                and not isinstance(config.get("kd_weight"), bool)
                and config.get("kd_weight") > 0
                and _validate_weighted_pool(request, data, aux),
            )
        else:
            check("stage3_known_pair_role", False)

    devices = runtime.get("devices")
    expected_uuids = request.get("expected_device_uuid_set")
    device_uuid = (
        expected_uuids[0] if isinstance(expected_uuids, list) and len(expected_uuids) == 1 else None
    )
    check(
        "device_uuid_singleton",
        _canonical_gpu_uuid(device_uuid) is not None
        and isinstance(devices, list)
        and len(devices) == 1
        and _same_gpu_uuid(_mapping(devices[0]).get("uuid"), device_uuid),
    )
    check(
        "device_is_admitted_b300",
        isinstance(devices, list)
        and len(devices) == 1
        and "B300" in str(_mapping(devices[0]).get("name", "")).upper()
        and request.get("requested_device_set") == [device_uuid],
    )
    torch_total = request.get("expected_device_torch_total_memory_bytes")
    nvml_total = request.get("expected_device_total_memory_bytes")
    check(
        "torch_memory_observation",
        type(torch_total) is int
        and _mapping(devices[0] if isinstance(devices, list) and devices else {}).get(
            "total_memory_bytes"
        )
        == torch_total
        and isinstance(profile.get("device_total_memory_bytes"), list)
        and profile["device_total_memory_bytes"] == [torch_total],
    )
    check(
        "nvml_memory_observation",
        type(nvml_total) is int
        and nvml_total == _mapping(policy.get("memory_identity")).get("nvml_total_memory_bytes")
        and torch_total == _mapping(policy.get("memory_identity")).get("torch_total_memory_bytes"),
    )
    memory_rows = profile.get("memory_by_rank")
    row = (
        _mapping(memory_rows[0]) if isinstance(memory_rows, list) and len(memory_rows) == 1 else {}
    )
    allocated, reserved = row.get("peak_allocated_bytes"), row.get("peak_reserved_bytes")
    rss = row.get("host_peak_rss_bytes")
    check("one_rank_memory", row.get("rank") == 0)
    check(
        "allocated_cap",
        type(allocated) is int and 0 < allocated <= profile_policy.get("max_allocated_bytes", 0),
    )
    check(
        "reserved_cap",
        type(reserved) is int
        and allocated <= reserved <= profile_policy.get("max_reserved_bytes", 0),
    )
    check(
        "host_rss_cap",
        type(rss) is int
        and 0 < rss <= profile_policy.get("max_host_rss_bytes", 0)
        and profile.get("host_peak_rss_bytes") == [rss],
    )

    nvml = _mapping(attempt.get("nvml_pre_profile"))
    check(
        "preprofile_nvml_device",
        nvml.get("status") == "observed"
        and nvml.get("device_uuid") == device_uuid
        and nvml.get("total_memory_bytes") == nvml_total
        and _mapping(worker.get("nvml_pre_profile")) == nvml,
    )
    free = nvml.get("free_memory_bytes")
    check(
        "nvml_free_headroom",
        type(free) is int
        and type(reserved) is int
        and free >= profile_policy.get("min_nvml_free_before_bytes", 0) + reserved
        and free - reserved >= profile_policy.get("min_nvml_free_after_peak_bytes", 0),
    )
    check(
        "torch_memory_uuid_match",
        row.get("device_total_memory_bytes") == torch_total
        and _same_gpu_uuid(
            _mapping(devices[0] if isinstance(devices, list) and devices else {}).get("uuid"),
            device_uuid,
        ),
    )

    expected_cpu = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    expected_env = _mapping(profile_policy.get("effective_thread_environment"))
    check(
        "thread_environment_request_worker",
        request.get("cpu_thread_env") == expected_cpu
        and request.get("effective_thread_environment") == expected_env
        and worker.get("cpu_thread_env") == expected_cpu
        and worker.get("effective_thread_environment") == expected_env,
    )

    owner_path_text = request.get("worker_owner_receipt_path")
    try:
        owner_path = Path(str(owner_path_text))
        owner_receipt = read_json(owner_path)
        owner_hash_ok = _file_hash_equal(owner_path, request.get("worker_owner_receipt_sha256"))
    except (OSError, ValueError, json.JSONDecodeError):
        owner_receipt, owner_hash_ok = {}, False
    owner, child = _mapping(worker.get("owner")), _mapping(worker.get("child"))
    check(
        "worker_receipt_identity",
        worker.get("schema") == "wnn.multistage-owned-process.v1"
        and worker.get("stage") == stage
        and worker.get("trial_number") == trial
        and worker.get("mode") == "profile",
    )
    check(
        "worker_terminal_success",
        worker.get("status") == "completed"
        and worker.get("exit_code") == 0
        and worker.get("timed_out") is False
        and _process_tuple(child)
        and child.get("pid") == child.get("pgid"),
    )
    check(
        "owned_parent_process_bound",
        owner_hash_ok
        and _same_process(request.get("worker_owned_process"), owner)
        and _same_process(owner_receipt, owner)
        and owner_receipt.get("command_sha256")
        == request.get("worker_owned_process_command_sha256"),
    )
    check(
        "owned_profile_child_bound",
        _same_process(request.get("profile_child_process"), child)
        and worker.get("command_sha256") == request.get("profile_child_process_command_sha256"),
    )
    command = worker.get("command")
    command_ok = isinstance(command, list) and all(isinstance(v, str) for v in command)
    if command_ok:
        command_ok = hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest() == worker.get("command_sha256")
        command_text = " ".join(command).lower()
        command_ok = (
            command_ok
            and "scripts/train_sequence.py" in command_text
            and "--resume" not in command
            and "--profile-receipt" not in command
        )
        for flag, expected in (
            ("--config", request.get("config_path")),
            ("--data-root", request.get("data_root")),
            ("--split-manifest", request.get("auxiliary_split_manifest")),
            ("--output", request.get("profile_output_path")),
            ("--profile-updates", str(profile_policy.get("updates"))),
        ):
            try:
                actual = command[command.index(flag) + 1]
            except (ValueError, IndexError):
                command_ok = False
                continue
            if flag == "--profile-updates":
                command_ok = command_ok and actual == expected
            else:
                command_ok = command_ok and _path_equal(actual, Path(str(expected)))
        for flag, expected in (
            ("--initializer", request.get("initializer_path")),
            ("--teacher-cache", request.get("teacher_cache_path")),
        ):
            present = flag in command
            command_ok = command_ok and (present == (expected is not None))
            if present and expected is not None:
                try:
                    command_ok = command_ok and _path_equal(
                        command[command.index(flag) + 1], Path(str(expected))
                    )
                except IndexError:
                    command_ok = False
        forbidden = (
            "nvidia-cuda-mps-control",
            "set_active_thread_percentage",
            "--gpu-reset",
            "killall",
            "pkill",
        )
        command_ok = command_ok and not any(token in command_text for token in forbidden)
    check("exact_profile_command", bool(command_ok))

    attempt_terminal = _mapping(attempt.get("terminal_attempt"))
    check(
        "attempt_terminal_exact",
        attempt.get("schema") == "wnn.multistage-profile-attempt.v1"
        and attempt.get("stage") == stage
        and attempt.get("trial_number") == trial
        and attempt.get("profile_id") == f"stage{stage}-trial-{trial:06d}"
        and attempt.get("status") == "profile_complete"
        and attempt.get("exit_code") == 0
        and attempt_terminal.get("status") == "profile_complete"
        and attempt_terminal.get("exit_code") == 0
        and attempt.get("native_profile_sha256") == profile_sha,
    )
    check(
        "attempt_receipt_request_binding",
        request.get("profile_attempt_receipt_sha256") == attempt_sha
        and _path_equal(request.get("profile_attempt_receipt_path"), attempt_path)
        and attempt.get("worker_receipt_sha256") == worker_sha
        and _path_equal(attempt.get("worker_receipt_path"), worker_path),
    )
    plan_path_text = request.get("trial_plan_path")
    try:
        plan_path = Path(str(plan_path_text))
        plan_hash_ok = _file_hash_equal(plan_path, request.get("trial_plan_sha256"))
    except OSError:
        plan_hash_ok = False
    check(
        "immutable_trial_plan",
        plan_hash_ok
        and attempt.get("trial_plan_sha256") == request.get("trial_plan_sha256")
        and _path_equal(attempt.get("trial_plan_path"), plan_path),
    )
    try:
        budget_path = Path(str(attempt.get("budget_snapshot_path")))
        budget = read_json(budget_path)
        budget_ok = _file_hash_equal(budget_path, attempt.get("budget_snapshot_sha256"))
    except (OSError, ValueError, json.JSONDecodeError):
        budget, budget_ok = {}, False
    profiles, fits = budget.get("profile_attempts"), budget.get("fit_attempts")
    campaign_bounds = _mapping(policy.get("campaign_bounds"))
    max_profiles = campaign_bounds.get("max_profile_attempts")
    max_fits = campaign_bounds.get("max_fit_attempts")
    stage_fit_cap_keys = campaign_bounds.get("stage_fit_cap_keys", [])
    check(
        "attempt_budget_snapshot",
        budget_ok
        and budget.get("schema") == "wnn.multistage-budget.v1"
        and isinstance(profiles, list)
        and isinstance(fits, list)
        and len(profiles) == attempt.get("profile_attempt_count_at_completion")
        and len(fits) == attempt.get("fit_attempt_count_at_completion")
        and len(profiles) <= max_profiles
        and len(fits) <= max_fits
        and budget.get("max_profile_attempts") == max_profiles
        and budget.get("max_fit_attempts") == max_fits
        and set(_mapping(budget.get("stage_fit_caps")))
        == set(stage_fit_cap_keys),
    )
    check(
        "attempt_counts_match_budget",
        attempt.get("max_profile_attempts") == max_profiles
        and attempt.get("max_fit_attempts") == max_fits
        and attempt.get("stage_fit_caps") == budget.get("stage_fit_caps"),
    )
    stage_caps = _mapping(budget.get("stage_fit_caps"))
    stage_count_ok = (
        set(stage_caps) == set(stage_fit_cap_keys)
        and all(type(value) is int and value >= 0 for value in stage_caps.values())
        and sum(stage_caps.values()) <= max_fits
        and isinstance(fits, list)
        and all(
            sum(_mapping(fit).get("stage") == int(key) for fit in fits) <= cap
            for key, cap in stage_caps.items()
        )
    )
    check("stage_fit_cap_counts", stage_count_ok)

    check(
        "deadline_file_binding",
        request.get("deadline_receipt_sha256") == deadline_sha
        and _path_equal(request.get("deadline_receipt_path"), deadline_path)
        and attempt.get("campaign_deadline_sha256") == deadline_sha
        and _path_equal(attempt.get("campaign_deadline_path"), deadline_path),
    )
    start, end = worker.get("started_at_epoch"), worker.get("finished_at_epoch")
    actual_wall = attempt.get("actual_wall_seconds")
    check(
        "profile_wall_limit",
        _finite(start)
        and _finite(end)
        and _finite(actual_wall)
        and 0 < actual_wall <= profile_policy.get("max_wall_seconds", 0)
        and abs(float(end) - float(start) - float(actual_wall)) <= 1.0
        and attempt_terminal.get("wall_limit_seconds", profile_policy.get("max_wall_seconds"))
        <= profile_policy.get("max_wall_seconds", 0),
    )
    first_profile = deadline.get("first_profile_epoch")
    campaign_wall = _mapping(policy.get("campaign_bounds")).get(
        "campaign_wall_seconds_from_first_profile"
    )
    reserve_seconds = _mapping(policy.get("campaign_bounds")).get("collection_reserve_seconds")
    check(
        "deadline_and_collection_reserve",
        deadline.get("schema") == "wnn.multistage-deadline.v1"
        and deadline.get("campaign_wall_seconds") == campaign_wall
        and deadline.get("collection_reserve_seconds") == reserve_seconds
        and _finite(first_profile)
        and deadline.get("deadline_epoch") == first_profile + campaign_wall
        and deadline.get("reserve_start_epoch") == deadline.get("deadline_epoch") - reserve_seconds
        and _finite(end)
        and end <= deadline.get("reserve_start_epoch")
        and end <= deadline.get("deadline_epoch"),
    )

    rank_refs, ranks_ok = _rank_logs(request, int(profile_policy.get("updates", 0)))
    check("complete_finite_rank0_log", ranks_ok, rank_refs)
    if isinstance(source_files, dict) and isinstance(
        source_files.get("scripts/train_sequence.py"), str
    ):
        check("script_hash_present", _sha(source_files["scripts/train_sequence.py"]))
    else:
        check("script_hash_present", False)

    approved = not failures
    common = {
        "status": "approved" if approved else "rejected",
        "reviewer": "multistage_independent_critic" if stage == 1 else "independent",
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "trial_number": trial,
        "native_profile_sha256": profile_sha,
        "identity_sha256": canonical_sha256(identity),
        "trial_request_sha256": request_sha,
        "profile_attempt_receipt_sha256": attempt_sha,
        "config_sha256": request.get("config_sha256"),
        "expected_device_total_memory_bytes": nvml_total,
        "expected_device_torch_total_memory_bytes": torch_total,
        "initializer_parent_trial_result_sha256": request.get(
            "initializer_parent_trial_result_sha256"
        ),
        "auxiliary_split_sha256": request.get("auxiliary_split_sha256"),
        "primary_split_sha256": request.get("primary_split_sha256"),
        "source_sha256": request.get("source_sha256"),
        "initializer_path": request.get("initializer_path"),
        "initializer_sha256": request.get("initializer_sha256"),
        "teacher_cache_manifest_sha256": request.get("teacher_cache_manifest_sha256"),
        "review_policy_path": str(policy_path.resolve()),
        "review_policy_sha256": policy_sha,
        "reviewer_policy_sha256": policy_sha,
        "reviewer_policy_script_sha256": sha256_file(Path(__file__).resolve()),
        "request_subject_review_binding": {
            "profile_checker_policy_sha256": request.get("profile_checker_policy_sha256"),
            "profile_policy_sha256": request.get("profile_policy_sha256"),
            "reviewer_script_sha256": request_checker_sha,
            "read_only_compatibility_match": compatible_request_binding,
            "campaign_run_manifest_path": compat_evidence.get("run_manifest_path"),
            "campaign_run_manifest_sha256": compat_evidence.get("run_manifest_sha256"),
            "campaign_profile_checker_policy_path": compat_evidence.get(
                "profile_checker_policy_path"
            ),
            "campaign_profile_checker_policy_sha256": compat_evidence.get(
                "profile_checker_policy_sha256"
            ),
            "campaign_profile_reviewer_script_path": compat_evidence.get(
                "profile_reviewer_script_path"
            ),
            "campaign_profile_reviewer_script_sha256": compat_evidence.get(
                "profile_reviewer_script_sha256"
            ),
            "request_reviewer_script_key_absent": (
                compatible_request_binding and "reviewer_script_sha256" not in request
            ),
        },
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": contract_sha,
        "reviewer_script_sha256": sha256_file(Path(__file__).resolve()),
        "worker_receipt_sha256": worker_sha,
        "deadline_receipt_sha256": deadline_sha,
        "rank_log_refs": rank_refs,
        "expected_device_uuid_set": expected_uuids,
        "cpu_thread_env": request.get("cpu_thread_env"),
        "effective_thread_environment": request.get("effective_thread_environment"),
        "worker_owned_process": request.get("worker_owned_process"),
        "worker_owned_process_command_sha256": request.get("worker_owned_process_command_sha256"),
        "profile_child_process": request.get("profile_child_process"),
        "profile_child_process_command_sha256": request.get("profile_child_process_command_sha256"),
        "worker_owner_receipt_sha256": request.get("worker_owner_receipt_sha256"),
        "anchor_id": request.get("anchor_id"),
        "checks": checks,
        "failed_checks": failures,
        "native_profile_preserved": True,
        "engineering_admission_only": True,
        "scientific_candidate_promotion": False,
        "fixed_final_objective_gate": {
            "status": "required_not_attested",
            "objective": "inner-train control WP from the exact configured-final model.pt",
            "required_checkpoint_policy": "configured-final-epoch",
            "best_observed_control_metrics": "diagnostic_only",
            "profile_reviewer_attests_execution": False,
        },
    }
    if stage == 1:
        common["schema"] = "wnn.multistage-profile-review.v1"
        common["profile_policy_sha256"] = request.get("profile_policy_sha256")
        common["aux_split_sha256"] = request.get("auxiliary_split_sha256")
        common["anchor_review_receipt_required"] = False
    else:
        common["schema"] = "wnn.multistage-successor-profile-review.v1"
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--trial-request", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path, required=True)
    parser.add_argument("--worker-receipt", type=Path, required=True)
    parser.add_argument("--deadline-receipt", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.policy.name != POLICY_NAME:
        raise SystemExit(f"policy filename must be {POLICY_NAME}")
    result = review(
        profile_path=args.profile,
        request_path=args.trial_request,
        attempt_path=args.attempt_receipt,
        worker_path=args.worker_receipt,
        deadline_path=args.deadline_receipt,
        policy_path=args.policy,
        contract_path=args.contract,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "failed_checks": result["failed_checks"],
            }
        )
    )


if __name__ == "__main__":
    main()
