"""Review an unapproved Source4 quality-horizon engineering profile."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCE4 = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
HELPER_SHA = "c059d0d05711badb52779f31f9b69cfcfe7badb16628da6a0c1e1c2b3b3be279"
SCHEMA = "wnn.quality-horizon-profile-review.v1"
CPU4 = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
UUID_RE = re.compile(
    r"^(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", re.I
)


# fmt: off
def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def _canon(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()

def _m(value: object) -> Mapping[str, Any]: return value if isinstance(value, dict) else {}

def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

def _uuid(value: object) -> str | None:
    match = UUID_RE.fullmatch(value) if isinstance(value, str) else None
    return match.group(1).lower() if match else None

def _rank_log(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_000_000:
        return False
    try:
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    keys = ("total_loss", "label_loss", "gradient_norm_preclip", "learning_rate")
    return len(rows) == 50 and all(isinstance(row, dict) and row.get("rank") == 0 and row.get("step") == step and all(_finite(row.get(k)) and float(row[k]) >= 0 for k in keys) for step, row in enumerate(rows, 1))

def _precision(config: Mapping[str, Any], runtime: Mapping[str, Any], helper: Path) -> bool:
    if helper.is_symlink() or not helper.is_file() or _sha(helper) != HELPER_SHA:
        return False
    spec = importlib.util.spec_from_file_location("quality_profile_precision_contract", helper)
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.validate_precision_contract(config, {"precision": runtime.get("precision")}, dict(runtime))
    except (TypeError, ValueError):
        return False
    return all(runtime.get(k) is False for k in ("autocast", "amp_scaler", "matmul_tf32", "cudnn_tf32"))

def _members(data: Mapping[str, Any], path: Path) -> bool:
    try:
        split = _read(path)
        fit, evaluation = _m(split.get("fit")), _m(split.get("evaluation"))
        train, valid = fit.get("train"), evaluation.get("valid")
        expected_fit = [["train", n] for n in train] if isinstance(train, list) else []
        expected_valid = [["valid", n] for n in valid] if isinstance(valid, list) else []
        return (_sha(path) == SPLIT and fit.get("valid") == [] and len(expected_fit) == 10325 and len(expected_valid) == 1873
            and data.get("split_sha256") == SPLIT and data.get("primary_split_sha256") in (None, SPLIT)
            and data.get("fit_members") == expected_fit and data.get("evaluation_members") == expected_valid
            and data.get("evaluation_source") == data.get("evaluation_role") == "valid" and data.get("evaluation_mask") == "is_scored"
            and all(data.get(k) is None for k in ("teacher", "initializer", "hint_members")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False

def _config_ok(config: Mapping[str, Any], expected: Mapping[str, object], request: Mapping[str, Any], path: Path, identity: Mapping[str, Any]) -> bool:
    proposal = json.loads(json.dumps(expected, allow_nan=False))
    if isinstance(proposal.get("model"), dict):
        proposal["model"].pop("precision_mode", None)
    return (config == proposal and "precision_mode" not in _m(config.get("model")) and config.get("precision_mode") == "fp32"
        and identity.get("config_sha256") == _canon(config) and request.get("config_sha256") == _sha(path))

def _normalization(config: Mapping[str, Any], data: Mapping[str, Any]) -> bool:
    model, norm = _m(config.get("model")), _m(data.get("normalization"))
    return (model.get("input_feature_mode") == "raw112" and config.get("normalization_scope") == norm.get("scope") == "prefix"
        and config.get("normalization_sequences") == norm.get("fit_prefix_sequences") == norm.get("sequence_count") == 32
        and set(_m(norm.get("source_fingerprints"))) == {"datasets/train.parquet"})

def _device(profile: Mapping[str, Any], identity: Mapping[str, Any], request: Mapping[str, Any], inventory: Mapping[str, Any], config: Mapping[str, Any], helper: Path) -> bool:
    runtime = _m(identity.get("runtime"))
    devices, rows = runtime.get("devices"), profile.get("memory_by_rank")
    gpu = devices[0] if isinstance(devices, list) and len(devices) == 1 else {}
    row = rows[0] if isinstance(rows, list) and len(rows) == 1 else {}
    alloc, reserved, rss = (_m(row).get(k) for k in ("peak_allocated_bytes", "peak_reserved_bytes", "host_peak_rss_bytes"))
    wanted = _uuid(request.get("gpu_uuid"))
    indexed = inventory.get("uuid_by_index")
    values = indexed.values() if isinstance(indexed, dict) else indexed if isinstance(indexed, list) else []
    observed = [_uuid(v) for v in values]
    return (wanted is not None and wanted == _uuid(_m(gpu).get("uuid")) and wanted in observed and "B300" in str(_m(gpu).get("name", "")).upper()
        and identity.get("world_size") == runtime.get("world_size") == runtime.get("visible_devices") == 1 and _m(row).get("rank") == 0
        and type(alloc) is int and type(reserved) is int and 0 < alloc <= reserved <= 64 * (1 << 30)
        and type(rss) is int and 0 < rss <= 24 * (1 << 30) and _m(row).get("device_total_memory_bytes") == _m(gpu).get("total_memory_bytes")
        and profile.get("peak_memory_bytes") == alloc and profile.get("peak_reserved_bytes") == reserved and _precision(config, runtime, helper))

def _process(value: object) -> bool:
    row = _m(value)
    return (type(row.get("pid")) is int and row["pid"] > 0 and type(row.get("pgid")) is int and row["pgid"] > 0
        and str(row.get("birth", "")) not in ("", "0"))

def _worker(request: Mapping[str, Any], child: Mapping[str, Any]) -> bool:
    argv, native = request.get("argv"), request.get("native_argv")
    env, actual = _m(request.get("environment")), _m(child.get("effective_thread_environment"))
    start, end = child.get("started_at_epoch"), child.get("finished_at_epoch")
    bounded = isinstance(argv, list) and "175s" in argv and any(Path(str(v)).name == "timeout" for v in argv)
    updates = isinstance(native, list) and any((v == "--profile-updates" and i + 1 < len(native) and native[i+1] == "50") or v == "--profile-updates=50" for i, v in enumerate(native))
    return (child.get("schema") == "wnn.multistage-owned-process.v1" and child.get("status") == "completed" and child.get("exit_code") == 0 and child.get("timed_out") is False
        and child.get("command") == argv and bounded and updates and isinstance(argv, list) and isinstance(native, list) and argv[-len(native):] == native
        and _process(child.get("owner")) and _process(child.get("child"))
        and actual == CPU4 and all(env.get(k) == v for k, v in CPU4.items()) and _uuid(env.get("CUDA_VISIBLE_DEVICES")) == _uuid(request.get("gpu_uuid"))
        and isinstance(start, (int, float)) and isinstance(end, (int, float)) and 0 < end - start <= 180
        and not any(v in (native or []) for v in ("--resume", "--initializer", "--teacher-cache", "--teacher")))

def review_profile(profile_path: Path, rank0_path: Path, config_path: Path, request_path: Path, child_path: Path, split_path: Path, inventory_path: Path, expected_config: Mapping[str, object], precision_helper_path: Path | None = None) -> dict[str, Any]:
    """Return a separate engineering review without changing native approval."""
    try:
        profile, request, child, inventory = (_read(p) for p in (profile_path, request_path, child_path, inventory_path))
        config = _read(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"schema": SCHEMA, "status": "rejected", "approved": False, "failed_checks": ["input_files"], "error": str(exc), "native_profile_preserved": True}
    identity = _m(profile.get("identity"))
    data, native = _m(identity.get("data")), request.get("native_argv")
    helper = precision_helper_path or Path(__file__).with_name("primary_refit_precision_contract_r5.py")
    if not helper.is_file():
        helper = Path(request.get("source_root", "")) / "primary_refit_precision_contract_r5.py"
    if not helper.is_file():
        helper = Path(__file__).resolve().parents[1] / "research/experiment_drafts/multistage-optuna-b300-20261031/primary_refit_precision_contract_r5.py"
    losses = _m(profile.get("losses"))
    annotation = _m(profile.get("evaluation_annotation"))
    clean = isinstance(native, list) and not any(v in native for v in ("--resume", "--initializer", "--teacher-cache", "--teacher"))
    checks = {
        "source4_identity": identity.get("source_sha256") == SOURCE4 and _canon(identity.get("source_files")) == SOURCE4,
        "source4_config_identity": _config_ok(config, expected_config, request, config_path, identity),
        "official_membership_and_ancestry": _members(data, split_path),
        "train_only_prefix32_normalization": _normalization(config, data),
        "no_teacher_initializer_or_resume": config.get("kd_weight") == config.get("hint_weight") == 0 and _m(config.get("model")).get("hint_dim") == 0 and clean,
        "native_unapproved_no_eval_no_checkpoint": profile.get("receipt_schema") == "wnn.sequence-profile.v1" and profile.get("status") == "profile_complete" and profile.get("approved") is False and annotation.get("no_eval_during_fit") is True and profile.get("scientific_checkpoint_written") is False,
        "50_updates_finite_loss_gradient_lr": profile.get("updates") == 50 and profile.get("errors") == [] and profile.get("error_count") == 0 and set(losses) == {"label", "distillation", "hint", "total"} and all(_finite(v) and float(v) >= 0 for v in losses.values()) and _rank_log(rank0_path) and _finite(profile.get("elapsed_seconds")) and 0 < float(profile["elapsed_seconds"]) <= 180,
        "profiled_b300_uuid_memory_fp32": _device(profile, identity, request, inventory, config, helper),
        "owned_cpu4_exit0_bounded_profile": _worker(request, child),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    return {"schema": SCHEMA, "status": "engineering_checks_passed" if not failed else "rejected", "approved": False,
        "profile_approval": False, "scientific_candidate_promotion": False, "native_profile_preserved": True,
        "native_profile_sha256": _sha(profile_path), "config_sha256": _sha(config_path), "source_sha256": identity.get("source_sha256"),
        "primary_split_sha256": SPLIT, "checks": checks, "failed_checks": failed}
# fmt: on
