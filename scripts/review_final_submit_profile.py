"""Review one final-submit engineering profile without launching a worker.

The CLI accepts a pre-run request, the frozen TL3 plan, the final-fit manifest,
the receiver source audit, a source capsule, a terminal process receipt, and either
the native Source4 receipt or legacy TL3 profile log/provenance. It writes a
separate review and, only when every check passes, an approved receipt copy.
The native profile is never rewritten.

Request contract: ``wnn.final-submit-profile-request.v1``. Required fields are
``mode`` (``native_source4_lstm`` or ``legacy_tl3``), ``node_id``, ``plan_sha256``,
``recipe_sha256``, ``argv``/``argv_sha256``, ``source_capsule_sha256``,
``full_source_code_sha256``, ``config_file_sha256`` and
``config_identity_sha256`` for native mode, ``final_fit_manifest_sha256``,
``source_audit_sha256``, ``fit_members_sha256``, ``fit_member_count`` (12480),
``evaluation_member_count`` (0), ``profile_updates`` (50 native) or
``profile_batches`` (ceil(50 / updates-per-batch), legacy), ``warmup_batches``
(0 native, 1 legacy), and ``gpu_uuid``. The caller passes
the request's pre-frozen byte hash separately as ``expected_request_sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "wnn.final-submit-profile-request.v1"
REVIEW_SCHEMA = "wnn.final-submit-profile-review.v1"
APPROVAL_SCHEMA = "wnn.final-submit-profile-approval.v1"
PLAN_SCHEMA = "wnn.final-submit-recipes.v1"
MEMBERSHIP_SCHEMA = "wnn.final-fit-membership.v1"
SOURCE_AUDIT_SCHEMA = "wnn.final-submit-source-validation.v1"
SOURCE_AUDIT_SHA256 = "1e8a8a794a0b8d9677a76030e570681074503d38aa2d54b80e36067566b94ce0"
SOURCE_CAPSULE_SHA256 = "8381c0d6a6333cbc32dca2187516994c5a80832193bcf2531bd606511bc7ccb4"
TRAIN_GROUPS = 10_607
VALID_GROUPS = 1_873
FIT_GROUPS = TRAIN_GROUPS + VALID_GROUPS
TRAIN_ROWS = TRAIN_GROUPS * 20_000
VALID_ROWS = VALID_GROUPS * 20_000
HEX256 = re.compile(r"^[0-9a-f]{64}$")
UUID = re.compile(
    r"^(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", re.I
)


def _read_json(path: Path, limit: int = 8_000_000) -> dict[str, Any]:
    """Read a bounded, regular JSON object and reject links and oversized inputs."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > limit:
        raise ValueError(f"JSON input is empty or exceeds {limit} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    """Hash a file in bounded blocks."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_source_ok(request: Mapping[str, Any], capsule_path: Path) -> bool:
    """Verify the pre-run source map against the exact source archive bytes."""
    expected = _mapping(request.get("expected_source")).get("files")
    if (
        not isinstance(expected, Mapping)
        or not expected
        or request.get("full_source_code_sha256") != _canonical_sha256(expected)
        or capsule_path.is_symlink()
        or not capsule_path.is_file()
        or capsule_path.stat().st_size > 32_000_000
    ):
        return False
    names = list(expected)
    if any(
        not isinstance(name, str)
        or name.startswith("/")
        or ".." in Path(name).parts
        or not HEX256.fullmatch(str(expected[name]))
        for name in names
    ):
        return False
    try:
        with tarfile.open(capsule_path, mode="r:*") as archive:
            members = archive.getmembers()
            by_name = {member.name: member for member in members}
            if len(by_name) != len(members):
                return False
            for name, expected_sha256 in expected.items():
                member = by_name.get(name)
                if member is None or not member.isfile() or member.size > 32_000_000:
                    return False
                stream = archive.extractfile(member)
                if stream is None:
                    return False
                digest = hashlib.sha256()
                with stream:
                    while block := stream.read(1 << 20):
                        digest.update(block)
                if digest.hexdigest() != expected_sha256:
                    return False
    except (OSError, tarfile.TarError, ValueError):
        return False
    return True


def _canonical_sha256(value: object) -> str:
    """Hash JSON values with the repository's stable canonical encoding."""
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _native_config_identity_sha256(config: Mapping[str, Any]) -> str:
    """Match Source4's post-validation addition of model.precision_mode."""
    normalized = dict(config)
    model = dict(_mapping(config.get("model")))
    model["precision_mode"] = config.get("precision_mode")
    normalized["model"] = model
    return _canonical_sha256(normalized)


def _finite(value: object, *, positive: bool = False) -> bool:
    """Accept finite JSON numeric values, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (float(value) > 0 if positive else float(value) >= 0)
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _uuid_identity(value: object) -> str | None:
    """Normalize a GPU UUID while requiring its complete canonical shape."""
    if not isinstance(value, str):
        return None
    match = UUID.fullmatch(value)
    return match.group(1).lower() if match else None


def _member_hash(groups: Mapping[str, list[int]]) -> str:
    members = [[source, group] for source in ("train", "valid") for group in groups[source]]
    raw = json.dumps(members, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _expected_plan_node(plan: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("plan_sha256") != request.get("plan_sha256"):
        return False
    plan_copy = dict(plan)
    declared_sha = plan_copy.pop("plan_sha256", None)
    if declared_sha != _canonical_sha256(plan_copy):
        return False
    mode, node_id = request.get("mode"), request.get("node_id")
    if mode == "legacy_tl3":
        nodes = plan.get("nodes")
        if not isinstance(nodes, list):
            return False
        found = [node for node in nodes if isinstance(node, Mapping) and node.get("id") == node_id]
        return (
            len(found) == 1
            and request.get("recipe_sha256") == _canonical_sha256(found[0].get("args"))
            and request.get("recipe_argv") == found[0].get("args")
        )
    if mode == "native_source4_lstm":
        optional = _mapping(plan.get("optional_source4_lstm"))
        config_file_sha = next(
            (
                optional[key]
                for key in ("config_file_sha256", "config_path_sha256", "config_sha256")
                if key in optional
            ),
            None,
        )
        return (
            node_id == "source4_lstm192x1"
            and optional.get("included") is True
            and optional.get("node_id") == node_id
            and isinstance(optional.get("config_path"), str)
            and bool(optional.get("config_path"))
            and optional.get("config_path") == _argument_value(request.get("argv"), "--config")
            and config_file_sha == request.get("config_file_sha256")
            and optional.get("config_identity_sha256") == request.get("config_identity_sha256")
            and request.get("recipe_sha256") == request.get("config_identity_sha256")
            and optional.get("configured_epochs", optional.get("epochs")) == 8
            and optional.get("expected_updates", optional.get("updates")) == 7_840
            and optional.get("fixed_endpoint_completed_epochs") == 8
            and optional.get("run") == "final-lstm192x1"
            and plan.get("configured_epochs") == 54
            and plan.get("expected_total_updates") == 312_000
            and plan.get("total_configured_epochs") == 62
            and plan.get("total_expected_optimizer_updates") == 319_840
            and plan.get("total_scientific_fits") == 18
            and isinstance(plan.get("nodes"), list)
            and len(plan["nodes"]) == 17
        )
    return False


def _legacy_launch_matches_plan(plan: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    """Prove the profile launch is the frozen recipe with only authorized edits."""
    nodes = plan.get("nodes")
    if not isinstance(nodes, list):
        return False
    found = [
        node
        for node in nodes
        if isinstance(node, Mapping) and node.get("id") == request.get("node_id")
    ]
    if len(found) != 1 or not isinstance(found[0].get("args"), list):
        return False
    expected = list(found[0]["args"])
    if not all(isinstance(token, str) for token in expected):
        return False
    if request.get("recipe_argv") != expected:
        return False
    save_flags = [index for index, token in enumerate(expected) if token == "--save-training-state"]
    if len(save_flags) > 1:
        return False
    if save_flags:
        del expected[save_flags[0]]
    names = [index for index, token in enumerate(expected) if token == "--name"]
    if len(names) != 1 or names[0] + 1 >= len(expected):
        return False
    name_index = names[0] + 1
    if expected[name_index].endswith("_engineering_r2"):
        return False
    expected[name_index] += "_engineering_r2"
    batches = request.get("profile_batches")
    if type(batches) is not int or batches < 1:
        return False
    if "--profile-steps" in expected or "--profile-warmup" in expected:
        return False
    expected.extend(("--profile-steps", str(batches), "--profile-warmup", "1"))
    argv = request.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 8
        or not all(isinstance(token, str) for token in argv)
    ):
        return False
    return (
        argv[:4] == ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "590s"]
        and argv[4].startswith("/")
        and argv[5].startswith("/")
        and argv[5].endswith("profile_wrapper_r2.py")
        and bool(argv[6])
        and argv[7:] == expected
    )


def _membership_checks(
    membership: Mapping[str, Any], audit: Mapping[str, Any], audit_sha256: str
) -> dict[str, bool]:
    groups = _mapping(membership.get("fit_members"))
    train, valid = groups.get("train"), groups.get("valid")
    exact_groups = (
        isinstance(train, list)
        and isinstance(valid, list)
        and all(type(group) is int for group in train + valid)
        and train == list(range(TRAIN_GROUPS))
        and valid == list(range(VALID_GROUPS))
    )
    group_map = (
        {"train": train, "valid": valid}
        if isinstance(train, list) and isinstance(valid, list)
        else None
    )
    exact_hash = group_map is not None and membership.get("fit_members_sha256") == _member_hash(
        group_map
    )
    sources, audit_sources = _mapping(membership.get("sources")), _mapping(audit.get("sources"))
    source_hashes = all(
        _mapping(sources.get(name)).get("sha256")
        == _mapping(audit_sources.get(name)).get("file_sha256")
        and _mapping(sources.get(name)).get("path") == f"datasets/{name}.parquet"
        for name in ("train", "valid")
    )
    exact_receipt = (
        membership.get("source_validation", {}).get("receipt_sha256")
        if isinstance(membership.get("source_validation"), Mapping)
        else None
    )
    validation_member = _mapping(membership.get("source_validation"))
    audit_shape = (
        audit.get("schema") == SOURCE_AUDIT_SCHEMA
        and audit.get("fitting_member_count") == FIT_GROUPS
        and _mapping(audit_sources.get("train")).get("row_groups") == TRAIN_GROUPS
        and _mapping(audit_sources.get("valid")).get("row_groups") == VALID_GROUPS
        and _mapping(audit_sources.get("train")).get("rows") == TRAIN_ROWS
        and _mapping(audit_sources.get("valid")).get("rows") == VALID_ROWS
        and _mapping(sources.get("train")).get("row_groups") == TRAIN_GROUPS
        and _mapping(sources.get("valid")).get("row_groups") == VALID_GROUPS
        and validation_member.get("checked_member") == ["train", 528]
        and validation_member.get("rows") == 20_000
        and validation_member.get("decoded_feature_and_target_columns") == 114
        and validation_member.get("all_decoded_values_finite") is True
        and audit.get("purpose")
        == "all-labeled submission-only final fit; no independent validation score"
        and _mapping(audit.get("row_group_528")).get("source") == "train"
        and _mapping(audit.get("row_group_528")).get("row_group") == 528
        and _mapping(audit.get("row_group_528")).get("rows") == 20_000
        and _mapping(audit.get("row_group_528")).get("columns") == 117
        and _mapping(audit.get("row_group_528")).get("nonfinite") == 0
    )
    return {
        "complete_official_membership": membership.get("schema") == MEMBERSHIP_SCHEMA
        and membership.get("policy")
        == "all_official_train_and_valid_labels_fit_no_independent_score_v1"
        and membership.get("version") == 1
        and membership.get("fit_sequences") == FIT_GROUPS
        and exact_groups
        and exact_hash,
        "no_independent_evaluation": membership.get("independent_evaluation_members") == []
        and membership.get("independent_score_claim") == "forbidden"
        and membership.get("post_fit_primary_valid_scoring") == "forbidden",
        "source_audit_matches_membership": bool(
            source_hashes and audit_shape and exact_receipt == audit_sha256
        ),
    }


def _precision_ok(runtime: Mapping[str, Any]) -> bool:
    return (
        runtime.get("autocast") is False
        and runtime.get("amp_scaler") is False
        and runtime.get("matmul_tf32") is False
        and runtime.get("cudnn_tf32") is False
        and runtime.get("matmul_precision") == "highest"
    )


def _rank_log_ok(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_000_000:
        return False
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    required = ("total_loss", "label_loss", "gradient_norm_preclip", "learning_rate")
    return len(rows) == 50 and all(
        isinstance(row, dict)
        and row.get("rank") == 0
        and row.get("step") == step
        and all(_finite(row.get(key)) for key in required)
        and float(row["learning_rate"]) > 0
        for step, row in enumerate(rows, 1)
    )


def _native_checks(
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    rank_log: Path,
    config: Mapping[str, Any],
    membership: Mapping[str, Any],
) -> dict[str, bool]:
    identity = _mapping(profile.get("identity"))
    data = _mapping(identity.get("data"))
    final_fit = _mapping(data.get("final_fit"))
    runtime = _mapping(identity.get("runtime"))
    devices = runtime.get("devices")
    device = devices[0] if isinstance(devices, list) and len(devices) == 1 else {}
    resources = profile.get("memory_by_rank")
    row = resources[0] if isinstance(resources, list) and len(resources) == 1 else {}
    losses = _mapping(profile.get("losses"))
    expected_fit = _mapping(final_fit.get("fit_row_groups"))
    fields = {
        "manifest_sha256": request.get("final_fit_manifest_sha256"),
        "fit_members_sha256": request.get("fit_members_sha256"),
        "source_validation_sha256": request.get("source_audit_sha256"),
    }
    data_identity = all(final_fit.get(key) == value for key, value in fields.items())
    data_identity = (
        data_identity
        and expected_fit.get("train") == list(range(TRAIN_GROUPS))
        and expected_fit.get("valid") == list(range(VALID_GROUPS))
    )
    expected_members = [["train", group] for group in range(TRAIN_GROUPS)] + [
        ["valid", group] for group in range(VALID_GROUPS)
    ]
    data_identity = data_identity and data.get("fit_members") == expected_members
    data_identity = (
        data_identity
        and data.get("evaluation_members") == []
        and data.get("evaluation_row_groups") == []
    )
    data_identity = (
        data_identity
        and data.get("evaluation_role")
        == data.get("evaluation_source")
        == data.get("evaluation_mask")
        == "none"
    )
    sources = _mapping(data.get("sources"))
    expected_sources = _mapping(membership.get("sources"))
    data_identity = data_identity and all(
        _mapping(sources.get(f"datasets/{name}.parquet")).get("sha256")
        == _mapping(expected_sources.get(name)).get("sha256")
        for name in ("train", "valid")
    )
    normalization = _mapping(data.get("normalization"))
    data_identity = (
        data_identity
        and normalization.get("scope") == "prefix"
        and normalization.get("sequence_count") == 32
    )
    norm_sources = _mapping(normalization.get("source_fingerprints"))
    data_identity = data_identity and set(norm_sources) == {
        "datasets/train.parquet",
        "datasets/valid.parquet",
    }
    data_identity = data_identity and all(
        _mapping(norm_sources.get(f"datasets/{name}.parquet")).get("sha256")
        == _mapping(expected_sources.get(name)).get("sha256")
        for name in ("train", "valid")
    )
    precision = _precision_ok(runtime) and _mapping(runtime.get("precision")).get("mode") == "fp32"
    source_files = _mapping(identity.get("source_files"))
    uuid_match = (
        _uuid_identity(request.get("gpu_uuid")) is not None
        and isinstance(device, Mapping)
        and _uuid_identity(device.get("uuid")) == _uuid_identity(request.get("gpu_uuid"))
    )
    memory_ok = (
        isinstance(row, Mapping)
        and type(row.get("peak_allocated_bytes")) is int
        and type(row.get("peak_reserved_bytes")) is int
        and type(row.get("device_total_memory_bytes")) is int
        and type(row.get("host_peak_rss_bytes")) is int
        and 0
        < row["peak_allocated_bytes"]
        <= row["peak_reserved_bytes"]
        <= row["device_total_memory_bytes"]
        and row["host_peak_rss_bytes"] > 0
    )
    return {
        "native_profile_complete_unapproved": profile.get("receipt_schema")
        == "wnn.sequence-profile.v1"
        and profile.get("status") == "profile_complete"
        and profile.get("approved") is False
        and profile.get("updates") == 50
        and profile.get("errors") == []
        and profile.get("error_count") == 0
        and profile.get("scientific_checkpoint_written") is False
        and _finite(profile.get("elapsed_seconds"), positive=True)
        and profile.get("forwarded_rows", 0) >= profile.get("real_rows", 1) > 0
        and 0 < profile.get("valid_rows", 0) <= profile.get("real_rows", 0),
        "identity_matches_request": identity.get("source_sha256")
        == request.get("full_source_code_sha256")
        and identity.get("source_sha256") == _canonical_sha256(source_files)
        and source_files == _mapping(request.get("expected_source")).get("files")
        and identity.get("config_sha256") == request.get("config_identity_sha256"),
        "all_fit_membership_no_eval": data_identity,
        "strict_fp32_runtime": precision and config.get("precision_mode") == "fp32",
        "one_assigned_b300_and_resources": uuid_match
        and isinstance(device, Mapping)
        and "B300" in str(device.get("name", "")).upper()
        and runtime.get("world_size") == 1
        and memory_ok,
        "fixed_lstm192x1_recipe": _native_recipe_ok(config),
        "finite_losses_and_step_journal": set(losses) == {"label", "distillation", "hint", "total"}
        and all(_finite(value) for value in losses.values())
        and _rank_log_ok(rank_log),
    }


def _native_recipe_ok(config: Mapping[str, Any]) -> bool:
    expected = {
        "accumulate_chunks": 4,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.999,
        "adamw_eps": 1e-8,
        "batch_size": 256,
        "checkpoint_retention": "all_epochs",
        "chunk_rows": 256,
        "epochs": 8,
        "expected_fold": None,
        "grad_clip": 1.0,
        "hint_mode": "aligned",
        "hint_weight": 0.0,
        "input_prefetch_batches": 1,
        "kd_weight": 0.0,
        "lr": 0.0015008627097145002,
        "min_lr_ratio": 0.0,
        "model": {
            "architecture": "lstm",
            "attention_backend": "math",
            "checkpoint_blocks": True,
            "compile_blocks": False,
            "context_rows": 512,
            "dropout": 0.0,
            "ff_multiplier": 4,
            "heads": 12,
            "hidden": 192,
            "hint_dim": 0,
            "hint_projection": "learned",
            "input_feature_mode": "raw112",
            "layers": 1,
        },
        "muon_momentum": 0.95,
        "muon_ns_eps": 1e-7,
        "muon_ns_steps": 5,
        "normalization_scope": "prefix",
        "normalization_sequences": 32,
        "optimizer": "adamw",
        "precision_mode": "fp32",
        "required_device": "B300",
        "required_world_size": 1,
        "schedule": "legacy_warmup_cosine",
        "schema_version": 1,
        "seed": 1302,
        "training_mask": "all_after_warmup",
        "warmup_fraction": 0.05,
        "weight_decay": 0.01,
    }
    return dict(config) == expected


def _legacy_records(path: Path) -> dict[str, Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError("legacy profile log is missing, linked, or oversized")
    markers = ("PROFILE", "PROFILE_UPDATE_EVIDENCE", "PROFILE_RESOURCES")
    found: dict[str, Mapping[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                raw_record = json.loads(stripped)
            except json.JSONDecodeError:
                raw_record = None
            if isinstance(raw_record, Mapping):
                if "final_fit_manifest_sha256" in raw_record:
                    if "ARGS" in found:
                        raise ValueError("legacy log repeats parsed ARGS record")
                    found["ARGS"] = raw_record
                elif "source_train_sequences" in raw_record and "eval_sequences" in raw_record:
                    if "DATA_COUNTS" in found:
                        raise ValueError("legacy log repeats DATA_COUNTS record")
                    found["DATA_COUNTS"] = raw_record
        for marker in markers:
            prefix = marker + " "
            if line.startswith(prefix):
                if marker in found:
                    raise ValueError(f"legacy log repeats {marker}")
                value = json.loads(line[len(prefix) :])
                if not isinstance(value, Mapping):
                    raise ValueError(f"legacy {marker} record is not an object")
                found[marker] = value
    if set(found) != {*markers, "ARGS", "DATA_COUNTS"}:
        raise ValueError(
            "legacy log must contain actual parsed ARGS, DATA_COUNTS, PROFILE, "
            "PROFILE_UPDATE_EVIDENCE, and PROFILE_RESOURCES"
        )
    return found


def _legacy_checks(
    records: Mapping[str, Mapping[str, Any]],
    request: Mapping[str, Any],
    membership: Mapping[str, Any],
) -> dict[str, bool]:
    profile, updates, resources, args, counts = (
        records[key]
        for key in (
            "PROFILE",
            "PROFILE_UPDATE_EVIDENCE",
            "PROFILE_RESOURCES",
            "ARGS",
            "DATA_COUNTS",
        )
    )
    sources = _mapping(membership.get("sources"))
    logs_clean = True
    for record in (profile, updates, resources):
        if any(not _finite(value) for value in record.values() if type(value) in (int, float)):
            logs_clean = False
    chunk_rows = _argument_value(request.get("recipe_argv"), "--chunk")
    profile_batches = request.get("profile_batches")
    warmup_batches = request.get("warmup_batches")
    updates_per_batch = 20_000 // int(chunk_rows) if chunk_rows and chunk_rows.isdecimal() else 0
    measured_updates = profile_batches * updates_per_batch if type(profile_batches) is int else 0
    total_updates = (
        (profile_batches + warmup_batches) * updates_per_batch
        if type(profile_batches) is int and type(warmup_batches) is int
        else 0
    )
    resources_ok = (
        resources.get("oom_count") == 0
        and _uuid_identity(resources.get("gpu_uuid")) is not None
        and _uuid_identity(resources.get("gpu_uuid")) == _uuid_identity(request.get("gpu_uuid"))
        and "B300" in str(resources.get("gpu_name", "")).upper()
        and all(
            type(resources.get(key)) is int and resources[key] > 0
            for key in (
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "device_total_memory_bytes",
                "host_peak_rss_bytes",
            )
        )
        and resources["peak_allocated_bytes"]
        <= resources["peak_reserved_bytes"]
        <= resources["device_total_memory_bytes"]
    )
    precision_ok = (
        all(
            resources.get(key) is False
            for key in ("autocast", "amp_scaler", "matmul_tf32", "cudnn_tf32")
        )
        and resources.get("matmul_precision") == "highest"
        and args.get("precision") == "fp32"
    )
    update_ok = (
        updates.get("optimizer_attempts") == total_updates
        and updates.get("optimizer_completed_updates") == total_updates
        and updates.get("amp_skipped_updates") == 0
        and updates.get("loss_checks") == total_updates
        and updates.get("loss_finite") is True
        and updates.get("gradient_finite") is True
        and updates.get("gradient_stats_steps") == total_updates
        and updates.get("gradient_stats_nonfinite_steps") == 0
        and updates.get("counts_consistent") is True
        and updates.get("grad_scaler_enabled") is False
        and updates.get("valid_for_throughput") is True
        and type(profile_batches) is int
        and type(warmup_batches) is int
        and warmup_batches == 1
        and measured_updates >= 50
        and measured_updates < 50 + updates_per_batch
        and profile_batches == math.ceil(50 / updates_per_batch)
        and _argument_value(request.get("argv"), "--profile-steps") == str(profile_batches)
        and _argument_value(request.get("argv"), "--profile-warmup") == str(warmup_batches)
    )
    expected_name = _argument_value(request.get("argv"), "--name")
    expected_checkpoint_epoch = _argument_value(request.get("recipe_argv"), "--checkpoint-epoch")
    expected_train_hash = _mapping(sources.get("train")).get("sha256")
    expected_valid_hash = _mapping(sources.get("valid")).get("sha256")
    data_ok = (
        args.get("final_fit_manifest_sha256") == request.get("final_fit_manifest_sha256")
        and args.get("final_fit_members_sha256") == request.get("fit_members_sha256")
        and args.get("final_fit_train_sha256") == expected_train_hash
        and args.get("final_fit_valid_sha256") == expected_valid_hash
        and args.get("final_fit_source_validation_sha256") == request.get("source_audit_sha256")
        and args.get("name") == expected_name
        and args.get("final_fit_manifest")
        == _argument_value(request.get("argv"), "--final-fit-manifest")
        and args.get("split_manifest") is None
        and args.get("fit_groups_manifest") is None
        and args.get("resume_state") is None
        and args.get("save_training_state") is False
        and args.get("train_source") == "train+valid"
        and args.get("train_seqs") == TRAIN_GROUPS
        and args.get("include_heldout_valid") is True
        and args.get("valid_train_fraction") == 1
        and args.get("valid_repeat") == 1
        and args.get("valid_fold") == -1
        and args.get("no_eval_during_fit") is True
        and counts.get("source_train_sequences") == TRAIN_GROUPS
        and counts.get("eligible_valid_train_sequences") == VALID_GROUPS
        and counts.get("used_valid_train_sequences") == VALID_GROUPS
        and counts.get("eval_sequences") == 0
        and args.get("precision") == "fp32"
        and args.get("checkpoint_policy") == "fixed"
        and type(args.get("checkpoint_epoch")) is int
        and expected_checkpoint_epoch is not None
        and expected_checkpoint_epoch.isdecimal()
        and args.get("checkpoint_epoch") == int(expected_checkpoint_epoch)
        and args.get("propensity") in (None, "")
        and not args.get("class_balance", False)
    )
    return {
        "legacy_profile_markers_finite": logs_clean
        and _finite(profile.get("step_wall"), positive=True)
        and _finite(profile.get("seq_per_s"), positive=True),
        "all_optimizer_updates_and_gradients_finite": update_ok,
        "strict_fp32_runtime": precision_ok,
        "one_assigned_b300_and_resources": resources_ok,
        "actual_args_and_membership_counts": data_ok
        and request.get("recipe_sha256") == _canonical_sha256(request.get("recipe_argv")),
    }


def _process_ok(process: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    start, end = process.get("started_at_epoch"), process.get("finished_at_epoch")
    command = process.get("command")
    if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
        return False
    if request.get("mode") == "legacy_tl3":
        runtime_root = command[5].rsplit("/", 1)[0] if len(command) > 5 else ""
        argv_log_path_matches = len(command) > 6 and command[6] == request.get("log_path")
    else:
        config_path = _argument_value(command, "--config")
        runtime_root = (
            config_path.rsplit("/inputs/", 1)[0]
            if isinstance(config_path, str) and "/inputs/" in config_path
            else ""
        )
        argv_log_path_matches = isinstance(request.get("log_path"), str) and bool(
            request.get("log_path")
        )
    expected_threads = {
        key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    cwd_matches = bool(runtime_root) and process.get("cwd") == f"{runtime_root}/source"
    return (
        process.get("schema") == "wnn.multistage-owned-process.v1"
        and process.get("mode") == "final_submit_engineering_r2"
        and process.get("status") == "completed"
        and process.get("exit_code") == 0
        and process.get("timed_out") is False
        and command == request.get("argv")
        and process.get("command_sha256") == request.get("argv_sha256")
        and request.get("argv_sha256") == _canonical_sha256(command)
        and request.get("cpu_threads") == 4
        and process.get("timeout_seconds") == 600
        and _uuid_identity(process.get("gpu_key")) == _uuid_identity(request.get("gpu_uuid"))
        and process.get("output_dir") == request.get("profile_output")
        and process.get("effective_thread_environment") == expected_threads
        and process.get("child_environment") == expected_threads
        and argv_log_path_matches
        and cwd_matches
        and _argument_value(
            command,
            "--profile-updates"
            if request.get("mode") == "native_source4_lstm"
            else "--profile-steps",
        )
        == (
            "50"
            if request.get("mode") == "native_source4_lstm"
            else str(request.get("profile_batches"))
        )
        and _finite(start, positive=True)
        and _finite(end, positive=True)
        and float(end) > float(start)
        and isinstance(process.get("child"), Mapping)
        and type(process["child"].get("pid")) is int
        and type(process["child"].get("pgid")) is int
        and bool(str(process["child"].get("birth", "")))
    )


def _argument_value(argv: object, option: str) -> str | None:
    if not isinstance(argv, list):
        return None
    values: list[str] = []
    for index, token in enumerate(argv):
        if not isinstance(token, str):
            return None
        if token == option and index + 1 < len(argv):
            values.append(str(argv[index + 1]))
        elif token.startswith(option + "="):
            values.append(token.partition("=")[2])
    return values[0] if len(values) == 1 else None


def _exact_group_set(value: object, expected_count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected_count
        and all(type(group) is int for group in value)
        and set(value) == set(range(expected_count))
    )


def review_profile(
    *,
    request_path: Path,
    expected_request_sha256: str,
    plan_path: Path,
    membership_path: Path,
    source_audit_path: Path,
    source_capsule_path: Path,
    process_path: Path,
    output_dir: Path,
    profile_path: Path | None = None,
    rank_log_path: Path | None = None,
    config_path: Path | None = None,
    legacy_log_path: Path | None = None,
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Review evidence and create separate review/approval files on a full pass."""
    request, plan, membership, audit, process = (
        _read_json(request_path),
        _read_json(plan_path),
        _read_json(membership_path),
        _read_json(source_audit_path, 64_000),
        _read_json(process_path),
    )
    if request.get("schema") != REQUEST_SCHEMA or not HEX256.fullmatch(expected_request_sha256):
        raise ValueError("unsupported request or malformed frozen request SHA256")
    if _sha256(request_path) != expected_request_sha256:
        raise ValueError("profile request bytes differ from their pre-frozen SHA256")
    membership_results = _membership_checks(membership, audit, _sha256(source_audit_path))
    request_checks = {
        "plan_node_recipe_pinned": _expected_plan_node(plan, request),
        "source_capsule_pinned": _sha256(source_capsule_path) == SOURCE_CAPSULE_SHA256
        and request.get("source_capsule_sha256") == SOURCE_CAPSULE_SHA256
        and _expected_source_ok(request, source_capsule_path),
        "final_fit_sources_pinned": _sha256(membership_path)
        == request.get("final_fit_manifest_sha256")
        and _sha256(source_audit_path) == SOURCE_AUDIT_SHA256
        and request.get("source_audit_sha256") == SOURCE_AUDIT_SHA256,
        "complete_membership_pinned": request.get("fit_members_sha256")
        == membership.get("fit_members_sha256")
        and request.get("fit_member_count") == FIT_GROUPS
        and request.get("evaluation_member_count") == 0
        and request.get("no_eval_during_fit") is True
        and request.get("independent_score_claim") == "forbidden",
        "profile_budget_pinned": (
            request.get("profile_updates") == 50 and request.get("warmup_batches") == 0
            if request.get("mode") == "native_source4_lstm"
            else type(request.get("profile_batches")) is int and request.get("warmup_batches") == 1
        ),
        "gpu_identity_well_formed": _uuid_identity(request.get("gpu_uuid")) is not None,
        "owned_process_exit0": _process_ok(process, request),
    }
    if request.get("mode") == "legacy_tl3":
        request_checks["legacy_launch_matches_frozen_recipe"] = _legacy_launch_matches_plan(
            plan, request
        )
    checks = dict(membership_results) | request_checks
    approved_payload: dict[str, Any]
    if request.get("mode") == "native_source4_lstm":
        if profile_path is None or rank_log_path is None or config_path is None:
            raise ValueError("native review requires profile, rank log, and config files")
        profile = _read_json(profile_path)
        config = _read_json(config_path, 64_000)
        checks.update(_native_checks(profile, request, rank_log_path, config, membership))
        checks["config_bytes_and_identity_pinned"] = _sha256(config_path) == request.get(
            "config_file_sha256"
        ) and _native_config_identity_sha256(config) == request.get("config_identity_sha256")
        checks["no_observed_oom"] = not _contains_oom(profile) and not _file_contains_oom(
            rank_log_path
        )
        approved_payload = dict(profile)
    elif request.get("mode") == "legacy_tl3":
        if legacy_log_path is None:
            raise ValueError("legacy review requires the actual stdout profile log")
        records = _legacy_records(legacy_log_path)
        checks.update(_legacy_checks(records, request, membership))
        checks["no_observed_oom"] = not _contains_oom(records) and not _file_contains_oom(
            legacy_log_path
        )
        approved_payload = {
            "receipt_schema": APPROVAL_SCHEMA,
            "mode": request["mode"],
            "node_id": request["node_id"],
            "profile_batches": request["profile_batches"],
            "warmup_batches": request["warmup_batches"],
            "request_sha256": expected_request_sha256,
            "profile_log_sha256": _sha256(legacy_log_path),
            "source_capsule_sha256": request.get("source_capsule_sha256"),
            "full_source_code_sha256": request.get("full_source_code_sha256"),
            "approved": False,
        }
    else:
        raise ValueError("unsupported profile mode")
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    review: dict[str, Any] = {
        "schema": REVIEW_SCHEMA,
        "status": "approved" if not failed else "rejected",
        "approved_for_full_fit": not failed,
        "request_sha256": expected_request_sha256,
        "profile_mode": request["mode"],
        "node_id": request["node_id"],
        "plan_sha256": request.get("plan_sha256"),
        "source_capsule_sha256": request.get("source_capsule_sha256"),
        "full_source_code_sha256": request.get("full_source_code_sha256"),
        "evidence_sha256": {
            "request": _sha256(request_path),
            "plan": _sha256(plan_path),
            "membership": _sha256(membership_path),
            "source_audit": _sha256(source_audit_path),
            "source_capsule": _sha256(source_capsule_path),
            "process": _sha256(process_path),
            **(
                {
                    "native_profile": _sha256(profile_path),
                    "rank_log": _sha256(rank_log_path),
                    "config": _sha256(config_path),
                }
                if profile_path is not None
                and rank_log_path is not None
                and config_path is not None
                else {}
            ),
            **(
                {
                    "legacy_log": _sha256(legacy_log_path),
                    **(
                        {"provenance": _sha256(provenance_path)}
                        if provenance_path is not None
                        else {}
                    ),
                }
                if legacy_log_path is not None
                else {}
            ),
        },
        "checks": checks,
        "failed_checks": failed,
        "native_profile_preserved": True,
        "approved_profile_file": "approved_profile_receipt.json" if not failed else None,
    }
    if output_dir.exists():
        raise FileExistsError(f"review output is create-only: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".final-profile-review-", dir=output_dir.parent
    ) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        review_path = staged / "profile_review.json"
        review_bytes = json.dumps(review, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with review_path.open("x", encoding="utf-8") as stream:
            stream.write(review_bytes)
        if not failed:
            approved_payload["approved"] = True
            approved_payload["approval_review_sha256"] = _sha256(review_path)
            approved_path = staged / "approved_profile_receipt.json"
            with approved_path.open("x", encoding="utf-8") as stream:
                json.dump(approved_payload, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        os.rename(staged, output_dir)
    return review


def _contains_oom(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_oom(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_oom(item) for item in value)
    return isinstance(value, str) and bool(
        re.search(r"\b(?:CUDA\s+)?out of memory\b|\bOOM[- ]kill\b", value, re.I)
    )


def _file_contains_oom(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 20_000_000:
        return True
    return bool(
        re.search(
            r"\b(?:CUDA\s+)?out of memory\b|\bOOM[- ]kill\b",
            path.read_text(encoding="utf-8", errors="replace"),
            re.I,
        )
    )


def main() -> None:
    """Run the offline review; this command never launches or contacts a worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--expected-request-sha256", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--source-capsule", required=True, type=Path)
    parser.add_argument("--process", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--rank-log", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--legacy-log", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = review_profile(
        request_path=args.request,
        expected_request_sha256=args.expected_request_sha256,
        plan_path=args.plan,
        membership_path=args.membership,
        source_audit_path=args.source_audit,
        source_capsule_path=args.source_capsule,
        process_path=args.process,
        output_dir=args.output_dir,
        profile_path=args.profile,
        rank_log_path=args.rank_log,
        config_path=args.config,
        legacy_log_path=args.legacy_log,
        provenance_path=args.provenance,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
