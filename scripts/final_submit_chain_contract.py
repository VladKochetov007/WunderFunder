"""Frozen input, graph, admission, and profile checks for the finite final-fit director."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wnn.final_submit_manifest import FinalFitMembership, load_final_fit_membership
from wnn.training.final_submit_bridge import validate_source_validation_receipt

PLAN_SHA256 = "29802499ab6ba62586b612cd10164d27c9737a96cf8de41c09fef3f6d0402cf4"
SOURCE_CAPSULE_SHA256 = "8381c0d6a6333cbc32dca2187516994c5a80832193bcf2531bd606511bc7ccb4"
SOURCE_COMMIT = "39d83f73926a459269bcd8511eec08ce9a6e02fd"
SOURCE_AUDIT_SHA256 = "1e8a8a794a0b8d9677a76030e570681074503d38aa2d54b80e36067566b94ce0"
SOURCE_AUDIT_SCHEMA = "wnn.final-submit-source-validation.v1"
PLAN_SCHEMA = "wnn.final-submit-recipes.v1"
ADMISSION_SCHEMA = "wnn.final-submit-root-admission.v1"
TOTAL_FITS = 18
MAX_PROFILES = 24
MAX_FIT_SECONDS = 21_600
MAX_PROFILE_SECONDS = 600
MAX_LANES = 8
HARD_DEADLINE = 1_793_523_959.0380824
COLLECTION_RESERVE = 1_800
FIT_MEMBERS = 12_480
TRAIN_MEMBERS = 10_607
VALID_MEMBERS = 1_873
SEQUENCE_ROWS = 20_000
NATIVE_PROFILE_SHA256 = "cf777b3d198c9239a1df6e2d9d983218676f7b35b5d4069a99b4b405b103853a"
FIN256_PROFILE_SHA256 = "9e01f3c052912052538916c7fcabe5084facdb043c5c74e9cfeb638588795e34"
W18_TEACHERS = (
    ("tl3final_fbc1k_s5", "strong", 0.3, 3),
    ("tl3final_ffull_s3", "full_finetuned", 0.1, 1),
    ("tl3final_ffull_s4", "full_finetuned", 0.1, 1),
    ("tl3final_frawW_s3ft", "raw_wmse", 0.125, 1),
    ("tl3final_frawW_s4ft", "raw_wmse", 0.125, 1),
    ("tl3final_orgbase", "samefold_baseline", 0.0625, 3),
)
TL3_IDS = (
    "fin256_s3",
    "fmu128_s3",
    "frawW_s3",
    "fin256_s4",
    "fmu128_s4",
    "frawW_s4",
    "orgbase",
    "fin256ft_s3",
    "ffull_s3",
    "frawW_s3ft",
    "fin256ft_s4",
    "ffull_s4",
    "frawW_s4ft",
    "fbc1k_s5",
    "fttall_tlead_s5",
    "fttall_tlead_s6",
    "w18final_distilled_s5",
)
EXPECTED_PARENTS = {
    "fin256_s3": (),
    "fmu128_s3": (),
    "frawW_s3": (),
    "fin256_s4": (),
    "fmu128_s4": (),
    "frawW_s4": (),
    "orgbase": (),
    "fin256ft_s3": ("fin256_s3",),
    "ffull_s3": ("fmu128_s3",),
    "frawW_s3ft": ("frawW_s3",),
    "fin256ft_s4": ("fin256_s4",),
    "ffull_s4": ("fmu128_s4",),
    "frawW_s4ft": ("frawW_s4",),
    "fbc1k_s5": ("fin256_s3", "fin256_s4"),
    "fttall_tlead_s5": ("fin256ft_s3", "fin256ft_s4"),
    "fttall_tlead_s6": ("fin256ft_s3", "fin256ft_s4"),
    "w18final_distilled_s5": tuple(name.removeprefix("tl3final_") for name, *_ in W18_TEACHERS),
}


def sha256_file(path: Path) -> str:
    """Hash a regular file in bounded blocks and reject symbolic links."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, max_bytes: int = 8_000_000) -> dict[str, Any]:
    """Read a bounded JSON object from a regular file."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"unsafe, missing, or oversized JSON input: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def canonical_sha256(value: object) -> str:
    """Hash a JSON value using the repository's compact canonical encoding."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def option_values(argv: Sequence[str], option: str) -> list[str]:
    """Return all values for one CLI option, supporting `--key=value` syntax."""
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif token.startswith(option + "="):
            values.append(token.partition("=")[2])
    return values


def verify_plan(plan: Mapping[str, Any], plan_path: Path) -> dict[str, dict[str, Any]]:
    """Enforce the finite 17-node TL3 graph and its separate native 18th fit."""
    if plan.get("schema") != PLAN_SCHEMA or plan.get("plan_sha256") != PLAN_SHA256:
        raise ValueError("plan schema or frozen plan identity differs")
    plan_copy = dict(plan)
    declared = plan_copy.pop("plan_sha256", None)
    if declared != canonical_sha256(plan_copy):
        raise ValueError("plan canonical hash does not match its declared identity")
    if (
        plan.get("configured_epochs") != 54
        or plan.get("expected_total_updates") != 312_000
        or plan.get("total_configured_epochs") != 62
        or plan.get("total_expected_optimizer_updates") != 319_840
        or plan.get("total_scientific_fits") != TOTAL_FITS
    ):
        raise ValueError("frozen TL3 subtotal or amended 18-fit aggregate changed")
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != len(TL3_IDS):
        raise ValueError("the final TL3 graph must contain exactly 17 original nodes")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw_node in enumerate(nodes):
        if not isinstance(raw_node, dict) or raw_node.get("id") != TL3_IDS[index]:
            raise ValueError("TL3 node order or identity changed")
        node = raw_node
        node_id = str(node["id"])
        args = node.get("args")
        if not isinstance(args, list) or not all(isinstance(token, str) for token in args):
            raise ValueError(f"invalid frozen argument list for {node_id}")
        if tuple(node.get("parents", ())) != EXPECTED_PARENTS[node_id]:
            raise ValueError(f"frozen parent graph changed for {node_id}")
        if node.get("run") != f"tl3final_{node_id}":
            raise ValueError(f"final run prefix changed for {node_id}")
        for key, value in (
            ("--train-source", "train+valid"),
            ("--train-seqs", str(TRAIN_MEMBERS)),
            ("--final-fit-manifest", None),
            ("--final-fit-source-validation-receipt", None),
            ("--final-fit-source-validation-sha256", SOURCE_AUDIT_SHA256),
            ("--no-eval-during-fit", None),
            ("--include-heldout-valid", None),
            ("--checkpoint-policy", "fixed"),
            ("--checkpoint-epoch", str(node.get("endpoint"))),
        ):
            is_flag = key in {"--no-eval-during-fit", "--include-heldout-valid"}
            occurrences = sum(token == key or token.startswith(key + "=") for token in args)
            values = [] if is_flag else option_values(args, key)
            if (
                is_flag and (occurrences != 1 or any(token.startswith(key + "=") for token in args))
            ) or (not is_flag and len(values) != 1):
                raise ValueError(f"frozen arguments for {node_id} must contain one {key}")
            if not is_flag and value is not None and values != [value]:
                raise ValueError(f"frozen {key} differs for {node_id}")
            if not is_flag and value is None and not values[0]:
                raise ValueError(f"frozen {key} must have a nonempty path for {node_id}")
        for forbidden in ("--eval-source", "--full-valid", "--split-manifest", "--train-eval-seqs"):
            if forbidden in args:
                raise ValueError(f"final node {node_id} must not contain {forbidden}")
        batch, chunk = option_values(args, "--batch"), option_values(args, "--chunk")
        if len(batch) != 1 or len(chunk) != 1:
            raise ValueError(f"batch/chunk is ambiguous for {node_id}")
        expected_updates = (
            FIT_MEMBERS // int(batch[0]) * (SEQUENCE_ROWS // int(chunk[0])) * int(node["epochs"])
        )
        if node.get("expected_updates") != expected_updates:
            raise ValueError(f"expected update count changed for {node_id}")
        by_id[node_id] = node
    if sum(int(node["expected_updates"]) for node in by_id.values()) != 312_000:
        raise ValueError("TL3 node update subtotal changed")
    native = plan.get("optional_source4_lstm")
    if (
        not isinstance(native, dict)
        or native.get("included") is not True
        or native.get("node_id") != "source4_lstm192x1"
        or native.get("run") != "final-lstm192x1"
        or native.get("configured_epochs") != 8
        or native.get("expected_updates") != 7_840
        or native.get("fixed_endpoint_completed_epochs") != 8
        or not isinstance(native.get("config_path"), str)
        or not isinstance(native.get("config_file_sha256"), str)
        or not isinstance(native.get("config_identity_sha256"), str)
    ):
        raise ValueError("native Source4 LSTM fit identity or fixed endpoint changed")
    return by_id


def verify_sources(
    *,
    data_root: Path,
    membership_path: Path,
    audit_path: Path,
    audit_sha256: str,
) -> FinalFitMembership:
    """Verify the reviewed receiver receipt and exact all-labeled parquet membership."""
    if audit_sha256 != SOURCE_AUDIT_SHA256 or sha256_file(audit_path) != SOURCE_AUDIT_SHA256:
        raise ValueError("receiver source-validation receipt bytes differ")
    audit = read_json(audit_path, 64_000)
    if audit.get("schema") != SOURCE_AUDIT_SCHEMA:
        raise ValueError("receiver source-validation receipt schema differs")
    membership = load_final_fit_membership(membership_path, data_root, SOURCE_AUDIT_SHA256)
    if (
        len(membership.train_groups) != TRAIN_MEMBERS
        or len(membership.valid_groups) != VALID_MEMBERS
    ):
        raise ValueError(
            "final-fit membership must contain exactly 10,607 train and 1,873 valid groups"
        )
    validate_source_validation_receipt(audit_path, SOURCE_AUDIT_SHA256, membership)
    return membership


def verify_root_admission(
    *,
    admission_path: Path,
    grader_path: Path,
    director_path: Path,
    helper_path: Path,
    runtime_helper_path: Path,
    execution_helper_path: Path,
    plan_sha256: str,
    source_capsule_sha256: str,
    now_epoch: float,
) -> dict[str, Any]:
    """Require the exact root-issued execution receipt and pinned grader/director bytes."""
    admission = read_json(admission_path)
    files = admission.get("director_files")
    expected_files = {
        "scripts/final_submit_chain.py": sha256_file(director_path),
        "scripts/final_submit_chain_contract.py": sha256_file(helper_path),
        "scripts/final_submit_chain_runtime.py": sha256_file(runtime_helper_path),
        "scripts/final_submit_chain_execution.py": sha256_file(execution_helper_path),
    }
    expected = {
        "schema": ADMISSION_SCHEMA,
        "approved": True,
        "plan_sha256": plan_sha256,
        "source_capsule_sha256": source_capsule_sha256,
        "grader_sha256": sha256_file(grader_path),
        "director_sha256": sha256_file(director_path),
        "max_scientific_fits": TOTAL_FITS,
        "max_profiles": MAX_PROFILES,
        "max_fit_wall_seconds": MAX_FIT_SECONDS,
        "max_profile_wall_seconds": MAX_PROFILE_SECONDS,
        "max_lanes": MAX_LANES,
        "hard_deadline_epoch": HARD_DEADLINE,
        "collection_reserve_seconds": COLLECTION_RESERVE,
        "foreign_process_signals": False,
    }
    for key, value in expected.items():
        if admission.get(key) != value:
            raise ValueError(f"root execution admission field differs: {key}")
    if files != expected_files:
        raise ValueError("root admission director_files mapping differs from current helper bytes")
    if now_epoch >= HARD_DEADLINE - COLLECTION_RESERVE:
        raise ValueError("contract collection reserve has begun; no new fit may start")
    return admission


def validate_profile_approval(
    *,
    node_id: str,
    approval_path: Path,
    expected_sha256: str,
    plan_sha256: str,
    membership: FinalFitMembership,
    mode: str,
) -> dict[str, Any]:
    """Require an independent passing review bound to this exact node and source identity."""
    if sha256_file(approval_path) != expected_sha256:
        raise ValueError(f"profile approval bytes differ for {node_id}")
    fixed_hash = NATIVE_PROFILE_SHA256 if node_id == "source4_lstm192x1" else FIN256_PROFILE_SHA256
    if node_id in {"source4_lstm192x1", "fin256_s3"} and expected_sha256 != fixed_hash:
        raise ValueError(f"profile approval hash differs from the reviewed R2 {node_id} artifact")
    approval = read_json(approval_path)
    review_path = approval_path.parent / "profile_review.json"
    review = read_json(review_path)
    checks = review.get("checks")
    required = (
        approval.get("approved") is True
        and approval.get("approval_review_sha256") == sha256_file(review_path)
        and review.get("status") == "approved"
        and review.get("approved_for_full_fit") is True
        and review.get("node_id") == node_id
        and review.get("plan_sha256") == plan_sha256
        and review.get("source_capsule_sha256") == SOURCE_CAPSULE_SHA256
        and review.get("evidence_sha256", {}).get("membership") == membership.manifest_sha256
        and review.get("evidence_sha256", {}).get("source_audit") == SOURCE_AUDIT_SHA256
        and review.get("profile_mode") == mode
        and (mode == "native_source4_lstm" or isinstance(approval.get("request_sha256"), str))
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values())
        and review.get("failed_checks") == []
    )
    if not required:
        raise ValueError(f"independent profile review is missing or mismatched for {node_id}")
    if mode == "native_source4_lstm" and (
        approval.get("receipt_schema") != "wnn.sequence-profile.v1"
        or approval.get("status") != "profile_complete"
        or approval.get("scientific_checkpoint_written") is not False
    ):
        raise ValueError("native full fit requires the reviewed completed profile receipt")
    return approval
