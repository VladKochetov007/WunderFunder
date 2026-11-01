"""Build a single-member full-corpus W18 package after verifying its full lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CHAIN_SCRIPTS = (
    ROOT
    / "research"
    / "experiment_drafts"
    / "final-submit-all-labeled-20261031"
    / "receiving"
    / "control-code-r2"
    / "scripts"
)
sys.path.insert(0, str(CHAIN_SCRIPTS))

import final_submit_chain_contract as chain_contract  # noqa: E402

from scripts.package_final_tl3 import (  # noqa: E402
    FINAL_FIXED_EPOCHS,
    FINAL_RUNS,
    _load_member,
    _mapping,
    _validate_final_checkpoint,
    canonical_sha256,
    sha256_file,
)
from wnn.final_submit_manifest import (  # noqa: E402
    FINAL_FIT_POLICY,
    FINAL_FIT_SCHEMA,
    TOTAL_SEQUENCES,
    TRAIN_SEQUENCES,
    VALID_SEQUENCES,
    FinalFitMembership,
    load_final_fit_membership,
)
from wnn.training.checkpoints import serialized_args  # noqa: E402
from wnn.training.cli import build_parser  # noqa: E402
from wnn.training.final_submit_bridge import (  # noqa: E402
    SOURCE_VALIDATION_SHA256,
    final_fit_identity,
    load_final_fit_arguments,
    validate_final_fit_training_options,
    validate_source_validation_receipt,
)
from wnn.training.final_submit_bridge_cache import (  # noqa: E402
    validate_final_distillation_cache,
)
from wnn.training.final_submit_bridge_contract import (  # noqa: E402
    FINAL_W18_TEACHERS,
)
from wnn.training.final_submit_bridge_teachers import (  # noqa: E402
    audit_final_fit_ancestor_closure,
)

MAX_ZIP_BYTES = 20_000_000
MAX_RECEIPT_BYTES = 1_000_000
W18_RECEIPT_SCHEMA = "wnn.final-w18-model-receipt.v1"
W18_LINEAGE_SCHEMA = "wnn.final-w18-lineage-closure.v1"
W18_CACHE_SCHEMA = "wnn.final-w18-teacher-cache-binding.v1"
W18_PACKAGE_SCHEMA = "wnn.final-w18-package-manifest.v1"
W18_BUILD_SCHEMA = "wnn.final-w18-package-build-receipt.v1"
W18_SCOPE = "approved_full_corpus_final_w18"
W18_RUN = FINAL_RUNS[0]
W18_NODE_ID = "w18final_distilled_s5"
W18_PLAN_SHA256 = "29802499ab6ba62586b612cd10164d27c9737a96cf8de41c09fef3f6d0402cf4"
W18_ENDPOINT = FINAL_FIXED_EPOCHS["w18"]
W18_TEACHER_CACHE = "final-fit-w18-mixed-targets"
_SHA256 = set("0123456789abcdef")
_LINEAGE_CHECKPOINTS = {
    "tl3final_w18final_distilled_s5": "model_ep1.pt",
    "tl3final_fbc1k_s5": "model_ep3.pt",
    "tl3final_ffull_s3": "model_ep1.pt",
    "tl3final_ffull_s4": "model_ep1.pt",
    "tl3final_frawW_s3ft": "model_ep1.pt",
    "tl3final_frawW_s4ft": "model_ep1.pt",
    "tl3final_orgbase": "model_ep3.pt",
    "tl3final_fin256_s3": "model_ep3.pt",
    "tl3final_fin256_s4": "model_ep3.pt",
    "tl3final_fmu128_s3": "model_ep3.pt",
    "tl3final_fmu128_s4": "model_ep3.pt",
    "tl3final_frawW_s3": "model_ep3.pt",
    "tl3final_frawW_s4": "model_ep3.pt",
}
_LINEAGE_RUNS = tuple(sorted(_LINEAGE_CHECKPOINTS))
_CACHE_FILES = {
    "families_train",
    "families_valid",
    "main_source_train",
    "main_source_valid",
}


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256
    )


def _fit_members_digest(groups: Mapping[str, list[int]]) -> str:
    members = [
        [source, group]
        for source in ("train", "valid")
        for group in groups[source]
    ]
    encoded = json.dumps(members, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_final_fit_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("W18 receipt must bind the exact final-fit identity")
    expected_fields = {
        "schema",
        "policy",
        "manifest_sha256",
        "fit_members_sha256",
        "source_sha256",
        "source_validation_sha256",
        "fit_row_groups",
        "independent_evaluation_members",
        "independent_score_claim",
        "post_fit_primary_valid_scoring",
    }
    fit = dict(value)
    if set(fit) != expected_fields or fit.get("schema") != FINAL_FIT_SCHEMA:
        raise ValueError("W18 receipt final-fit fields or schema differ")
    if fit.get("policy") != FINAL_FIT_POLICY:
        raise ValueError("W18 receipt does not bind the approved all-labeled fit policy")
    if any(
        not _valid_sha256(fit.get(name))
        for name in ("manifest_sha256", "fit_members_sha256", "source_validation_sha256")
    ):
        raise ValueError("W18 receipt contains a malformed final-fit hash")
    sources = fit.get("source_sha256")
    if (
        not isinstance(sources, Mapping)
        or set(sources) != {"train", "valid"}
        or not all(_valid_sha256(sources.get(name)) for name in ("train", "valid"))
        or fit.get("source_validation_sha256") != SOURCE_VALIDATION_SHA256
    ):
        raise ValueError("W18 receipt source identities differ from the reviewed sources")
    groups = fit.get("fit_row_groups")
    if (
        not isinstance(groups, Mapping)
        or set(groups) != {"train", "valid"}
        or groups.get("train") != list(range(TRAIN_SEQUENCES))
        or groups.get("valid") != list(range(VALID_SEQUENCES))
        or any(
            type(group) is not int
            for source in ("train", "valid")
            for group in groups[source]
        )
        or fit.get("fit_members_sha256") != _fit_members_digest(groups)
    ):
        raise ValueError("W18 receipt fit membership is not all 12,480 whole sequences")
    if (
        fit.get("independent_evaluation_members") != []
        or fit.get("independent_score_claim") != "forbidden"
        or fit.get("post_fit_primary_valid_scoring") != "forbidden"
    ):
        raise ValueError("W18 receipt cannot claim independent evaluation or a score")
    return fit


def _validate_lineage_closure(
    value: object, fit: Mapping[str, object], w18_checkpoint_sha256: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"schema", "roles", "sha256"}:
        raise ValueError("W18 receipt lineage closure has missing or unsupported fields")
    roles = value.get("roles")
    if (
        value.get("schema") != W18_LINEAGE_SCHEMA
        or not isinstance(roles, list)
        or len(roles) != len(_LINEAGE_RUNS)
        or value.get("sha256") != canonical_sha256(roles)
    ):
        raise ValueError("W18 receipt lineage closure is malformed or not hash-bound")
    expected_record_fields = {
        "run",
        "checkpoint_file",
        "checkpoint_sha256",
        "final_fit_manifest_sha256",
        "final_fit_members_sha256",
        "source_validation_sha256",
    }
    records: dict[str, dict[str, object]] = {}
    for raw in roles:
        if not isinstance(raw, Mapping) or set(raw) != expected_record_fields:
            raise ValueError("W18 lineage entry has missing or unsupported fields")
        record = dict(raw)
        run = record.get("run")
        if (
            not isinstance(run, str)
            or run in records
            or run not in _LINEAGE_CHECKPOINTS
            or record.get("checkpoint_file") != _LINEAGE_CHECKPOINTS[run]
            or not _valid_sha256(record.get("checkpoint_sha256"))
            or record.get("final_fit_manifest_sha256") != fit["manifest_sha256"]
            or record.get("final_fit_members_sha256") != fit["fit_members_sha256"]
            or record.get("source_validation_sha256") != fit["source_validation_sha256"]
        ):
            raise ValueError("W18 lineage entry is duplicated or has a mismatched fit identity")
        records[run] = record
    if tuple(sorted(records)) != _LINEAGE_RUNS:
        raise ValueError("W18 receipt must bind exactly the W18 model and 12 actual ancestors")
    if records[W18_RUN]["checkpoint_sha256"] != w18_checkpoint_sha256:
        raise ValueError("W18 checkpoint differs between its member and lineage closure")
    return dict(value)


def validate_final_w18_receipt(
    value: object, receipt_sha256: str | None = None
) -> dict[str, object]:
    """Validate a one-member W18 receipt; the four-member receipt schema is rejected."""
    if not isinstance(value, Mapping) or value.get("schema") != W18_RECEIPT_SCHEMA:
        raise ValueError("unsupported single-W18 receipt schema")
    receipt = dict(value)
    required = {
        "schema",
        "final_fit",
        "fit_sequences",
        "independent_evaluation_members",
        "independent_evaluation_sequences",
        "independent_score_claim",
        "fixed_endpoint",
        "members",
        "plan_sha256",
        "teacher_recipe",
        "teacher_cache",
        "lineage_closure",
    }
    if set(receipt) != required:
        raise ValueError("single-W18 receipt has missing or unsupported fields")
    if receipt_sha256 is not None and not _valid_sha256(receipt_sha256):
        raise ValueError("single-W18 receipt SHA256 is malformed")
    fit = _validate_final_fit_identity(receipt.get("final_fit"))
    if (
        type(receipt.get("fit_sequences")) is not int
        or receipt["fit_sequences"] != TOTAL_SEQUENCES
        or receipt.get("independent_evaluation_members") != []
        or type(receipt.get("independent_evaluation_sequences")) is not int
        or receipt["independent_evaluation_sequences"] != 0
        or receipt.get("independent_score_claim") != "forbidden"
        or receipt.get("plan_sha256") != W18_PLAN_SHA256
        or receipt.get("fixed_endpoint") != W18_ENDPOINT
    ):
        raise ValueError("single-W18 receipt differs from the fixed all-corpus endpoint")

    members = receipt.get("members")
    if not isinstance(members, Mapping) or set(members) != {"w18"}:
        raise ValueError("single-W18 receipt must contain exactly the W18 member")
    member = members["w18"]
    expected_member_fields = {
        "run",
        "checkpoint_sha256",
        "configuration_sha256",
        "provenance_sha256",
    }
    if (
        not isinstance(member, Mapping)
        or set(member) != expected_member_fields
        or member.get("run") != W18_RUN
        or any(
            not _valid_sha256(member.get(field))
            for field in (
                "checkpoint_sha256",
                "configuration_sha256",
                "provenance_sha256",
            )
        )
    ):
        raise ValueError("single-W18 receipt member identity is incomplete or unsupported")

    closure = _validate_lineage_closure(
        receipt.get("lineage_closure"), fit, str(member["checkpoint_sha256"])
    )
    closure_by_run = {record["run"]: record for record in closure["roles"]}
    recipe = receipt.get("teacher_recipe")
    if not isinstance(recipe, list) or len(recipe) != len(FINAL_W18_TEACHERS):
        raise ValueError("single-W18 receipt must bind all six frozen teachers")
    expected_recipe_fields = {
        "run",
        "family",
        "coefficient",
        "epoch",
        "checkpoint_file",
        "checkpoint_sha256",
    }
    for item, (run, family, coefficient, epoch) in zip(
        recipe, FINAL_W18_TEACHERS, strict=True
    ):
        closure_record = closure_by_run[run]
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_recipe_fields
            or item.get("run") != run
            or item.get("family") != family
            or item.get("coefficient") != coefficient
            or item.get("epoch") != epoch
            or item.get("checkpoint_file") != _LINEAGE_CHECKPOINTS[run]
            or item.get("checkpoint_sha256") != closure_record["checkpoint_sha256"]
        ):
            raise ValueError("single-W18 teacher recipe differs from its audited closure")

    teacher_cache = receipt.get("teacher_cache")
    if (
        not isinstance(teacher_cache, Mapping)
        or set(teacher_cache) != {"schema", "metadata_sha256", "verified_files"}
        or teacher_cache.get("schema") != W18_CACHE_SCHEMA
        or not _valid_sha256(teacher_cache.get("metadata_sha256"))
        or not isinstance(teacher_cache.get("verified_files"), Mapping)
        or set(teacher_cache["verified_files"]) != _CACHE_FILES
        or not all(_valid_sha256(digest) for digest in teacher_cache["verified_files"].values())
    ):
        raise ValueError("single-W18 receipt teacher-cache identity is incomplete")
    return receipt


def _unique_lineage_records(
    raw_records: list[dict[str, object]], membership: FinalFitMembership
) -> list[dict[str, object]]:
    """Collapse repeated train/valid audit traversal without losing identity checks."""
    required = {
        "run",
        "checkpoint_file",
        "checkpoint_sha256",
        "final_fit_manifest_sha256",
        "final_fit_members_sha256",
        "source_validation_sha256",
    }
    unique: dict[str, dict[str, object]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("teacher ancestry audit returned an unsupported record")
        record = dict(raw)
        run = record.get("run")
        if (
            not isinstance(run, str)
            or run not in _LINEAGE_CHECKPOINTS
            or record.get("checkpoint_file") != _LINEAGE_CHECKPOINTS[run]
            or record.get("final_fit_manifest_sha256") != membership.manifest_sha256
            or record.get("final_fit_members_sha256") != membership.fit_members_sha256
            or record.get("source_validation_sha256") != membership.source_validation_sha256
        ):
            raise ValueError("teacher ancestry audit escaped the exact W18 final-fit closure")
        previous = unique.setdefault(run, record)
        if previous != record:
            raise ValueError(f"teacher ancestry audit disagrees for duplicate run {run}")
    if tuple(sorted(unique)) != _LINEAGE_RUNS:
        missing = sorted(set(_LINEAGE_RUNS) - set(unique))
        extra = sorted(set(unique) - set(_LINEAGE_RUNS))
        raise ValueError(f"W18 closure must contain the exact 13 roles; missing={missing}, extra={extra}")
    return [unique[run] for run in _LINEAGE_RUNS]


def _build_final_w18_receipt(
    *,
    checkpoint_path: Path,
    membership: FinalFitMembership,
    expected_args: dict[str, object],
    cache_root: Path,
    data_root: Path,
    plan_sha256: str,
) -> dict[str, object]:
    import torch

    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = _mapping(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        "final W18 checkpoint",
    )
    checkpoint_args = _mapping(checkpoint.get("args"), "final W18 arguments")
    if checkpoint_args != expected_args:
        raise ValueError("final W18 checkpoint arguments differ from the frozen W18 plan")
    provenance = _mapping(checkpoint.get("provenance"), "final W18 provenance")
    if provenance.get("args") != checkpoint_args:
        raise ValueError("final W18 provenance arguments differ from its checkpoint")
    member = {
        "run": W18_RUN,
        "checkpoint_sha256": checkpoint_sha256,
        "configuration_sha256": canonical_sha256(checkpoint_args),
        "provenance_sha256": canonical_sha256(provenance),
    }
    _validate_final_checkpoint("w18", checkpoint, member, membership)

    cache_result = validate_final_distillation_cache(
        cache_root, membership, "source", require_families=True
    )
    cache_metadata = _mapping(cache_result.get("metadata"), "final W18 cache metadata")
    cache_sources = cache_metadata.get("sources")
    if not isinstance(cache_sources, list):
        raise ValueError("final W18 teacher cache lacks its source records")
    source_by_run = {
        source.get("run"): source for source in cache_sources if isinstance(source, Mapping)
    }
    if len(source_by_run) != len(cache_sources):
        raise ValueError("final W18 teacher cache source runs are duplicated or malformed")

    raw_closure = audit_final_fit_ancestor_closure(
        W18_RUN, "model_ep1.pt", membership, data_root
    )
    lineage_records = _unique_lineage_records(raw_closure, membership)
    lineage_by_run = {record["run"]: record for record in lineage_records}
    teacher_recipe: list[dict[str, object]] = []
    for run, family, coefficient, epoch in FINAL_W18_TEACHERS:
        source = source_by_run.get(run)
        lineage = lineage_by_run.get(run)
        if (
            not isinstance(source, Mapping)
            or lineage is None
            or source.get("checkpoint_sha256") != lineage["checkpoint_sha256"]
        ):
            raise ValueError(f"final W18 cache source is not in the audited closure: {run}")
        teacher_recipe.append(
            {
                "run": run,
                "family": family,
                "coefficient": coefficient,
                "epoch": epoch,
                "checkpoint_file": lineage["checkpoint_file"],
                "checkpoint_sha256": lineage["checkpoint_sha256"],
            }
        )

    fit = final_fit_identity(membership)
    receipt = {
        "schema": W18_RECEIPT_SCHEMA,
        "final_fit": fit,
        "fit_sequences": TOTAL_SEQUENCES,
        "independent_evaluation_members": [],
        "independent_evaluation_sequences": 0,
        "independent_score_claim": "forbidden",
        "fixed_endpoint": dict(W18_ENDPOINT),
        "members": {"w18": member},
        "plan_sha256": plan_sha256,
        "teacher_recipe": teacher_recipe,
        "teacher_cache": {
            "schema": W18_CACHE_SCHEMA,
            "metadata_sha256": cache_result["metadata_sha256"],
            "verified_files": cache_result["verified_files"],
        },
        "lineage_closure": {
            "schema": W18_LINEAGE_SCHEMA,
            "roles": lineage_records,
            "sha256": canonical_sha256(lineage_records),
        },
    }
    return validate_final_w18_receipt(receipt)


def _resolve_inputs(args: argparse.Namespace) -> tuple[FinalFitMembership, dict[str, object]]:
    data_root = args.data_root.resolve(strict=True)
    membership_path = args.membership.resolve(strict=True)
    source_audit_path = args.source_audit.resolve(strict=True)
    plan_path = args.plan.resolve(strict=True)
    membership = load_final_fit_membership(
        membership_path, data_root, SOURCE_VALIDATION_SHA256
    )
    validate_source_validation_receipt(
        source_audit_path, SOURCE_VALIDATION_SHA256, membership
    )
    plan = chain_contract.read_json(plan_path, 2_000_000)
    plan_nodes = chain_contract.verify_plan(plan, plan_path)
    if plan.get("plan_sha256") != W18_PLAN_SHA256:
        raise ValueError("W18 package plan differs from the pinned finite fit plan")
    node = plan_nodes.get(W18_NODE_ID)
    if (
        not isinstance(node, Mapping)
        or node.get("run") != W18_RUN
        or node.get("endpoint") != W18_ENDPOINT["checkpoint_epoch"]
        or node.get("epochs") != W18_ENDPOINT["configured_epochs"]
    ):
        raise ValueError("frozen final plan does not contain the fixed W18 endpoint")

    parser = build_parser()
    expected = parser.parse_args(node["args"])
    if (
        Path(expected.final_fit_manifest).resolve() != membership.path
        or Path(expected.final_fit_source_validation_receipt).resolve() != source_audit_path
        or expected.final_fit_source_validation_sha256 != SOURCE_VALIDATION_SHA256
    ):
        raise ValueError("W18 plan membership or source-validation paths differ")
    plan_membership = load_final_fit_arguments(parser, expected, data_root)
    if plan_membership is None or plan_membership != membership:
        raise ValueError("W18 plan does not resolve to the supplied complete-fit membership")
    validate_final_fit_training_options(expected)
    if (
        expected.epochs != W18_ENDPOINT["configured_epochs"]
        or expected.checkpoint_epoch != W18_ENDPOINT["checkpoint_epoch"]
        or expected.name != W18_RUN
    ):
        raise ValueError("W18 plan arguments differ from the pinned final endpoint")
    if Path(expected.final_fit_source_validation_receipt).resolve() != source_audit_path:
        raise ValueError("W18 plan source-validation receipt differs from supplied receipt")
    return membership, serialized_args(expected)


def _write_package(args: argparse.Namespace) -> dict[str, object]:
    if args.max_zip_bytes <= 0 or args.max_zip_bytes > MAX_ZIP_BYTES:
        raise ValueError(f"max zip bytes must lie in 1..{MAX_ZIP_BYTES}")
    output = args.output_zip.resolve()
    model_receipt_path = args.model_receipt.resolve()
    build_receipt_path = (args.build_receipt or output.with_name(
        f"{output.stem}_build_receipt.json"
    )).resolve()
    if len({output, model_receipt_path, build_receipt_path}) != 3:
        raise ValueError("W18 ZIP, model receipt and build receipt paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in (output, model_receipt_path, build_receipt_path)):
        raise FileExistsError("W18 package outputs are create-only")

    membership, expected_args = _resolve_inputs(args)
    checkpoint_path = args.w18.resolve(strict=True)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError("final W18 checkpoint must be a regular non-symlink file")
    expected_checkpoint_path = (
        args.data_root.resolve()
        / "artifacts"
        / "runs"
        / W18_RUN
        / f"model_ep{W18_ENDPOINT['checkpoint_epoch']}.pt"
    )
    if checkpoint_path != expected_checkpoint_path.resolve():
        raise ValueError("W18 checkpoint path differs from its receipt-bound final run")

    cache_root = args.data_root.resolve() / "artifacts" / "runs" / W18_TEACHER_CACHE
    expected_cache = Path(str(expected_args["distill_cache_root"])).resolve()
    if cache_root.resolve() != expected_cache:
        raise ValueError("W18 teacher cache path differs from the frozen recipe")
    final_receipt = _build_final_w18_receipt(
        checkpoint_path=checkpoint_path,
        membership=membership,
        expected_args=expected_args,
        cache_root=cache_root,
        data_root=args.data_root.resolve(),
        plan_sha256=W18_PLAN_SHA256,
    )
    final_receipt_bytes = (
        json.dumps(final_receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(final_receipt_bytes) > MAX_RECEIPT_BYTES:
        raise ValueError("single-W18 model receipt exceeds its bounded size")
    final_receipt_sha256 = hashlib.sha256(final_receipt_bytes).hexdigest()
    validate_final_w18_receipt(final_receipt, final_receipt_sha256)
    checked_final_receipt = {
        "receipt": final_receipt,
        "membership": membership,
        "sha256": final_receipt_sha256,
    }

    native_root = args.native_from.resolve(strict=True)
    native_paths = {
        level: native_root / level / "wnn_native.abi3.so"
        for level in ("v2", "avx2", "v3")
    }
    for path in native_paths.values():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing final W18 native library: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    model_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    build_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".final-w18-", dir=output.parent) as temporary:
        stage = Path(temporary)
        package = stage / "pkg"
        package.mkdir()
        shutil.copyfile(ROOT / "submission" / "solution_final_w18.py", package / "solution.py")
        (package / "final_w18_receipt.json").write_bytes(final_receipt_bytes)

        member_record = _load_member(checkpoint_path, "w18", package, checked_final_receipt)
        layout = member_record["layout"]
        single_member = {
            key: member_record[key]
            for key in (
                "role",
                "run_reference",
                "checkpoint_sha256",
                "configuration_sha256",
                "provenance_sha256",
                "weights",
                "weights_sha256",
            )
        } | {"input_width": layout["input_width"]}
        native_hashes: dict[str, str] = {}
        for level, source in native_paths.items():
            relative = f"{level}/wnn_native.abi3.so"
            target = package / relative
            target.parent.mkdir()
            shutil.copyfile(source, target)
            native_hashes[relative] = sha256_file(target)

        manifest = {
            "schema": W18_PACKAGE_SCHEMA,
            "candidate_scope": W18_SCOPE,
            "training_and_selection_scope": "all official train and valid labels fit; no independent evaluation",
            "independent_evaluation_sequences": 0,
            "independent_score_claim": "forbidden",
            "serving_policy": "w18_only",
            "precision": "q8vnni",
            "fast_activations": True,
            "active_member_indices": [0],
            "callback_output": {
                "schema": "wnn.single-member-output.v1",
                "role": "w18",
                "weight": 1.0,
                "clip": "official-global-clip-after-callback",
            },
            "final_model_receipt": {
                "path": "final_w18_receipt.json",
                "schema": W18_RECEIPT_SCHEMA,
                "sha256": final_receipt_sha256,
            },
            "members": [single_member],
            "native_libraries": native_hashes,
        }
        (package / "members.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        zip_members: dict[str, str] = {}
        staged_zip = stage / "submission.zip"
        with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(item for item in package.rglob("*") if item.is_file()):
                relative = path.relative_to(package).as_posix()
                archive.write(path, relative)
                zip_members[relative] = sha256_file(path)
        if staged_zip.stat().st_size > args.max_zip_bytes:
            raise ValueError(
                f"W18 submission is {staged_zip.stat().st_size} bytes; limit is {args.max_zip_bytes}"
            )
        with zipfile.ZipFile(staged_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("single-W18 ZIP failed CRC verification")

        build_receipt = {
            "schema": W18_BUILD_SCHEMA,
            "artifact": {
                "path": str(output),
                "bytes": staged_zip.stat().st_size,
                "sha256": sha256_file(staged_zip),
                "members": zip_members,
            },
            "candidate_scope": W18_SCOPE,
            "serving_policy": "w18_only",
            "active_member_indices": [0],
            "precision": "q8vnni",
            "fast_activations": True,
            "independent_evaluation_sequences": 0,
            "independent_score_claim": "forbidden",
            "final_w18_receipt": {
                "path": str(model_receipt_path),
                "schema": W18_RECEIPT_SCHEMA,
                "sha256": final_receipt_sha256,
            },
            "native_libraries": native_hashes,
            "solution_sha256": sha256_file(package / "solution.py"),
            "packager_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint_sha256": member_record["checkpoint_sha256"],
            "lineage_closure_sha256": final_receipt["lineage_closure"]["sha256"],
        }
        with output.open("xb") as destination:
            destination.write(staged_zip.read_bytes())
        with model_receipt_path.open("xb") as destination:
            destination.write(final_receipt_bytes)
        with build_receipt_path.open("x", encoding="utf-8") as destination:
            destination.write(
                json.dumps(build_receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
    return build_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w18", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--native-from", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--max-zip-bytes", type=int, default=MAX_ZIP_BYTES)
    args = parser.parse_args()
    print(json.dumps(_write_package(args), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
