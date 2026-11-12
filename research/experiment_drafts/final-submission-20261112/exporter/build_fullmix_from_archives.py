"""Build the fixed all-labeled-data 90/10 TL3/LSTM package from frozen archives.

This is a packaging/export step. It does not fit a model or produce a score.
The resulting full_mix callback still requires a fresh Python 3.11 CPU runtime
profile before any upload decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

CHECKPOINT_ARCHIVE_SHA256 = "207b9251b89ad27c814fe26110ac30fd2d1a209cbcabd1498ecb315392d783b8"
W18_ZIP_SHA256 = "932700e4e10721829641870f18198fca06ba3f9c5f06be5541d70bf38bf880f9"
W18_EVIDENCE_ARCHIVE_SHA256 = "2dc00d8a16f316775ddb979382312ee8a1d60b3c3555128726b8c4a44786c29e"
W18_CHECKPOINT_SHA256 = "70cfefcfeb14b5a5e1f2a181fc7ddf15330a24aaf32e7f324b122fb54327daf8"
PROJECT_COMMIT = "6493556e853e02538d50baed049073eeb717e9d7"
SOURCE_FILE_SHA256 = {
    "submission/solution_final_tl3.py": "e194efa82a83d1ebf745ec0f8027a73344c5be6e7ed2e81e4594b4cf17501c23",
    "scripts/package_final_tl3.py": "4ff9d3975285e3c41fa8fa82ceeffa724258b6bbc069c2f540a4f65f7dfda101",
}
COMPAT_HELPER_SHA256 = "cb094e932b7f04b6e3eb12621a3f28879bbd5a2a1a9f3f2271e7f6b72ded75b9"
COMPAT_REPAIR_RECEIPT_SHA256 = "e1853c576e02f9d26a9e5aaf8c3c05aa5507f31812150ae3d22b8c030b4152a1"
COMPAT_REPAIR_SCHEMA = "wnn.final-submit-source4-history-compatibility-repair.r1"
W18_EVIDENCE_CHECKPOINT = "models/tl3final_w18final_distilled_s5/model_ep1.pt"
W18_EVIDENCE_RECEIPT = "receipts/final-w18-model-receipt.json"
W18_EVIDENCE_MANIFEST = "MANIFEST.json"
W18_EVIDENCE_MEMBERSHIP = "inputs/final-fit-membership-r1.json"
W18_EVIDENCE_VALIDATION = "inputs/SOURCE_VALIDATION_R1.json"
CHECKPOINT_ARCHIVE_SCHEMA = "wnn.final-fit-checkpoint-archive.v1"
W18_PACKAGE_MEMBERS = frozenset(
    {
        "solution.py",
        "members.json",
        "final_w18_receipt.json",
        "w18_fit_process_receipt.json",
        "weights_w18.npz",
        "v2/wnn_native.abi3.so",
        "avx2/wnn_native.abi3.so",
        "v3/wnn_native.abi3.so",
    }
)
FINAL_CHECKPOINTS = {
    "tiedlead_s5": ("tl3final_fttall_tlead_s5", "models/fttall_tlead_s5/model.pt"),
    "tiedlead_s6": ("tl3final_fttall_tlead_s6", "models/fttall_tlead_s6/model.pt"),
    "source4_lstm": ("final-lstm192x1", "models/source4_lstm192x1/model.pt"),
}
MAX_ZIP_BYTES = 20_000_000


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular input file in bounded memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular input file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a bounded JSON object from a regular file."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError(f"expected a regular JSON file under 2 MB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    """Serialize an evidence object deterministically with no non-finite values."""
    return json.dumps(value, allow_nan=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def safe_member(name: str) -> None:
    """Reject unsafe archive paths before reading or writing a member."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe ZIP member path: {name}")


def find_archive_item(manifest: dict[str, Any], run: str) -> dict[str, Any]:
    """Return the unique source-qualified checkpoint record for a final run."""
    items = manifest.get("items")
    matches = [item for item in items if isinstance(item, dict) and item.get("run") == run]
    if len(matches) != 1:
        raise ValueError(f"checkpoint archive must contain exactly one record for {run}")
    return matches[0]


def checkpoint_identity(checkpoint_bytes: bytes, role: str, package: Any) -> dict[str, Any]:
    """Read checkpoint metadata only and derive hashes bound into final receipt."""
    import torch

    torch.set_num_threads(1)
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{role} checkpoint envelope is not a mapping")
    config, _ = package._source_config(dict(checkpoint), role)
    provenance = package._mapping(checkpoint.get("provenance"), f"{role} provenance")
    identity: dict[str, Any] = {
        "run": package.FINAL_RUNS[package.ROLES.index(role)],
        "checkpoint_sha256": sha256_bytes(checkpoint_bytes),
        "configuration_sha256": package.canonical_sha256(config),
        "provenance_sha256": package.canonical_sha256(provenance),
    }
    if role == "source4_lstm":
        data = package._mapping(checkpoint.get("data"), "Source4 data provenance")
        identity["data_sha256"] = package.canonical_sha256(data)
    return identity


def load_package_helper(
    project: Path,
    helper_path: Path | None,
    repair_receipt_path: Path | None,
) -> tuple[Any, dict[str, str] | None]:
    """Load the pinned base validator or an explicitly selected pinned overlay."""
    if (helper_path is None) != (repair_receipt_path is None):
        raise ValueError("--compat-helper and --compat-repair-receipt must be provided together")
    if helper_path is None:
        from scripts import package_final_tl3 as package

        return package, None

    helper = helper_path.resolve(strict=True)
    repair_path = repair_receipt_path.resolve(strict=True)
    helper_sha = sha256_file(helper)
    repair_sha = sha256_file(repair_path)
    if helper_sha != COMPAT_HELPER_SHA256:
        raise ValueError("compatibility helper bytes differ from the pinned Source4 overlay")
    if repair_sha != COMPAT_REPAIR_RECEIPT_SHA256:
        raise ValueError("compatibility repair receipt differs from the pinned receipt")
    repair = read_json(repair_path)
    if (
        repair.get("schema") != COMPAT_REPAIR_SCHEMA
        or repair.get("base_source")
        != {
            "git_commit": PROJECT_COMMIT,
            "path": "scripts/package_final_tl3.py",
            "sha256": SOURCE_FILE_SHA256["scripts/package_final_tl3.py"],
        }
        or repair.get("overlay_source", {}).get("sha256") != helper_sha
    ):
        raise ValueError("compatibility repair receipt does not bind the pinned base and overlay")
    spec = importlib.util.spec_from_file_location("package_final_tl3_source4_compat_r1", helper)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load compatibility helper: {helper}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    if not callable(getattr(package, "validate_source4_final_history", None)):
        raise ImportError("compatibility helper lacks the pinned Source4 history validator")
    return package, {"helper_sha256": helper_sha, "repair_receipt_sha256": repair_sha}


def main() -> None:
    """Create one immutable full_mix ZIP and its source/ancestry receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="exact final-submit source checkout")
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--w18-zip", type=Path, required=True)
    parser.add_argument("--w18-build-receipt", type=Path, required=True)
    parser.add_argument("--w18-evidence-archive", type=Path, required=True)
    parser.add_argument("--final-fit-membership", type=Path, required=True)
    parser.add_argument("--source-validation-receipt", type=Path, required=True)
    parser.add_argument("--compat-helper", type=Path)
    parser.add_argument("--compat-repair-receipt", type=Path)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--max-zip-bytes", type=int, default=MAX_ZIP_BYTES)
    args = parser.parse_args()

    project = args.project.resolve(strict=True)
    sys.path[:0] = [str(project), str(project / "src")]

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("run the packager in the pinned offline Python 3.11 environment")
    if args.max_zip_bytes <= 0 or args.max_zip_bytes > MAX_ZIP_BYTES:
        raise ValueError(f"max ZIP bytes must be in 1..{MAX_ZIP_BYTES}")
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PROJECT_COMMIT:
        raise ValueError(f"source checkout must be pinned at {PROJECT_COMMIT}, got {commit}")
    source_hashes: dict[str, str] = {}
    for relative, expected_sha in SOURCE_FILE_SHA256.items():
        source_path = project / relative
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError(f"pinned package source is missing: {source_path}")
        actual_sha = sha256_file(source_path)
        if actual_sha != expected_sha:
            raise ValueError(f"package source bytes differ from pinned source: {relative}")
        source_hashes[relative] = actual_sha
    package, compatibility_overlay = load_package_helper(
        project, args.compat_helper, args.compat_repair_receipt
    )

    archive_sha = sha256_file(args.checkpoint_archive)
    if archive_sha != CHECKPOINT_ARCHIVE_SHA256:
        raise ValueError("final checkpoint archive bytes differ from the pinned PR54 artifact")
    w18_sha = sha256_file(args.w18_zip)
    if w18_sha != W18_ZIP_SHA256:
        raise ValueError("W18 ZIP bytes differ from the verified final W18 artifact")
    w18_build = read_json(args.w18_build_receipt)
    if w18_build.get("artifact", {}).get("sha256") != w18_sha:
        raise ValueError("W18 build receipt does not bind the supplied W18 ZIP")
    membership_raw = args.final_fit_membership.read_bytes()
    membership_doc = json.loads(membership_raw)
    if sha256_bytes(membership_raw) != "3f766945e00a2fa5edc6230d9c0bed125da112584f11e3d82458167a011bf543":
        raise ValueError("final-fit membership differs from the pinned full-fit manifest")
    validation_sha = sha256_file(args.source_validation_receipt)
    source_validation_sha = package.SOURCE_VALIDATION_SHA256
    if validation_sha != source_validation_sha:
        raise ValueError("source-validation receipt differs from the package source pin")
    w18_evidence_sha = sha256_file(args.w18_evidence_archive)
    if w18_evidence_sha != W18_EVIDENCE_ARCHIVE_SHA256:
        raise ValueError("W18 evidence archive bytes differ from the authenticated full-evidence artifact")

    output = args.output_zip.resolve()
    receipt_path = (args.build_receipt or output.with_name(f"{output.stem}_build_receipt.json")).resolve()
    if output.exists() or receipt_path.exists() or output == receipt_path:
        raise FileExistsError("full_mix ZIP and build receipt are create-only outputs")
    output.parent.mkdir(parents=True, exist_ok=True)

    with (
        zipfile.ZipFile(args.w18_zip) as w18_archive,
        zipfile.ZipFile(args.checkpoint_archive) as checkpoint_archive,
        zipfile.ZipFile(args.w18_evidence_archive) as w18_evidence_archive,
    ):
        if (
            w18_archive.testzip() is not None
            or checkpoint_archive.testzip() is not None
            or w18_evidence_archive.testzip() is not None
        ):
            raise ValueError("an input artifact failed ZIP CRC validation")
        if set(w18_archive.namelist()) != W18_PACKAGE_MEMBERS:
            raise ValueError("W18 package has missing or unexpected members")
        for name in w18_archive.namelist() + checkpoint_archive.namelist() + w18_evidence_archive.namelist():
            safe_member(name)
        build_members = w18_build.get("artifact", {}).get("members")
        if not isinstance(build_members, dict) or set(build_members) != W18_PACKAGE_MEMBERS:
            raise ValueError("W18 build receipt member manifest differs from the actual ZIP")
        for name, expected in build_members.items():
            if sha256_bytes(w18_archive.read(name)) != expected:
                raise ValueError(f"W18 ZIP member hash mismatch: {name}")

        archive_manifest = json.loads(checkpoint_archive.read("MANIFEST.json"))
        w18_evidence_manifest = json.loads(w18_evidence_archive.read(W18_EVIDENCE_MANIFEST))
        w18_evidence_receipt = json.loads(w18_evidence_archive.read(W18_EVIDENCE_RECEIPT))
        evidence_items = {
            item.get("path"): item
            for item in w18_evidence_manifest.get("evidence_files", [])
            if isinstance(item, dict)
        }
        w18_checkpoint_bytes = w18_evidence_archive.read(W18_EVIDENCE_CHECKPOINT)
        w18_checkpoint_sha = sha256_bytes(w18_checkpoint_bytes)
        if (
            w18_evidence_manifest.get("schema") != "wnn.final-w18-closure-evidence.v1"
            or w18_evidence_manifest.get("membership_sha256") != sha256_bytes(membership_raw)
            or w18_evidence_manifest.get("source_validation_sha256") != validation_sha
            or w18_evidence_manifest.get("fit_sequences") != 12_480
            or w18_evidence_manifest.get("independent_evaluation_sequences") != 0
            or w18_evidence_manifest.get("independent_score_claim") != "forbidden"
            or evidence_items.get(W18_EVIDENCE_CHECKPOINT, {}).get("sha256") != W18_CHECKPOINT_SHA256
            or w18_checkpoint_sha != W18_CHECKPOINT_SHA256
            or w18_evidence_receipt.get("schema") != "wnn.final-w18-model-receipt.v1"
            or w18_evidence_receipt.get("members", {}).get("w18", {}).get("checkpoint_sha256") != W18_CHECKPOINT_SHA256
        ):
            raise ValueError("W18 full-evidence archive does not bind the exact all-fit checkpoint and closure")
        if (
            w18_evidence_archive.read(W18_EVIDENCE_MEMBERSHIP) != membership_raw
            or w18_evidence_archive.read(W18_EVIDENCE_VALIDATION) != args.source_validation_receipt.read_bytes()
        ):
            raise ValueError("supplied membership or source-validation bytes differ from the W18 evidence closure")
        if (
            archive_manifest.get("schema") != CHECKPOINT_ARCHIVE_SCHEMA
            or archive_manifest.get("fit_sequences") != 12_480
            or archive_manifest.get("independent_evaluation_sequences") != 0
            or archive_manifest.get("completed_model_artifacts") != 17
            or archive_manifest.get("planned_scientific_fit_roles") != 18
            or archive_manifest.get("partial") is not True
            or archive_manifest.get("membership_sha256") != sha256_bytes(membership_raw)
        ):
            raise ValueError("PR54 archive is not the pinned 17-of-18 all-labeled fit archive")

        w18_receipt = w18_evidence_receipt
        w18_receipt_bytes = w18_evidence_archive.read(W18_EVIDENCE_RECEIPT)
        w18_manifest = json.loads(w18_archive.read("members.json"))
        w18_member = w18_receipt.get("members", {}).get("w18")
        w18_manifest_member = w18_manifest.get("members", [None])[0]
        if (
            w18_receipt.get("schema") != "wnn.final-w18-model-receipt.v1"
            or not isinstance(w18_member, dict)
            or w18_member.get("run") != package.FINAL_RUNS[0]
            or sha256_bytes(w18_receipt_bytes) != w18_build.get("final_w18_receipt", {}).get("sha256")
            or w18_archive.read("final_w18_receipt.json") != w18_receipt_bytes
            or w18_manifest.get("independent_evaluation_sequences") != 0
            or w18_manifest.get("independent_score_claim") != "forbidden"
            or not isinstance(w18_manifest_member, dict)
            or w18_manifest_member.get("weights") != "weights_w18.npz"
        ):
            raise ValueError("W18 package lacks its exact full-fit member/receipt binding")
        if sha256_bytes(membership_raw) != w18_receipt["final_fit"]["manifest_sha256"]:
            raise ValueError("final-fit membership bytes differ from the W18 receipt closure")
        if w18_receipt["final_fit"].get("source_validation_sha256") != validation_sha:
            raise ValueError("final-fit receipt does not bind the supplied source-validation receipt")

        fit = {
            "schema": package.FINAL_FIT_SCHEMA,
            "policy": package.FINAL_FIT_POLICY,
            "manifest_sha256": sha256_bytes(membership_raw),
            "fit_members_sha256": membership_doc["fit_members_sha256"],
            "source_sha256": {
                "train": membership_doc["sources"]["train"]["sha256"],
                "valid": membership_doc["sources"]["valid"]["sha256"],
            },
            "source_validation_sha256": validation_sha,
            "fit_row_groups": membership_doc["fit_members"],
            "independent_evaluation_members": [],
            "independent_score_claim": "forbidden",
            "post_fit_primary_valid_scoring": "forbidden",
        }
        if w18_receipt["final_fit"] != fit:
            raise ValueError("W18 and checkpoint archive do not share the exact all-fit identity")

        checkpoint_items = {
            role: find_archive_item(archive_manifest, run) for role, (run, _) in FINAL_CHECKPOINTS.items()
        }
        identities: dict[str, dict[str, Any]] = {}
        checkpoint_bytes: dict[str, bytes] = {"w18": w18_checkpoint_bytes}
        identities["w18"] = checkpoint_identity(w18_checkpoint_bytes, "w18", package)
        if identities["w18"] != w18_member:
            raise ValueError("raw W18 checkpoint config/provenance identity differs from its final receipt")
        for role, (run, member_name) in FINAL_CHECKPOINTS.items():
            item = checkpoint_items[role]
            if (
                item.get("run") != run
                or item.get("fixed_epoch") != package.FINAL_FIXED_EPOCHS[role]["checkpoint_epoch"]
                or item.get("configured_epochs") != package.FINAL_FIXED_EPOCHS[role]["configured_epochs"]
            ):
                raise ValueError(f"archive endpoint is not the fixed final checkpoint: {run}")
            raw = checkpoint_archive.read(member_name)
            if sha256_bytes(raw) != item.get("checkpoint_sha256"):
                raise ValueError(f"checkpoint bytes differ from archive manifest: {run}")
            checkpoint_bytes[role] = raw
            identities[role] = checkpoint_identity(raw, role, package)

        final_receipt_value = {
            "schema": package.FINAL_RECEIPT_SCHEMA,
            "final_fit": fit,
            "fit_sequences": 12_480,
            "independent_evaluation_members": [],
            "independent_evaluation_sequences": 0,
            "independent_score_claim": "forbidden",
            "fixed_epochs": package.FINAL_FIXED_EPOCHS,
            "members": identities,
        }

        with tempfile.TemporaryDirectory(prefix=".fullmix-package-", dir=output.parent) as temporary:
            stage = Path(temporary)
            package_root = stage / "package"
            package_root.mkdir()
            final_receipt_path = stage / "final_model_set_receipt.json"
            final_receipt_bytes = json_bytes(final_receipt_value)
            final_receipt_path.write_bytes(final_receipt_bytes)
            final_receipt = package.load_final_model_receipt(
                final_receipt_path, sha256_bytes(final_receipt_bytes)
            )

            solution_path = project / "submission" / "solution_final_tl3.py"
            shutil.copyfile(solution_path, package_root / "solution.py")
            (package_root / "precision.txt").write_text("q8vnni\n", encoding="utf-8")
            import numpy as np

            member_records = []
            for role in package.ROLES:
                if role == "w18":
                    checkpoint_path = stage / "w18.pt"
                else:
                    _, member_name = FINAL_CHECKPOINTS[role]
                    checkpoint_path = stage / f"{role}.pt"
                    checkpoint_path.write_bytes(checkpoint_bytes[role])
                if role == "w18":
                    checkpoint_path.write_bytes(checkpoint_bytes[role])
                member_records.append(package._load_member(checkpoint_path, role, package_root, final_receipt))

            # The previously delivered W18 package provides an independent exact
            # conversion reference; compare tensors without constructing a model.
            w18_reference = w18_archive.read("weights_w18.npz")
            with np.load(io.BytesIO(w18_reference), allow_pickle=False) as reference, np.load(
                package_root / "weights_w18.npz", allow_pickle=False
            ) as exported:
                if set(reference.files) != set(exported.files) or any(
                    not np.array_equal(reference[name], exported[name]) for name in reference.files
                ):
                    raise ValueError("re-exported W18 checkpoint tensors differ from the verified W18 package")

            members_manifest = {
                "schema": "wnn.final-tl3-lstm-members.v1",
                "precision": "q8vnni",
                "fast_activations": True,
                "serving_policy": "full_mix",
                "active_member_indices": [0, 1, 2, 3],
                "candidate_blend": package.BLEND,
                "blend": package.BLEND,
                "callback_output": {"schema": "wnn.nested-fp64-fp32.v1", "blend": package.BLEND},
                "candidate_scope": "approved_full_corpus_final_model_set",
                "training_and_selection_scope": "all official train and valid labels fit; no independent evaluation",
                "independent_evaluation_sequences": 0,
                "independent_score_claim": "forbidden",
                "final_fit": fit,
                "final_model_receipt": {"schema": package.FINAL_RECEIPT_SCHEMA, "sha256": sha256_bytes(final_receipt_bytes)},
                "members": [
                    {
                        key: record[key]
                        for key in ("role", "run_reference", "checkpoint_sha256", "configuration_sha256", "model_configuration", "provenance_sha256", "data_sha256", "weights", "weights_sha256")
                    }
                    | {"input_width": record["layout"]["input_width"]}
                    for record in member_records
                ],
            }
            (package_root / "members.json").write_bytes(json_bytes(members_manifest))
            native_hashes: dict[str, str] = {}
            for level in ("v2", "avx2", "v3"):
                name = f"{level}/wnn_native.abi3.so"
                payload = w18_archive.read(name)
                if sha256_bytes(payload) != w18_build["native_libraries"][name]:
                    raise ValueError(f"W18 native-library hash mismatch: {name}")
                target = package_root / name
                target.parent.mkdir()
                target.write_bytes(payload)
                native_hashes[name] = sha256_bytes(payload)

            staged_zip = stage / "fullmix.zip"
            zip_member_hashes: dict[str, str] = {}
            with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as destination:
                for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
                    relative = path.relative_to(package_root).as_posix()
                    destination.write(path, relative)
                    zip_member_hashes[relative] = sha256_file(path)
            if staged_zip.stat().st_size > args.max_zip_bytes:
                raise ValueError(f"full_mix ZIP exceeds {args.max_zip_bytes} bytes")
            with zipfile.ZipFile(staged_zip) as built:
                if built.testzip() is not None:
                    raise ValueError("full_mix ZIP failed CRC validation")
                if {name for name in built.namelist() if not name.endswith("/")} != set(zip_member_hashes):
                    raise ValueError("full_mix ZIP member inventory changed while writing")

            receipt = {
                "schema": "wnn.final-tl3-lstm-build-receipt.v1",
                "artifact": {"path": str(output), "bytes": staged_zip.stat().st_size, "sha256": sha256_file(staged_zip), "members": zip_member_hashes},
                "native_libraries": native_hashes,
                "solution_sha256": sha256_file(solution_path),
                "packager_sha256": sha256_file(Path(__file__).resolve()),
                "candidate_blend": package.BLEND,
                "applied_blend": package.BLEND,
                "serving_policy": "full_mix",
                "active_member_indices": [0, 1, 2, 3],
                "precision": "q8vnni",
                "fast_activations": True,
                "candidate_scope": members_manifest["candidate_scope"],
                "training_and_selection_scope": members_manifest["training_and_selection_scope"],
                "final_model_receipt": members_manifest["final_model_receipt"],
                "input_artifacts": {
                    "checkpoint_archive_sha256": archive_sha,
                    "checkpoint_archive_manifest_sha256": sha256_bytes(checkpoint_archive.read("MANIFEST.json")),
                    "w18_zip_sha256": w18_sha,
                    "w18_build_receipt_sha256": sha256_file(args.w18_build_receipt),
                    "w18_evidence_archive_sha256": w18_evidence_sha,
                    "w18_checkpoint_sha256": w18_checkpoint_sha,
                    "w18_evidence_manifest_sha256": sha256_bytes(w18_evidence_archive.read(W18_EVIDENCE_MANIFEST)),
                    "w18_final_model_receipt_sha256": sha256_bytes(w18_evidence_archive.read(W18_EVIDENCE_RECEIPT)),
                    "final_fit_membership_sha256": sha256_bytes(membership_raw),
                    "source_validation_receipt_sha256": validation_sha,
                },
                "source": {"git_commit": commit, "files": source_hashes},
                "compatibility_overlay": compatibility_overlay,
                "runtime_gate": "pending_fresh_python311_cpu_fullmix_profile",
                "independent_score_claim": "forbidden",
                "model_construction_tests": "not run by this packager",
            }
            if receipt_path.parent != output.parent:
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as destination:
                shutil.copyfileobj(staged_zip.open("rb"), destination)
            with receipt_path.open("xb") as destination:
                destination.write(json_bytes(receipt))
    print(json.dumps({"status": "built_unprofiled", "zip": str(output), "receipt": str(receipt_path), "zip_bytes": receipt["artifact"]["bytes"], "zip_sha256": receipt["artifact"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
