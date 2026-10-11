"""Run one fixed OOD student arm against a shared compact fold cache.

This retry path reuses completed teacher predictions, builds one immutable
homogeneous-only cache for the four R2 OOD arms, and preserves the original
training settings. It never removes the shared cache or existing artifacts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", "")).resolve()
FROZEN_SOURCE_ROOT = Path(os.environ.get("WNN_FROZEN_SOURCE_ROOT", "")).resolve()
BUNDLE_MANIFEST_PATH = SCRIPT_DIR / "OOD_R2_BUNDLE_MANIFEST.json"
FROZEN_MANIFEST_COPY = SCRIPT_DIR / "frozen_source_manifest_r1.json"
FROZEN_MANIFEST_SHA256 = "382beb68d556dd63215a9f7467bda09a23626ec3e1a2195739cffcab6819d9cc"
PROJECT_ROOT_REQUIRED = Path("/root/wnn")
PYTHON_REQUIRED = "/opt/conda/bin/python"
TIMEOUT_PREFIX = ["timeout", "--signal=TERM", "--kill-after=10s", "14400s"]


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory before importing project code."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(expected_manifest_sha256: str) -> dict[str, Any]:
    """Verify the frozen 58-file source snapshot and staged R2 scripts pre-import."""
    if PROJECT_ROOT != PROJECT_ROOT_REQUIRED:
        raise ValueError("WNN_PROJECT_ROOT must be the audited /root/wnn data/artifact root")
    if FROZEN_SOURCE_ROOT != PROJECT_ROOT_REQUIRED / "lr377_source_r1":
        raise ValueError("R2 code imports must use /root/wnn/lr377_source_r1")
    if sha256_file(BUNDLE_MANIFEST_PATH) != expected_manifest_sha256:
        raise ValueError("R2 bundle manifest hash differs from the launch receipt")
    bundle = json.loads(BUNDLE_MANIFEST_PATH.read_text())
    if bundle.get("schema") != "wnn.ood-shared-cache-repair-bundle.v2":
        raise ValueError("unsupported R2 bundle manifest")
    if bundle.get("frozen_source_manifest_sha256") != FROZEN_MANIFEST_SHA256:
        raise ValueError("R2 bundle does not bind the audited 58-file source snapshot")
    if sha256_file(FROZEN_MANIFEST_COPY) != FROZEN_MANIFEST_SHA256:
        raise ValueError("bundled frozen-source manifest copy changed")
    remote_manifest = FROZEN_SOURCE_ROOT / "SOURCE_MANIFEST.json"
    if sha256_file(remote_manifest) != FROZEN_MANIFEST_SHA256:
        raise ValueError("remote frozen SOURCE_MANIFEST.json changed")
    frozen = json.loads(FROZEN_MANIFEST_COPY.read_text())
    actual_manifest = json.loads(remote_manifest.read_text())
    expected_sources = frozen.get("sources")
    if (
        frozen.get("schema") != "wnn.frozen-source-snapshot.v1"
        or frozen.get("remote_root") != str(FROZEN_SOURCE_ROOT)
        or actual_manifest != frozen
        or not isinstance(expected_sources, dict)
        or len(expected_sources) != 58
    ):
        raise ValueError("frozen source manifest schema or file count differs")
    expected_file_set = set(expected_sources)
    actual_file_set = {
        str(path.relative_to(FROZEN_SOURCE_ROOT))
        for base in (FROZEN_SOURCE_ROOT / "scripts", FROZEN_SOURCE_ROOT / "src/wnn")
        for path in base.rglob("*.py")
    }
    if actual_file_set != expected_file_set:
        raise ValueError("frozen source snapshot Python file set differs from its manifest")
    for relative, expected_sha in expected_sources.items():
        path = FROZEN_SOURCE_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"frozen source file differs from manifest: {relative}")

    bundle_files = bundle.get("bundle_files")
    if not isinstance(bundle_files, dict):
        raise ValueError("R2 bundle source hash map is missing")
    expected_names = {
        "ood_student_pipeline_shared_r2.py",
        "build_distill_cache_r2.py",
        "frozen_source_manifest_r1.json",
    }
    if set(bundle_files) != expected_names:
        raise ValueError("R2 bundle file set differs")
    for name, expected_sha in bundle_files.items():
        path = SCRIPT_DIR / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"staged R2 source changed: {name}")
    if bundle_files["ood_student_pipeline_shared_r2.py"] != sha256_file(Path(__file__)):
        raise ValueError("running R2 runner bytes are not the frozen bundle bytes")

    aliases = frozen.get("expected_aliases", {
        "artifacts": "/root/wnn/artifacts",
        "cache": "/root/wnn/cache",
        "datasets": "/root/wnn/datasets",
        "research": "/root/wnn/research",
    })
    for name, target in aliases.items():
        alias = FROZEN_SOURCE_ROOT / name
        if not alias.is_symlink() or str(alias.resolve()) != target:
            raise ValueError(f"frozen snapshot data alias differs: {name}")
    split_path = PROJECT_ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    registry_path = PROJECT_ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1/portable_inputs_fold1.json"
    if sha256_file(split_path) != bundle.get("split_manifest_sha256"):
        raise ValueError("fold-1 split manifest changed")
    if sha256_file(registry_path) != bundle.get("portable_input_manifest_sha256"):
        raise ValueError("fold-1 portable input manifest changed")
    return bundle


def _preimport_bootstrap(expected_manifest_sha256: str) -> dict[str, Any]:
    """Verify all source bytes, then make frozen project modules resolve first."""
    bundle = verify_bundle(expected_manifest_sha256)
    frozen_src = str(FROZEN_SOURCE_ROOT / "src")
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != PROJECT_ROOT / "src"]
    sys.path.insert(0, frozen_src)
    os.environ["WNN_OOD_R2_BUNDLE_MANIFEST_SHA256"] = expected_manifest_sha256
    return bundle


# The CLI's pinned manifest hash is checked before importing NumPy, torch, or wnn.
_PREIMPORT_ARGS = argparse.ArgumentParser(add_help=False)
_PREIMPORT_ARGS.add_argument("--bundle-manifest-sha256", required=True)
_PREIMPORT, _ = _PREIMPORT_ARGS.parse_known_args()
BUNDLE = _preimport_bootstrap(_PREIMPORT.bundle_manifest_sha256)

import build_distill_cache_r2 as cache_builder  # noqa: E402
import numpy as np  # noqa: E402

from wnn.distillation_cache import validate_distillation_cache  # noqa: E402
from wnn.splits import CampaignSplit, load_campaign_split  # noqa: E402

ROOT = PROJECT_ROOT

FOLD = 1
TEACHER_EPOCH = 2
TEACHERS = (
    "combo_scratch_ep3s4_onecycle_f1",
    "combo_scratch_ep3s5_onecycle_f1",
)
PARENT = "c20261003_combo_s3_f1@ep3"
PROPENSITY = "c20260929_prop_f1"
CACHE_NAME = "dc_ood_fold1_strong2_ep2_homogeneous_r2"
ALIAS_NAME = "ood_shared_fold1_strong2_ep2_homogeneous_r2"
LOCK_NAME = "ood_shared_fold1_strong2_ep2_homogeneous_r2.lock"
RECEIPT_NAME = "shared_cache_receipt.json"
MIN_FREE_BYTES = 20 * (1 << 30)
OUTPUT_BUDGET_BYTES = 16 * (1 << 30)
METADATA_RESERVE_BYTES = 16 << 20
CHECKPOINT_STATE_RESERVE_PER_ARM = 64 << 20
EXPECTED_FIT_GROUP_COUNTS = {"train": 10533, "valid": 1416}
EXPECTED_VALID_TARGET_ROWS = 1873
EXPECTED_COMPACT_CACHE_BYTES = 1_911_840_000
SHARED_CACHE_LOCK_PATH = Path("/root/run/locks/ood_shared_fold1_strong2_ep2_homogeneous_r2.lock")
RUN_SEQUENCE_LOCK_PATH = Path("/root/run/locks/ood_shared_fold1_strong2_ep2_homogeneous_r2.sequence.lock")
OOD_ARMS = {
    "splice_r2": ("splice", 0.5),
    "spmix_r2": ("splice,mixup_paths", 0.5),
    "swn_r2": ("splice,warp,noise", 0.5),
    "swn25_r2": ("splice,warp,noise", 0.25),
}
def teacher_specs() -> list[str]:
    """Return the frozen four-family alias mixture for the two OOD teachers."""
    first, second = TEACHERS
    return [
        f"strong={first}@ep{TEACHER_EPOCH}*0.150000",
        f"strong={second}@ep{TEACHER_EPOCH}*0.150000",
        f"full_finetuned={first}_a1@ep{TEACHER_EPOCH}*0.200000",
        f"raw_wmse={second}_a2@ep{TEACHER_EPOCH}*0.250000",
        f"samefold_baseline={first}_a3@ep{TEACHER_EPOCH}*0.062500",
    ]


def _target_groups(data_root: Path, split: CampaignSplit, split_name: str) -> list[int]:
    """Return the frozen target-index row order restricted to fitting groups."""
    order = cache_builder._target_order(data_root, split_name)
    allowed = set(cache_builder._expected_groups(split, split_name))
    selected = [group for group in order if group in allowed]
    if set(selected) != allowed:
        raise ValueError(f"target index omits {split_name} fitting groups")
    return selected


def _student_complete(student: str) -> bool:
    """Return whether a sibling OOD output set is already fully materialized."""
    run_root = ROOT / "artifacts" / "runs" / student
    prediction = ROOT / "cache" / f"pred_{student}_valid.npy"
    required = (
        run_root / "model_ep1.pt",
        run_root / "provenance.json",
        run_root / "training_state.pt",
        prediction,
        prediction.with_suffix(".json"),
    )
    return all(path.is_file() for path in required)


def _remaining_output_reservation(valid_rows: int) -> int:
    """Reserve missing predictions and checkpoint/state outputs for all four arms."""
    prediction_bytes = valid_rows * cache_builder.SEQUENCE_ROWS * 2 * 4
    total = 0
    for arm in OOD_ARMS:
        student = f"vt_f1_ood_{arm}_homshare_r2"
        run_root = ROOT / "artifacts" / "runs" / student
        prediction = ROOT / "cache" / f"pred_{student}_valid.npy"
        partials = (prediction.with_name(prediction.name + ".partial"),)
        if any(path.exists() for path in partials):
            raise FileExistsError(f"existing prediction partial must be inspected: {partials[0]}")
        if _student_complete(student):
            continue
        if run_root.exists() or prediction.exists() or prediction.with_suffix(".json").exists():
            raise FileExistsError(
                f"incomplete sibling output exists for {student}; refusing to treat it as free space"
            )
        total += prediction_bytes + CHECKPOINT_STATE_RESERVE_PER_ARM
    return total


def preflight_cache_space(
    cache_root: Path,
    split_path: Path,
    other_live_reservations_bytes: int,
) -> dict[str, int]:
    """Reserve cache, all remaining arm outputs, external reservations, and the 20 GiB floor."""
    data_root = ROOT
    split = load_campaign_split(split_path, data_root)
    fit_rows = sum(
        len(_target_groups(data_root, split, split_name)) for split_name in ("train", "valid")
    )
    payload_bytes = cache_builder.estimate_output_bytes(fit_rows, "homogeneous-only")
    observed_fit_counts = {
        split_name: len(_target_groups(data_root, split, split_name))
        for split_name in ("train", "valid")
    }
    if observed_fit_counts != EXPECTED_FIT_GROUP_COUNTS:
        raise ValueError(f"fold-1 fitting membership count changed: {observed_fit_counts}")
    if payload_bytes != EXPECTED_COMPACT_CACHE_BYTES:
        raise ValueError(f"compact cache payload changed: {payload_bytes}")
    with np.load(data_root / "cache" / "valid_targets.npz", allow_pickle=False) as targets:
        valid_rows = len(targets["row_group"])
    if valid_rows != EXPECTED_VALID_TARGET_ROWS:
        raise ValueError(f"student prediction row count changed: {valid_rows}")
    output_bytes = _remaining_output_reservation(valid_rows)
    cache_bytes_to_reserve = 0 if cache_root.exists() else payload_bytes
    required_bytes = (
        cache_bytes_to_reserve
        + output_bytes
        + METADATA_RESERVE_BYTES
        + other_live_reservations_bytes
    )
    if payload_bytes > OUTPUT_BUDGET_BYTES:
        raise OSError(f"homogeneous cache payload {payload_bytes} exceeds the 16 GiB output budget")
    if not cache_root.parent.is_dir():
        raise FileNotFoundError(f"cache parent is absent: {cache_root.parent}")
    free_bytes = shutil.disk_usage(cache_root.parent).free
    if free_bytes - required_bytes < MIN_FREE_BYTES:
        raise OSError(
            f"R2 requires {required_bytes} bytes including all output/external reserves; "
            f"available while preserving 20 GiB free is {max(0, free_bytes - MIN_FREE_BYTES)}"
        )
    return {
        "compact_cache_payload_bytes": payload_bytes,
        "cache_bytes_to_reserve_now": cache_bytes_to_reserve,
        "remaining_arm_outputs_bytes": output_bytes,
        "metadata_reserve_bytes": METADATA_RESERVE_BYTES,
        "other_live_reservations_bytes": other_live_reservations_bytes,
        "minimum_free_after_reservations_bytes": MIN_FREE_BYTES,
        "observed_free_bytes": free_bytes,
        "required_reservation_bytes": required_bytes,
    }


def _link(source: Path, destination: Path) -> None:
    """Create a source-bound symlink without replacing an existing file."""
    resolved = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace alias input: {destination}")
    destination.symlink_to(resolved)


def create_alias_roots(checkpoint_root: Path, teacher_cache_root: Path) -> tuple[Path, Path]:
    """Create unique audited aliases atomically, preserving existing paths."""
    base = Path("/root/run") / f"alias_{ALIAS_NAME}"
    partial = base.with_name(base.name + ".partial")
    if base.exists() or base.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(f"alias root already exists; inspect it before reuse: {base}")
    temporary_ckpt = partial / "ckpt"
    temporary_cache = partial / "cache"
    temporary_ckpt.mkdir(parents=True)
    temporary_cache.mkdir()
    _link(checkpoint_root / PROPENSITY, temporary_ckpt / PROPENSITY)
    for teacher in TEACHERS:
        _link(checkpoint_root / teacher, temporary_ckpt / teacher)
        for split_name in ("train", "valid"):
            for suffix in ("npy", "json"):
                _link(
                    teacher_cache_root / f"pred_{teacher}_{split_name}.{suffix}",
                    temporary_cache / f"pred_{teacher}_{split_name}.{suffix}",
                )

    aliases = (
        (f"{TEACHERS[0]}_a1", TEACHERS[0]),
        (f"{TEACHERS[1]}_a2", TEACHERS[1]),
        (f"{TEACHERS[0]}_a3", TEACHERS[0]),
    )
    for alias, source in aliases:
        alias_root = temporary_ckpt / alias
        alias_root.mkdir()
        for filename in (f"model_ep{TEACHER_EPOCH}.pt", "provenance.json"):
            _link(checkpoint_root / source / filename, alias_root / filename)
        for split_name in ("train", "valid"):
            _link(
                teacher_cache_root / f"pred_{source}_{split_name}.npy",
                temporary_cache / f"pred_{alias}_{split_name}.npy",
            )
            sidecar = json.loads(
                (teacher_cache_root / f"pred_{source}_{split_name}.json").read_text()
            )
            sidecar["run"] = alias
            (temporary_cache / f"pred_{alias}_{split_name}.json").write_text(json.dumps(sidecar))
    partial.rename(base)
    return base / "ckpt", base / "cache"


def _verify_metadata_identity(metadata: dict[str, Any], split: CampaignSplit) -> None:
    """Fail closed unless cache metadata matches the fixed OOD recipe."""
    expected = [
        {
            "family": item.family,
            "run": item.run,
            "epoch": item.epoch,
            "coefficient": item.coefficient,
        }
        for item in (cache_builder.parse_teacher(value) for value in teacher_specs())
    ]
    if metadata.get("output_profile") != "homogeneous-only":
        raise ValueError("shared cache does not declare the homogeneous-only profile")
    if metadata.get("stored_targets") != ["main_homogeneous"]:
        raise ValueError("shared cache target manifest is not homogeneous-only")
    if metadata.get("fold") != FOLD or metadata.get("split_manifest_sha256") != split.sha256:
        raise ValueError("shared cache does not match the fixed fold-1 split")
    if metadata.get("homogeneous_family") != "strong":
        raise ValueError("shared cache homogeneous family is not strong")
    if metadata.get("source_coefficients") != expected:
        raise ValueError("shared cache source runs, epochs or coefficients changed")
    if set(metadata.get("files", {})) != {
        "main_homogeneous_train",
        "main_homogeneous_valid",
    }:
        raise ValueError("shared cache contains an unexpected target set")


def write_or_verify_receipt(
    cache_root: Path, validated: dict[str, Any], split: CampaignSplit
) -> None:
    """Bind the exact validated compact cache to its source and split identity."""
    metadata = validated["metadata"]
    _verify_metadata_identity(metadata, split)
    receipt_path = cache_root / RECEIPT_NAME
    receipt: dict[str, Any] = {
        "schema": "wnn.ood-shared-homogeneous-cache-receipt.v1",
        "cache_name": CACHE_NAME,
        "fold": FOLD,
        "split_manifest_sha256": split.sha256,
        "teachers": list(TEACHERS),
        "teacher_epoch": TEACHER_EPOCH,
        "homogeneous_family": "strong",
        "coefficient_sum": 0.8125,
        "output_profile": metadata["output_profile"],
        "stored_targets": metadata["stored_targets"],
        "source_coefficients": metadata["source_coefficients"],
        "source_records": metadata["sources"],
        "cache_metadata_sha256": validated["metadata_sha256"],
        "verified_target_sha256": validated["verified_files"],
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if receipt_path.exists():
        if receipt_path.read_text() != encoded:
            raise ValueError("existing shared-cache receipt does not match the validated cache")
        return
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(encoded)


def run_frozen(argv: list[str], log: str, *, stage: str, manifest_sha256: str) -> None:
    """Run one bounded subprocess with exact frozen Python/source environment."""
    if not argv or argv[0] != PYTHON_REQUIRED:
        raise ValueError(f"{stage} must use {PYTHON_REQUIRED}")
    command = [*TIMEOUT_PREFIX, *argv]
    environment = os.environ.copy()
    environment.update(
        {
            "WNN_PROJECT_ROOT": str(PROJECT_ROOT_REQUIRED),
            "WNN_FROZEN_SOURCE_ROOT": str(FROZEN_SOURCE_ROOT),
            "WNN_OOD_R2_BUNDLE_MANIFEST_SHA256": manifest_sha256,
            "PYTHONPATH": str(FROZEN_SOURCE_ROOT / "src"),
        }
    )
    print(json.dumps({"event": "r2_subprocess", "stage": stage, "argv": command}), flush=True)
    with Path(log).open("ab") as handle:
        subprocess.run(
            command,
            check=True,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def require_completed_teacher_inputs() -> None:
    """Require exact teacher, propensity, and warm-start files before cache allocation."""
    require_completed_teacher_inputs_base()
    parent_run, _, parent_epoch = PARENT.partition("@ep")
    parent_checkpoint = ROOT / "artifacts" / "runs" / parent_run / f"model_ep{int(parent_epoch)}.pt"
    parent_provenance = ROOT / "artifacts" / "runs" / parent_run / "provenance.json"
    if not parent_checkpoint.is_file() or not parent_provenance.is_file():
        raise FileNotFoundError(f"fixed parent checkpoint/provenance is incomplete: {parent_run}")


def require_completed_teacher_inputs_base() -> None:
    """Require all audited teacher and propensity artifacts; no export is permitted."""
    propensity_root = ROOT / "artifacts" / "runs" / PROPENSITY
    for path in (propensity_root / "model.pt", propensity_root / "provenance.json"):
        if not path.is_file():
            raise FileNotFoundError(f"completed propensity source is required: {path}")
    for teacher in TEACHERS:
        run_root = ROOT / "artifacts" / "runs" / teacher
        for path in (run_root / f"model_ep{TEACHER_EPOCH}.pt", run_root / "provenance.json"):
            if not path.is_file():
                raise FileNotFoundError(f"completed teacher artifact is required: {path}")
        for split_name in ("train", "valid"):
            prediction = ROOT / "cache" / f"pred_{teacher}_{split_name}.npy"
            sidecar = prediction.with_suffix(".json")
            if not prediction.is_file() or not sidecar.is_file():
                raise FileNotFoundError(
                    f"completed {split_name} prediction is required for {teacher}; exports are disabled"
                )


def ensure_shared_cache(
    cache_root: Path,
    split_path: Path,
    log: str,
    *,
    other_live_reservations_bytes: int,
    manifest_sha256: str,
) -> dict[str, int]:
    """Build or validate the one immutable compact cache under an exclusive lock."""
    lock_root = Path("/root/run/locks")
    lock_root.mkdir(parents=True, exist_ok=True)
    with SHARED_CACHE_LOCK_PATH.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        split = load_campaign_split(split_path, ROOT)
        disk_reservation = preflight_cache_space(
            cache_root, split_path, other_live_reservations_bytes
        )
        if cache_root.exists():
            validated = validate_distillation_cache(
                cache_root, ROOT, split, "homogeneous", require_families=False
            )
            write_or_verify_receipt(cache_root, validated, split)
            return disk_reservation
        if cache_root.with_name(cache_root.name + ".partial").exists():
            raise FileExistsError(
                "partial shared cache exists; preserve and inspect it before retry"
            )

        checkpoint_root, teacher_cache_root = create_alias_roots(
            ROOT / "artifacts" / "runs", ROOT / "cache"
        )
        builder_path = SCRIPT_DIR / "build_distill_cache_r2.py"
        if not builder_path.is_file():
            raise FileNotFoundError(f"frozen compact builder is missing: {builder_path}")
        build = [
            PYTHON_REQUIRED,
            str(builder_path),
            "--data-root",
            str(ROOT),
            "--checkpoint-root",
            str(checkpoint_root),
            "--teacher-cache-root",
            str(teacher_cache_root),
            "--split-manifest",
            str(split_path),
            "--output",
            str(cache_root),
            "--homogeneous-family",
            "strong",
            "--chunk-sequences",
            "4",
            "--output-profile",
            "homogeneous-only",
            "--output-budget-gb",
            "16",
            "--min-free-gb",
            "20",
            "--expected-coefficient-sum",
            "0.8125",
            "--additional-reserve-bytes",
            str(
                disk_reservation["remaining_arm_outputs_bytes"]
                + disk_reservation["metadata_reserve_bytes"]
                + disk_reservation["other_live_reservations_bytes"]
            ),
        ]
        for teacher in teacher_specs():
            build.extend(("--teacher", teacher))
        print(
            json.dumps(
                {
                    "event": "building_shared_homogeneous_cache",
                    "cache": str(cache_root),
                    "disk_reservation": disk_reservation,
                }
            ),
            flush=True,
        )
        run_frozen(build, log, stage="compact-cache-builder", manifest_sha256=manifest_sha256)
        validated = validate_distillation_cache(
            cache_root, ROOT, split, "homogeneous", require_families=False
        )
        write_or_verify_receipt(cache_root, validated, split)
        fcntl.flock(lock, fcntl.LOCK_UN)
        return disk_reservation


def main() -> int:
    """Execute one of the four frozen OOD arms with a shared read-only cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(OOD_ARMS))
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--other-live-reservations-bytes", required=True, type=int)
    args = parser.parse_args()
    if args.bundle_manifest_sha256 != _PREIMPORT.bundle_manifest_sha256:
        raise ValueError("pre-import and runtime bundle manifest hashes disagree")
    if args.other_live_reservations_bytes < 0:
        raise ValueError("external live reservation must be a nonnegative byte count")

    cache_root = ROOT / "cache" / CACHE_NAME
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    student = f"vt_f1_ood_{args.arm}_homshare_r2"
    student_root = ROOT / "artifacts" / "runs" / student
    if student_root.exists() or student_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing student run: {student_root}")
    prediction = ROOT / "cache" / f"pred_{student}_valid.npy"
    for path in (
        prediction,
        prediction.with_suffix(".json"),
        prediction.with_name(prediction.name + ".partial"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing student prediction output: {path}")
    log_root = Path("/root/run/logs")
    log_root.mkdir(parents=True, exist_ok=True)
    log = str(log_root / f"pipeline_{args.arm}.log")
    if Path(log).exists():
        raise FileExistsError(f"refusing to append to an existing OOD attempt log: {log}")

    require_completed_teacher_inputs()
    RUN_SEQUENCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_SEQUENCE_LOCK_PATH.open("a") as sequence_lock:
        try:
            fcntl.flock(sequence_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("R2 OOD arm sequence is serial; another R2 arm currently owns the lock") from error
        disk_reservation = ensure_shared_cache(
            cache_root,
            split_path,
            log,
            other_live_reservations_bytes=args.other_live_reservations_bytes,
            manifest_sha256=args.bundle_manifest_sha256,
        )
        print(json.dumps({"event": "r2_disk_reservation", **disk_reservation}), flush=True)

        augmentation, fraction = OOD_ARMS[args.arm]
        distill_flags = [
            "--distill-cache-root",
            str(cache_root),
            "--distill-cache-variant",
            "homogeneous",
            "--distill-alpha",
            "0.7",
            "--distill-boot-h",
            "0",
        ]
        ood_flags = [
            "--ood-teachers",
            ",".join(f"{name}@ep{TEACHER_EPOCH}" for name in TEACHERS),
            "--ood-aug",
            augmentation,
            "--ood-frac",
            str(fraction),
            "--ood-mode",
            "mix",
            "--ood-scale",
            "0.8125",
            "--ood-weight",
            "teacher",
            "--ood-sources",
            "24",
        ]
        trainer = FROZEN_SOURCE_ROOT / "scripts" / "train_rnn.py"
        fit = [
            PYTHON_REQUIRED,
            str(trainer),
            "--name",
            student,
            "--device",
            "cuda",
            "--arch",
            "gru",
            "--cell",
            "gru",
            "--hidden",
            "256",
            "--layers",
            "2",
            "--dropout",
            "0",
            "--seed",
            "3",
            "--split-manifest",
            str(split_path),
            "--valid-fold",
            str(FOLD),
            "--eval-source",
            "vfold",
            "--train-source",
            "train+valid",
            "--train-seqs",
            "10607",
            "--init",
            PARENT,
            "--epochs",
            "2",
            "--checkpoint-policy",
            "fixed",
            "--checkpoint-epoch",
            "1",
            "--no-eval-during-fit",
            "--batch",
            "16",
            "--chunk",
            "2000",
            "--loss",
            "wmse",
            "--loss-norm",
            "chunk",
            "--lr",
            "0.0001",
            "--wd",
            "0.01",
            "--optim",
            "adamw",
            "--sched",
            "onecycle",
            "--precision",
            "fp16",
            "--propensity",
            PROPENSITY,
            "--prop-power",
            "2",
            "--prop-eps",
            "0.03",
            "--valid-true-mask",
            "0.03",
            "--aux-struct",
            "0.3",
            "--aux-next",
            "0.3",
            *distill_flags,
            *ood_flags,
            "--fast",
            "--save-training-state",
        ]
        run_frozen(fit, log, stage="ood-student-fit", manifest_sha256=args.bundle_manifest_sha256)
        predictor = FROZEN_SOURCE_ROOT / "scripts" / "predict_run.py"
        predict = [
            PYTHON_REQUIRED,
            str(predictor),
            student,
            "valid",
            "--epoch",
            "1",
        ]
        run_frozen(predict, log, stage="student-valid-prediction", manifest_sha256=args.bundle_manifest_sha256)
        print(
            json.dumps(
                {
                    "event": "OOD_SHARED_PIPELINE_DONE",
                    "student": student,
                    "trainer": str(trainer),
                    "predictor": str(predictor),
                    "bundle_manifest_sha256": args.bundle_manifest_sha256,
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
