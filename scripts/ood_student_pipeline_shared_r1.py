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
import sys
from pathlib import Path
from typing import Any

# Cache metadata is versioned JSON and retains producer-defined nested records.
ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

import build_distill_cache as cache_builder  # noqa: E402
import distill_pipeline as dp  # noqa: E402

from wnn.distillation_cache import validate_distillation_cache  # noqa: E402
from wnn.splits import CampaignSplit, load_campaign_split  # noqa: E402

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
OUTPUT_RESERVE_BYTES = 16 << 20
OOD_ARMS = {
    "splice_r2": ("splice", 0.5),
    "spmix_r2": ("splice,mixup_paths", 0.5),
    "swn_r2": ("splice,warp,noise", 0.5),
    "swn25_r2": ("splice,warp,noise", 0.25),
}
DEPLOYED_SOURCE_SHA256 = {
    "scripts/ood_student_pipeline.py": "1a3e680851415685982ac60f46ca1d8fd25cd0545c165154b6d8426193c3d6e0",
    "scripts/distill_pipeline.py": "dd663a27e92ec7c0fb11542935a89e5011e83a32097335185ed14e39c019c449",
    "scripts/train_rnn.py": "8c041bf2a49b43ef5a014a5f7cee59302605d464f5b82fe8c7dc12267c386a40",
    "scripts/predict_run.py": "46279845540e680940c4444124164102d2499e0468aa7767fc5ba64f90854018",
    "scripts/build_distill_cache.py": "09f7110f21507195199d1f50fad6d1074fef1cd4d670a91093935be333f9dd26",
    "src/wnn/distillation_cache.py": "403fd74db91f08bb255db147fb50f7f6b6b9702280350aaa2479c01a7f972f19",
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_deployed_sources() -> None:
    """Require the exact already-deployed pipeline, trainer, and cache consumer."""
    for relative, expected in DEPLOYED_SOURCE_SHA256.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"deployed source changed since read-only audit: {relative}")


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


def preflight_cache_space(cache_root: Path, split_path: Path) -> int:
    """Check the compact cache budget before aliases or exports are created."""
    data_root = ROOT
    split = load_campaign_split(split_path, data_root)
    fit_rows = sum(
        len(_target_groups(data_root, split, split_name)) for split_name in ("train", "valid")
    )
    payload_bytes = cache_builder.estimate_output_bytes(fit_rows, "homogeneous-only")
    required_bytes = payload_bytes + OUTPUT_RESERVE_BYTES
    if payload_bytes > OUTPUT_BUDGET_BYTES:
        raise OSError(f"homogeneous cache payload {payload_bytes} exceeds the 16 GiB output budget")
    if not cache_root.parent.is_dir():
        raise FileNotFoundError(f"cache parent is absent: {cache_root.parent}")
    free_bytes = shutil.disk_usage(cache_root.parent).free
    if free_bytes - required_bytes < MIN_FREE_BYTES:
        raise OSError(
            f"homogeneous cache requires {required_bytes} bytes including reserve; "
            f"available while preserving 20 GiB free is {max(0, free_bytes - MIN_FREE_BYTES)}"
        )
    return payload_bytes


def require_completed_teacher_inputs() -> None:
    """Require all source artifacts to exist; this retry path never exports them."""
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


def ensure_shared_cache(cache_root: Path, split_path: Path, log: str) -> None:
    """Build or validate the one immutable compact cache under an exclusive lock."""
    lock_root = Path("/root/run/locks")
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / LOCK_NAME).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        split = load_campaign_split(split_path, ROOT)
        if cache_root.exists():
            validated = validate_distillation_cache(
                cache_root, ROOT, split, "homogeneous", require_families=False
            )
            write_or_verify_receipt(cache_root, validated, split)
            return
        if cache_root.with_name(cache_root.name + ".partial").exists():
            raise FileExistsError(
                "partial shared cache exists; preserve and inspect it before retry"
            )

        payload_bytes = preflight_cache_space(cache_root, split_path)
        require_completed_teacher_inputs()
        checkpoint_root, teacher_cache_root = create_alias_roots(
            ROOT / "artifacts" / "runs", ROOT / "cache"
        )
        builder_path = SCRIPT_DIR / "build_distill_cache.py"
        if not builder_path.is_file():
            raise FileNotFoundError(f"frozen compact builder is missing: {builder_path}")
        build = [
            sys.executable,
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
        ]
        for teacher in teacher_specs():
            build.extend(("--teacher", teacher))
        print(
            json.dumps(
                {
                    "event": "building_shared_homogeneous_cache",
                    "cache": str(cache_root),
                    "estimated_payload_bytes": payload_bytes,
                }
            ),
            flush=True,
        )
        dp.run(build, log)
        validated = validate_distillation_cache(
            cache_root, ROOT, split, "homogeneous", require_families=False
        )
        write_or_verify_receipt(cache_root, validated, split)
        fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> int:
    """Execute one of the four frozen OOD arms with a shared read-only cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(OOD_ARMS))
    args = parser.parse_args()

    cache_root = ROOT / "cache" / CACHE_NAME
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    student = f"vt_f1_ood_{args.arm}_homshare_r2"
    student_root = ROOT / "artifacts" / "runs" / student
    if student_root.exists() or student_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing student run: {student_root}")
    for split_name in ("valid",):
        for suffix in ("npy", "json"):
            prediction = ROOT / "cache" / f"pred_{student}_{split_name}.{suffix}"
            if prediction.exists() or prediction.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite existing student prediction: {prediction}"
                )
    verify_deployed_sources()
    log_root = Path("/root/run/logs")
    log_root.mkdir(parents=True, exist_ok=True)
    log = str(log_root / f"pipeline_{args.arm}.log")
    if Path(log).exists():
        raise FileExistsError(f"refusing to append to an existing OOD attempt log: {log}")
    ensure_shared_cache(cache_root, split_path, log)

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
    trainer = ROOT / "scripts" / "train_rnn.py"
    fit = [
        sys.executable,
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
    dp.run(fit, log)
    dp.predict(student, "valid", 1, None, log)
    print(f"OOD_SHARED_PIPELINE_DONE {student}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
