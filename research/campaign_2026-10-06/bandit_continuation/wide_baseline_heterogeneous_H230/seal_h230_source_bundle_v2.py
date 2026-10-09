"""Seal H230 source artifacts using the pinned cache builder's checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
WORKTREE = Path("/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007")
BUILDER = WORKTREE / "scripts/build_distill_cache.py"
EXPECTED_WORKTREE_COMMIT = "dde32d0f5e39efe65c0497fec06449d38e18b369"
EXPECTED_SOURCE_BUNDLE_SHA256 = "6984c736f4683b557a5db5008cd9b79bccb552ce565126a0788a5d8e2b1c8cef"
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_E521_PLAN_SHA256 = "e6e51ab739c4d9fcda5a37e76993b60941ad0bbd108ae07863ce265952235bf3"
V4_REVIEW_MAP_SHA256 = "df4e994e5a152eead157772cda8c83ab0a3abe62703d81c1dbe7b1126cda9b08"
V4_REVIEW_SHA256 = "bf395ec97e3d24fefe271613e3125fff0814ec5320d899824449922de3c43a31"


class SourceSealError(ValueError):
    """Raised when a frozen source identity or cache-builder check fails."""


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Hash a file incrementally without materializing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object, *, replace: bool = True) -> None:
    """Durably write a JSON receipt through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() and not replace:
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _resource_snapshot() -> dict[str, object]:
    """Record process and current cgroup memory when those readings exist."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    snapshot: dict[str, object] = {
        "process_max_rss_bytes": int(max_rss * (1024 if os.name == "posix" else 1)),
        "cgroup_memory_current_bytes": None,
        "cgroup_memory_peak_bytes": None,
        "cgroup_observation_error": None,
    }
    try:
        rows = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        groups = [row.split(":", 2)[2] for row in rows if row.startswith("0::")]
        if len(groups) != 1:
            raise OSError("cannot resolve one unified cgroup path")
        directory = (Path("/sys/fs/cgroup") / groups[0].lstrip("/")).resolve()
        root = Path("/sys/fs/cgroup").resolve()
        if not directory.is_relative_to(root):
            raise OSError("cgroup path escaped unified hierarchy")
        for filename, key in (
            ("memory.current", "cgroup_memory_current_bytes"),
            ("memory.peak", "cgroup_memory_peak_bytes"),
        ):
            file_path = directory / filename
            if file_path.exists():
                snapshot[key] = int(file_path.read_text(encoding="ascii").strip())
    except OSError as error:
        snapshot["cgroup_observation_error"] = f"{type(error).__name__}: {error}"
    return snapshot


def _verify_worktree(bundle: dict[str, Any]) -> list[dict[str, object]]:
    """Rehash every pinned source file before invoking the cache builder."""
    result = subprocess.run(
        ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit = result.stdout.strip()
    if commit != EXPECTED_WORKTREE_COMMIT:
        raise SourceSealError(f"worktree commit {commit} != frozen commit")
    entries = bundle.get("worktree", {}).get("tracked_source_files")
    if not isinstance(entries, list) or len(entries) != 32:
        raise SourceSealError("source bundle must bind all 32 worktree source files")
    observed: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise SourceSealError("worktree source entry is malformed")
        path = WORKTREE / str(entry["path"])
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != entry.get("sha256") or size != entry.get("bytes"):
            raise SourceSealError(f"pinned worktree source changed: {path}")
        observed.append({"path": str(path), "bytes": size, "sha256": digest})
    builder_entry = next(
        (item for item in entries if item.get("path") == "scripts/build_distill_cache.py"),
        None,
    )
    if builder_entry is None or sha256_file(BUILDER) != builder_entry["sha256"]:
        raise SourceSealError("cache builder is not the source-bundle-pinned file")
    return observed


def _load_builder() -> Any:
    """Import only the exact pinned cache-builder module in this worktree."""
    spec = importlib.util.spec_from_file_location("h230_pinned_build_distill_cache", BUILDER)
    if spec is None or spec.loader is None:
        raise SourceSealError(f"cannot import pinned builder {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_expected(label: str, expected: object, actual: object) -> None:
    """Compare a previously frozen value when V1 recorded one."""
    if expected is not None and expected != actual:
        raise SourceSealError(f"{label} changed: expected {expected!r}, observed {actual!r}")


def _seed_diagnostic(
    entry: dict[str, Any], provenance: dict[str, Any], builder_record: dict[str, Any]
) -> dict[str, object]:
    """Report source labels separately from the actual trainer seed arguments."""
    trainer_seed = provenance.get("args", {}).get("seed")
    checkpoint_seed = builder_record.get("seed")
    return {
        "source_bundle_seed_label": entry.get("seed"),
        "provenance_trainer_seed": trainer_seed,
        "checkpoint_trainer_seed": checkpoint_seed,
        "label_matches_trainer_seed": entry.get("seed") == trainer_seed,
        "provenance_matches_checkpoint_seed": trainer_seed == checkpoint_seed,
    }


def _verify_sequence(root: Path) -> dict[str, object]:
    """Require the terminal seed-paired E521 train-export sequence."""
    relative = Path("research/campaign_2026-10-06/bandit_continuation/local_wide_teacher_E521")
    plan_path = root / relative / "TRAIN_EXPORT_PAIR_SEQUENCE_V3.json"
    receipt_path = root / relative / "TRAIN_EXPORT_PAIR_SEQUENCE_RECEIPT_V3.json"
    plan_sha = sha256_file(plan_path)
    if plan_sha != EXPECTED_E521_PLAN_SHA256:
        raise SourceSealError("E521 paired export plan changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "finished"
        or receipt.get("worker_status") != "finished"
        or receipt.get("launch_returncode") != 0
        or receipt.get("plan_sha256") != plan_sha
    ):
        raise SourceSealError("E521 paired train-export sequence is not terminal success")
    return {
        "plan_path": str(plan_path.relative_to(root)),
        "plan_sha256": plan_sha,
        "receipt_path": str(receipt_path.relative_to(root)),
        "receipt_sha256": sha256_file(receipt_path),
        "status": receipt["status"],
        "last_attempt": receipt.get("attempt"),
        "last_step_index": receipt.get("index"),
    }


def _verify_teacher(
    root: Path,
    builder: Any,
    split: Any,
    target_orders: dict[str, list[int]],
    entry: dict[str, Any],
) -> dict[str, object]:
    """Use pinned builder checks and release each source mmap before continuing."""
    run = str(entry["run"])
    epoch = int(entry["epoch_zero_based"])
    family = str(entry["family"])
    coefficient = float(entry["coefficient"])
    source = builder.TeacherSource(family, run, coefficient, epoch)
    checkpoint_root = root / "artifacts/runs"
    cache_root = root / "cache"
    provenance_path = checkpoint_root / run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    ancestor_closure = builder._audit_ancestor_closure(
        run, split, checkpoint_root, f"model_ep{epoch}.pt"
    )
    record, arrays, cache_records = builder._inspect_teacher(
        run, split, cache_root, checkpoint_root, target_orders, epoch
    )
    # _inspect_teacher verifies checkpoint/provenance membership, sidecar identity,
    # full SHA-256, ordered groups, shape/dtype, and finite values in bounded rows.
    # Drop the two read-only mmap handles before moving to another teacher.
    del arrays

    checkpoint_sha = str(record["checkpoint_sha256"])
    provenance_sha = sha256_file(provenance_path)
    _verify_expected(
        f"{run} checkpoint hash",
        entry.get("observed_checkpoint_sha256")
        or entry.get("expected_checkpoint_sha256_from_root_draft"),
        checkpoint_sha,
    )
    _verify_expected(
        f"{run} provenance hash",
        entry.get("observed_provenance_sha256")
        or entry.get("expected_provenance_sha256_from_root_draft"),
        provenance_sha,
    )
    _verify_expected(f"{run} family", entry.get("family"), source.family)
    _verify_expected(f"{run} coefficient", entry.get("coefficient"), source.coefficient)
    _verify_expected(f"{run} fit counts", entry.get("fit_group_counts"), record["fit_group_counts"])

    exports: dict[str, dict[str, object]] = {}
    old_exports = entry.get("prediction_exports")
    if not isinstance(old_exports, dict):
        raise SourceSealError(f"{run}: frozen prediction export list is malformed")
    for split_name, actual in cache_records.items():
        previous = old_exports.get(split_name)
        if not isinstance(previous, dict):
            raise SourceSealError(f"{run}: frozen {split_name} export is missing")
        sidecar_path = cache_root / f"pred_{run}_{split_name}.json"
        prediction_path = cache_root / f"pred_{run}_{split_name}.npy"
        _verify_expected(
            f"{run}/{split_name} sidecar SHA",
            previous.get("sidecar_sha256"),
            actual["sidecar_sha256"],
        )
        _verify_expected(
            f"{run}/{split_name} prediction SHA",
            previous.get("sidecar_prediction_sha256"),
            actual["prediction_sha256"],
        )
        _verify_expected(
            f"{run}/{split_name} target order SHA",
            previous.get("ordered_row_groups_sha256"),
            actual["row_groups_sha256"],
        )
        _verify_expected(f"{run}/{split_name} epoch", previous.get("epoch"), epoch)
        _verify_expected(
            f"{run}/{split_name} row count",
            previous.get("row_count"),
            len(target_orders[split_name]),
        )
        _verify_expected(
            f"{run}/{split_name} prediction bytes",
            previous.get("observed_prediction_bytes_no_content_read"),
            prediction_path.stat().st_size,
        )
        exports[split_name] = {
            **previous,
            **actual,
            "prediction_path": str(prediction_path.relative_to(root)),
            "sidecar_path": str(sidecar_path.relative_to(root)),
            "observed_prediction_bytes": prediction_path.stat().st_size,
            "observed_prediction_sha256": actual["prediction_sha256"],
            "observed_sidecar_sha256": actual["sidecar_sha256"],
            "row_count": len(target_orders[split_name]),
            "ordered_row_groups_sha256": actual["row_groups_sha256"],
            "shape": [len(target_orders[split_name]), 20_000, 2],
            "dtype": "float32",
            "finite": True,
            "status": "builder_hash_shape_order_finite_verified",
        }
    result = dict(entry)
    result.update(
        observed_checkpoint_sha256=checkpoint_sha,
        observed_provenance_sha256=provenance_sha,
        fit_group_counts=record["fit_group_counts"],
        fit_row_group_hashes=record["fit_row_group_hashes"],
        ancestor_closure=ancestor_closure,
        seed_identity_diagnostic=_seed_diagnostic(entry, provenance, record),
        prediction_exports=exports,
        artifact_status="checkpoint_provenance_ancestry_predictions_verified_by_pinned_builder",
    )
    return result


def seal_source_bundle(
    root: Path, bundle_path: Path, output_path: Path, receipt_path: Path
) -> None:
    """Verify the H230 sources and atomically write actual-byte identity records."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite source bundle {output_path}")
    receipt: dict[str, Any] = {
        "id": "H230-source-seal-v2",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "stage": "verify_sources_with_pinned_cache_builder",
        "source_bundle_path": str(bundle_path.relative_to(root)),
        "expected_source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "completed_sources": [],
        "output_path": str(output_path),
        "resource_heartbeat": _resource_snapshot(),
    }
    _atomic_json(receipt_path, receipt)
    try:
        bundle_sha = sha256_file(bundle_path)
        if bundle_sha != EXPECTED_SOURCE_BUNDLE_SHA256:
            raise SourceSealError("frozen H230 source bundle V1 changed")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        receipt["observed_source_bundle_sha256"] = bundle_sha
        receipt["worktree_source_files"] = _verify_worktree(bundle)
        receipt["resource_heartbeat"] = _resource_snapshot()
        _atomic_json(receipt_path, receipt)

        sequence = _verify_sequence(root)
        split_path = root / str(bundle["fold"]["manifest_path"])
        split_sha = sha256_file(split_path)
        if split_sha != EXPECTED_SPLIT_SHA256:
            raise SourceSealError("frozen H230 fold manifest changed")
        builder = _load_builder()
        split = builder.load_campaign_split(split_path, root)
        if split.fold != 0 or split.sha256 != split_sha:
            raise SourceSealError("loaded campaign split does not match fold-0 identity")
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        target_orders = {name: builder._target_order(root, name) for name in ("train", "valid")}
        manifest_orders = {
            name: [int(row["row_group"]) for row in split_manifest["sequences"][name]]
            for name in ("train", "valid")
        }
        expected_order_hashes: dict[str, str] = {}
        for entry in bundle["current_teacher_artifact_snapshot"]:
            for name, record in entry["prediction_exports"].items():
                digest = record.get("ordered_row_groups_sha256")
                if digest is not None:
                    expected_order_hashes.setdefault(name, str(digest))
        group_records: dict[str, dict[str, object]] = {}
        for name, order in target_orders.items():
            if order != manifest_orders[name] or len(order) != len(set(order)):
                raise SourceSealError(
                    f"{name} target index order differs from frozen fold manifest"
                )
            digest = builder.row_groups_sha256(order)
            _verify_expected(
                f"{name} ordered target digest", expected_order_hashes.get(name), digest
            )
            fit_count = len(split.train_groups) if name == "train" else len(split.valid_groups)
            group_records[name] = {
                "count": len(order),
                "ordered_sha256": digest,
                "fit_count": fit_count,
                "evaluation_count": len(split.evaluation_groups) if name == "valid" else 0,
            }
        sources = bundle.get("current_teacher_artifact_snapshot")
        if not isinstance(sources, list) or len(sources) != 8:
            raise SourceSealError("expected exactly eight frozen source family members")
        receipt.update(
            source_sequence=sequence,
            split_manifest_sha256=split_sha,
            target_row_groups=group_records,
            total_sources=8,
            total_prediction_exports=16,
            resource_heartbeat=_resource_snapshot(),
        )
        _atomic_json(receipt_path, receipt)

        sealed_sources: list[dict[str, object]] = []
        for entry in sources:
            if not isinstance(entry, dict):
                raise SourceSealError("source entry is malformed")
            sealed = _verify_teacher(root, builder, split, target_orders, entry)
            sealed_sources.append(sealed)
            receipt["completed_sources"] = [str(item["run"]) for item in sealed_sources]
            receipt["resource_heartbeat"] = _resource_snapshot()
            _atomic_json(receipt_path, receipt)

        result = dict(bundle)
        result.update(
            id="H230-actual-source-hash-bundle-v2",
            captured_utc=datetime.now(UTC).isoformat(),
            supersedes={
                "path": str(bundle_path.relative_to(root)),
                "sha256": bundle_sha,
                "reason": "V1 had pending raw-family exports; V2 records actual producer hashes and pinned cache-builder validation after the paired E521 sequence finished.",
            },
            static_contract_review={
                "map_path": "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_REVIEW_TARGETS_V4.json",
                "map_sha256": V4_REVIEW_MAP_SHA256,
                "receipt_path": "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_INDEPENDENT_STATIC_EXECUTION_REVIEW_V4_20261007_R1.json",
                "receipt_sha256": V4_REVIEW_SHA256,
            },
            source_sequence=sequence,
            actual_target_row_groups=group_records,
            current_teacher_artifact_snapshot=sealed_sources,
            prediction_validation={
                "implementation": "pinned build_distill_cache.py::_inspect_teacher and _audit_ancestor_closure",
                "builder_path": str(BUILDER),
                "builder_sha256": sha256_file(BUILDER),
                "checkpoint_provenance_ancestry_checked": True,
                "ordered_target_row_groups_checked": True,
                "prediction_sha256_checked_against_sidecars": True,
                "shape_dtype_and_finiteness_checked_in_64_sequence_memmap_chunks": True,
                "source_mmaps_released_after_each_teacher": True,
                "prediction_arrays_materialized": False,
                "all_16_prediction_files_verified": True,
            },
            raw_parquet_full_sha256_status="not_computed_here; pinned split loader rechecks manifest source identity and each later stage must refresh its live input receipt",
        )
        _atomic_json(output_path, result, replace=False)
        receipt.update(
            status="finished",
            finished_utc=datetime.now(UTC).isoformat(),
            output_sha256=sha256_file(output_path),
            observed_source_count=len(sealed_sources),
            observed_prediction_count=sum(
                len(item["prediction_exports"]) for item in sealed_sources
            ),
            resource_heartbeat=_resource_snapshot(),
        )
        _atomic_json(receipt_path, receipt)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        receipt.update(
            status="failed",
            finished_utc=datetime.now(UTC).isoformat(),
            error=f"{type(error).__name__}: {error}",
            resource_heartbeat=_resource_snapshot(),
        )
        _atomic_json(receipt_path, receipt)
        raise


def parse_args() -> argparse.Namespace:
    """Parse frozen paths; this command only verifies sources and writes receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT
        / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_HASH_BUNDLE_V1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_HASH_BUNDLE_V2.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT
        / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_SEAL_PROGRESS_V3.json",
    )
    return parser.parse_args()


def main() -> None:
    """Verify H230 source artifacts without scoring, fitting or cache writing."""
    args = parse_args()
    seal_source_bundle(
        args.root.resolve(), args.bundle.resolve(), args.output.resolve(), args.receipt.resolve()
    )
    print(json.dumps(json.loads(args.receipt.read_text(encoding="utf-8")), sort_keys=True))


if __name__ == "__main__":
    main()
