"""Wait for final L4 parquet files, rebuild source-bound caches, and admit profiles."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from journal import Journal, write_json_fsync
from policy import sha256_file
from prep_common import (
    PREP_SECONDS,
    _atomic_copy,
    _copy_pinned_script,
    _run_checked,
)


def prepare_inputs(
    *,
    root: Path,
    packet: Path,
    source: Path,
    state: Path,
    python: str,
    journal: Journal,
    cpus: tuple[int, ...] = (0, 1),
) -> dict[str, Any]:
    """Reconstruct and strictly verify L4 caches, graph, folds, and four manifests."""
    data = root / "data"
    datasets = data / "datasets"
    cache = data / "cache"
    scripts = data / "scripts"
    canonical = packet / "canonical"
    receipts = state / "receipts"
    logs = state / "logs"
    attempt = state / "prep"
    splits_out = attempt / "splits"
    if not canonical.is_dir() or not source.is_dir():
        raise FileNotFoundError("pinned source tree or canonical manifest directory is missing")
    for fold in range(4):
        if not (canonical / f"fold{fold}.json").is_file():
            raise FileNotFoundError(f"canonical fold{fold} manifest is missing")
    if splits_out.exists():
        raise FileExistsError(f"attempt preparation directory already exists: {splits_out}")
    for directory in (cache, scripts, receipts, logs):
        directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + PREP_SECONDS
    source_cache = source / "scripts" / "cache_targets.py"
    source_graph = source / "scripts" / "sequence_chains.py"
    script_hashes = {
        "cache_targets.py": _copy_pinned_script(source_cache, scripts / source_cache.name),
        "sequence_chains.py": _copy_pinned_script(source_graph, scripts / source_graph.name),
    }
    targets = [cache / "train_targets.npz", cache / "valid_targets.npz"]
    if all(path.is_file() for path in targets):
        pass
    elif any(path.exists() for path in targets):
        raise RuntimeError(
            "only one target cache exists; refusing mixed or partial cache provenance"
        )
    else:
        for split in ("train", "valid"):
            _run_checked(
                name=f"cache_targets_{split}",
                argv=[python, str(scripts / "cache_targets.py"), split],
                cwd=data,
                env={"PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
                cpus=cpus,
                log=logs / f"cache_targets_{split}.log",
                journal=journal,
                root=root,
                deadline=deadline,
            )
        if not all(path.is_file() and path.stat().st_size > 0 for path in targets):
            raise RuntimeError("target cache generation returned without both complete archives")
    preflight = receipts / "input-preflight.json"
    _run_checked(
        name="validate_l4_targets",
        argv=[
            python,
            str(packet / "code" / "validate_l4_targets.py"),
            "--train-targets",
            str(targets[0]),
            "--train-parquet",
            str(datasets / "train.parquet"),
            "--valid-targets",
            str(targets[1]),
            "--valid-parquet",
            str(datasets / "valid.parquet"),
            "--valid-mask-parquet",
            str(datasets / "valid_mask.parquet"),
            "--canonical-dir",
            str(canonical),
            "--receipt",
            str(preflight),
        ],
        cwd=root,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        cpus=cpus,
        log=logs / "validate_l4_targets.log",
        journal=journal,
        root=root,
        deadline=deadline,
    )
    preflight_data = json.loads(preflight.read_text(encoding="utf-8"))
    if preflight_data.get("preflight_status") != "verified_r4_raw_parquets":
        raise RuntimeError(
            "raw Parquet target-cache preflight did not verify exact frozen R4 inputs"
        )
    mask_identity = preflight_data.get("valid", {}).get("valid_mask_parquet", {})
    if (
        mask_identity.get("bytes") != 91_972_846
        or mask_identity.get("sha256")
        != "301430165527a6003d44486a8c5051288c31112d327df94fb420e82c018e5251"
    ):
        raise RuntimeError("valid-mask parquet identity differs from the staged frozen input")
    graph = cache / "chains.npz"
    if not graph.is_file():
        _run_checked(
            name="sequence_chains",
            argv=[python, str(scripts / "sequence_chains.py")],
            cwd=data,
            env={
                "PYTHONPATH": str(source / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "WNN_THREADS": "2",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "2",
                "TMPDIR": str(root / "tmp"),
            },
            cpus=cpus,
            log=logs / "sequence_chains.log",
            journal=journal,
            root=root,
            deadline=deadline,
        )
    folds = cache / "chain_folds.npz"
    folds_receipt = receipts / "folds.json"
    _run_checked(
        name="derive_l4_folds",
        argv=[
            python,
            str(packet / "code" / "derive_l4_folds.py"),
            "--graph",
            str(graph),
            "--canonical-dir",
            str(canonical),
            "--source-root",
            str(source),
            "--executed-script",
            str(scripts / "sequence_chains.py"),
            "--input-preflight",
            str(preflight),
            "--output",
            str(folds),
            "--receipt",
            str(folds_receipt),
        ],
        cwd=root,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        cpus=cpus,
        log=logs / "derive_l4_folds.log",
        journal=journal,
        root=root,
        deadline=deadline,
    )
    _run_checked(
        name="build_campaign_splits",
        argv=[
            python,
            str(source / "scripts" / "build_campaign_splits.py"),
            "--root",
            str(data),
            "--output-dir",
            str(splits_out),
        ],
        cwd=root,
        env={"PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        cpus=cpus,
        log=logs / "build_campaign_splits.log",
        journal=journal,
        root=root,
        deadline=deadline,
    )
    comparison = subprocess_compare(
        python=python,
        packet=packet,
        canonical=canonical,
        fresh=splits_out,
        root=root,
        state=state,
        journal=journal,
        deadline=deadline,
        cpus=cpus,
    )
    manifest_records = []
    for fold in range(4):
        fresh = splits_out / f"fold{fold}.json"
        if not fresh.is_file():
            raise FileNotFoundError(f"fresh split builder omitted fold{fold}: {fresh}")
        _atomic_copy(fresh, root / "splits" / fresh.name)
        manifest_records.append(
            {"fold": fold, "bytes": fresh.stat().st_size, "sha256": sha256_file(fresh)}
        )
    result = {
        "schema": "tabdata-b300-controller-prep.v1",
        "status": "verified_all_frozen_memberships",
        "identity_class": "fresh_b300_source_bound_attempt",
        "inputs": {
            "train_targets_sha256": sha256_file(targets[0]),
            "valid_targets_sha256": sha256_file(targets[1]),
            "graph_sha256": sha256_file(graph),
            "fold_cache_sha256": sha256_file(folds),
            "preflight_receipt": str(preflight),
            "fold_receipt": str(folds_receipt),
            "staged_script_sha256": script_hashes,
        },
        "membership_comparison": comparison,
        "fresh_manifests": manifest_records,
    }
    write_json_fsync(receipts / "prep.json", result)
    journal.record("data_prep_pass", receipt=str(receipts / "prep.json"))
    return result


def subprocess_compare(
    *,
    python: str,
    packet: Path,
    canonical: Path,
    fresh: Path,
    root: Path,
    state: Path,
    journal: Journal,
    deadline: float,
    cpus: tuple[int, ...],
) -> dict[str, Any]:
    """Run and parse the immutable full-membership comparison gate."""
    output = state / "receipts" / "membership-comparison.json"
    if output.exists():
        raise FileExistsError(output)
    log = state / "logs" / "membership-comparison.log"
    result = _run_checked(
        name="canonical_manifest_checks",
        argv=[
            python,
            str(packet / "code" / "canonical_manifest_checks.py"),
            "--canonical-dir",
            str(canonical),
            "--fresh-dir",
            str(fresh),
        ],
        cwd=root,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        cpus=cpus,
        log=log,
        journal=journal,
        root=root,
        deadline=deadline,
    )
    del result
    lines = log.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.startswith("{")]
    if len(records) != 1 or records[0].get("status") != "exact_membership_match":
        raise RuntimeError("fresh manifest membership comparison emitted no unique pass receipt")
    write_json_fsync(output, records[0])
    return records[0]
