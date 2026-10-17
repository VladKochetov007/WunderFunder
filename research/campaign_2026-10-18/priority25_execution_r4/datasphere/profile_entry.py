#!/usr/bin/env python3
"""Stage sealed rank-1 inputs and run only the two bounded GPU profiles."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from profile_recipe import (
    EXPECTED_ARMS,
    EXPECTED_COMMIT,
    EXPECTED_RECIPE,
    EXPECTED_SUPPLEMENTAL_MEMBERS,
    check_runtime,
    run_arm,
)
from profile_staging import (
    assemble,
    copy_mounted,
    relative_path,
    safe_extract,
    sealed_hash,
    sha256_file,
    write_bundle,
)


def _prepare(
    spec: dict[str, Any],
    args: argparse.Namespace,
    input_identities: dict[str, Any],
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    if spec.get("schema_version") != 1 or spec.get("state") != "SEALED_FOR_PROFILE":
        raise ValueError("profile spec is not sealed for execution")
    if spec.get("source_commit") != EXPECTED_COMMIT or spec.get("recipe") != EXPECTED_RECIPE:
        raise ValueError("profile spec changes the authorized source or recipe")
    if spec.get("arms") != EXPECTED_ARMS:
        raise ValueError("profile spec changes the authorized model pair")
    if (
        spec.get("internal_deadline_seconds") != 3300
        or spec.get("external_cancel_deadline_seconds_from_submit") != 3600
        or int(spec.get("minimum_free_bytes_after_staging", 0)) < 10 * (1 << 30)
    ):
        raise ValueError("profile spec changes the time or disk safety bounds")
    if spec.get("dataset_slots") != ["DS1", "DS2", "DS3", "DS4"]:
        raise ValueError("profile spec changes the historical dataset slots")
    if set(spec.get("supplemental_members", {})) != EXPECTED_SUPPLEMENTAL_MEMBERS:
        raise ValueError("supplemental inputs must contain only graph/fold/manifests")
    if set(spec.get("mounted_inputs", {})) != {"train", "valid"}:
        raise ValueError("profile must stage exactly the train and valid parquet splits")
    if set(spec.get("mounted_files", {})) != {"cache/train_targets.npz", "cache/valid_targets.npz"}:
        raise ValueError("profile must stage only the two target-index files")
    for split, count in (("train", 6), ("valid", 2)):
        parts = spec["mounted_inputs"][split]["parts"]
        if (
            len(parts) != count
            or spec["mounted_inputs"][split]["assembled_path"] != f"datasets/{split}.parquet"
        ):
            raise ValueError(f"profile requires the historical ordered {split} parquet parts")
        for index, part in enumerate(parts):
            if not part["relative_path"].endswith(f"{split}.parquet.part{index:02d}"):
                raise ValueError(f"{split} parquet parts are missing or out of order")
    for field in ("code_archive_sha256", "supplemental_archive_sha256"):
        if input_identities[field] != sealed_hash(spec.get(field), field):
            raise ValueError(f"archive hash mismatch: {field}")
    if len(args.datasets) != len(spec["dataset_slots"]):
        raise ValueError("dataset argument count differs from sealed slot list")
    dataset_roots = dict(zip(spec["dataset_slots"], map(Path, args.datasets), strict=True))
    work = Path.cwd() / "rank01-profile-work"
    source_root = work.joinpath(*relative_path(spec["source_root"]).parts)
    data_root = work.joinpath(*relative_path(spec["data_root"]).parts)
    source_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    safe_extract(args.code, source_root)
    required = ("scripts/train_rnn.py", "src/wnn/training/runner.py", "pyproject.toml", "uv.lock")
    if not all((source_root / name).is_file() for name in required):
        raise ValueError("source archive lacks the canonical trainer or locked metadata")
    runtime = check_runtime(spec["expected_environment"])
    safe_extract(args.supplemental, data_root, spec["supplemental_members"])
    fold_cache = data_root / "cache/chain_folds.npz"
    fold_cache.parent.mkdir(parents=True, exist_ok=True)
    if not fold_cache.exists():
        shutil.copyfile(data_root / "derivedcache/chain_folds.npz", fold_cache)
    manifest_path = data_root.joinpath(*relative_path(spec["split_manifest"]).parts)
    registry_path = data_root.joinpath(*relative_path(spec["portable_registry"]).parts)
    for path, field in (
        (manifest_path, "expected_split_manifest_sha256"),
        (registry_path, "expected_portable_registry_sha256"),
    ):
        if sha256_file(path) != sealed_hash(spec[field], field):
            raise ValueError(f"staged manifest identity mismatch: {path.name}")
    projected_bytes = sum(int(item["bytes"]) for item in spec["mounted_inputs"].values())
    projected_bytes += sum(int(item["bytes"]) for item in spec["mounted_files"].values())
    disk_before = shutil.disk_usage(work.parent).free
    reserve = int(spec["minimum_free_bytes_after_staging"])
    if disk_before < projected_bytes + reserve:
        raise OSError("insufficient free disk for parquet assembly plus required reserve")
    mounted = {
        split: assemble(split, item, data_root, dataset_roots)
        for split, item in spec["mounted_inputs"].items()
    }
    indexes = {
        name: copy_mounted(name, item, data_root, dataset_roots)
        for name, item in spec["mounted_files"].items()
    }
    disk_after = shutil.disk_usage(work.parent).free
    if disk_after < reserve:
        raise OSError("post-staging free disk fell below the sealed reserve")
    staging = {
        "mounted_inputs": mounted,
        "mounted_files": indexes,
        "free_bytes_before_assembly": disk_before,
        "free_bytes_after_staging": disk_after,
    }
    return source_root, data_root, manifest_path, runtime, staging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = Path("profile-results").resolve()
    output.mkdir(exist_ok=True)
    receipt: dict[str, Any] = {"status": "STARTING", "quality_result": None, "profiles": []}

    def timeout(*_):
        raise TimeoutError("rank-1 profile reached its internal deadline")

    signal.signal(signal.SIGALRM, timeout)
    try:
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "WNN_THREADS",
            "WNN_LOADER_WORKERS",
        ):
            os.environ[name] = "0" if name == "WNN_LOADER_WORKERS" else "1"
        spec = json.loads(args.spec.read_bytes())
        if spec.get("state") != "SEALED_FOR_PROFILE":
            raise ValueError("profile spec is not sealed for execution")
        inputs = {
            "profile_spec_sha256": sha256_file(args.spec),
            "code_archive_sha256": sha256_file(args.code),
            "supplemental_archive_sha256": sha256_file(args.supplemental),
            "source_commit": spec.get("source_commit"),
            "recipe": EXPECTED_RECIPE,
        }
        receipt["inputs"] = inputs
        source_root, data_root, split_path, runtime, staging = _prepare(spec, args, inputs)
        sys.path.insert(0, str(source_root / "src"))
        os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(data_root / spec["portable_registry"])
        os.environ["WNN_PROJECT_ROOT"] = str(data_root)
        os.environ["WNN_GPU_FRACTION"] = "0.90"
        signal.setitimer(signal.ITIMER_REAL, int(spec["internal_deadline_seconds"]))
        from wnn.splits import load_campaign_split

        validated = load_campaign_split(split_path, data_root)
        if validated.sha256 != sealed_hash(
            spec["expected_split_manifest_sha256"], "split manifest"
        ):
            raise ValueError("canonical split validation returned a different identity")
        import torch

        profiles = []
        for arm in EXPECTED_ARMS:
            profiles.append(run_arm(arm, split_path, source_root, data_root, output))
            torch.cuda.empty_cache()
            gc.collect()
        receipt.update(
            status="PROFILE_ONLY_COMPLETE",
            profiles=profiles,
            runtime=runtime,
            split_sha256=validated.sha256,
            **staging,
        )
    except BaseException as error:
        receipt.update(
            status="FAILED",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        print(receipt["traceback"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt["allocated_seconds"] = round(time.monotonic() - started, 3)
        receipt["quality_result"] = None
        (output / "profile-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        write_bundle(args.out, output)
        print(f"Profile bundle saved: {args.out} status={receipt['status']}", flush=True)
    return 0 if receipt["status"] == "PROFILE_ONLY_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
