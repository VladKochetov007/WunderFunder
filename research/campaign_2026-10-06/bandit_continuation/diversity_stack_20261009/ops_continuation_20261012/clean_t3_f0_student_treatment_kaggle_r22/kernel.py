from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

SOURCE_DATASET = "vladkochetov07/wnn-clean-t3-f0-treatment-source-20261012-r22"
ASSET_DATASET = "vladkochetov07/wnn-clean-t3-f0-treatment-assets-20261012-r22"
KERNEL_ID = "vladkochetov07/wnn-clean-t3-f0-muonall-student-20261012-r22-runner"
PAIR_PLAN_SHA256 = "__PAIR_PLAN_SHA256__"
ASSET_MANIFEST_SHA256 = "__ASSET_MANIFEST_SHA256__"
BUNDLE_SHA256 = "88641b3c9fc629a7a2ce8b11cfd98ecf8b53fe943ffb2d7f632e1dfb997fa549"
SOURCE_BUNDLE_MANIFEST_SHA256 = "__SOURCE_BUNDLE_MANIFEST_SHA256__"
WALL_SECONDS = 14_400
FINALIZATION_SECONDS = 300
PROFILE_BATCHES = 30
PROFILE_WARMUP = 5
MAX_PROFILE_RAM = 6_000_000_000
MAX_PROFILE_GPU = 5_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_root_release(receipt: dict, plan: dict, source_manifest_sha: str) -> None:
    if receipt.get("schema") != "wnn.clean_f0_student_treatment_root_release.r2":
        raise PermissionError("missing root release schema")
    if receipt.get("status") != "admitted" or receipt.get("authorized") is not True:
        raise PermissionError("root release is not admitted")
    if receipt.get("target") != {"kernel_id": KERNEL_ID, "source_dataset": SOURCE_DATASET, "asset_dataset": ASSET_DATASET}:
        raise PermissionError("root release target does not match this packet")
    binding = receipt.get("binding", {})
    expected = {
        "pair_plan_sha256": PAIR_PLAN_SHA256,
        "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
        "bundle_sha256": BUNDLE_SHA256,
        "bundle_manifest_sha256": SOURCE_BUNDLE_MANIFEST_SHA256,
        "source_manifest_sha256": source_manifest_sha,
        "cache_metadata_sha256": plan["cache"]["metadata"]["sha256"],
        "cache_train_sha256": plan["cache"]["train"]["sha256"],
        "cache_valid_sha256": plan["cache"]["fit_valid"]["sha256"],
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            raise PermissionError(f"root release does not bind {field}")
    parent_hashes = {arm: item["checkpoint"]["sha256"] for arm, item in plan["parents"].items()}
    parent_provenance = {arm: item["provenance"]["sha256"] for arm, item in plan["parents"].items()}
    if binding.get("parent_checkpoint_sha256") != parent_hashes or binding.get("parent_provenance_sha256") != parent_provenance:
        raise PermissionError("root release parent identities differ from the frozen pair")
    budget = receipt.get("quota_budget", {})
    if type(budget.get("wall_ceiling_seconds")) is not int or budget["wall_ceiling_seconds"] != WALL_SECONDS:
        raise PermissionError("root release wall ceiling differs from the packet")
    if type(budget.get("single_device_gpu_seconds_upper_bound")) is not int or budget["single_device_gpu_seconds_upper_bound"] != WALL_SECONDS:
        raise PermissionError("root release single-device GPU bound differs from the packet")
    if budget.get("paid_spending_authorized") is not False:
        raise PermissionError("this packet does not authorize paid spending")
    if receipt.get("profile_authorized") is not True:
        raise PermissionError("root release does not authorize the required device profile")
    if type(receipt.get("fit_authorized")) is not bool:
        raise PermissionError("root release must explicitly authorize or withhold full fits")


def safe_rel(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe relative path: {value}")
    return str(path)


def extract_bundle(archive: Path, manifest: dict, root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"working root already exists: {root}")
    if sha256(archive) != BUNDLE_SHA256 or manifest.get("archive_sha256") != BUNDLE_SHA256:
        raise ValueError("sealed trainer bundle SHA mismatch")
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    root.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = safe_rel(member.name)
            if not member.isfile() or name not in allowed or name in seen:
                raise ValueError(f"undeclared, unsafe, duplicate or non-file bundle member: {name}")
            seen.add(name)
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"missing member body: {name}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if seen != allowed:
        raise ValueError("bundle inventory differs from the sealed manifest")
    for item in manifest["files"]:
        path = root / safe_rel(item["path"])
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"source archive member identity mismatch: {item['path']}")


def validate_and_mount_external(root: Path, input_roots: dict[str, Path], manifest: dict, plan: dict) -> dict:
    expected = manifest.get("external_mounts")
    if not isinstance(expected, dict) or expected != plan["external_mounts"]:
        raise ValueError("source and paired-plan external mount maps differ")
    asset_root = input_roots[ASSET_DATASET]
    asset_manifest_path = asset_root / "assets-manifest.json"
    if sha256(asset_manifest_path) != ASSET_MANIFEST_SHA256:
        raise ValueError("flat asset manifest SHA mismatch")
    asset_manifest = load_json(asset_manifest_path)
    records = {item["filename"]: item for item in asset_manifest["files"]}
    if records != plan["asset_files"]:
        raise ValueError("flat asset inventory differs from the paired plan")
    expected_asset_files = {"assets-manifest.json", *records}
    observed_asset_files = {path.name for path in asset_root.iterdir() if path.is_file()}
    if observed_asset_files != expected_asset_files:
        raise ValueError("flat asset dataset contains missing or undeclared files")
    observations = []
    for logical, mapping in expected.items():
        safe_rel(logical)
        if isinstance(mapping, str):
            dataset = "vladkochetov07/wunder-funder"
            filename = mapping
        else:
            dataset = mapping["dataset"]
            filename = mapping["filename"]
        if dataset not in input_roots:
            raise ValueError(f"undeclared dataset mount: {dataset}")
        source = input_roots[dataset] / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing mounted input: {dataset}/{filename}")
        descriptor = plan["external_input_identities"].get(logical)
        if descriptor is None or source.stat().st_size != descriptor["size"]:
            raise ValueError(f"external mounted size mismatch: {logical}")
        # The frozen trainer performs portable full-byte verification for the two raw
        # parquet inputs before constructing its fitting data. All other declared
        # inputs are SHA-checked here once, before a worker can start.
        raw = logical in {"datasets/train.parquet", "datasets/valid.parquet"}
        observed_sha = None if raw else sha256(source)
        if not raw and observed_sha != descriptor["sha256"]:
            raise ValueError(f"external mounted SHA mismatch: {logical}")
        destination = root / safe_rel(logical)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output collision at external mount: {logical}")
        destination.symlink_to(source)
        observations.append({"path": logical, "size": source.stat().st_size, "sha256": observed_sha,
                             "status": "deferred_to_frozen_trainer" if raw else "verified"})
    return {"schema": "wnn.clean_f0_pair_external_mount_preflight.r1", "status": "pass",
            "observations": observations, "raw_parquet_sha_producer": "sealed train_rnn.py portable input verification"}


def build_spec(root: Path, plan: dict, name: str, argv: list[str], artifacts: list[str], limits: dict) -> Path:
    code = plan["worker_code"]
    inputs = plan["worker_inputs"]
    spec = {
        "version": 1,
        "name": name,
        "kind": "gpu",
        "argv": ["{python}", *argv],
        "code": code,
        "code_sha256": plan["worker_code_sha256"],
        "inputs": inputs,
        "artifacts": artifacts,
        "limits": limits,
        "environment": {
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "WNN_PROJECT_ROOT": str(root),
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "4",
            "WNN_PORTABLE_DATA_MANIFEST": str(root / "research/campaign_2026-10-04/portable_inputs.json"),
            "WNN_STARTUP_DIAGNOSTICS": "1",
            "WNN_STARTUP_STACK_DUMP_SECONDS": "60",
            "WNN_DEFERRED_IDENTITY_PATHS": ",".join(item["path"] for item in inputs),
        },
    }
    path = root / "kaggle_specs" / f"{name}.json"
    write_json(path, spec)
    return path


def run_one(root: Path, spec: Path, output: Path, deadline: float) -> dict:
    if output.exists():
        raise FileExistsError(f"stage output already exists: {output}")
    remaining = deadline - time.monotonic()
    if remaining <= FINALIZATION_SECONDS:
        raise TimeoutError("not enough kernel time remains to start another stage")
    runner = root / "scripts/kaggle_gpu_runner.py"
    env = os.environ.copy()
    env["WNN_KERNEL_DEADLINE_MONOTONIC"] = str(deadline)
    command = [sys.executable, str(runner), "--root", str(root), "--spec", str(spec), "--output", str(output)]
    started = time.monotonic()
    result = subprocess.run(command, cwd=root, env=env, timeout=max(1, int(remaining - FINALIZATION_SECONDS)), check=False)
    summary_path = output / "summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    if result.returncode != 0 or summary.get("incident") is not None or summary.get("not_started"):
        raise RuntimeError(f"worker stage failed ({result.returncode}): {summary.get('incident')}")
    finished = summary.get("finished", [])
    if len(finished) != 1 or finished[0].get("status") != "complete" or finished[0].get("returncode") != 0:
        raise RuntimeError("worker stage did not produce one successful terminal record")
    return {"name": output.name, "status": "complete", "wall_seconds": time.monotonic() - started,
            "worker": finished[0], "summary_sha256": sha256(summary_path)}


def profile_guard(result: dict, output: Path) -> dict:
    worker = result["worker"]
    device = worker.get("device", {})
    if "T4" not in device.get("name", "") or not device.get("uuid"):
        raise RuntimeError("required one-device T4 profile not observed")
    if worker.get("peak_ram_bytes", MAX_PROFILE_RAM + 1) > MAX_PROFILE_RAM:
        raise RuntimeError("profile exceeded declared worker RAM guard")
    if worker.get("peak_gpu_bytes", MAX_PROFILE_GPU + 1) > MAX_PROFILE_GPU:
        raise RuntimeError("profile exceeded observed VRAM guard")
    log = output / worker["name"] / "worker.log"
    markers = {}
    for line in log.read_text(errors="replace").splitlines():
        for key in ("PROFILE_UPDATE_COUNTS ", "GRAD_STATS ", "PROFILE "):
            if line.startswith(key):
                if key in markers:
                    raise RuntimeError(f"duplicate {key.strip()} record")
                markers[key] = json.loads(line[len(key):])
    counts = markers.get("PROFILE_UPDATE_COUNTS ", {}).get("epoch", {})
    attempts = counts.get("optimizer_attempts")
    completed = counts.get("optimizer_completed_updates")
    skipped = counts.get("amp_skipped_updates")
    if counts.get("completed_batches") != PROFILE_BATCHES or not all(type(x) is int for x in (attempts, completed, skipped)):
        raise RuntimeError("profile did not complete the exact 30-batch warmup/measurement prefix")
    if attempts <= 0 or completed <= 0 or completed + skipped != attempts:
        raise RuntimeError("optimizer update counters do not reconcile")
    grad = markers.get("GRAD_STATS ", {})
    if grad.get("steps") != attempts or grad.get("nonfinite_steps", attempts + 1) > skipped:
        raise RuntimeError("finite-gradient summary is inconsistent with AMP skips")
    timing = markers.get("PROFILE ", {})
    if not isinstance(timing.get("step_wall"), (int, float)) or timing["step_wall"] <= 0:
        raise RuntimeError("profile has no positive measured batch timing")
    return {"status": "pass", "device": device, "completed_batches": PROFILE_BATCHES,
            "warmup_batches": PROFILE_WARMUP, "optimizer_attempts": attempts,
            "optimizer_completed_updates": completed, "amp_skipped_updates": skipped,
            "gradient_nonfinite_steps": grad["nonfinite_steps"], "profile": timing,
            "peak_ram_bytes": worker["peak_ram_bytes"], "peak_gpu_bytes": worker["peak_gpu_bytes"],
            "actual_t4_devices_profiled": 1, "two_device_concurrency_claim": False}


def main() -> None:
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    source_root = Path("/kaggle/input") / SOURCE_DATASET.rsplit("/", 1)[1]
    assets_root = Path("/kaggle/input") / ASSET_DATASET.rsplit("/", 1)[1]
    data_root = Path("/kaggle/input/wunder-funder")
    expected_source_files = {"bundle.wnn", "bundle-manifest.json", "pair-plan.json", "ROOT_TREATMENT_RELEASE_R2.json"}
    if {p.name for p in source_root.iterdir() if p.is_file()} != expected_source_files:
        raise RuntimeError("source dataset file inventory differs from the frozen four-file stage")
    plan_path = source_root / "pair-plan.json"
    if sha256(plan_path) != PAIR_PLAN_SHA256:
        raise RuntimeError("paired plan SHA mismatch")
    plan = load_json(plan_path)
    asset_path = assets_root / "assets-manifest.json"
    if sha256(asset_path) != ASSET_MANIFEST_SHA256:
        raise RuntimeError("asset manifest SHA mismatch")
    load_json(asset_path)
    manifest_path = source_root / "bundle-manifest.json"
    bundle_manifest = load_json(manifest_path)
    if bundle_manifest.get("dataset") != SOURCE_DATASET or bundle_manifest.get("archive_sha256") != BUNDLE_SHA256:
        raise RuntimeError("source bundle manifest identity mismatch")
    if sha256(manifest_path) != SOURCE_BUNDLE_MANIFEST_SHA256:
        raise RuntimeError("source bundle manifest SHA mismatch")
    release = load_json(source_root / "ROOT_TREATMENT_RELEASE_R2.json")
    validate_root_release(release, plan, plan["source_manifest_sha256"])
    root = Path("/kaggle/working/experiment")
    if Path("/kaggle/working/results").exists():
        raise FileExistsError("results output already exists")
    if shutil.disk_usage("/kaggle/working").free < 8_000_000_000:
        raise RuntimeError("less than 8 GB free for staged checkpoints and predictions")
    extract_bundle(source_root / "bundle.wnn", bundle_manifest, root)
    inputs = {SOURCE_DATASET: source_root, ASSET_DATASET: assets_root, "vladkochetov07/wunder-funder": data_root,
              "vladkochetov07/wnn-c20261004-fold0-assets": Path("/kaggle/input/wnn-c20261004-fold0-assets")}
    mount_receipt = validate_and_mount_external(root, inputs, bundle_manifest, plan)
    write_json(root / "results_preflight.json", {"mounts": mount_receipt, "source_bundle_sha256": BUNDLE_SHA256,
                                                   "pair_plan_sha256": PAIR_PLAN_SHA256,
                                                   "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
                                                   "root_release_sha256": sha256(source_root / "ROOT_TREATMENT_RELEASE_R2.json")})
    output_root = Path("/kaggle/working/results")
    output_root.mkdir(parents=True, exist_ok=False)
    pipeline = {"schema": "wnn.clean_f0_student_treatment_pipeline.r2", "status": "running",
                "started_monotonic": started, "wall_ceiling_seconds": WALL_SECONDS, "stages": []}
    write_json(output_root / "pipeline.json", pipeline)
    profile_spec = build_spec(root, plan, plan["profile"]["name"], plan["profile"]["argv"], [], plan["profile"]["limits"])
    profile_output = output_root / "profile"
    profile_result = run_one(root, profile_spec, profile_output, deadline)
    profile_result["guard"] = profile_guard(profile_result, profile_output)
    pipeline["stages"].append(profile_result)
    pipeline["profile_guard"] = profile_result["guard"]
    pipeline["status"] = "profile_passed"
    write_json(output_root / "pipeline.json", pipeline)
    if release.get("fit_authorized") is not True:
        pipeline["status"] = "profile_complete_treatment_fits_not_authorized"
        write_json(output_root / "pipeline.json", pipeline)
        return
    pipeline["status"] = "fits_running"
    write_json(output_root / "pipeline.json", pipeline)
    for arm in plan["arms"]:
        for stage in arm["fit_stages"]:
            spec = build_spec(root, plan, stage["name"], stage["argv"], stage["artifacts"], stage["limits"])
            result = run_one(root, spec, output_root / stage["name"], deadline)
            pipeline["stages"].append(result)
            write_json(output_root / "pipeline.json", pipeline)
        export = arm["export"]
        spec = build_spec(root, plan, export["name"], export["argv"], export["artifacts"], export["limits"])
        result = run_one(root, spec, output_root / export["name"], deadline)
        pipeline["stages"].append(result)
        write_json(output_root / "pipeline.json", pipeline)
    pipeline["status"] = "complete"
    pipeline["elapsed_seconds"] = time.monotonic() - started
    write_json(output_root / "pipeline.json", pipeline)


if __name__ == "__main__":
    main()
