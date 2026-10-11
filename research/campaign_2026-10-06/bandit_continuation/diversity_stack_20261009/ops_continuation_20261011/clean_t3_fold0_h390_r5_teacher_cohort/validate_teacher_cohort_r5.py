"""Validate R5 package bytes against the actual R4 failure/profile receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

CHECKOUT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_teacher_cohort_r5 as builder  # noqa: E402

R4_FIXTURE_BINDING_SHA256 = "f3770c483ee72c2a0b0fa6f751b2e4072f9d40bcd23d934ac06019230841ae1a"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runner(files: dict[str, bytes]):
    """Import the exact packaged runner and adjacent helper in a private temp dir."""
    with tempfile.TemporaryDirectory(prefix="wnn-r5-runner-fixture-") as temp:
        scripts = Path(temp) / "scripts"
        scripts.mkdir()
        (scripts / "kaggle_gpu_runner.py").write_bytes(files["scripts/kaggle_gpu_runner.py"])
        (scripts / "kaggle_common.py").write_bytes(files["scripts/kaggle_common.py"])
        sys.path.insert(0, str(scripts))
        sys.modules.pop("kaggle_common", None)
        spec = importlib.util.spec_from_file_location(
            "r5_actual_kaggle_gpu_runner", scripts / "kaggle_gpu_runner.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the packaged Kaggle worker runner")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            yield module
        finally:
            sys.path.remove(str(scripts))
            sys.modules.pop("kaggle_common", None)


def validate(package_dir: Path, fixture_dir: Path, r4_packet: Path, diagnostics_path: Path) -> dict:
    package_dir = package_dir.resolve(strict=True)
    fixture_dir = fixture_dir.resolve(strict=True)
    r4_packet = r4_packet.resolve(strict=True)
    diagnostics_path = diagnostics_path.resolve(strict=True)
    bundle_dir = package_dir / "bundle_teacher_cohort_r5"
    manifest_path = bundle_dir / "bundle-manifest.json"
    archive_path = bundle_dir / "bundle.wnn"
    manifest = read_json(manifest_path)
    if manifest.get("dataset") != builder.R5_DATASET or digest(archive_path) != manifest.get(
        "archive_sha256"
    ):
        raise ValueError("R5 archive or dataset identity does not match its manifest")
    r5_files = builder.read_tar(archive_path)
    mounted = set(manifest["external_mounts"])
    declared = {row["path"] for row in manifest["files"] if row["path"] not in mounted} | set(
        manifest["spec_paths"]
    )
    if set(r5_files) != declared:
        raise ValueError("R5 archive members differ from its sealed file/spec manifest")
    for row in manifest["files"]:
        if row["path"] in mounted:
            continue
        payload = r5_files[row["path"]]
        if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"R5 member identity failed: {row['path']}")
    bootstrap_path = bundle_dir / "kernel/kernel.py"
    driver_path = package_dir / "kernel_support/teacher_cohort_r5_driver.py"
    if (
        r5_files[f"{builder.R5_PACKET}/kernel_support/teacher_cohort_r5_driver.py"]
        != driver_path.read_bytes()
    ):
        raise ValueError("R5 standalone and embedded driver bytes differ")
    if manifest.get("archive_bytes") != archive_path.stat().st_size:
        raise ValueError("R5 archive size differs from manifest")

    contract = read_json(package_dir / "R5_RESOURCE_HANDOFF_CONTRACT_R1.json")
    fixture_binding = read_json(fixture_dir / "R4_ACTUAL_FAILURE_FIXTURE_R1.json")
    if (
        fixture_binding.get("schema") != "wnn.kaggle.r4_actual_failure_profile_fixture.r1"
        or fixture_binding.get("r4_terminal_receipt_sha256") != builder.R4_TERMINAL_RECEIPT_SHA
        or fixture_binding.get("r4_diagnostics_sha256") != builder.R4_DIAGNOSTICS_SHA
        or digest(fixture_dir / "R4_ACTUAL_FAILURE_FIXTURE_R1.json") != R4_FIXTURE_BINDING_SHA256
        or len(fixture_binding.get("r4_actual_output_files", [])) != 7
    ):
        raise ValueError("R4 actual fixture is not bound to the reviewed terminal receipts")
    preseal_bindings = read_json(package_dir / "R5_PRESEAL_BINDINGS_R1.json")
    if preseal_bindings.get("r4_actual_failure_fixture_sha256") != digest(
        fixture_dir / "R4_ACTUAL_FAILURE_FIXTURE_R1.json"
    ):
        raise ValueError("R5 preseal does not bind the actual R4 failed-state fixture")
    for row in fixture_binding.get("r4_actual_output_files", []):
        path = fixture_dir / row["fixture_path"]
        if (
            path.stat().st_size != row["size_bytes"]
            or digest(path) != row["sha256"]
            or row["source_sha256"] != row["sha256"]
        ):
            raise ValueError(f"actual R4 fixture bytes changed: {row['fixture_path']}")
    if len(contract.get("allowed_paths", [])) != 10:
        raise ValueError("R5 resource allowlist is not the exact 10 raw/cache files")
    if {row["path"] for row in contract["allowed_paths"]} != {
        "datasets/train.parquet",
        "datasets/valid.parquet",
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/propcache_c20260929_prop_f0_train.json",
        "cache/propcache_c20260929_prop_f0_train.npy",
        "cache/propcache_c20260929_prop_f0_valid.json",
        "cache/propcache_c20260929_prop_f0_valid.npy",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    }:
        raise ValueError("R5 allowlist contains a missing or undeclared path")

    driver_spec = importlib.util.spec_from_file_location("r5_driver_under_review", driver_path)
    if driver_spec is None or driver_spec.loader is None:
        raise RuntimeError("could not import R5 driver helper functions")
    driver = importlib.util.module_from_spec(driver_spec)
    driver_spec.loader.exec_module(driver)
    fixture_profile_dir = fixture_dir / "paired_profiles"
    producer_evidence = driver._profile_input_receipts(fixture_profile_dir, contract)
    if len(producer_evidence) != 6:
        raise ValueError(
            "actual R4 producer fixture did not yield six expected profile/input bindings"
        )

    r4_archive_path = r4_packet / "bundle_teacher_cohort_r3/bundle.wnn"
    r4_files = builder.read_tar(r4_archive_path)
    if manifest.get("external_mounts") != read_json(
        r4_packet / "bundle_teacher_cohort_r3/bundle-manifest.json"
    ).get("external_mounts"):
        raise ValueError("R5 changed the exact external source/input mount map")
    r4_manifest = read_json(r4_packet / "bundle_teacher_cohort_r3/bundle-manifest.json")
    for old_spec_path in r4_manifest["spec_paths"]:
        expected_spec = builder.changed_spec(r4_files[old_spec_path])
        new_spec_path = f"kaggle_specs/{expected_spec['name']}.json"
        if json.loads(r5_files[new_spec_path]) != expected_spec:
            raise ValueError(
                f"R5 changed a scientific worker spec beyond unique IDs: {old_spec_path}"
            )

    mutation_refusals = 0
    with tempfile.TemporaryDirectory(prefix="wnn-r5-actual-receipt-mutation-") as temp:
        temp_fixture = Path(temp) / "paired_profiles"
        shutil.copytree(fixture_profile_dir, temp_fixture)
        s3_name = "profile_clean_t3_f0_muonall_h384_e3_s3_paired_t4_r5"
        identity_path = temp_fixture / f"{s3_name}.input_identity.json"
        identity = read_json(identity_path)
        cache_row = next(
            row for row in identity["observations"] if row.get("path") == "cache/chains.npz"
        )
        cache_row["observed"]["sha256"] = "0" * 64
        identity_path.write_text(json.dumps(identity))
        try:
            driver._profile_input_receipts(temp_fixture, contract)
        except RuntimeError:
            mutation_refusals += 1
        else:
            raise ValueError("R5 input verifier accepted a mutated actual cache identity receipt")
    with tempfile.TemporaryDirectory(prefix="wnn-r5-actual-log-mutation-") as temp:
        temp_fixture = Path(temp) / "paired_profiles"
        shutil.copytree(fixture_profile_dir, temp_fixture)
        s3_name = "profile_clean_t3_f0_muonall_h384_e3_s3_paired_t4_r5"
        worker_log = temp_fixture / s3_name / "worker.log"
        original_lines = worker_log.read_text().splitlines()
        kept = []
        for line in original_lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if (
                event.get("event") == "portable_sha256_complete"
                and event.get("file") == "valid.parquet"
            ):
                continue
            kept.append(line)
        worker_log.write_text("\n".join(kept) + "\n")
        try:
            driver._profile_input_receipts(temp_fixture, contract)
        except RuntimeError:
            mutation_refusals += 1
        else:
            raise ValueError("R5 input verifier accepted a missing actual raw-Parquet hash event")
    if mutation_refusals != 2:
        raise ValueError("R5 actual producer receipt mutation tests did not both refuse")

    actual_pipeline = read_json(fixture_dir / "R4_teacher_cohort_pipeline.json")
    actual_guard = read_json(fixture_dir / "R4_paired_profile_guard.json")
    diagnostics = read_json(diagnostics_path)
    if (
        actual_pipeline.get("error_type") != "ValueError"
        or actual_pipeline.get("error") != "Worker reservation exceeds host operational budget"
        or actual_guard.get("status") != "pass"
        or actual_guard.get("runner_concurrency") != 2
        or actual_pipeline.get("stages", [{}])[0].get("name") != "paired_profiles"
    ):
        raise ValueError(
            "actual R4 failed-state fixture differs from the reviewed terminal incident"
        )
    if (
        diagnostics.get("pipeline", {}).get("fit_stage_started") is not False
        or diagnostics.get("pipeline", {}).get("export_stage_started") is not False
    ):
        raise ValueError("R4 fixture does not prove the error preceded all full fits/exports")

    if r5_files["scripts/kaggle_gpu_runner.py"] != r4_files["scripts/kaggle_gpu_runner.py"]:
        raise ValueError("R5 changed the strict R4 worker admission implementation")
    fit_specs = [
        json.loads(r4_files[f"kaggle_specs/fit_clean_t3_f0_muonall_h384_e3_s{seed}_r3.json"])
        for seed in (3, 4, 5)
    ]
    if any(spec["limits"]["ram_bytes"] != 8_000_000_000 for spec in fit_specs):
        raise ValueError("R5 changed one or more 8 GB fit reservations")

    runner_context = load_runner(r4_files)
    runner = next(runner_context)
    try:
        preflight = read_json(fixture_dir / "R4_profile_preflight_inventory.json")
        high_host = preflight["host"]
        devices = preflight["devices"]
        accepted = runner.admit_specs(fit_specs, high_host, devices)
        if accepted != 2:
            raise ValueError(
                "actual strict runner unexpectedly serialized the high-memory boundary fixture"
            )
        capacity = int(high_host["capacity_bytes"])
        reserve = int(capacity * 0.2)
        boundary_host = {**high_host, "available_bytes": reserve + 8_000_000_000 - 1}
        try:
            runner.admit_specs(fit_specs, boundary_host, devices)
        except ValueError as error:
            if str(error) != "Worker reservation exceeds host operational budget":
                raise
        else:
            raise ValueError(
                "strict runner failed to refuse the sub-8 GB operational-budget fixture"
            )
    finally:
        runner_context.close()

    prep = read_json(package_dir / "R5_PREPARATION_RECEIPT_R1.json")
    if (
        prep.get("r5_archive_sha256") != manifest["archive_sha256"]
        or prep.get("r5_bootstrap_sha256") != digest(bootstrap_path)
        or prep.get("root_authority_created") is not False
        or prep.get("provider_action_performed") is not False
    ):
        raise ValueError(
            "R5 preparation receipt does not bind the built package or overclaims authority"
        )
    if (
        diagnostics.get("kernel") != builder.R4_KERNEL
        or diagnostics.get("generic_terminal_receipt", {}).get("sha256")
        != builder.R4_TERMINAL_RECEIPT_SHA
        or diagnostics.get("pipeline", {}).get("fit_stage_started") is not False
        or diagnostics.get("pipeline", {}).get("export_stage_started") is not False
    ):
        raise ValueError("external R4 terminal diagnostics do not match the actual no-fit failure")

    return {
        "schema": "wnn.kaggle.teacher_cohort_r5_actual_failure_validation.r1",
        "status": "pass_static_actual_producer_fixture",
        "r4_actual_failure": {
            "kernel": builder.R4_KERNEL,
            "terminal_status": "ERROR",
            "paired_profiles_passed": True,
            "fit_workers_started": False,
            "error": actual_pipeline["error"],
            "post_profile_memory_inventory": "not persisted in R4; no root-cause claim",
            "cache_pressure_cause": "unproven hypothesis",
        },
        "actual_fixture_binding_sha256": digest(fixture_dir / "R4_ACTUAL_FAILURE_FIXTURE_R1.json"),
        "actual_profile_producer_evidence": producer_evidence,
        "actual_producer_fixture_mutations_refused": mutation_refusals,
        "strict_admission_fixture": {
            "runner_sha256": hashlib.sha256(r4_files["scripts/kaggle_gpu_runner.py"]).hexdigest(),
            "fit_worker_ram_reservation_bytes": 8_000_000_000,
            "high_memory_fixture_concurrency": accepted,
            "low_memory_fixture_result": "same strict admission refused with Worker reservation exceeds host operational budget",
            "low_memory_fixture_is_synthetic_boundary_not_reconstruction_of_R4_post-profile_state": True,
        },
        "r5": {
            "dataset": builder.R5_DATASET,
            "kernel_id": builder.R5_KERNEL,
            "archive_sha256": manifest["archive_sha256"],
            "manifest_sha256": digest(manifest_path),
            "bootstrap_sha256": digest(bootstrap_path),
            "driver_sha256": digest(driver_path),
            "resource_contract_sha256": digest(
                package_dir / "R5_RESOURCE_HANDOFF_CONTRACT_R1.json"
            ),
            "allowlist_count": len(contract["allowed_paths"]),
            "root_authorized": False,
            "provider_action_performed": False,
        },
        "r4_diagnostics_sha256": digest(diagnostics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--r4-packet", type=Path, required=True)
    parser.add_argument("--r4-diagnostics", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.package_dir, args.fixture_dir, args.r4_packet, args.r4_diagnostics)
    builder.write_json(args.receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
