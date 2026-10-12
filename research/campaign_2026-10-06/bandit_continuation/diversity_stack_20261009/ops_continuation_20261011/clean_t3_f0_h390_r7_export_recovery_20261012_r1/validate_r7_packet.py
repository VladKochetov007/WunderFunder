from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

OPS_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011"
)
PACKET_NAME = "clean_t3_f0_h390_r7_export_recovery_20261012_r1"
R5_RELATIVE = Path("clean_t3_fold0_h390_r5_teacher_cohort/package_r9")
R6_RELATIVE = Path(
    "clean_t3_fold0_h390_r5_teacher_cohort/r6_bootstrap_guard_successor_20261012_r1/"
    "terminal_collection_v2"
)
R5_SOURCE_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot"
)
R5_SOURCE_MANIFEST_SHA256 = "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa"
R5_PREDICTOR_SHA256 = "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0"
R5_TRAINER_SHA256 = "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import packet file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_actual_model_consumer(
    main_root: Path, source_root: Path, bindings: dict[str, Any]
) -> list[dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    scripts = source_root / "scripts"
    source_lib = source_root / "src"
    sys.path.insert(0, str(source_lib))
    sys.path.insert(0, str(scripts))
    predictor_path = scripts / "predict_run.py"
    trainer_path = scripts / "train_rnn.py"
    if sha256_file(predictor_path) != R5_PREDICTOR_SHA256:
        raise ValueError("extracted R5 predictor differs from the independently pinned source")
    if sha256_file(trainer_path) != R5_TRAINER_SHA256:
        raise ValueError("extracted R5 trainer differs from the independently pinned source")
    predictor_source = predictor_path.read_text(encoding="utf-8")
    if "out[i : i + len(p)] = p[..., :2]" not in predictor_source:
        raise ValueError("R5 predictor no longer writes only the two official target columns")
    predictor = import_file("r7_offline_r5_predictor", predictor_path)
    results = []
    for run_name, expected in sorted(bindings["checkpoints"].items()):
        checkpoint_path = main_root / expected["model_ep2_path"]
        if checkpoint_path.stat().st_size != expected["model_ep2_bytes"]:
            raise ValueError(f"checkpoint size mismatch: {run_name}")
        if sha256_file(checkpoint_path) != expected["model_ep2_sha256"]:
            raise ValueError(f"checkpoint digest mismatch: {run_name}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("args", {}).get("seed") != expected["seed"]
            or checkpoint.get("args", {}).get("hidden") != 384
            or checkpoint.get("args", {}).get("layers") != 2
            or checkpoint.get("n_in") != 211
            or checkpoint.get("model", {}).get("head.2.weight", torch.empty(0)).shape[0] != 23
        ):
            raise ValueError(f"R5 predictor checkpoint interface differs: {run_name}")
        model = predictor.load_model(checkpoint, torch.device("cpu"))
        nonfinite = [
            name
            for name, tensor in model.state_dict().items()
            if not torch.isfinite(tensor).all().item()
        ]
        if nonfinite:
            raise ValueError(
                f"R6 checkpoint has non-finite model tensors: {run_name}: {nonfinite[:3]}"
            )
        results.append(
            {
                "run": run_name,
                "selected_epoch": 2,
                "checkpoint_sha256": expected["model_ep2_sha256"],
                "provenance_sha256": expected["provenance_sha256"],
                "seed": expected["seed"],
                "n_in": int(checkpoint["n_in"]),
                "hidden": int(checkpoint["args"]["hidden"]),
                "layers": int(checkpoint["args"]["layers"]),
                "checkpoint_head_outputs": int(checkpoint["model"]["head.2.weight"].shape[0]),
                "prediction_output_columns": 2,
                "loaded_by": "unchanged R5 predict_run.load_model on CPU; no inference or data loading",
                "all_loaded_model_tensors_finite": True,
            }
        )
        del model, checkpoint
    return results


def verify_npy_scanner(kernel: Any, temporary: Path) -> dict[str, Any]:
    import numpy as np

    good_path = temporary / "finite.npy"
    bad_path = temporary / "nonfinite.npy"
    good = np.arange(24, dtype=np.float32).reshape(2, 6, 2)
    np.save(good_path, good, allow_pickle=False)
    expected_digest = sha256_file(good_path)
    header = kernel.read_npy_header(good_path)
    if header != ((2, 6, 2), "<f4"):
        raise ValueError(f"NPY scanner accepted an unexpected header: {header}")
    scanned_digest = kernel.finite_npy_sha256(good_path, (2, 6, 2))
    if scanned_digest != expected_digest:
        raise ValueError("streaming NPY finite/hash scanner changed the file identity")
    bad = good.copy()
    bad[0, 0, 0] = np.nan
    np.save(bad_path, bad, allow_pickle=False)
    try:
        kernel.finite_npy_sha256(bad_path, (2, 6, 2))
    except ValueError as error:
        if "non-finite" not in str(error):
            raise
    else:
        raise ValueError("NPY scanner accepted a non-finite prediction")
    return {"finite_fixture_sha256": expected_digest, "nonfinite_rejected": True}


def validate(main_root: Path, packet: Path) -> dict[str, Any]:
    started = time.monotonic()
    main_root = main_root.resolve(strict=True)
    packet = packet.resolve(strict=True)
    kernel_dir = packet / "kernel"
    bindings_path = kernel_dir / "R7_EXPORT_BINDINGS_R1.json"
    kernel_path = kernel_dir / "kernel.py"
    metadata_path = kernel_dir / "kernel-metadata.json"
    release_path = kernel_dir / "ROOT_EXPORT_RELEASE_R7.json"
    kernel = import_file("r7_offline_kernel", kernel_path)
    bindings = kernel.load_bindings(kernel_dir)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if release.get("status") != "pending" or release.get("authorized") is not False:
        raise PermissionError("offline validator accepts only the unadmitted pending packet")
    try:
        kernel.require_root_release(kernel_dir, bindings)
    except PermissionError:
        pass
    else:
        raise AssertionError("pending R7 packet unexpectedly passed the worker-start release gate")
    if len(bindings["export_names"]) != 6 or len(bindings["r5_declared_export_names"]) != 9:
        raise ValueError("R7 must retain the nine-spec source catalog and run only six exports")
    if len(bindings["local_full_valid_counterparts"]) != 3:
        raise ValueError("R7 must bind all three completed local full-valid counterparts")
    if bindings["execution_scope"] != {
        "fit_workers": 0,
        "profile_workers": 0,
        "exports_only": True,
    }:
        raise ValueError("R7 execution scope is not export-only")
    if bindings["expected_total_npy_bytes"] != 5_731_200_768:
        raise ValueError("R7 six-array payload differs from the reviewed target")
    expected_datasets = {bindings["r5_dataset_id"]}
    for mapping in bindings["r5_external_mounts"].values():
        expected_datasets.add(
            mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        )
    if (
        metadata.get("id") != bindings["target"]["kernel_id"]
        or metadata.get("title") != bindings["target"]["title"]
        or metadata.get("kernel_sources") != [bindings["r6_kernel_id"]]
        or set(metadata.get("dataset_sources", [])) != expected_datasets
        or metadata.get("is_private") is not True
        or metadata.get("enable_internet") is not False
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_tpu") is not False
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
    ):
        raise ValueError("R7 Kaggle metadata differs from the frozen private T4 export packet")
    if (
        bindings["worker_limits"]["max_concurrent_workers"] != 1
        or bindings["runtime_limits"]["worst_case_serial_worker_seconds"]
        > bindings["runtime_limits"]["kernel_wall_ceiling_seconds"]
        - bindings["runtime_limits"]["finalization_reserve_seconds"]
    ):
        raise ValueError("R7 serial worker/runtime reservation exceeds its frozen wall budget")

    ops = main_root / OPS_RELATIVE
    r5 = ops / R5_RELATIVE
    r6 = ops / R6_RELATIVE
    archive = r5 / "bundle_teacher_cohort_r5" / "bundle.wnn"
    manifest_path = r5 / "bundle_teacher_cohort_r5" / "bundle-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if hashlib.sha256(manifest_raw).hexdigest() != bindings["r5_bundle_manifest_sha256"]:
        raise ValueError("R5 manifest changed after R7 binding generation")
    if sha256_file(archive) != bindings["r5_bundle_archive_sha256"]:
        raise ValueError("R5 source archive changed after R7 binding generation")
    provider_root = r6 / "provider_outputs" / "attempt_001" / "outputs"
    if not provider_root.is_dir():
        raise FileNotFoundError(
            "R6 collected provider outputs are not available for the actual source consumer"
        )

    with tempfile.TemporaryDirectory(prefix="wnn-r7-offline-validate-") as temporary_name:
        temporary = Path(temporary_name)
        extracted_root = temporary / "project"
        kernel.checked_extract_bundle(
            archive,
            manifest,
            hashlib.sha256(manifest_raw).hexdigest(),
            bindings["r5_bundle_archive_sha256"],
            extracted_root,
            bindings,
            time.monotonic() + 100,
        )
        driver, _helpers, specs = kernel.verify_export_specs(extracted_root, bindings)
        if len(specs) != 6 or len(driver.EXPORT_NAMES) != 9:
            raise ValueError(
                "actual R5 driver/spec consumer did not preserve the 9→6 export contract"
            )
        source_root = extracted_root / R5_SOURCE_RELATIVE
        source_manifest = source_root / "SOURCE_MANIFEST_H390.json"
        if sha256_file(source_manifest) != R5_SOURCE_MANIFEST_SHA256:
            raise ValueError("extracted source manifest differs from R5 frozen source binding")
        with tempfile.TemporaryDirectory(prefix="wnn-r7-checkpoint-mount-") as mount_name:
            mount_root = Path(mount_name) / "experiment"
            mounted = kernel.mount_verified_checkpoints(
                mount_root, provider_root, bindings["checkpoints"]
            )
            if len(mounted) != 6 or not all(path.is_symlink() for path in mounted):
                raise ValueError(
                    "actual R6 source mount did not expose exactly three checkpoint/provenance pairs"
                )
        model_loads = verify_actual_model_consumer(main_root, source_root, bindings)
        npy_scanner = verify_npy_scanner(kernel, temporary)

    local_valid = {
        name: {
            "receipt_sha256": row["local_handoff_receipt_sha256"],
            "npy_sha256": row["prediction_sha256"],
            "sidecar_sha256": row["sidecar_sha256"],
            "bytes": row["prediction_bytes"],
            "logical_equivalence": row["logical_equivalence"],
        }
        for name, row in sorted(bindings["local_full_valid_counterparts"].items())
    }
    result = {
        "schema": "wnn.kaggle.clean_t3_f0_export_only_r7_offline_validation.r1",
        "status": "PASS_OFFLINE_ACTUAL_PRODUCER_CONSUMER_PREFLIGHT",
        "packet_path": str(packet),
        "support_envelope": {
            "slice": "wnn-campaign.slice",
            "cpu_quota": "50%",
            "threads": 1,
            "memory_max_bytes": 1_500_000_000,
            "runtime_max_seconds": 120,
            "launcher": "MAIN scripts/limited",
        },
        "packet_pins": {
            "bindings_sha256": sha256_file(bindings_path),
            "kernel_sha256": sha256_file(kernel_path),
            "metadata_sha256": sha256_file(metadata_path),
            "pending_release_sha256": sha256_file(release_path),
            "packet_builder_sha256": sha256_file(packet / "prepare_r7_packet.py"),
            "offline_validator_sha256": sha256_file(Path(__file__).resolve()),
            "readme_sha256": sha256_file(packet / "README_R1.md"),
            "pre_final_diagnostics_sha256": sha256_file(packet / "PRE_FINAL_DIAGNOSTICS_R1.json"),
            "r5_archive_sha256": sha256_file(archive),
            "r5_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "r6_operator_receipt_sha256": bindings["r6_operator_receipt_sha256"],
            "r6_stage_receipt_sha256": bindings["r6_root_stage_receipt_sha256"],
            "r6_collector_receipt_sha256": bindings["r6_complete_collector_receipt_sha256"],
        },
        "offline_consumer_checks": {
            "actual_r5_archive_members_extracted_and_hash_checked": True,
            "actual_r5_driver_imported": True,
            "r5_export_names_preserved": len(driver.EXPORT_NAMES),
            "r7_export_names": bindings["export_names"],
            "active_specs_validated_by_actual_r5_consumer": len(specs),
            "actual_r5_source_manifest_sha256": R5_SOURCE_MANIFEST_SHA256,
            "actual_predictor_sha256": R5_PREDICTOR_SHA256,
            "actual_trainer_sha256": R5_TRAINER_SHA256,
            "actual_r6_checkpoint_mounts_verified": len(mounted) // 2,
            "actual_r5_predictor_loaded_all_three_checkpoints": model_loads,
            "prediction_npy_finite_scanner_fixture": npy_scanner,
            "pending_root_release_refused_before_worker_start": True,
            "fit_workers_started": 0,
            "profile_workers_started": 0,
            "export_workers_started": 0,
        },
        "local_full_valid_counterparts": local_valid,
        "limitations": [
            "No Kaggle/provider calls, GPU use, exports, fits, or score calculation were performed.",
            "The private source-kernel version-2 readback and mounted payload hashes must be checked at root admission; kernel metadata addresses its slug, not an immutable version.",
            "The actual Kaggle host/device resource gate is evaluated separately before every single-export worker by the unchanged R5 runner.",
            "The three local full-valid arrays are logically equivalent under the exact R5 export specs and source/checkpoint/split/FP32 identities; they are not claimed byte-identical to unrun Kaggle output arrays.",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "VALIDATION_R1.json")
    args = parser.parse_args()
    result = validate(args.main_root, args.packet_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
