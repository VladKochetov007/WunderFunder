from __future__ import annotations

import json
import math
import os
import shutil
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

import capacity_worker as runtime

PACKET = Path(__file__).resolve().parent


def verify_authority() -> dict[str, object]:
    """Require the root's exact packet admission before any model computation."""
    authority = json.loads((PACKET / "ROOT_AUTHORIZATION.json").read_text())
    specification = json.loads((PACKET / "worker_spec.json").read_text())
    if authority.get("authorized") is not True:
        raise PermissionError("r5 root admission pending")
    if authority.get("attempt_id") != os.environ["WNN_ATTEMPT_ID"]:
        raise PermissionError("r5 admission attempt mismatch")
    if authority.get("recipe_sha256") != runtime.sha256_file(PACKET / "CAPACITY_R5_RECIPE.json"):
        raise PermissionError("r5 recipe binding mismatch")
    if authority.get("worker_spec_sha256") != runtime.sha256_file(PACKET / "worker_spec.json"):
        raise PermissionError("r5 specification binding mismatch")
    for path, expected in specification["packet_files"].items():
        if runtime.sha256_file(PACKET / path) != expected:
            raise ValueError("r5 packet member binding mismatch")
    return specification


def prepare_project(
    output: Path, specification: dict[str, object]
) -> tuple[Path, dict[str, str], Path]:
    """Extract sealed own source; verify and bind the six exact cloud inputs once."""
    project = output / "project"
    source_hashes = runtime._safe_extract_source(PACKET, project)
    input_root = Path(os.environ["WNN_INPUT_DIR"]).resolve(strict=True)
    raw_sources = []
    bindings = {}
    for logical, expected in specification["inputs"].items():
        source = input_root / logical
        if source.is_symlink() or not source.is_file():
            raise ValueError("r5 input must be a regular staged file")
        if source.stat().st_size != expected["bytes"]:
            raise ValueError("r5 input byte-size mismatch")
        identity = runtime._sha_matches(source, expected["sha256"], expected["bytes"])
        destination = project / logical
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("r5 source archive unexpectedly contains an input")
        destination.symlink_to(source)
        item = {
            "mount_path": str(source),
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
            "identity": asdict(identity),
        }
        bindings[logical] = item
        if logical.startswith("datasets/"):
            raw_sources.append(
                {
                    "logical_name": logical,
                    "path": str(source),
                    "bytes": expected["bytes"],
                    "sha256": expected["sha256"],
                    "identity": asdict(identity),
                }
            )
    split = project / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PACKET / "fold0.json", split)
    portable = project / "research/campaign_2026-10-04/portable_inputs.json"
    portable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PACKET / "portable_inputs.json", portable)
    identity_path = project / "input_identity_receipt.json"
    runtime.write_json(
        identity_path,
        {
            "schema": "wnn-capacity-r5-input-identity-v1",
            "raw_sources": raw_sources,
            "all_sources": bindings,
            "split_manifest_sha256": runtime.sha256_file(split),
        },
    )
    binding_path = output / "shared_input_binding.json"
    runtime.write_json(
        binding_path,
        {
            "schema": "wnn-gru-capacity-shared-input-binding-v1",
            "input_identity_receipt_sha256": runtime.sha256_file(identity_path),
            "sources": bindings,
            "reuse_contract": {
                "project_inputs_are_read_only_symlinks": True,
                "cohort_identity_verification_count": 1,
                "parsed_loader_shared": False,
            },
        },
    )
    os.environ["WNN_INPUT_BINDING_PATH"] = str(binding_path)
    os.environ["WNN_INPUT_BINDING_SHA256"] = runtime.sha256_file(binding_path)
    runtime.validate_shared_input_binding(project, binding_path, runtime.sha256_file(binding_path))
    return project, source_hashes, identity_path


def finalize(output: Path, result: dict[str, object]) -> None:
    """Preserve partial per-arm outputs and exclude training states/data from delivery."""
    runtime.write_json(output / "result.json", result)
    members = []
    for path in output.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(output)
        if path.name in {"artifact_bundle.tar", "worker.log"}:
            continue
        if (
            relative.parts[0] != "project"
            or path.name == "input_identity_receipt.json"
            or "artifacts/runs" in str(relative)
            and path.name
            in {
                "model_ep3.pt",
                "provenance.json",
                "history.json",
            }
            or (
                relative.parts[1] == "cache"
                and path.name.startswith("pred_")
                and path.suffix in {".npy", ".json"}
            )
        ):
            members.append((path, str(relative)))
    total = sum(path.stat().st_size for path, _ in members)
    if total > 5_000_000_000:
        raise ValueError("r5 artifact closure exceeds five GB")
    with tarfile.open(output / "artifact_bundle.tar", "w") as archive:
        for path, name in sorted(members, key=lambda item: item[1]):
            archive.add(path, arcname=name, recursive=False)


def export_teacher_arrays(
    arm: dict[str, object],
    common: dict[str, object],
    output: Path,
    checkpoint_sha256: str,
) -> dict[str, object]:
    """Export complete feature-only teacher arrays without another model fit."""
    import numpy as np
    import pyarrow.parquet as pq

    project = common["project_root"]
    exports = {}
    for split in ("train", "valid"):
        output_name = arm["run_name"] + "_teacher"
        argv = [
            arm["run_name"],
            split,
            "--epoch",
            "3",
            "--output-name",
            output_name,
            "--inference-precision",
            "fp32",
        ]
        child = runtime._run_child(
            arm=arm["id"],
            phase="predict",
            run_name=arm["run_name"],
            argv=argv,
            receipt_root=output / "receipts" / arm["id"] / ("teacher_" + split),
            **{key: value for key, value in common.items() if key not in {"batch", "split_path"}},
        )
        if child["status"] != "complete" or child["exit_code"] != 0:
            raise RuntimeError("full teacher export failed")
        path = project / "cache" / f"pred_{output_name}_{split}.npy"
        sidecar_path = path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        expected_groups = pq.ParquetFile(
            project / "datasets" / f"{split}.parquet"
        ).metadata.num_row_groups
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (expected_groups, 20_000, 2) or array.dtype != np.float32:
            raise ValueError("full teacher array shape/dtype mismatch")
        for start in range(0, expected_groups, 8):
            if not np.isfinite(array[start : start + 8]).all():
                raise ValueError("full teacher array contains nonfinite values")
        if (
            sidecar["row_groups"] != list(range(expected_groups))
            or sidecar["checkpoint_sha256"] != checkpoint_sha256
            or sidecar["run"] != arm["run_name"]
            or sidecar["epoch"] != 3
            or sidecar["inference_precision"] != "fp32"
        ):
            raise ValueError("full teacher export sidecar binding mismatch")
        exports[split] = {
            "subprocess": child,
            "prediction_path": str(path.relative_to(output)),
            "prediction_bytes": path.stat().st_size,
            "prediction_sha256": runtime.sha256_file(path),
            "sidecar_sha256": runtime.sha256_file(sidecar_path),
            "row_group_count": expected_groups,
            "label_mask_reads": False,
            "ancestry": "own seed9 cleanfold0 fit; complete feature-only rows do not change fitting membership",
        }
    return exports


def main() -> int:
    """Run fresh sequential Muon512/768 profiles, fits and fixed-fold exports."""
    output = Path(os.environ["WNN_OUTPUT_DIR"]).resolve(strict=True)
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "attempt_id": os.environ["WNN_ATTEMPT_ID"],
        "packet_sha256": os.environ["WNN_PACKET_SHA256"],
        "status": "failed",
        "arm_results": {},
        "profiles": {},
        "scientific_claim": False,
    }
    try:
        specification = verify_authority()
        os.environ.update(
            {"OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "ARROW_NUM_THREADS": "1"}
        )
        import psutil

        host_memory = psutil.virtual_memory()
        result["host_memory_before_worker"] = {
            "total_bytes": host_memory.total,
            "available_bytes": host_memory.available,
        }
        if host_memory.available < 35_000_000_000:
            raise MemoryError("r5 host lacks 35GB available headroom")
        deadline = started + min(int(os.environ["WNN_MAX_RUNTIME_SECONDS"]), 35_400)
        hardware = runtime._gpu_hardware()
        result["hardware"] = hardware
        if not any(model in hardware["name"] for model in ("3090", "4090")):
            raise ValueError("r5 hardware family differs from allocation")
        import torch

        image = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "hardware": hardware,
            "status": "failed",
        }
        if torch.__version__ == "2.14.0+cu130" and torch.version.cuda == "13.0":
            image["status"] = "passed"
        runtime.write_json(output / "image_preflight.json", image)
        if image["status"] != "passed":
            raise RuntimeError("r5 pinned image software gate failed")
        project, hashes, identity = prepare_project(output, specification)
        split_path = project / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        split = json.loads(split_path.read_text())
        common = {
            "batch": 8,
            "split_path": split_path,
            "packet_root": PACKET,
            "project_root": project,
            "input_identity": identity,
            "worker_started": started,
            "worker_deadline": deadline,
        }
        for hidden in (512, 768):
            arm_id = f"muon{hidden}"
            arm = {
                "id": arm_id,
                "hidden": hidden,
                "optimizer": "muon",
                "primary": True,
                "run_name": f"r5_capacity_{arm_id}_seed9_f0",
            }
            profile_arm = dict(arm, run_name=arm["run_name"] + "_profile")
            argv = runtime._profile_argv(profile_arm, 8, split_path)
            argv[argv.index("--profile-warmup") + 1] = "1"
            argv[argv.index("--profile-steps") + 1] = "2"
            smoke = runtime._run_child(
                arm=arm_id,
                phase="profile",
                run_name=profile_arm["run_name"],
                argv=argv,
                receipt_root=output / "receipts" / arm_id / "profile",
                **{
                    key: value
                    for key, value in common.items()
                    if key not in {"batch", "split_path"}
                },
            )
            if smoke["status"] != "complete" or smoke["exit_code"] != 0:
                raise RuntimeError("r5 actual-device smoke failed")
            evidence = smoke["markers"]["profile_update_evidence"]
            if any(
                evidence.get(key) != expected
                for key, expected in {
                    "optimizer_attempts": 30,
                    "optimizer_completed_updates": 30,
                    "amp_skipped_updates": 0,
                    "loss_checks": 30,
                    "loss_finite": True,
                    "gradient_finite": True,
                    "valid_for_throughput": True,
                }.items()
            ):
                raise ValueError("r5 actual-device smoke numeric gate failed")
            profile = {
                "step_wall_ms": smoke["markers"]["profile"]["step_wall"],
                "optimizer_precision": smoke["markers"]["optimizer_precision"],
                "gpu_samples": smoke["gpu"],
                "subprocess": smoke,
                "qualification": "one warmup/two timed batches on actual device; successful V4 230-update T4 profiles retained separately",
            }
            runtime._validate_precision_policy(arm, profile, hardware)
            result["profiles"][arm_id] = profile
            projection = runtime._profile_projection_seconds(profile, 8)
            if (
                not math.isfinite(projection)
                or projection <= 0
                or not profile["gpu_samples"].get("gpu_peak_used_mb")
                or profile["gpu_samples"].get("gpu_min_free_mb", 0) < 1024
            ):
                raise ValueError("r5 actual-device timing/memory evidence invalid")
            export_budget = 5400 - float(result.get("teacher_export_wall_seconds", 0))
            if export_budget <= 0:
                raise TimeoutError("r5 aggregate teacher export budget exhausted")
            if projection + 900 + export_budget > deadline - time.monotonic():
                raise TimeoutError("r5 actual-device projection exceeds remaining hard stop")
            runtime.write_json(output / "result.json", result)
            fit = runtime._fit_arm(
                arm=arm,
                receipt_root=output / "receipts" / arm_id / "fit",
                split_manifest=split,
                source_hashes=hashes,
                evaluation_groups=split["evaluation"]["valid"],
                **common,
            )
            result["arm_results"][arm_id] = fit
            runtime.write_json(output / "result.json", result)
            if fit["status"] != "complete":
                raise RuntimeError("r5 arm fit/export failed; preserve partial delivery")
            export_started = time.monotonic()
            export_common = dict(common)
            export_common["worker_deadline"] = min(
                deadline, export_started + export_budget + runtime.FINALIZATION_RESERVE_SECONDS
            )
            try:
                fit["teacher_exports"] = export_teacher_arrays(
                    arm, export_common, output, fit["fit_validation"]["checkpoint_sha256"]
                )
            finally:
                result["teacher_export_wall_seconds"] = (
                    float(result.get("teacher_export_wall_seconds", 0))
                    + time.monotonic()
                    - export_started
                )
            runtime.write_json(output / "result.json", result)
        result["paired_fit_consistency"] = runtime._validate_pair_consistency(result["arm_results"])
        if result["paired_fit_consistency"].get("status") != "pass":
            raise ValueError("r5 paired fit consistency failed")
        result["status"] = "delivered_capacity_pair"
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    finally:
        result["wall_seconds"] = time.monotonic() - started
        finalize(output, result)
    return 0 if result["status"] == "delivered_capacity_pair" else 1


if __name__ == "__main__":
    raise SystemExit(main())
