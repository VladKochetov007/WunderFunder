from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

OPS_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011"
)
R5_RELATIVE = Path("clean_t3_fold0_h390_r5_teacher_cohort/package_r9")
R6_RELATIVE = Path(
    "clean_t3_fold0_h390_r5_teacher_cohort/r6_bootstrap_guard_successor_20261012_r1/"
    "terminal_collection_v2"
)
TARGET_KERNEL_ID = "vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r7"
TARGET_TITLE = "wnn-clean-t3-f0-h384-exp-20261012-r7"
R6_KERNEL_ID = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner"
R6_VERSION = 2
EXPECTED_NPY_BYTES = 5_731_200_768
S3_LOCAL_HANDOFF_SHA256 = "0eabad07a2bea44f0d4691575ccc7105526c39049c56dbf32ba948d7efb717c8"
S45_LOCAL_HANDOFF_SHA256 = "6f11b22d113cb363351a5dd3dbe6da29271a350bfe17bc6cd39d6def36e4b195"
EXPECTED_LOCAL_FULL_VALID_NPY_BYTES = 299_680_128
MAX_KERNEL_WALL_SECONDS = 16_800
FINALIZATION_RESERVE_SECONDS = 300
WORKER_TIMEOUT_SECONDS = 1_800
WORKER_RAM_BYTES = 8_000_000_000
WORKER_GPU_MEMORY_BYTES = 5_000_000_000
WORKER_GPU_STOP_BYTES = 6_000_000_000
WORKER_CPUS = 2
MAX_CONCURRENT_WORKERS = 1
FREE_DISK_RESERVE_BYTES = 4_294_967_296
COLLECTOR_CAP_BYTES = 8_589_934_592
MAX_SINGLE_NPY_COPY_BYTES = 2_000_000_000
NON_ARRAY_OUTPUT_RESERVE_BYTES = 268_435_456


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def checked_sha(path: Path, expected: str, expected_size: int | None = None) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"size mismatch for {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def build_local_full_valid_counterparts(
    main_root: Path,
    ops: Path,
    checkpoints: dict[str, Any],
    spec_catalog: dict[str, Any],
    r5_preseal: dict[str, Any],
) -> dict[str, Any]:
    s3_path = ops / "clean_t3_local_s3_terminal_r1" / "LOCAL_TERMINAL_HANDOFF_R1.json"
    s45_path = ops / "clean_t3_local_s4_s5_terminal_r1" / "LOCAL_EXPORT_TERMINAL_HANDOFF_R1.json"
    checked_sha(s3_path, S3_LOCAL_HANDOFF_SHA256)
    checked_sha(s45_path, S45_LOCAL_HANDOFF_SHA256)
    s3 = json.loads(s3_path.read_text(encoding="utf-8"))
    s45 = json.loads(s45_path.read_text(encoding="utf-8"))
    if (
        s3.get("schema") != "wnn.clean_t3_local_full_valid_export_terminal_handoff.r1"
        or s3.get("status") != "terminal_export_validated_grade_pending_root"
        or s3.get("attempt", {}).get("exit_status") != 0
        or s45.get("schema") != "wnn.clean_t3_local_f0_s4_s5_export_terminal_handoff.r1"
        or s45.get("status") != "PASS_S4_S5_TERMINAL_FULL_VALID_EXPORTS"
    ):
        raise ValueError("local full-valid producer receipts are not terminal successful handoffs")

    source_root = ops / "clean_t3_fold0_h390_r3_teacher_cohort" / "source_snapshot"
    source_manifest_path = source_root / "SOURCE_MANIFEST_H390.json"
    predictor_path = source_root / "scripts" / "predict_run.py"
    trainer_path = source_root / "scripts" / "train_rnn.py"
    source_manifest_sha = sha256_file(source_manifest_path)
    predictor_sha = sha256_file(predictor_path)
    trainer_sha = sha256_file(trainer_path)
    if source_manifest_sha != r5_preseal["source_manifest_sha256"]:
        raise ValueError("local export source manifest differs from R5's frozen source binding")
    if s3["source_and_model"].get("predictor_sha256") != predictor_sha:
        raise ValueError("s3 local full-valid receipt does not bind the frozen predictor")
    if s3["source_and_model"].get("trainer_sha256_from_root_actual_consumer_review") != trainer_sha:
        raise ValueError("s3 local full-valid receipt does not bind the frozen trainer")
    if (
        s45["source_bindings"].get("predictor_sha256") != predictor_sha
        or s45["source_bindings"].get("trainer_sha256") != trainer_sha
    ):
        raise ValueError(
            "s4/s5 local full-valid receipt does not bind the frozen predictor/trainer"
        )
    if (
        s45["source_bindings"].get("split_sha256")
        != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
    ):
        raise ValueError("s4/s5 local full-valid receipts use a different split manifest")

    outputs: dict[int, dict[str, Any]] = {}
    s3_source = s3["source_and_model"]
    s3_output = s3["output_validation"]
    outputs[3] = {
        "run": s3_source["run"],
        "seed": s3_source["seed"],
        "selected_epoch": s3_source["selected_epoch"],
        "checkpoint_sha256": s3_source["checkpoint_sha256"],
        "provenance_sha256": s3_source["provenance_sha256"],
        "prediction_path": s3_output["npy_path"],
        "prediction_bytes": s3_output["npy_bytes"],
        "prediction_sha256": s3_output["npy_sha256"],
        "dtype": s3_output["dtype"],
        "shape": s3_output["shape"],
        "sidecar_path": s3_output["sidecar_path"],
        "sidecar_bytes": s3_output["sidecar_bytes"],
        "sidecar_sha256": s3_output["sidecar_sha256"],
        "sidecar_epoch": s3_output["sidecar_epoch"],
        "sidecar_checkpoint_sha256": s3_output["sidecar_checkpoint_sha256"],
        "sidecar_split_manifest_sha256": s3_output["sidecar_split_manifest_sha256"],
        "sidecar_inference_precision": s3_output["sidecar_inference_precision"],
        "sidecar_campaign_fit_scope": s3_output["sidecar_campaign_fit_scope"],
        "all_values_finite": s3_output["all_values_finite"],
        "rowgroups_exact_0_to_1872": s3_output["sidecar_row_groups"]["exactly_full_integer_range"],
        "partial_absent": s3_output["partial_file_absent"],
        "local_handoff_receipt_path": str(s3_path.relative_to(main_root)),
        "local_handoff_receipt_sha256": S3_LOCAL_HANDOFF_SHA256,
        "local_handoff_status": s3["status"],
    }
    for output in s45.get("outputs", []):
        seed = output.get("seed")
        if seed in {4, 5}:
            outputs[seed] = {
                "run": output["run"],
                "seed": seed,
                "selected_epoch": output["sidecar_epoch"],
                "checkpoint_sha256": output["checkpoint_sha256"],
                "provenance_sha256": output["provenance_sha256"],
                "prediction_path": output["prediction_path"],
                "prediction_bytes": output["prediction_bytes"],
                "prediction_sha256": output["prediction_sha256"],
                "dtype": output["dtype"],
                "shape": output["shape"],
                "sidecar_path": output["sidecar_path"],
                "sidecar_bytes": output["sidecar_bytes"],
                "sidecar_sha256": output["sidecar_sha256"],
                "sidecar_epoch": output["sidecar_epoch"],
                "sidecar_checkpoint_sha256": output["checkpoint_sha256"],
                "sidecar_split_manifest_sha256": output["sidecar_split_manifest_sha256"],
                "sidecar_inference_precision": output["sidecar_inference_precision"],
                "sidecar_campaign_fit_scope": output["sidecar_campaign_fit_scope"],
                "all_values_finite": output["all_values_finite"],
                "rowgroups_exact_0_to_1872": output["full_valid_rowgroups_exact_0_to_1872"],
                "partial_absent": output["partial_absent"],
                "local_handoff_receipt_path": str(s45_path.relative_to(main_root)),
                "local_handoff_receipt_sha256": S45_LOCAL_HANDOFF_SHA256,
                "local_handoff_status": s45["status"],
            }
    if set(outputs) != {3, 4, 5}:
        raise ValueError("local full-valid handoff receipts do not cover seeds 3, 4, and 5")

    counterparts = {}
    for seed, output in sorted(outputs.items()):
        run_name = f"clean_t3_f0_muonall_h384_e3_s{seed}_r5"
        checkpoint = checkpoints[run_name]
        if (
            output["run"] != run_name
            or output["seed"] != seed
            or output["selected_epoch"] != 2
            or output["checkpoint_sha256"] != checkpoint["model_ep2_sha256"]
            or output["provenance_sha256"] != checkpoint["provenance_sha256"]
            or output["prediction_bytes"] != EXPECTED_LOCAL_FULL_VALID_NPY_BYTES
            or output["dtype"] != "float32"
            or output["shape"] != [1873, 20_000, 2]
            or output["sidecar_epoch"] != 2
            or output["sidecar_split_manifest_sha256"] != checkpoint["split_manifest_sha256"]
            or output["sidecar_inference_precision"] != "fp32"
            or output["sidecar_campaign_fit_scope"] != "full-valid-feature-only"
            or output["all_values_finite"] is not True
            or output["rowgroups_exact_0_to_1872"] is not True
            or output["partial_absent"] is not True
        ):
            raise ValueError(
                f"local full-valid counterpart is not bound to the exact R5 source: s{seed}"
            )
        npy_path = output["prediction_path"]
        sidecar_path = output["sidecar_path"]
        for relative in (npy_path, sidecar_path):
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"unsafe local full-valid artifact path: {relative}")
        npy_file = main_root / npy_path
        sidecar_file = main_root / sidecar_path
        checked_sha(npy_file, output["prediction_sha256"], output["prediction_bytes"])
        checked_sha(sidecar_file, output["sidecar_sha256"], output["sidecar_bytes"])
        if npy_file.with_name(npy_file.name + ".partial").exists():
            raise FileExistsError(f"local full-valid prediction has a partial sibling: {npy_path}")
        sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
        if (
            sidecar.get("run") != run_name
            or sidecar.get("split") != "valid"
            or sidecar.get("epoch") != 2
            or sidecar.get("checkpoint_sha256") != checkpoint["model_ep2_sha256"]
            or sidecar.get("prediction_sha256") != output["prediction_sha256"]
            or sidecar.get("campaign_split_manifest_sha256") != checkpoint["split_manifest_sha256"]
            or sidecar.get("inference_precision") != "fp32"
            or sidecar.get("campaign_fit_scope") != "full-valid-feature-only"
            or sidecar.get("row_groups") != list(range(1873))
        ):
            raise ValueError(f"local full-valid sidecar differs from its receipt: s{seed}")
        spec_name = f"export_clean_t3_f0_muonall_h384_e3_s{seed}_r5_full_valid"
        counterparts[spec_name] = {
            **output,
            "local_npy_path": npy_path,
            "local_sidecar_path": sidecar_path,
            "logical_equivalence": "same frozen R5 export spec and exact model_ep2/predictor/split/FP32/full-valid scope; produced by the reviewed local consumer, not byte-identical to an unrun Kaggle output",
            "r5_full_valid_spec_sha256": spec_catalog[spec_name]["spec_sha256"],
            "r5_full_valid_spec_path": spec_catalog[spec_name]["spec_path"],
            "predictor_sha256": predictor_sha,
            "trainer_sha256": trainer_sha,
            "source_manifest_sha256": source_manifest_sha,
            "split_manifest_sha256": checkpoint["split_manifest_sha256"],
        }
    return counterparts


def build(main_root: Path, packet: Path) -> dict[str, Any]:
    main_root = main_root.resolve(strict=True)
    packet = packet.resolve(strict=True)
    existing_release_path = packet / "kernel" / "ROOT_EXPORT_RELEASE_R7.json"
    if existing_release_path.exists():
        existing_release = json.loads(existing_release_path.read_text(encoding="utf-8"))
        if (
            existing_release.get("status") != "pending"
            or existing_release.get("authorized") is not False
        ):
            raise PermissionError("refusing to reseal an R7 packet after root authorization")
    ops = main_root / OPS_RELATIVE
    r5 = ops / R5_RELATIVE
    r6 = ops / R6_RELATIVE
    r5_kernel = r5 / "bundle_teacher_cohort_r5"
    r5_archive = r5_kernel / "bundle.wnn"
    r5_manifest_path = r5_kernel / "bundle-manifest.json"
    r5_preseal_path = r5 / "R5_PRESEAL_BINDINGS_R1.json"
    r6_operator_path = r6 / "R6_V2_TERMINAL_OPERATOR_RECEIPT_R1.json"
    r6_stage_path = r6 / "ROOT_ACTUAL_COLLECTION_AND_LOCAL_STAGE_R1.json"
    r6_collector_path = r6 / "COMPLETE_COLLECTOR_RECEIPT_R1.json"

    r5_manifest = json.loads(r5_manifest_path.read_text(encoding="utf-8"))
    r5_preseal = json.loads(r5_preseal_path.read_text(encoding="utf-8"))
    r6_operator = json.loads(r6_operator_path.read_text(encoding="utf-8"))
    r6_stage = json.loads(r6_stage_path.read_text(encoding="utf-8"))
    r6_collector = json.loads(r6_collector_path.read_text(encoding="utf-8"))
    if r5_manifest.get("private") is not True:
        raise ValueError("R5 bundle manifest is not private")
    if (
        sha256_file(r5_archive)
        != "88641b3c9fc629a7a2ce8b11cfd98ecf8b53fe943ffb2d7f632e1dfb997fa549"
    ):
        raise ValueError("R5 source archive no longer matches its R9 seal")
    if r6_operator.get("schema") != "wnn.kaggle.r6-v2-terminal-operator-receipt.r1":
        raise ValueError("unexpected R6 terminal operator receipt")
    if r6_operator.get("authority", {}).get("kernel") != R6_KERNEL_ID:
        raise ValueError("R6 terminal receipt source kernel mismatch")
    if r6_operator.get("authority", {}).get("version") != R6_VERSION:
        raise ValueError("R6 terminal receipt version mismatch")
    if r6_operator.get("collection", {}).get("complete") is not True:
        raise ValueError("R6 complete collector gate is not satisfied")
    if (
        r6_stage.get("decision")
        != "PASS_COMPLETE_COLLECTION_AND_THREE_FIT_ARTIFACTS_FOR_EXPORT_ONLY_RECOVERY"
    ):
        raise ValueError("R6 root local-stage receipt is not an export-only recovery pass")
    if r6_stage.get("provider_action") is not False or r6_stage.get("new_spend_usd") != 0:
        raise ValueError("R6 collection receipt does not preserve the no-provider-action boundary")
    if r6_collector.get("terminal_collection", {}).get("collection_complete") is not True:
        raise ValueError("R6 provider output collection is incomplete")
    if r6_collector.get("terminal_status") != "KernelWorkerStatus.ERROR":
        raise ValueError("R6 historical terminal failure status changed")
    pipeline = r6_operator.get("pipeline", {})
    if (
        pipeline.get("teacher_fits", {}).get("status") != "complete"
        or pipeline.get("teacher_fits", {}).get("returncode") != 0
        or pipeline.get("exports", {}).get("workers_started") != 0
        or pipeline.get("exports", {}).get("prediction_npy_and_sidecar_files_present") != 0
    ):
        raise ValueError("R6 must be a fit-complete, export-not-started source")

    r5_binding_sha = sha256_file(r5_preseal_path)
    r5_manifest_sha = sha256_file(r5_manifest_path)
    r5_archive_sha = sha256_file(r5_archive)
    r5_driver_path = (
        "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        "ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/"
        "kernel_support/teacher_cohort_r5_driver.py"
    )
    # R5's original driver is outside the inference bundle; verify its sealed byte pin
    # before referring only to the byte-identical copy inside the R5 package archive.
    r5_driver_local = r5 / "kernel_support" / "teacher_cohort_r5_driver.py"
    r5_driver_sha = r5_preseal["r5_driver_sha256"]
    checked_sha(r5_driver_local, r5_driver_sha)
    spec_sha_map = r5_preseal["worker_specs_sha256"]
    spec_catalog = {}
    r5_declared_export_names = [
        f"export_clean_t3_f0_muonall_h384_e3_s{seed}_r5_{kind}"
        for seed in (3, 4, 5)
        for kind in ("fit_train", "fit_valid", "full_valid")
    ]
    export_names = [
        f"export_clean_t3_f0_muonall_h384_e3_s{seed}_r5_{kind}"
        for seed in (3, 4, 5)
        for kind in ("fit_train", "fit_valid")
    ]
    split_path = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split_mapping = r5_manifest["external_mounts"][split_path]
    split_sha = split_mapping["filename"].removesuffix(".wnn")
    for name in r5_declared_export_names:
        spec_path = f"kaggle_specs/{name}.json"
        expected_sha = spec_sha_map[name]
        archive_member = next(
            (row for row in r5_manifest["files"] if row["path"] == spec_path), None
        )
        if archive_member is None or spec_path not in r5_manifest["spec_paths"]:
            raise ValueError(f"R5 bundle manifest omits the declared spec: {spec_path}")
        if archive_member["sha256"] != expected_sha:
            raise ValueError(f"R5 spec digest differs between preseal and bundle manifest: {name}")
        spec_info = {
            "spec_path": spec_path,
            "spec_sha256": expected_sha,
            "size_bytes": archive_member["size"],
        }
        seed = int(name.split("_s")[-1].split("_r5_")[0])
        run_name = f"clean_t3_f0_muonall_h384_e3_s{seed}_r5"
        kind = name.rsplit("_r5_", 1)[1]
        split = "train" if kind == "fit_train" else "valid"
        campaign_scope = "full-valid-feature-only" if kind == "full_valid" else split
        artifacts = [
            f"cache/pred_{run_name}_{kind}_{split}.npy",
            f"cache/pred_{run_name}_{kind}_{split}.json",
        ]
        spec_catalog[name] = {
            **spec_info,
            "run_name": run_name,
            "seed": seed,
            "split": split,
            "campaign_fit_scope": campaign_scope,
            "campaign_split_manifest_sha256": split_sha,
            "checkpoint_sha256": None,
            "artifacts": artifacts,
            "kind": kind,
            "limits": {
                "cpus": WORKER_CPUS,
                "ram_bytes": WORKER_RAM_BYTES,
                "gpu_memory_bytes": WORKER_GPU_MEMORY_BYTES,
                "gpu_stop_bytes": WORKER_GPU_STOP_BYTES,
                "timeout_seconds": WORKER_TIMEOUT_SECONDS,
            },
        }
    exports = {name: spec_catalog[name] for name in export_names}

    checkpoints = {}
    provider_root = r6 / "provider_outputs" / "attempt_001" / "outputs"
    local_runs_root = main_root / "artifacts" / "runs"
    stage_artifacts = {row["path"]: row for row in r6_stage.get("local_staged_artifacts", [])}
    collected = r6_collector["terminal_collection"]["files"]
    for seed in (3, 4, 5):
        run_name = f"clean_t3_f0_muonall_h384_e3_s{seed}_r5"
        cp_rel = f"artifacts/runs/{run_name}/model_ep2.pt"
        prov_rel = f"artifacts/runs/{run_name}/provenance.json"
        cp_stage = stage_artifacts.get(cp_rel)
        prov_stage = stage_artifacts.get(prov_rel)
        if cp_stage is None or prov_stage is None:
            raise ValueError(
                f"R6 root stage receipt omits the exact selected checkpoint pair: {run_name}"
            )
        if not (local_runs_root / run_name / "model_ep2.pt").is_file():
            raise FileNotFoundError(f"staged R6 checkpoint missing: {run_name}")
        if not (local_runs_root / run_name / "provenance.json").is_file():
            raise FileNotFoundError(f"staged R6 provenance missing: {run_name}")
        cp_remote_key = f"attempt_001/outputs/experiment/{cp_rel}"
        prov_remote_key = f"attempt_001/outputs/experiment/{prov_rel}"
        cp_remote = collected.get(cp_remote_key)
        prov_remote = collected.get(prov_remote_key)
        if cp_remote is None or prov_remote is None:
            raise ValueError(
                f"R6 complete collector receipt omits canonical source files: {run_name}"
            )
        if (
            cp_remote["sha256"] != cp_stage["sha256"]
            or cp_remote["size_bytes"] != cp_stage["bytes"]
        ):
            raise ValueError(f"R6 checkpoint collector/root-stage identities disagree: {run_name}")
        if (
            prov_remote["sha256"] != prov_stage["sha256"]
            or prov_remote["size_bytes"] != prov_stage["bytes"]
        ):
            raise ValueError(f"R6 provenance collector/root-stage identities disagree: {run_name}")
        provider_cp = provider_root / "experiment" / cp_rel
        provider_prov = provider_root / "experiment" / prov_rel
        checked_sha(provider_cp, cp_stage["sha256"], cp_stage["bytes"])
        checked_sha(provider_prov, prov_stage["sha256"], prov_stage["bytes"])
        local_cp = local_runs_root / run_name / "model_ep2.pt"
        local_prov = local_runs_root / run_name / "provenance.json"
        checked_sha(local_cp, cp_stage["sha256"], cp_stage["bytes"])
        checked_sha(local_prov, prov_stage["sha256"], prov_stage["bytes"])
        provenance = json.loads(local_prov.read_text(encoding="utf-8"))
        if (
            provenance.get("selected_epoch") != 2
            or provenance.get("args", {}).get("name") != run_name
            or provenance.get("args", {}).get("seed") != seed
            or provenance.get("args", {}).get("hidden") != 384
            or provenance.get("args", {}).get("layers") != 2
            or provenance.get("args", {}).get("epochs") != 3
            or provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") != cp_stage["sha256"]
        ):
            raise ValueError(f"R6 staged provenance does not bind model_ep2/recipe: {run_name}")
        source_hash = provenance.get("run_identity_source_sha256")
        split_hash = provenance.get("split_manifest_sha256")
        checkpoints[run_name] = {
            "seed": seed,
            "selected_epoch": 2,
            "model_ep2_path": cp_rel,
            "model_ep2_bytes": cp_stage["bytes"],
            "model_ep2_sha256": cp_stage["sha256"],
            "provenance_path": prov_rel,
            "provenance_bytes": prov_stage["bytes"],
            "provenance_sha256": prov_stage["sha256"],
            "run_identity_source_sha256": source_hash,
            "split_manifest_sha256": split_hash,
        }
        for _name, entry in spec_catalog.items():
            if entry["run_name"] == run_name:
                entry["checkpoint_sha256"] = cp_stage["sha256"]

    # Exact spec bytes are listed in both the bundle `files` table and `spec_paths`.
    # Confirm the archived bytes agree with the same sealed size and digest.
    import tarfile

    with tarfile.open(r5_archive, mode="r:gz") as archive:
        for name, entry in spec_catalog.items():
            member = archive.getmember(entry["spec_path"])
            if member.size != entry["size_bytes"]:
                raise ValueError(f"R5 export spec size differs from bundle manifest: {name}")
            raw = archive.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != entry["spec_sha256"]:
                raise ValueError(f"R5 export spec digest differs from preseal: {name}")

    local_full_valid = build_local_full_valid_counterparts(
        main_root,
        ops,
        checkpoints,
        spec_catalog,
        r5_preseal,
    )
    s3_handoff_path = ops / "clean_t3_local_s3_terminal_r1" / "LOCAL_TERMINAL_HANDOFF_R1.json"
    s3_handoff = json.loads(s3_handoff_path.read_text(encoding="utf-8"))
    fit_membership = s3_handoff["source_and_model"]["source_qualified_fit_and_eval_membership"]
    for _name, entry in exports.items():
        source_prefix = "train" if entry["split"] == "train" else "valid"
        entry["expected_row_groups_count"] = fit_membership[f"{source_prefix}_fit_groups"]
        entry["expected_ordered_row_groups_sha256"] = fit_membership[
            f"{source_prefix}_fit_ordered_int64_le_sha256"
        ]
        entry["expected_row_groups_source"] = (
            "source-qualified fit membership in local s3 terminal handoff"
        )

    bindings: dict[str, Any] = {
        "schema": "wnn.kaggle.clean_t3_f0_export_only_r7_bindings.r1",
        "target": {"kernel_id": TARGET_KERNEL_ID, "title": TARGET_TITLE},
        "r5_dataset_id": "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5",
        "r5_bundle_manifest_sha256": r5_manifest_sha,
        "r5_bundle_archive_sha256": r5_archive_sha,
        "r5_preseal_bindings_sha256": r5_binding_sha,
        "r5_source_manifest_sha256": r5_preseal["source_manifest_sha256"],
        "r5_driver_relative_path": r5_driver_path,
        "r5_driver_sha256": r5_driver_sha,
        "r6_kernel_id": R6_KERNEL_ID,
        "r6_kernel_version": R6_VERSION,
        "r6_operator_receipt_sha256": sha256_file(r6_operator_path),
        "r6_root_stage_receipt_sha256": sha256_file(r6_stage_path),
        "r6_complete_collector_receipt_sha256": sha256_file(r6_collector_path),
        "r6_terminal_status": r6_collector["terminal_status"],
        "r6_exports_started": False,
        "r6_provider_outputs_root_suffix": "experiment/artifacts/runs",
        "r5_external_mounts": r5_manifest["external_mounts"],
        "checkpoints": checkpoints,
        "r5_declared_export_names": r5_declared_export_names,
        "r5_export_spec_catalog": spec_catalog,
        "export_names": export_names,
        "export_specs": exports,
        "fit_membership": {
            "source_handoff_path": str(s3_handoff_path.relative_to(main_root)),
            "source_handoff_sha256": S3_LOCAL_HANDOFF_SHA256,
            "train_fit_groups": fit_membership["train_fit_groups"],
            "train_fit_ordered_int64_le_sha256": fit_membership[
                "train_fit_ordered_int64_le_sha256"
            ],
            "valid_fit_groups": fit_membership["valid_fit_groups"],
            "valid_fit_ordered_int64_le_sha256": fit_membership[
                "valid_fit_ordered_int64_le_sha256"
            ],
        },
        "local_full_valid_counterparts": local_full_valid,
        "expected_total_npy_bytes": EXPECTED_NPY_BYTES,
        "array_schema": {
            "dtype": "float32",
            "sequence_length": 20_000,
            "columns": 2,
            "layout": "C",
        },
        "worker_limits": {
            "cpus": WORKER_CPUS,
            "ram_bytes": WORKER_RAM_BYTES,
            "gpu_memory_bytes": WORKER_GPU_MEMORY_BYTES,
            "gpu_stop_bytes": WORKER_GPU_STOP_BYTES,
            "timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "max_concurrent_workers": MAX_CONCURRENT_WORKERS,
            "fresh_host_gate_each_worker": True,
        },
        "runtime_limits": {
            "kernel_wall_ceiling_seconds": MAX_KERNEL_WALL_SECONDS,
            "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "worst_case_serial_worker_seconds": len(export_names) * WORKER_TIMEOUT_SECONDS,
            "maximum_single_npy_copy_bytes": MAX_SINGLE_NPY_COPY_BYTES,
        },
        "disk_and_collection": {
            "expected_prediction_npy_bytes": EXPECTED_NPY_BYTES,
            "minimum_free_disk_bytes": FREE_DISK_RESERVE_BYTES,
            "collector_cap_bytes": COLLECTOR_CAP_BYTES,
            "non_array_output_reserve_bytes": NON_ARRAY_OUTPUT_RESERVE_BYTES,
        },
        "execution_scope": {"fit_workers": 0, "profile_workers": 0, "exports_only": True},
    }
    root_binding = {
        "target_kernel_id": TARGET_KERNEL_ID,
        "r5_bundle_manifest_sha256": r5_manifest_sha,
        "r5_bundle_archive_sha256": r5_archive_sha,
        "r5_preseal_bindings_sha256": r5_binding_sha,
        "r6_kernel_id": R6_KERNEL_ID,
        "r6_kernel_version": R6_VERSION,
        "r6_operator_receipt_sha256": bindings["r6_operator_receipt_sha256"],
        "r6_root_stage_receipt_sha256": bindings["r6_root_stage_receipt_sha256"],
        "r6_complete_collector_receipt_sha256": bindings["r6_complete_collector_receipt_sha256"],
        "checkpoints_sha256": {name: row["model_ep2_sha256"] for name, row in checkpoints.items()},
        "provenance_sha256": {name: row["provenance_sha256"] for name, row in checkpoints.items()},
        "r5_export_specs_sha256": {name: row["spec_sha256"] for name, row in spec_catalog.items()},
        "export_specs_sha256": {name: row["spec_sha256"] for name, row in exports.items()},
        "fit_membership": bindings["fit_membership"],
        "local_full_valid_receipts_sha256": {
            name: row["local_handoff_receipt_sha256"] for name, row in local_full_valid.items()
        },
        "local_full_valid_npy_sha256": {
            name: row["prediction_sha256"] for name, row in local_full_valid.items()
        },
        "export_count": len(export_names),
        "r5_declared_export_count": len(r5_declared_export_names),
        "fit_workers": 0,
        "max_concurrent_export_workers": 1,
        "worker_limits": bindings["worker_limits"],
        "runtime_limits": bindings["runtime_limits"],
        "disk_and_collection": bindings["disk_and_collection"],
        "fresh_r6_version_readback_required": True,
        "no_provider_mutation_by_bootstrap": True,
    }
    bindings["root_release_binding"] = root_binding
    kernel_dir = packet / "kernel"
    bindings_path = kernel_dir / "R7_EXPORT_BINDINGS_R1.json"
    write_json(bindings_path, bindings)
    bindings_sha = sha256_file(bindings_path)
    kernel_path = kernel_dir / "kernel.py"
    source = kernel_path.read_text(encoding="utf-8")
    if 'EXPECTED_BINDINGS_SHA256 = "TO_BE_FILLED"' not in source:
        raise ValueError("kernel binding digest marker was already modified")
    kernel_path.write_text(
        source.replace(
            'EXPECTED_BINDINGS_SHA256 = "TO_BE_FILLED"',
            f'EXPECTED_BINDINGS_SHA256 = "{bindings_sha}"',
            1,
        ),
        encoding="utf-8",
    )
    metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [
            "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5",
            "vladkochetov07/wnn-c20261004-fold0-assets",
            "vladkochetov07/wunder-funder",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": TARGET_KERNEL_ID,
        "is_private": True,
        "kernel_sources": [R6_KERNEL_ID],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": TARGET_TITLE,
    }
    declared_datasets = (
        {metadata["r5_dataset_id"]}
        if "r5_dataset_id" in metadata
        else {"vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5"}
    )
    for mapping in bindings["r5_external_mounts"].values():
        if isinstance(mapping, dict):
            declared_datasets.add(mapping["dataset"])
        else:
            declared_datasets.add("vladkochetov07/wunder-funder")
    metadata["dataset_sources"] = sorted(declared_datasets)
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    root_release = {
        "schema": "wnn.root.clean_t3_f0_export_only_release.r7.r1",
        "status": "pending",
        "authorized": False,
        "binding": root_binding,
        "note": "Root must refresh quota, read back R6 kernel version 2, review the R7 source, then create a new admitted receipt; this pending template authorizes nothing.",
    }
    write_json(kernel_dir / "ROOT_EXPORT_RELEASE_R7.json", root_release)
    return {
        "bindings_sha256": bindings_sha,
        "kernel_sha256": sha256_file(kernel_path),
        "metadata_sha256": sha256_file(kernel_dir / "kernel-metadata.json"),
        "root_release_template_sha256": sha256_file(kernel_dir / "ROOT_EXPORT_RELEASE_R7.json"),
        "r5_archive_sha256": r5_archive_sha,
        "r5_manifest_sha256": r5_manifest_sha,
        "r6_operator_sha256": bindings["r6_operator_receipt_sha256"],
        "r6_root_stage_sha256": bindings["r6_root_stage_receipt_sha256"],
        "r6_collector_sha256": bindings["r6_complete_collector_receipt_sha256"],
        "export_count": len(exports),
        "checkpoint_count": len(checkpoints),
        "fit_workers": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    print(json.dumps(build(args.main_root, args.packet_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
