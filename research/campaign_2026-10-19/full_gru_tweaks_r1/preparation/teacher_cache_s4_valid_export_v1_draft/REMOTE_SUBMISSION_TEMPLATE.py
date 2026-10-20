"""Admit one S4 fit-valid prediction export after accepted fit-train exports."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import subprocess
import zlib
from pathlib import Path

PAYLOAD = json.loads(base64.b64decode("__PAYLOAD_B64__", validate=True))


def main() -> None:
    """Validate accepted S3 and S4 train exports, then launch S4 fit-valid."""
    assert PAYLOAD["schema"] == "fg19.teacher-cache-s4-valid-export-admission.v1"
    required_root_bindings = (
        "s4_train_export_acceptance_b64",
        "s4_train_export_acceptance_sha256",
        "s4_train_export_stop_b64",
        "s4_train_export_stop_sha256",
        "s4_train_export_invocation_id",
    )
    if any(
        PAYLOAD[name] == "__PENDING_ROOT_S4_TRAIN_ACCEPTANCE_AND_STOP_BINDING__"
        for name in required_root_bindings
    ):
        raise SystemExit(
            "Bind root-captured S4-train acceptance and exact-stop bytes before admission."
        )
    gate_raw = zlib.decompress(base64.b64decode(PAYLOAD["gates_zlib"], validate=True))
    assert hashlib.sha256(gate_raw).hexdigest() == PAYLOAD["gates_sha256"]
    gates = {"__name__": "reused_teacher_cache_profile_gates"}
    exec(compile(gate_raw, "reviewed_teacher_cache_profile_gates", "exec"), gates)
    gates["GPU"] = PAYLOAD["gpu"]
    gates["MIN_GPU_FREE"] = PAYLOAD["minimum_gpu_free_bytes"]

    b = gates["B"]
    source = Path(PAYLOAD["source_root"])
    prediction_dir = b / PAYLOAD["prediction_dir"]
    evidence_dir = b / PAYLOAD["evidence_dir"]
    scratch_root = b / "research_tmp"
    unit = PAYLOAD["unit"]
    metadata_root = b / "research/full_gru_tweaks_r1/teacher_cache_v1"
    assert b.is_dir() and not b.is_symlink()
    assert (b / "research").is_dir() and not (b / "research").is_symlink()
    assert metadata_root.parent.is_dir() and not metadata_root.parent.is_symlink()
    assert metadata_root.is_dir() and not metadata_root.is_symlink()
    assert prediction_dir.is_dir() and not prediction_dir.is_symlink()
    assert not os.path.lexists(evidence_dir)
    assert not os.path.lexists(b / "runtime/STOP_ADMISSION.json")
    assert gates["_unit"](unit, ["LoadState"])["LoadState"] == "not-found"
    assert source.parent == b and source.is_dir() and not source.is_symlink()
    assert scratch_root.is_dir() and not scratch_root.is_symlink()

    contract_raw = base64.b64decode(PAYLOAD["contract_b64"], validate=True)
    assert hashlib.sha256(contract_raw).hexdigest() == PAYLOAD["contract_sha256"]
    contract = json.loads(contract_raw)
    assert contract["schema"] == "fg19.teacher-cache-execution-contract.v1"
    assert contract["split_sha256"] == PAYLOAD["split_sha256"]
    assert [item["seed"] for item in contract["teachers"]] == [3, 4, 5]

    manifest_raw = base64.b64decode(PAYLOAD["source_manifest_b64"], validate=True)
    assert hashlib.sha256(manifest_raw).hexdigest() == PAYLOAD["source_manifest_sha256"]
    manifest = json.loads(manifest_raw)
    assert (
        manifest["source_commit"] == PAYLOAD["source_commit"]
        and manifest["source_root"] == PAYLOAD["source_root"]
        and len(manifest["files"]) == 167
    )
    expected_files = {
        item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for item in manifest["files"]
    }
    assert expected_files == PAYLOAD["source_files"]

    acceptance_bytes: dict[str, bytes] = {}
    for teacher in contract["teachers"]:
        relative = Path(teacher["acceptance_path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        raw = base64.b64decode(PAYLOAD["teacher_acceptances_b64"][teacher["run"]], validate=True)
        assert hashlib.sha256(raw).hexdigest() == teacher["acceptance_sha256"]
        accepted = json.loads(raw)
        assert (
            accepted["status"] == "ACCEPTED"
            and accepted["run_name"] == teacher["run"]
            and accepted["selected_epoch"] == teacher["epoch"] == 2
            and accepted["checkpoint_sha256"] == teacher["checkpoint_sha256"]
            and accepted["model_sha256"] == teacher["model_sha256"]
            and accepted["source_commit"] == contract["source_teacher_commit"]
        )
        acceptance_bytes[str(relative)] = raw

    teacher = contract["teachers"][0]
    target_teacher = contract["teachers"][1]
    for selected in (teacher, target_teacher):
        run_dir = b / "data/artifacts/runs" / selected["run"]
        checkpoint = run_dir / "model_ep2.pt"
        provenance = run_dir / "provenance.json"
        for path, expected in (
            (checkpoint, selected["checkpoint_sha256"]),
            (provenance, selected["provenance_sha256"]),
        ):
            assert path.is_file() and not path.is_symlink() and path.stat().st_size < 32_000_000
            assert gates["_sha"](path) == expected

    profile_acceptance_raw = base64.b64decode(PAYLOAD["profile_acceptance_b64"], validate=True)
    assert (
        hashlib.sha256(profile_acceptance_raw).hexdigest() == PAYLOAD["profile_acceptance_sha256"]
    )
    profile_acceptance = json.loads(profile_acceptance_raw)
    profile = profile_acceptance["profile"]
    assert (
        profile_acceptance["schema"] == "fg19.teacher-cache-profile-root-acceptance.v1"
        and profile_acceptance["status"] == "ACCEPTED"
        and profile_acceptance["capture_payloads_length_sha_validated"] is True
        and profile_acceptance["source_commit"] == PAYLOAD["source_commit"]
        and profile_acceptance["source_manifest_sha256"] == PAYLOAD["source_manifest_sha256"]
        and profile_acceptance["source_semantic_pins_match_manifest"] is True
        and profile_acceptance["normalization_matches_stored_s3_native_logs"] is True
        and profile_acceptance["profile_path_b"] == PAYLOAD["profile_path_b"]
        and profile_acceptance["unit"]["Result"] == "success"
        and profile_acceptance["unit"]["ExecMainStatus"] == "0"
        and profile_acceptance["unit"]["SubState"] == "exited"
        and profile["status"] == "PASS"
        and profile["record_type"] == "PROFILE"
        and profile["checkpoint_sha256"] == teacher["checkpoint_sha256"]
        and profile["split_manifest_sha256"] == PAYLOAD["split_sha256"]
        and profile["prediction_shape"] == [16, 20_000, 2]
        and profile["prediction_dtype"] == "float32"
        and profile["prediction_finite"] is True
        and profile["finite_output_count"] == 640_000
        and profile["profile_contract"]["batch_size"] == 16
        and profile["profile_contract"]["chunk_size"] == 4000
        and profile["profile_contract"]["raw_main_outputs"] is True
        and profile["profile_contract"]["strict_fp32"] is True
        and profile["profile_contract"]["autocast_enabled"] is False
        and profile["profile_contract"]["matmul_tf32"] is False
        and profile["profile_contract"]["cudnn_tf32"] is False
    )
    profile_path = Path(PAYLOAD["profile_path_b"])
    assert profile_path.is_file() and not profile_path.is_symlink()
    assert gates["_sha"](profile_path) == PAYLOAD["profile_sha256"]
    assert json.loads(profile_path.read_bytes()) == profile

    stop_raw = base64.b64decode(PAYLOAD["profile_stop_b64"], validate=True)
    assert hashlib.sha256(stop_raw).hexdigest() == PAYLOAD["profile_stop_sha256"]
    stop = json.loads(stop_raw)
    profile_unit = PAYLOAD["profile_unit"]
    assert stop["stop"][profile_unit]["returncode"] == 0
    assert (
        profile_acceptance["unit"]["InvocationID"] == PAYLOAD["profile_invocation_id"]
        and stop["before"][profile_unit]["InvocationID"] == PAYLOAD["profile_invocation_id"]
    )
    assert (
        stop["after"][profile_unit]["LoadState"] == "not-found"
        and stop["after"][profile_unit]["ActiveState"] == "inactive"
        and stop["after"][profile_unit]["SubState"] == "dead"
        and stop["populated"] == {}
    )

    train_acceptance_raw = base64.b64decode(PAYLOAD["train_export_acceptance_b64"], validate=True)
    assert (
        hashlib.sha256(train_acceptance_raw).hexdigest()
        == PAYLOAD["train_export_acceptance_sha256"]
    )
    train_acceptance = json.loads(train_acceptance_raw)
    train_unit = PAYLOAD["train_export_unit"]
    train_invocation = PAYLOAD["train_export_invocation_id"]
    native = train_acceptance["native_unit"]
    assert (
        train_acceptance["schema"] == "fg19.teacher-s3-train-export-terminal-acceptance.v1"
        and train_acceptance["status"] == "ACCEPTED"
        and train_acceptance["capture_payloads_length_sha_validated"] is True
        and train_acceptance["run_name"] == "fg19_t3_h384x2_e3_s3_v1"
        and train_acceptance["split"] == "train"
        and train_acceptance["selected_epoch"] == 2
        and train_acceptance["source_commit"] == PAYLOAD["source_commit"]
        and train_acceptance["source_manifest_sha256"] == PAYLOAD["source_manifest_sha256"]
        and train_acceptance["source_files_verified"] == 167
        and train_acceptance["runtime_semantic_pins_verified"] == 18
        and train_acceptance["completed_sequences"] == 10_535
        and train_acceptance["prediction_bytes"] == 1_685_600_128
        and train_acceptance["prediction_sha256"]
        == "218e3c41e91ff45e691b85e6999412841ca05c68aebdfc25c1c7659b0cc6bec5"
        and train_acceptance["prediction_metadata_sha256"]
        == "491767c8a914cf945d0284d29506e41cb74e2977c7b2b78baec524eabae859f6"
        and train_acceptance["checkpoint_sha256"]
        == "58e62541b9305169b94301d2e4f6643741d6a25f0712bc7b7cbf020de6b76241"
        and train_acceptance["split_manifest_sha256"] == PAYLOAD["split_sha256"]
        and train_acceptance["ordered_fit_membership_verified"] is True
        and train_acceptance["prediction_finite_independently_checked"] is False
        and train_acceptance["finite_check_deferred_to_streamed_cpu_cache_builder"] is True
        and train_acceptance["new_scientific_fits"] == 0
        and train_acceptance["quality_grade"] is False
        and native["InvocationID"] == train_invocation
        and native["Result"] == "success"
        and native["ExecMainStatus"] == "0"
        and native["SubState"] == "exited"
    )
    train_stop_raw = base64.b64decode(PAYLOAD["train_export_stop_b64"], validate=True)
    assert hashlib.sha256(train_stop_raw).hexdigest() == PAYLOAD["train_export_stop_sha256"]
    train_stop = json.loads(train_stop_raw)
    assert train_stop["stop"][train_unit]["returncode"] == 0
    assert train_stop["before"][train_unit]["InvocationID"] == train_invocation
    assert train_stop["after"][train_unit]["LoadState"] == "not-found"
    assert train_stop["after"][train_unit]["ActiveState"] == "inactive"
    assert train_stop["after"][train_unit]["SubState"] == "dead"
    assert train_stop["populated"] == {}

    train_prediction = prediction_dir / f"pred_{teacher['run']}_train.npy"
    train_partial = train_prediction.with_name(train_prediction.name + ".partial")
    train_metadata = train_prediction.with_suffix(".json")
    assert train_acceptance["prediction_path_b"] == str(train_prediction)
    assert train_acceptance["prediction_metadata_path_b"] == str(train_metadata)
    assert train_prediction.is_file() and not train_prediction.is_symlink()
    assert train_prediction.stat().st_size == train_acceptance["prediction_bytes"]
    assert not os.path.lexists(train_partial)
    assert train_metadata.is_file() and not train_metadata.is_symlink()
    assert train_metadata.stat().st_size <= 1_048_576
    assert gates["_sha"](train_metadata) == train_acceptance["prediction_metadata_sha256"]

    valid_acceptance_raw = base64.b64decode(PAYLOAD["valid_export_acceptance_b64"], validate=True)
    assert (
        hashlib.sha256(valid_acceptance_raw).hexdigest()
        == PAYLOAD["valid_export_acceptance_sha256"]
    )
    valid_acceptance = json.loads(valid_acceptance_raw)
    valid_unit = PAYLOAD["valid_export_unit"]
    valid_invocation = PAYLOAD["valid_export_invocation_id"]
    valid_native = valid_acceptance["native_unit"]
    assert (
        valid_acceptance["schema"] == "fg19.teacher-s3-valid-export-terminal-acceptance.v1"
        and valid_acceptance["status"] == "ACCEPTED"
        and valid_acceptance["capture_payloads_length_sha_validated"] is True
        and valid_acceptance["run_name"] == teacher["run"] == "fg19_t3_h384x2_e3_s3_v1"
        and valid_acceptance["split"] == "valid"
        and valid_acceptance["selected_epoch"] == 2
        and valid_acceptance["completed_sequences"] == 1_405
        and valid_acceptance["progress_records"] == 88
        and valid_acceptance["source_commit"] == PAYLOAD["source_commit"]
        and valid_acceptance["source_manifest_sha256"] == PAYLOAD["source_manifest_sha256"]
        and valid_acceptance["source_files_verified"] == 167
        and valid_acceptance["runtime_semantic_pins_verified"] == 18
        and valid_acceptance["checkpoint_sha256"] == teacher["checkpoint_sha256"]
        and valid_acceptance["teacher_provenance_sha256"] == teacher["provenance_sha256"]
        and valid_acceptance["split_manifest_sha256"] == PAYLOAD["split_sha256"]
        and valid_acceptance["fit_membership_sha256"] == PAYLOAD["valid_fit_membership_sha256"]
        and valid_acceptance["prediction_shape_by_source_membership"] == [1_405, 20_000, 2]
        and valid_acceptance["prediction_dtype_authority"].startswith("float32")
        and valid_acceptance["prediction_finite_independently_checked"] is False
        and valid_acceptance["finite_check_deferred_to_streamed_cpu_cache_builder"] is True
        and valid_acceptance["ordered_fit_membership_verified"] is True
        and valid_acceptance["same_valid_namespace_eval_disjoint_verified"] is True
        and valid_acceptance["feature_only"] is True
        and valid_acceptance["new_scientific_fits"] == 0
        and valid_acceptance["quality_grade"] is False
        and valid_native["InvocationID"] == valid_invocation
        and valid_native["Result"] == "success"
        and valid_native["ExecMainStatus"] == "0"
        and valid_native["SubState"] == "exited"
    )
    valid_stop_raw = base64.b64decode(PAYLOAD["valid_export_stop_b64"], validate=True)
    assert hashlib.sha256(valid_stop_raw).hexdigest() == PAYLOAD["valid_export_stop_sha256"]
    valid_stop = json.loads(valid_stop_raw)
    assert valid_stop["stop"][valid_unit]["returncode"] == 0
    assert valid_stop["before"][valid_unit]["InvocationID"] == valid_invocation
    assert valid_stop["after"][valid_unit]["LoadState"] == "not-found"
    assert valid_stop["after"][valid_unit]["ActiveState"] == "inactive"
    assert valid_stop["after"][valid_unit]["SubState"] == "dead"
    assert valid_stop["populated"] == {}

    valid_prediction = prediction_dir / f"pred_{teacher['run']}_valid.npy"
    valid_partial = valid_prediction.with_name(valid_prediction.name + ".partial")
    valid_metadata = valid_prediction.with_suffix(".json")
    assert PAYLOAD["valid_fit_rows"] == 1_405
    assert valid_acceptance["prediction_path_b"] == str(valid_prediction)
    assert valid_acceptance["prediction_metadata_path_b"] == str(valid_metadata)
    assert valid_prediction.is_file() and not valid_prediction.is_symlink()
    assert valid_prediction.stat().st_size == valid_acceptance["prediction_bytes"] == 224_800_128
    assert not os.path.lexists(valid_partial)
    assert valid_metadata.is_file() and not valid_metadata.is_symlink()
    assert valid_metadata.stat().st_size <= 1_048_576
    assert gates["_sha"](valid_metadata) == valid_acceptance["prediction_metadata_sha256"]

    s4_train_acceptance_raw = base64.b64decode(
        PAYLOAD["s4_train_export_acceptance_b64"], validate=True
    )
    assert (
        hashlib.sha256(s4_train_acceptance_raw).hexdigest()
        == PAYLOAD["s4_train_export_acceptance_sha256"]
    )
    s4_train_acceptance = json.loads(s4_train_acceptance_raw)
    s4_train_unit = PAYLOAD["s4_train_export_unit"]
    s4_train_invocation = PAYLOAD["s4_train_export_invocation_id"]
    s4_train_native = s4_train_acceptance["native_unit"]
    assert (
        s4_train_acceptance["schema"] == "fg19.teacher-s4-train-export-terminal-acceptance.v1"
        and s4_train_acceptance["status"] == "ACCEPTED"
        and s4_train_acceptance["capture_payloads_length_sha_validated"] is True
        and s4_train_acceptance["run_name"] == target_teacher["run"]
        and s4_train_acceptance["split"] == "train"
        and s4_train_acceptance["selected_epoch"] == 2
        and s4_train_acceptance["source_commit"] == PAYLOAD["source_commit"]
        and s4_train_acceptance["source_manifest_sha256"] == PAYLOAD["source_manifest_sha256"]
        and s4_train_acceptance["source_files_verified"] == 167
        and s4_train_acceptance["runtime_semantic_pins_verified"] == 18
        and s4_train_acceptance["completed_sequences"] == 10_535
        and s4_train_acceptance["checkpoint_sha256"] == target_teacher["checkpoint_sha256"]
        and s4_train_acceptance["teacher_provenance_sha256"] == target_teacher["provenance_sha256"]
        and s4_train_acceptance["split_manifest_sha256"] == PAYLOAD["split_sha256"]
        and s4_train_acceptance["ordered_fit_membership_verified"] is True
        and s4_train_acceptance["prediction_finite_independently_checked"] is False
        and s4_train_acceptance["finite_check_deferred_to_streamed_cpu_cache_builder"] is True
        and s4_train_acceptance["new_scientific_fits"] == 0
        and s4_train_acceptance["quality_grade"] is False
        and s4_train_native["InvocationID"] == s4_train_invocation
        and s4_train_native["Result"] == "success"
        and s4_train_native["ExecMainStatus"] == "0"
        and s4_train_native["SubState"] == "exited"
    )
    s4_train_normalization = s4_train_acceptance["normalization_identity"]
    for statistic in ("mean", "std"):
        stat_identity = s4_train_normalization[statistic]
        assert stat_identity["shape"] == [211]
        assert stat_identity["dtype"] == "torch.float32"
        norm_sha = stat_identity["bytes_sha256"]
        assert len(norm_sha) == 64 and all(char in "0123456789abcdef" for char in norm_sha)

    s4_train_stop_raw = base64.b64decode(PAYLOAD["s4_train_export_stop_b64"], validate=True)
    assert hashlib.sha256(s4_train_stop_raw).hexdigest() == PAYLOAD["s4_train_export_stop_sha256"]
    s4_train_stop = json.loads(s4_train_stop_raw)
    assert s4_train_stop["stop"][s4_train_unit]["returncode"] == 0
    assert s4_train_stop["before"][s4_train_unit]["InvocationID"] == s4_train_invocation
    assert s4_train_stop["after"][s4_train_unit]["LoadState"] == "not-found"
    assert s4_train_stop["after"][s4_train_unit]["ActiveState"] == "inactive"
    assert s4_train_stop["after"][s4_train_unit]["SubState"] == "dead"
    assert s4_train_stop["populated"] == {}

    s4_train_prediction = prediction_dir / f"pred_{target_teacher['run']}_train.npy"
    s4_train_partial = s4_train_prediction.with_name(s4_train_prediction.name + ".partial")
    s4_train_metadata = s4_train_prediction.with_suffix(".json")
    assert s4_train_acceptance["prediction_path_b"] == str(s4_train_prediction)
    assert s4_train_acceptance["prediction_metadata_path_b"] == str(s4_train_metadata)
    assert s4_train_prediction.is_file() and not s4_train_prediction.is_symlink()
    assert (
        s4_train_prediction.stat().st_size
        == s4_train_acceptance["prediction_bytes"]
        == 1_685_600_128
    )
    assert not os.path.lexists(s4_train_partial)
    assert s4_train_metadata.is_file() and not s4_train_metadata.is_symlink()
    assert s4_train_metadata.stat().st_size <= 1_048_576
    assert gates["_sha"](s4_train_metadata) == s4_train_acceptance["prediction_metadata_sha256"]

    target_prediction = prediction_dir / f"pred_{target_teacher['run']}_valid.npy"
    target_partial = target_prediction.with_name(target_prediction.name + ".partial")
    target_metadata = target_prediction.with_suffix(".json")
    assert PAYLOAD["valid_fit_rows"] == 1_405
    assert not os.path.lexists(target_prediction)
    assert not os.path.lexists(target_partial)
    assert not os.path.lexists(target_metadata)

    source_files = PAYLOAD["source_files"]
    assert len(source_files) == 167
    actual_files = set()
    for path in source.rglob("*"):
        assert not path.is_symlink()
        if path.is_file():
            name = str(path.relative_to(source))
            actual_files.add(name)
            item = source_files[name]
            assert path.stat().st_size == item["bytes"]
            assert gates["_sha"](path) == item["sha256"]
    assert actual_files == set(source_files)

    split_path = b / "splits/fold0.json"
    assert split_path.is_file() and not split_path.is_symlink()
    assert gates["_sha"](split_path) == PAYLOAD["split_sha256"]
    data = gates["_data_identity"]()
    parent, populated, cgroup = gates["_parent"]()
    gpu, foreign = gates["_gpu"]()
    scratch = gates["_run"](["findmnt", "-T", str(b / "research_tmp"), "-n", "-o", "FSTYPE"])
    assert scratch.strip() not in ("tmpfs", "ramfs")
    disk = os.statvfs(b / "research_tmp")
    free_bytes = disk.f_bavail * disk.f_frsize
    reserve_bytes = (
        25 * (1 << 30) + contract["disk_estimates"]["three_teacher_plus_combined_payload_bytes"]
    )
    assert contract["disk_estimates"]["three_teacher_plus_combined_payload_bytes"] == 7_641_600_000
    assert free_bytes >= reserve_bytes

    metadata_files = {
        "preparation/TEACHER_CACHE_EXECUTION_CONTRACT_R1.json": contract_raw,
        **acceptance_bytes,
    }
    assert len(metadata_files) == 4
    for name, raw in metadata_files.items():
        path = metadata_root / name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == len(raw)
        assert gates["_sha"](path) == hashlib.sha256(raw).hexdigest()
    actual_metadata_json = {
        path.relative_to(metadata_root).as_posix()
        for path in metadata_root.rglob("*.json")
        if path.is_file()
    }
    assert actual_metadata_json == set(metadata_files)

    argv = [
        str(gates["PYTHON"]),
        "-u",
        str(source / "scripts/predict_run.py"),
        target_teacher["run"],
        "valid",
        "--epoch",
        "2",
        "--campaign-split-manifest",
        str(split_path),
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
        "--loader-workers",
        "0",
        "--output-dir",
        str(prediction_dir),
        "--create-only",
    ]
    command = ["systemd-run", "--user", "--unit=" + unit]
    properties = [
        "Type=exec",
        "RemainAfterExit=yes",
        "Slice=wnn-campaign.slice",
        "CPUQuota=100%",
        "MemoryHigh=7500000000",
        "MemoryMax=8000000000",
        "MemorySwapMax=0",
        "RuntimeMaxSec=10800",
        "StandardOutput=append:" + str(evidence_dir / "unit.log"),
        "StandardError=append:" + str(evidence_dir / "unit.log"),
    ]
    command += ["--property=" + item for item in properties]
    environment = dict(
        CUDA_VISIBLE_DEVICES=gates["GPU"],
        WNN_PROJECT_ROOT=str(b / "data"),
        PYTHONPATH=str(source / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        WNN_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        WNN_GPU_MEMORY_BYTES="4294967296",
        WNN_GPU_MEMORY_RESERVE_BYTES="1000000000",
        WNN_LOADER_WORKERS="0",
        WNN_LOADER_RAM_GB="1",
        TMPDIR=str(b / "research_tmp"),
    )
    command += ["--setenv=" + key + "=" + value for key, value in environment.items()]
    command += ["--working-directory=" + str(source), *argv]
    admission = dict(
        utc=dt.datetime.now(dt.UTC).isoformat(),
        unit=unit,
        source_commit=PAYLOAD["source_commit"],
        source_manifest_sha256=PAYLOAD["source_manifest_sha256"],
        source_files={name: item["sha256"] for name, item in source_files.items()},
        contract_sha256=PAYLOAD["contract_sha256"],
        existing_metadata_sha256={
            name: hashlib.sha256(raw).hexdigest() for name, raw in metadata_files.items()
        },
        train_export_acceptance_sha256=PAYLOAD["train_export_acceptance_sha256"],
        train_export_stop_sha256=PAYLOAD["train_export_stop_sha256"],
        train_prediction_producer_sha256=train_acceptance["prediction_sha256"],
        train_prediction_metadata_sha256=train_acceptance["prediction_metadata_sha256"],
        train_prediction_stat_bytes=train_prediction.stat().st_size,
        train_prediction_payload_read_or_hashed=False,
        valid_export_acceptance_sha256=PAYLOAD["valid_export_acceptance_sha256"],
        valid_export_stop_sha256=PAYLOAD["valid_export_stop_sha256"],
        valid_prediction_producer_sha256=valid_acceptance["prediction_sha256"],
        valid_prediction_metadata_sha256=valid_acceptance["prediction_metadata_sha256"],
        valid_prediction_stat_bytes=valid_prediction.stat().st_size,
        prior_prediction_payloads_read_or_hashed=False,
        s4_train_export_acceptance_sha256=PAYLOAD["s4_train_export_acceptance_sha256"],
        s4_train_export_stop_sha256=PAYLOAD["s4_train_export_stop_sha256"],
        s4_train_prediction_producer_sha256=s4_train_acceptance["prediction_sha256"],
        s4_train_prediction_metadata_sha256=s4_train_acceptance["prediction_metadata_sha256"],
        s4_train_prediction_stat_bytes=s4_train_prediction.stat().st_size,
        s4_normalization_identity=s4_train_normalization,
        teacher_acceptance_sha256={
            item["run"]: item["acceptance_sha256"] for item in contract["teachers"]
        },
        selected_teacher={
            "run": target_teacher["run"],
            "epoch": 2,
            "checkpoint_sha256": target_teacher["checkpoint_sha256"],
            "provenance_sha256": target_teacher["provenance_sha256"],
        },
        profile_acceptance_sha256=PAYLOAD["profile_acceptance_sha256"],
        profile_sha256=PAYLOAD["profile_sha256"],
        profile_stop_sha256=PAYLOAD["profile_stop_sha256"],
        data=data,
        parent=parent,
        populated=populated,
        gpu=gpu,
        foreign_preserved=foreign,
        disk={"free_bytes": free_bytes, "required_reserve_bytes": reserve_bytes},
        memory_events=(cgroup / "memory.events").read_text(),
        memory_events_local=(cgroup / "memory.events.local").read_text(),
        command=command,
        target_prediction_paths={
            "prediction": str(target_prediction),
            "partial": str(target_partial),
            "metadata": str(target_metadata),
        },
        expected_normalization_identity=s4_train_normalization,
        retry_policy="Never retry; inspect exact unit and durable receipts after any unknown outcome.",
    )
    evidence_dir.mkdir()
    gates["_write"](evidence_dir / "ADMISSION.json", admission)
    try:
        result = subprocess.run(command, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        stdout_raw = error.stdout or b""
        stderr_raw = error.stderr or b""
        launch = {
            "utc": dt.datetime.now(dt.UTC).isoformat(),
            "unit": unit,
            "command": command,
            "outcome": "UNKNOWN_TIMEOUT_DO_NOT_RETRY",
            "stdout_raw_base64": base64.b64encode(stdout_raw).decode("ascii"),
            "stderr_raw_base64": base64.b64encode(stderr_raw).decode("ascii"),
        }
        gates["_write"](evidence_dir / "LAUNCH.json", launch)
        raise SystemExit("systemd-run outcome is unknown; do not retry") from error
    launch = dict(
        utc=dt.datetime.now(dt.UTC).isoformat(),
        unit=unit,
        command=command,
        returncode=result.returncode,
        stdout_raw_base64=base64.b64encode(result.stdout).decode("ascii"),
        stderr_raw_base64=base64.b64encode(result.stderr).decode("ascii"),
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),
        post=gates["_unit"](
            unit, ["InvocationID", "ActiveState", "SubState", "MainPID", "ControlGroup"]
        ),
        unknown_outcome_policy="Never retry; inspect exact unit and durable receipts.",
    )
    gates["_write"](evidence_dir / "LAUNCH.json", launch)
    print(json.dumps({"admission": admission, "launch": launch}))
    if result.returncode:
        raise SystemExit("systemd-run returned nonzero; do not retry before inspection")


if __name__ == "__main__":
    main()
