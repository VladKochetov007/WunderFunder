#!/usr/bin/env python3
"""Check a local full-valid teacher export's frozen inputs; never launches it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import zipfile
from datetime import UTC, datetime
from pathlib import Path

MAIN_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011"
)
R5 = MAIN_REL / "clean_t3_fold0_h390_r5_teacher_cohort"
R3 = MAIN_REL / "clean_t3_fold0_h390_r3_teacher_cohort"
RUN = "clean_t3_f0_muonall_h384_e3_s3_r5"
OUTPUT_NAME = "clean_t3_f0_muonall_h384_e3_s3_r5_local_full_valid_r1"
EXPECTED = {
    "checkpoint": "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
    "provenance": "740b0d96ee311f215153db62272262e0153ce11f8f91560b03efae8e739f9ed0",
    "history": "c64927a708207dcaff25505c622e352d73790e808a9200899f5f2b2dbaf94d49",
    "training_state": "a9b6eed9f52a2a8fd350d1355bdd04a15a3b6d4af9886f608d715a709e006edb",
    "split": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    "valid_targets": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
    "source_manifest": "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa",
    "predictor": "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0",
    "trainer": "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6",
    "r6_operator_receipt": "4573b36e7493f038f0e92264627fc682bb0041d74c050120323632bd84b43b17",
    "r6_stage_receipt": "b77fda744f2e81a4218bd689681d4586f339a7674122fba6b4aa861758951a12",
}
EXPECTED_BYTES = {
    "checkpoint": 8_692_775,
    "provenance": 159_434,
    "history": 4_586,
    "training_state": 17_344_686,
    "split": 1_388_846,
    "valid_targets": 337_186_198,
    "valid_parquet": 5_615_509_128,
}
EXPECTED_VALID_PARQUET = {
    "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    "footer_bytes": 10_686_926,
    "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
    "mtime_ns": 1_577_826_000_000_000_000,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tail(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(-(size + 8), os.SEEK_END)
        remaining = size + 8
        while remaining:
            block = source.read(min(1 << 20, remaining))
            if not block:
                raise EOFError("truncated Parquet footer range")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def row_group_header(path: Path) -> dict[str, object]:
    """Read only the small row_group member from valid_targets.npz."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("row_group.npy")
        with archive.open(info) as stream:
            prefix = stream.read(256)
            match = re.search(rb"'descr': '([^']+)'.*?'shape': \((\d+),\)", prefix)
            if not match:
                raise ValueError("could not parse row_group.npy header")
            dtype, rows = match.groups()
            if prefix[:8] != b"\x93NUMPY\x01\x00":
                raise ValueError("row_group.npy is not NPY version 1.0")
            header_end = 10 + struct.unpack("<H", prefix[8:10])[0]
            payload = prefix[header_end:] + stream.read()
    if dtype != b"<i8" or len(payload) != int(rows) * 8:
        raise ValueError("row_group.npy has unexpected dtype or byte count")
    return {
        "member": info.filename,
        "dtype": dtype.decode("ascii"),
        "rows": int(rows),
        "payload_bytes": len(payload),
        "ordered_int64_le_sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    packet = root / R5
    source = root / R3 / "source_snapshot"
    run_dir = root / "artifacts/runs" / RUN
    paths = {
        "checkpoint": run_dir / "model_ep2.pt",
        "provenance": run_dir / "provenance.json",
        "history": run_dir / "history.json",
        "training_state": run_dir / "training_state.pt",
        "split": root / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "valid_targets": root / "cache/valid_targets.npz",
        "valid_parquet": root / "datasets/valid.parquet",
        "source_manifest": source / "SOURCE_MANIFEST_H390.json",
        "predictor": source / "scripts/predict_run.py",
        "trainer": source / "scripts/train_rnn.py",
        "r6_operator_receipt": packet / "r6_bootstrap_guard_successor_20261012_r1/terminal_collection_v2/R6_V2_TERMINAL_OPERATOR_RECEIPT_R1.json",
        "r6_stage_receipt": packet / "r6_bootstrap_guard_successor_20261012_r1/terminal_collection_v2/ROOT_ACTUAL_COLLECTION_AND_LOCAL_STAGE_R1.json",
    }
    observed: dict[str, dict[str, object]] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required input is absent: {key}: {path}")
        size = path.stat().st_size
        if key in EXPECTED_BYTES and size != EXPECTED_BYTES[key]:
            raise ValueError(f"byte count changed for {key}: {size}")
        row: dict[str, object] = {"path": str(path), "bytes": size}
        if key in EXPECTED:
            digest = sha256_file(path)
            if digest != EXPECTED[key]:
                raise ValueError(f"SHA256 changed for {key}: {digest}")
            row["sha256"] = digest
        observed[key] = row

    provenance = json.loads(Path(observed["provenance"]["path"]).read_text())
    args_record = provenance["args"]
    fit_identity = provenance["fit_membership_identity"]
    if (
        provenance.get("selected_epoch") != 2
        or provenance.get("split_manifest_sha256") != EXPECTED["split"]
        or provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") != EXPECTED["checkpoint"]
        or args_record.get("name") != RUN
        or args_record.get("seed") != 3
        or args_record.get("valid_fold") != 0
        or args_record.get("muon_scope") != "all"
        or args_record.get("propensity") != "c20260929_prop_f0"
        or args_record.get("train_source") != "train+valid"
        or args_record.get("include_heldout_valid") is not False
        or fit_identity.get("train", {}).get("count") != 10535
        or fit_identity.get("valid", {}).get("count") != 1405
        or fit_identity.get("dataset_length") != 11940
        or len(provenance.get("evaluation_row_groups", [])) != 468
    ):
        raise ValueError("checkpoint provenance does not match the frozen clean F0 recipe")

    stage = json.loads(Path(observed["r6_stage_receipt"]["path"]).read_text())
    if (
        stage.get("decision") != "PASS_COMPLETE_COLLECTION_AND_THREE_FIT_ARTIFACTS_FOR_EXPORT_ONLY_RECOVERY"
        or stage.get("provider_action") is not False
        or stage.get("fit_status") != "three complete fits, no prediction exports or quality grades"
    ):
        raise ValueError("R6 stage receipt does not authorize interpreting these as completed fits")
    staged = {row["path"]: row for row in stage.get("local_staged_artifacts", [])}
    for name in ("model_ep2.pt", "provenance.json", "history.json", "training_state.pt"):
        relative = f"artifacts/runs/{RUN}/{name}"
        row = staged.get(relative)
        key = {"model_ep2.pt": "checkpoint", "provenance.json": "provenance", "history.json": "history", "training_state.pt": "training_state"}[name]
        if row is None or row.get("sha256") != observed[key].get("sha256"):
            raise ValueError(f"R6 staged artifact closure is missing or differs: {relative}")

    source_binding_path = root / R3 / "SOURCE_BINDING_R3.json"
    source_binding = json.loads(source_binding_path.read_text())
    if (
        source_binding.get("source_manifest_sha256") != EXPECTED["source_manifest"]
        or source_binding.get("predictor_sha256") != EXPECTED["predictor"]
        or source_binding.get("trainer_sha256") != EXPECTED["trainer"]
    ):
        raise ValueError("frozen R3 source binding differs from the local predictor bytes")
    observed["source_binding"] = {
        "path": str(source_binding_path),
        "sha256": sha256_file(source_binding_path),
    }

    input_binding_path = root / R3 / "INPUT_BINDING_R3.json"
    input_binding = json.loads(input_binding_path.read_text())
    input_rows = {row["path"]: row for row in input_binding["external_inputs"]}
    valid_input = input_rows.get("datasets/valid.parquet")
    if (
        valid_input is None
        or valid_input.get("sha256") != EXPECTED_VALID_PARQUET["sha256"]
        or valid_input.get("size") != EXPECTED_BYTES["valid_parquet"]
    ):
        raise ValueError("R3 input binding does not pin the expected validation source")
    split = json.loads(paths["split"].read_text())
    valid_source = split["sources"]["datasets/valid.parquet"]
    if (
        valid_source.get("bytes") != EXPECTED_BYTES["valid_parquet"]
        or valid_source.get("footer_bytes") != EXPECTED_VALID_PARQUET["footer_bytes"]
        or valid_source.get("footer_sha256") != EXPECTED_VALID_PARQUET["footer_sha256"]
        or paths["valid_parquet"].stat().st_mtime_ns != EXPECTED_VALID_PARQUET["mtime_ns"]
        or sha256_tail(paths["valid_parquet"], EXPECTED_VALID_PARQUET["footer_bytes"])
        != EXPECTED_VALID_PARQUET["footer_sha256"]
    ):
        raise ValueError("local valid.parquet footer/metadata identity differs from fold0 producer")
    observed["valid_parquet_source_binding"] = {
        "path": str(paths["valid_parquet"]),
        "expected_full_sha256_from_R3_input_binding": valid_input["sha256"],
        "full_content_rehashed": False,
        "verified_file_size": paths["valid_parquet"].stat().st_size,
        "verified_mtime_ns": paths["valid_parquet"].stat().st_mtime_ns,
        "verified_trailing_footer_plus_length_and_PAR1_sha256": EXPECTED_VALID_PARQUET[
            "footer_sha256"
        ],
        "footer_bytes": EXPECTED_VALID_PARQUET["footer_bytes"],
    }

    spec_path = packet / "terminal_collection_r1/provider_output/attempt_001/outputs/experiment/kaggle_specs/export_clean_t3_f0_muonall_h384_e3_s3_r5_full_valid.json"
    spec = json.loads(spec_path.read_text())
    spec_inputs = {row["path"]: row for row in spec.get("inputs", [])}
    if (
        spec.get("name") != "export_clean_t3_f0_muonall_h384_e3_s3_r5_full_valid"
        or spec.get("limits", {}).get("ram_bytes") != 8_000_000_000
        or spec_inputs.get("datasets/valid.parquet", {}).get("sha256")
        != valid_input["sha256"]
        or not spec.get("argv", [])[-1:] == ["--campaign-full-valid-feature-only"]
        or "--inference-precision" not in spec.get("argv", [])
        or "fp32" not in spec.get("argv", [])
    ):
        raise ValueError("R5 full-valid export producer spec differs from local recipe")
    observed["r5_full_valid_export_spec"] = {
        "path": str(spec_path),
        "sha256": sha256_file(spec_path),
        "name": spec["name"],
        "ram_reservation_bytes": spec["limits"]["ram_bytes"],
        "argv": spec["argv"],
    }

    row_header = row_group_header(paths["valid_targets"])
    if row_header["rows"] != 1873:
        raise ValueError("official valid-target row order does not contain 1873 groups")
    observed["valid_target_row_group_header"] = row_header

    output_base = root / "cache" / f"pred_{OUTPUT_NAME}_valid"
    output_paths = [output_base.with_suffix(".npy"), output_base.with_suffix(".json")]
    partial_path = output_base.with_name(output_base.name + ".npy.partial")
    collisions = [str(path) for path in [*output_paths, partial_path] if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing output collision: {collisions}")

    argv = [
        str(root / ".venv/bin/python"),
        str(source / "scripts/predict_run.py"),
        RUN,
        "valid",
        "--epoch", "2",
        "--output-name", OUTPUT_NAME,
        "--campaign-split-manifest", "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "--device", "cuda",
        "--inference-precision", "fp32",
        "--campaign-full-valid-feature-only",
    ]
    command = {
        "status": "prepared_not_launched",
        "cwd": str(root),
        "argv": argv,
        "environment": {
            "WNN_PROJECT_ROOT": str(root),
            "PYTHONPATH": str(source / "src"),
            "WNN_LOADER_WORKERS": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "output_npy": str(output_paths[0]),
        "output_json": str(output_paths[1]),
        "expected_npy_shape": [1873, 20000, 2],
        "expected_npy_dtype": "float32",
        "expected_npy_bytes_including_header": 299_680_128,
        "semantics": "full-valid feature-only, official valid_targets row order; not a held-out score",
        "proposed_resource_envelope_pending_root_admission": {
            "cpu_quota_percent": 100,
            "threads": 1,
            "memory_max_bytes": 6_800_000_000,
            "gpu_allocator_target_bytes": 4_300_000_000,
            "observed_vram_guard_bytes": 5_000_000_000,
            "timeout_seconds": 1800,
            "loader_workers": 0,
        },
        "pending_root_gates": [
            "fresh local aggregate RAM/CPU/GPU admission",
            "actual local-device export admission/profile",
            "root reconciles the local raw valid.parquet bytes to the frozen full-content binding; this preflight checks size/mtime/footer only",
        ],
    }
    receipt = {
        "schema": "wnn.local_clean_t3_f0_s3_full_valid_export_preflight.r1",
        "created_utc": datetime.now(UTC).isoformat(),
        "decision": "PASS_STATIC_AND_ACTUAL_LOCAL_ARTIFACT_PREFLIGHT; NO EXPORT LAUNCHED",
        "scope": "read-only hash/metadata/CLI-contract preparation; no model import, CUDA, inference, grading, or provider action",
        "project_root": str(root),
        "run": RUN,
        "selected_checkpoint_epoch": 2,
        "expected_sha256": EXPECTED,
        "observed_inputs": observed,
        "valid_parquet_local_hash": "full file not rehashed; local size, mtime, producer footer+length+PAR1 hash, and exact frozen R3 full-content hash binding are checked separately",
        "command": command,
        "full_valid_predictions_are_not_a_heldout_grade": True,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": receipt["decision"], "receipt": str(args.receipt), "run": RUN, "outputs_absent": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
