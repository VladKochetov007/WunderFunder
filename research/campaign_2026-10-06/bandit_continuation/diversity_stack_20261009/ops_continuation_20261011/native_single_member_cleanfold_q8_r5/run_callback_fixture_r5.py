from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
R3_PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    "/ops_continuation_20261011/native_single_member_cleanfold_q8_r3"
)
R4_PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    "/ops_continuation_20261011/native_single_member_cleanfold_q8_r4"
)
FOLDS = ("0", "1")
ROLES = ("student", "hft")
WALL_SECONDS = 180


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def import_pinned(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned R4 consumer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_deadline(started: float, phase: str) -> None:
    if time.monotonic() - started >= WALL_SECONDS:
        raise TimeoutError(f"R5 callback fixture exceeded 180-second cap during {phase}")


def main() -> None:
    started = time.monotonic()
    if tuple(sys.version_info[:2]) != (3, 11):
        raise RuntimeError(f"fixture requires CPython 3.11, got {sys.version.split()[0]}")
    expected_python = Path("/tmp/wnn-projection-precompute-r1/py311/bin/python").resolve(
        strict=True
    )
    if Path(sys.executable).resolve(strict=True) != expected_python:
        raise RuntimeError(f"fixture interpreter differs: {sys.executable}")
    package_versions = {name: importlib.metadata.version(name) for name in ("numpy", "pyarrow")}
    if package_versions != {"numpy": "2.2.6", "pyarrow": "19.0.1"}:
        raise RuntimeError(f"Py311 package versions differ: {package_versions}")

    import numpy as np
    import pyarrow.parquet as pq

    main_root = MAIN_ROOT.resolve(strict=True)
    r3_path = main_root / R3_PACKET / "CONTRACT_SINGLE_R3.json"
    r3_preflight_path = main_root / R3_PACKET / "INPUT_PREFLIGHT_MAIN_R1.json"
    r4_contract_path = main_root / R4_PACKET / "CONTRACT_SINGLE_R4.json"
    r4_freeze_path = main_root / R4_PACKET / "PREPARATION_FREEZE_R4.json"
    r3 = read_json(r3_path)
    preflight = read_json(r3_preflight_path)
    r4_contract = read_json(r4_contract_path)
    r4_freeze = read_json(r4_freeze_path)
    r4_sha = sha256_file(r4_contract_path)
    if r4_freeze.get("contract_sha256") != r4_sha:
        raise RuntimeError("R4 contract differs from its freeze receipt")
    r4_consumer_path = main_root / R4_PACKET / "run_single_controls_r4.py"
    r4_consumer_sha = sha256_file(r4_consumer_path)
    if (
        r4_contract.get("source_bindings", {}).get("r4_consumer", {}).get("sha256")
        != r4_consumer_sha
    ):
        raise RuntimeError("R4 consumer source hash differs from its contract")
    if preflight.get("contract_sha256") != sha256_file(r3_path):
        raise RuntimeError("R3 preflight is not bound to its exact contract")

    r5_runtime_path = Path(__file__).resolve(strict=True).with_name("callback_runtime_r5.py")
    r5_runtime_sha = sha256_file(r5_runtime_path)
    runtime = import_pinned("r5_callback_runtime", r5_runtime_path)
    consumer = runtime.import_pinned_r4(r4_consumer_path)
    consumer.verify_source_bindings(r4_contract, main_root)
    verified_r3, verified_preflight, verified_valid_path = consumer.verify_r3_receipts(
        r4_contract, r3_preflight_path, main_root
    )
    if verified_r3 != r3 or verified_preflight != preflight:
        raise RuntimeError("R4 consumer verified different R3 receipt contents")
    valid_path = main_root / r3["valid_dataset"]["repo_path"]
    if verified_valid_path != valid_path.resolve(strict=True):
        raise RuntimeError("R4 consumer resolved a different valid.parquet")
    if valid_path.resolve(strict=True) != (main_root / "datasets/valid.parquet").resolve(
        strict=True
    ):
        raise RuntimeError("valid dataset does not resolve to the bound MAIN path")

    runtime.import_bound_native_package(r4_contract, consumer)
    predictors, callback_runner, _, utils, _ = consumer.build_predictors(r4_contract, main_root)
    for fold in FOLDS:
        for role in ROLES:
            predictor = predictors[fold][role]
            if str(predictor.model.precision) != "q8vnni":
                raise RuntimeError(f"wrong native precision for {fold}/{role}")
    check_deadline(started, "predictor construction")

    r1_binding = r3["producer_receipts"]["shared_contract_r1"]
    r1_path = main_root / r1_binding["repo_path"]
    if sha256_file(r1_path) != r1_binding["sha256"]:
        raise RuntimeError("R1 split-membership producer changed")
    r1 = read_json(r1_path)
    data_stat = valid_path.stat()
    if data_stat.st_size != r3["valid_dataset"]["bytes"]:
        raise RuntimeError("valid.parquet size differs from the R3 input identity")

    results: dict[str, Any] = {}
    with valid_path.open("rb") as source:
        identity = consumer.file_footer_identity(
            source, valid_path, int(r3["valid_dataset"]["footer_bytes"])
        )
        consumer.validate_live_identity(identity, r3, preflight)
        parquet = pq.ParquetFile(source)
        identity = consumer.file_footer_identity(
            source, valid_path, int(r3["valid_dataset"]["footer_bytes"])
        )
        consumer.validate_live_identity(identity, r3, preflight)
        metadata_validation = runtime.check_stable_metadata(
            parquet, r3, preflight, list(utils.FEATURE_COLUMNS)
        )
        wrong_footer = dict(identity)
        wrong_footer["footer_sha256_including_8_byte_trailer"] = "0" * 64
        try:
            consumer.validate_live_identity(wrong_footer, r3, preflight)
        except consumer.GateError:
            footer_drift_rejected = True
        else:
            raise RuntimeError("R5 identity guard accepted a mutated Parquet footer hash")

        mutated_preflight = json.loads(json.dumps(preflight))
        mutated_preflight["metadata_only_validation"]["schema_names_sha256"] = "0" * 64
        try:
            runtime.check_stable_metadata(
                parquet, r3, mutated_preflight, list(utils.FEATURE_COLUMNS)
            )
        except ValueError:
            schema_drift_rejected = True
        else:
            raise RuntimeError("R5 metadata guard accepted mutated schema names")
        for fold in FOLDS:
            groups = r1["folds"][fold]["split_manifest"]["evaluation_row_groups_sorted"]
            if (
                len(groups) != r3["folds"][fold]["evaluation_groups"]
                or groups != sorted(set(groups))
                or consumer.canonical_group_hash(groups)
                != r3["folds"][fold]["evaluation_group_list_sha256"]
            ):
                raise RuntimeError(f"fold {fold} evaluation group list differs")
            group = int(groups[0])
            columns = [
                "seq_ix",
                "step_in_seq",
                "need_prediction",
                *utils.FEATURE_COLUMNS,
            ]
            table = parquet.read_row_group(group, columns=columns, use_threads=False)
            sequence_id, need = utils.validate_sequence(table)
            features = np.column_stack(
                [table[name].to_numpy(zero_copy_only=False) for name in utils.FEATURE_COLUMNS]
            ).astype(np.float32, copy=False)
            if features.shape != (20_000, 112) or not np.isfinite(features).all():
                raise RuntimeError(f"fold {fold} fixture sequence geometry/data is invalid")
            sequence = {"sequence_id": sequence_id, "features": features, "need": need}
            results[fold] = {}
            for role in ROLES:
                check_deadline(started, f"{fold}/{role} callback")
                predictor = predictors[fold][role]
                first_ns, first = callback_runner.run_callbacks(predictor, sequence)
                second_ns, second = callback_runner.run_callbacks(predictor, sequence)
                if first.shape != (20_000, 2) or second.shape != (20_000, 2):
                    raise RuntimeError(f"unexpected output shape for {fold}/{role}")
                if not np.isfinite(first[need]).all() or not np.isfinite(second[need]).all():
                    raise RuntimeError(f"nonfinite required output for {fold}/{role}")
                first_bytes = np.asarray(first, dtype=np.float32).tobytes(order="C")
                second_bytes = np.asarray(second, dtype=np.float32).tobytes(order="C")
                if first_bytes != second_bytes:
                    raise RuntimeError(f"reset replay is not bitwise equal for {fold}/{role}")
                results[fold][role] = {
                    "member_count": 1,
                    "precision": "q8vnni",
                    "complete_rows": 20_000,
                    "sequence_id": int(sequence_id),
                    "valid_row_group": group,
                    "labels_loaded": False,
                    "required_outputs_finite": True,
                    "reset_replay_bitwise_equal": True,
                    "first_pass_callback_ns": int(first_ns),
                    "replay_callback_ns": int(second_ns),
                    "prediction_sha256": sha256_bytes(first_bytes),
                }
                print(
                    json.dumps(
                        {
                            "phase": "callback_fixture",
                            "fold": fold,
                            "role": role,
                            "sequence_id": int(sequence_id),
                            "utc": datetime.now(UTC).isoformat(),
                        }
                    ),
                    flush=True,
                )
                del first, second, first_bytes, second_bytes
            del table, features, sequence

    cgroup = consumer.read_cgroup_limits()
    if cgroup["cpu_quota"] > cgroup["cpu_period"]:
        raise RuntimeError("fixture cgroup exceeded the one-core limit")
    if cgroup["memory_max_bytes"] > 650_000_000 or cgroup["memory_swap_max_bytes"] != 0:
        raise RuntimeError("fixture cgroup memory/swap limit differs from its admission")
    if "wnn-campaign.slice" not in cgroup["path"].split("/"):
        raise RuntimeError("fixture process is outside the campaign cgroup")
    receipt = {
        "schema": "wnn.native.single-member-cleanfold-q8-callback-fixture.r5",
        "attempt_id": "native-q8-single-model-controls-r5-py311-callback-fixture-r3",
        "status": "PASS_ACTUAL_SINGLE_MEMBER_Q8_RESET_REPLAY",
        "created_utc": datetime.now(UTC).isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "executable": str(Path(sys.executable).resolve()),
            "prefix": str(Path(sys.prefix).resolve()),
            "packages": package_versions,
            "torch_imported": False,
        },
        "r3_contract_sha256": sha256_file(r3_path),
        "r3_input_preflight_sha256": sha256_file(r3_preflight_path),
        "r4_contract_sha256": r4_sha,
        "r4_consumer_sha256": r4_consumer_sha,
        "r5_callback_runtime_sha256": r5_runtime_sha,
        "valid_dataset": {
            "repo_path": r3["valid_dataset"]["repo_path"],
            "bytes": data_stat.st_size,
            "footer_sha256_including_8_byte_trailer": identity[
                "footer_sha256_including_8_byte_trailer"
            ],
            "footer_matches_r3_preflight": True,
            "schema_validation": metadata_validation,
            "mutated_footer_hash_rejected": footer_drift_rejected,
            "mutated_schema_names_hash_rejected": schema_drift_rejected,
            "row_data_columns": [
                "seq_ix",
                "step_in_seq",
                "need_prediction",
                *utils.FEATURE_COLUMNS,
            ],
            "label_columns_loaded": False,
        },
        "native_elf_sha256": r4_contract["runtime_assets"]["native_elf"]["sha256"],
        "weights_archive_sha256": r4_contract["weights"]["archive_sha256"],
        "folds": results,
        "cgroup": cgroup,
        "elapsed_wall_seconds_including_predictor_construction": time.monotonic() - started,
        "full_fold_score_performed": False,
        "full_fold_rows_scored": 0,
        "provider_or_gpu_action": False,
    }
    out = Path(__file__).with_name("CALLBACK_FIXTURE_SINGLE_R5_PY311_R3.json")
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
