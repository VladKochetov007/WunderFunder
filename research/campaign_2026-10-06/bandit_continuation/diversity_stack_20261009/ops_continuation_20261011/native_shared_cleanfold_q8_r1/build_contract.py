#!/usr/bin/env python3
"""Freeze the two-fold native-Q8 shared-output evaluation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = Path(__file__).resolve().parent
SPLIT_ROOT = MAIN / "research/campaign_2026-09-29/artifacts/splits"
SHARED_ROOT = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    / "ops_continuation_20261011/cpu_shared_backbone_r1"
)
PANEL_ROOT = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1"
    / "native_f1_precision_panel_r1"
)
RUNNER = Path(
    "/home/vlad/development/wunder/wnn-agent-worktrees/zen4-native-cpu-optimization-20261012"
    "/scripts/benchmark_submission_callbacks.py"
)
PY311 = Path("/tmp/wnn-projection-precompute-r1/py311/bin/python")
NATIVE_DIR = Path("/tmp/wnn-shared-f0-py311-package-r1")
WHEEL = Path(
    "/tmp/wnn-shared-f0-py311-wheel-r1/wnn_native-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl"
)
ELF = NATIVE_DIR / "wnn_native/wnn_native.abi3.so"

MODEL_PINS: dict[int, dict[str, dict[str, Any]]] = {
    0: {
        "student": {
            "run": "oldz4_h230_student_f0_s3_r3",
            "epoch": 1,
            "checkpoint_sha256": "664643b2653a85d192f74a976d76afa933ae56c02b2621387d2f576fd61695eb",
            "provenance_sha256": "5af59cf6dd5ed263e45560d164f99c6103301076908c297f1dab3c645e3df13c",
            "fit_source": "train+valid",
        },
        "hft": {
            "run": "oldz4_h230_hft_f0_s3_r5",
            "epoch": 0,
            "checkpoint_sha256": "2bde556119ededc3693733280444a42f0f64138a2e3759bed6b068cb72f18ec8",
            "provenance_sha256": "bf6105bfb1d137345bf775b09f06f432b5d993c61cf019aa927c2a896ebddac2",
            "fit_source": "valid",
        },
    },
    1: {
        "student": {
            "run": "oldz4_h230_student_f1_s3_r4",
            "epoch": 1,
            "checkpoint_sha256": "cd8886589e8664082da1f27aee4f2c4925b61c41da8c740598831ab2f774bbd6",
            "provenance_sha256": "8aa0e2af944a52a34d0c0622705071e7ca60a978ec429dc71269fa67b40354da",
            "fit_source": "train+valid",
        },
        "hft": {
            "run": "oldz4_h230_hft_f1_s3_r2",
            "epoch": 0,
            "checkpoint_sha256": "41e7aa3fb27024f2612f8f6b8f4232ca446ba92022dbf7ba90893ffee2292410",
            "provenance_sha256": "468ac58e9f59e9f32ea425f8556b26073e61b108eafa468ccf06f1cd0d3d1b9d",
            "fit_source": "valid",
        },
    },
}

SUPPORT_RECEIPTS = {
    "f0_shared_pair_preflight": (
        SHARED_ROOT / "PRODUCER_PREFLIGHT_R1.json",
        "3e6f0d13f98a6fdd3b4730b79ea733831af79323213b7c4e8dd541668c2255d9",
    ),
    "f0_fixed_half_grade": (
        MAIN / "research/campaign_2026-10-06/bandit_continuation/runtime/receipts/"
        "oldz4-r5-f0-s3-hft-export/F0_STUDENT_HFT_FIXED_HALF_GRADE_R5.json",
        "953cfb19c2fd15cec68569fd5c29a29520261706e33f2a142417fccd30040e1b",
    ),
    "f1_student_cp1_gate": (
        MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        "ops_continuation_20261011/oldz4_f1_core_hft_r2/ROOT_F1_STUDENT_CP1_TERMINAL_GATE_R2.json",
        "cb483a87aab7b902f8d73ac9b1d0651d5f5ddbf3b3cb776eaef372c50ac7a4e7",
    ),
    "f1_hft_resource_only_admission": (
        MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        "ops_continuation_20261011/oldz4_f1_hft_resource_fit_r3/"
        "ROOT_F1_HFT_RESOURCE_ONLY_FIT_ADMISSION_R3.json",
        "a147ddf216082bf58195d7479d346b930a74e891cd1c71cea06b393200198912",
    ),
    "f1_source_qualified_cache_review": (
        MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        "ops_continuation_20261011/old_z4_f1_pipeline_successor_20261012_r4/"
        "ROOT_ACTUAL_F1_CACHE_CONSUMER_R1.json",
        "8cafeeea7e3ac2453907fa248a60ff4b7ee824dca869ff4e03918f550ed1aafc",
    ),
}


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON bytes for a fixed ordered integer list or receipt."""
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def parquet_footer_sha256(path: Path, footer_bytes: int) -> str:
    """Hash the serialized footer and eight-byte Parquet trailer."""
    remaining = footer_bytes + 8
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(-remaining, 2)
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError("truncated Parquet footer range")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    """Require one existing file to match a frozen SHA-256 digest."""
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"frozen input hash mismatch: {path}")


def make_fold(fold: int) -> dict[str, Any]:
    """Bind producer checkpoints and their source-qualified outer-fold split."""
    split_path = SPLIT_ROOT / f"fold{fold}.json"
    split_bytes = split_path.read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    split = json.loads(split_bytes)
    if split.get("fold") != fold:
        raise ValueError(f"fold {fold} split manifest names another fold")
    if len(split["sequences"]["valid"]) != 1873:
        raise ValueError(f"fold {fold} split does not bind all valid sequences")
    group_ids = sorted(int(item) for item in split["evaluation"]["valid"])
    expected_count = 468 if fold == 0 else 457
    if len(group_ids) != expected_count or len(set(group_ids)) != expected_count:
        raise ValueError(f"fold {fold} outer-evaluation count or uniqueness mismatch")
    if set(group_ids) & set(split["fit"]["valid"]):
        raise ValueError(f"fold {fold} fit-valid and evaluation membership overlap")
    if set(group_ids) != set(split["excluded"]["valid"]):
        raise ValueError(f"fold {fold} exclusion and outer-evaluation membership differ")
    train_components = {
        int(item["component"])
        for item in split["sequences"]["train"]
        if int(item["row_group"]) in set(split["fit"]["train"])
    }
    valid_components = {
        int(item["component"])
        for item in split["sequences"]["valid"]
        if int(item["row_group"]) in set(split["fit"]["valid"])
    }
    evaluation_components = {
        int(item["component"])
        for item in split["sequences"]["valid"]
        if int(item["row_group"]) in set(group_ids)
    }
    fit_components = train_components | valid_components
    if fit_components & evaluation_components:
        raise ValueError(f"fold {fold} fit/evaluation component ancestry overlaps")
    if len(evaluation_components) != int(split["counts"]["evaluation_components"]):
        raise ValueError(f"fold {fold} evaluation component count differs from the split")

    valid_source = split["sources"]["datasets/valid.parquet"]
    valid_path = MAIN / "datasets/valid.parquet"
    if (
        valid_path.stat().st_size != valid_source["bytes"]
        or parquet_footer_sha256(valid_path, int(valid_source["footer_bytes"]))
        != valid_source["footer_sha256"]
    ):
        raise ValueError("current valid.parquet differs from split-manifest footer identity")

    models: dict[str, Any] = {}
    for role, pin in MODEL_PINS[fold].items():
        checkpoint = MAIN / "artifacts/runs" / pin["run"] / f"model_ep{pin['epoch']}.pt"
        provenance_path = MAIN / "artifacts/runs" / pin["run"] / "provenance.json"
        require_hash(checkpoint, pin["checkpoint_sha256"])
        require_hash(provenance_path, pin["provenance_sha256"])
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("split_manifest_sha256") != split_sha:
            raise ValueError(f"{pin['run']} provenance is bound to another split")
        if set(provenance.get("evaluation_row_groups", [])) != set(group_ids):
            raise ValueError(f"{pin['run']} evaluation membership differs from the split")
        if provenance.get("checkpoint_sha256", {}).get(checkpoint.name) != pin["checkpoint_sha256"]:
            raise ValueError(f"{pin['run']} checkpoint identity differs from provenance")
        fit = provenance.get("fit_row_groups", {})
        if role == "student":
            if set(fit.get("train", [])) != set(split["fit"]["train"]):
                raise ValueError(f"{pin['run']} training membership differs from split")
            if set(fit.get("valid", [])) != set(split["fit"]["valid"]):
                raise ValueError(f"{pin['run']} validation membership differs from split")
        else:
            if fit.get("train") != [] or set(fit.get("valid", [])) != set(split["fit"]["valid"]):
                raise ValueError(f"{pin['run']} HFT was not direct valid-only fit")
            if provenance.get("args", {}).get("train_source") != "valid":
                raise ValueError(f"{pin['run']} HFT provenance is not valid-only")
            if provenance.get("args", {}).get("mask_weight") is not True:
                raise ValueError(f"{pin['run']} HFT mask-weight policy differs")
        if (
            provenance.get("args", {}).get("hidden") != 256
            or provenance.get("args", {}).get("layers") != 2
        ):
            raise ValueError(f"{pin['run']} model dimensions differ from the pair contract")
        models[role] = {
            **pin,
            "checkpoint_path": str(checkpoint),
            "provenance_path": str(provenance_path),
            "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
            "fit_membership_identity": provenance.get("fit_membership_identity"),
        }

    return {
        "fold": fold,
        "exposure": "historically exposed source-qualified outer-fold evaluation; not a fresh holdout",
        "split_manifest": {
            "path": str(split_path),
            "sha256": split_sha,
            "evaluation_valid_count": expected_count,
            "evaluation_row_groups_sorted": group_ids,
            "evaluation_row_groups_sorted_canonical_sha256": canonical_sha256(group_ids),
            "fit_train_count": int(split["counts"]["fit_train"]),
            "fit_valid_count": int(split["counts"]["fit_valid"]),
            "excluded_train_count": int(split["counts"]["excluded_train"]),
            "fit_component_count": len(fit_components),
            "evaluation_component_count": len(evaluation_components),
            "fit_evaluation_component_overlap_count": 0,
        },
        "valid_dataset": {
            "path": str(valid_path),
            "bytes": int(valid_source["bytes"]),
            "footer_bytes": int(valid_source["footer_bytes"]),
            "footer_sha256_including_8_byte_trailer": valid_source["footer_sha256"],
            "row_groups": len(split["sequences"]["valid"]),
            "rows": len(split["sequences"]["valid"]) * 20_000,
        },
        "models": models,
    }


def main() -> None:
    """Write a create-only contract after validating current producer evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKET / "CONTRACT_R1.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen contract: {args.output}")

    support = {}
    for name, (path, digest) in SUPPORT_RECEIPTS.items():
        require_hash(path, digest)
        support[name] = {"path": str(path), "sha256": digest}

    shared_summary_path = SHARED_ROOT / "BENCHMARK_SUMMARY_R1.json"
    abi_path = SHARED_ROOT / "ELF_ABI_PY311_R1.json"
    review_path = SHARED_ROOT / "ROOT_SHARED_BACKBONE_REVIEW_R1.json"
    summary = json.loads(shared_summary_path.read_text())
    abi = json.loads(abi_path.read_text())
    native_hash = "f8aceef7322ee4deb3bf49992afcd0fec01ce5934a27d5269112a2052b2f5326"
    wheel_hash = "b4d71f3e06b51027d46bb74a7b2317405a69221f883e4c9c0efe2c8ea68f979f"
    if (
        not abi.get("passed")
        or abi["libraries"]["wnn_native/wnn_native.abi3.so"]["sha256"] != native_hash
    ):
        raise ValueError("Python 3.11 shared-backbone ELF ABI receipt differs")
    needed = abi["libraries"]["wnn_native/wnn_native.abi3.so"]["needed"]
    if any("libpython" in item.lower() for item in needed):
        raise ValueError("the frozen native extension must not depend on libpython")
    require_hash(ELF, native_hash)
    require_hash(WHEEL, wheel_hash)

    source_paths = {
        "checkpoint_loader": SHARED_ROOT / "benchmark_shared_f0.py",
        "shared_benchmark_summary": shared_summary_path,
        "shared_py311_abi": abi_path,
        "shared_backbone_review": review_path,
        "callback_runner": RUNNER,
        "project_utils": MAIN / "utils.py",
        "official_metric": MAIN / "src/wnn/gmetric.py",
        "split_producer": MAIN / "scripts/build_campaign_splits.py",
        "native_python_binding_reference": MAIN / "native/src/python.rs",
        "native_f1_panel_runner_reference": PANEL_ROOT / "run_native_f1_panel.py",
        "native_f1_panel_grader_reference": PANEL_ROOT / "grade_native_f1_panel.py",
        "packet_contract_builder": PACKET / "build_contract.py",
        "packet_validator": PACKET / "validate_packet.py",
        "weight_exporter": PACKET / "export_pair_weights.py",
        "quality_consumer": PACKET / "run_quality.py",
    }
    sources = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    expected_runner = "8bc57f520b9b284ca863fb8a798be3a0e923ad9242c0fb7ffa58abc2e2a0af53"
    if sources["callback_runner"]["sha256"] != expected_runner:
        raise ValueError("callback runner source differs from the measured callback runner")

    contract: dict[str, Any] = {
        "schema": "wnn.native.shared-cleanfold-q8-quality-contract.r1",
        "status": "prepared_pending_root_execution_admission",
        "candidate_id": "oldz4_f0_f1_student_hft_native_q8_shared_half_r1",
        "project_root": str(MAIN),
        "scope": {
            "claim": "native Q8 fixed-output blend quality on each full, source-qualified fold-0 and fold-1 outer evaluation partition",
            "folds_are_exposed": True,
            "fresh_holdout": False,
            "full_valid_in_sample_score": False,
            "training": False,
            "public_forecast": False,
            "submission_recommendation": False,
        },
        "endpoint": {
            "member_order": ["student", "hft"],
            "member_count": 2,
            "output_blend": "fixed arithmetic mean of the two members' two-target predictions",
            "coefficients": [0.5, 0.5],
            "coefficient_sum": 1.0,
            "parameter_averaging": False,
            "shared_backbone": True,
            "shared_backbone_default_in_native_api": False,
            "required_runtime_precision_prefix": "q8vnni",
            "precision_request": "q8vnni",
            "fast_activations": True,
            "budget_s": None,
        },
        "folds": {str(fold): make_fold(fold) for fold in (0, 1)},
        "metric": {
            "id": "global_wp_clip2_v1",
            "implementation_path": str(MAIN / "src/wnn/gmetric.py"),
            "mask": "need_prediction AND is_scored",
            "clip": [-2.0, 2.0],
            "weight": "absolute value of clipped target",
            "aggregation": "official pooled weighted Pearson per target, averaged equally across targets",
            "training_target_clipping_changed": False,
        },
        "callback_contract": {
            "sequence_rows": 20_000,
            "feature_columns": 112,
            "rows_advanced": "every row, including warmup rows where predict must return None",
            "reset": "native predictor reset before each complete sequence; sequence ID and step order checked",
            "evaluation_output_shape_per_fold": "(outer_eval_groups, 20000, 2)",
            "warmup_output_storage": "zero-filled; never scored because need_prediction is false",
            "processing": "one full row group at a time; no full-valid prediction or label array",
        },
        "runtime": {
            "python_executable": str(PY311),
            "python_requirement": "CPython 3.11 ABI3 wheel; exact actual interpreter checked at load",
            "wheel_path": str(WHEEL),
            "wheel_sha256": wheel_hash,
            "native_package_dir": str(NATIVE_DIR),
            "package_init_path": str(NATIVE_DIR / "wnn_native/__init__.py"),
            "package_init_sha256": sha256_file(NATIVE_DIR / "wnn_native/__init__.py"),
            "loaded_elf_path": str(ELF),
            "loaded_elf_sha256": native_hash,
            "static_needed": needed,
            "libpython_dependency_forbidden": True,
            "abi_receipt_path": str(abi_path),
            "abi_receipt_sha256": sha256_file(abi_path),
            "cpu_callback_measurement": {
                "host": summary["runtime_host"]["cpu_model"],
                "single_cpu_shared_q8_us_per_row_observed": [28.4798432225, 29.9806670075],
                "measured_scope": "10 complete feature-only sequences, repeated reset passes; no labels or full-fold quality",
                "full_fold_callback_rows": 18_500_000,
                "one_cpu_callback_seconds_at_30us_per_row": 555.0,
                "projection_at_50_percent_cpu_seconds": 1110.0,
                "proposed_total_wall_cap_seconds": 1800,
                "projection_is_not_a_runtime_guarantee_or_AMD_forecast": True,
            },
        },
        "producer_support_receipts": support,
        "source_bindings": sources,
        "execution_gate": {
            "required_root_receipt_schema": "wnn.root.native-shared-cleanfold-quality-admission.r1",
            "required_decision": "PASS_FOR_ONE_FROZEN_FULL_CLEANFOLD_Q8_EVALUATION",
            "required_receipt_bindings": [
                "contract_sha256",
                "weights_archive_sha256",
                "weights_receipt_sha256",
                "native_elf_sha256",
                "valid_dataset_footer_identity",
            ],
            "full_evaluation_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "contract": str(args.output),
                "sha256": sha256_file(args.output),
                "fold_counts": {
                    f: len(contract["folds"][f]["split_manifest"]["evaluation_row_groups_sorted"])
                    for f in ("0", "1")
                },
                "status": contract["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
