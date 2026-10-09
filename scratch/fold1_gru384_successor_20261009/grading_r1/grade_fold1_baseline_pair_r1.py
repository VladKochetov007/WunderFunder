#!/usr/bin/env python3
"""Grade the frozen fold-1 seed3/seed4/raw equal-mean baseline endpoints.

This is a one-shot, preregistered diagnostic. It reads only selected labels from
the 457 frozen validation row groups and uses the official global WP metric and
the repository's paired whole-component bootstrap implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, WARMUP, GlobalAccumulator  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

SPEC_REL = Path(
    "scratch/fold1_gru384_successor_20261009/grading_r1/FROZEN_BASELINE_PAIR_R1.json"
)
SPEC_SHA256 = "f6de6f79fad019300ce95a5bcd96fadc1fdeaceae19ae272be76f6d88258428b"
SEED3_AUDIT_REL = Path(
    "scratch/fold1_gru384_successor_20261009/recovery_r3/RESUME_TERMINAL_AUDIT_R3_R4.json"
)
SEED4_AUDIT_REL = Path(
    "scratch/fold1_gru384_successor_20261009/recovery_r3/SEED4_TERMINAL_AUDIT_LUNA_R1.json"
)
ROWS_PER_GROUP = 20_000


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    """Fail closed with a concise diagnostic."""
    if not condition:
        raise ValueError(message)


def validate_spec(root: Path, spec_path: Path) -> tuple[dict[str, Any], str]:
    """Bind to the root-owned preregistration before opening result arrays."""
    require(spec_path.resolve() == (root / SPEC_REL).resolve(), "unexpected frozen spec path")
    observed_sha = sha256_file(spec_path)
    require(observed_sha == SPEC_SHA256, "frozen baseline-pair spec SHA-256 changed")
    spec = read_json(spec_path)
    require(spec.get("schema") == "wnn.fold1-baseline-pair.preregistration.r1", "wrong spec schema")
    require(spec.get("status") == "frozen_before_score_inspection", "spec was not frozen before scoring")
    require(spec.get("endpoints") == [
        "seed3",
        "seed4",
        "equal_raw_prediction_mean_fp64_cast_fp32_then_metric_clip",
    ], "endpoint set/order differs from preregistration")
    require(spec.get("primary_contrast") == "equal_pair_minus_seed3", "primary contrast changed")
    require(spec.get("metric") == METRIC_ID == "global_wp_clip2_v1", "official metric mismatch")
    require(WARMUP == 99, "official warmup constant changed")
    bootstrap = spec.get("bootstrap", {})
    require(
        bootstrap == {
            "unit": "whole_connected_component",
            "paired": True,
            "replicates": 2000,
            "seed": 271828,
        },
        "bootstrap differs from frozen preregistration",
    )
    selection = spec.get("selection_budget", {})
    require(
        selection == {"endpoints": 3, "weight_search": False, "epoch_search": False},
        "selection budget differs from frozen preregistration",
    )
    return spec, observed_sha


def validate_prior_audits(root: Path, members: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Require the independently recorded terminal fit/checkpoint audits."""
    seed3 = read_json(root / SEED3_AUDIT_REL)
    seed4 = read_json(root / SEED4_AUDIT_REL)
    require(seed3.get("status") == "recovery_fit_and_output_passed", "seed3 terminal audit did not pass")
    require(seed4.get("verdict") == "fit_integrity_pass_with_unrelated_controller_inventory_drift",
            "seed4 independent terminal audit did not pass")
    by_run = {member["run"]: member for member in members}
    seed3_member = by_run["bd26_w384raw_s3_f1_resume_r3"]
    seed4_member = by_run["bd26_w384raw_s4_f1"]
    require(
        seed3.get("output_audit", {}).get("checkpoint_sha256") == seed3_member["checkpoint_sha256"],
        "seed3 checkpoint differs from its terminal audit",
    )
    require(
        seed4.get("artifacts", {}).get("model_ep3_pt", {}).get("sha256")
        == seed4_member["checkpoint_sha256"],
        "seed4 checkpoint differs from its independent terminal audit",
    )
    return {
        "seed3": {"path": str(SEED3_AUDIT_REL), "sha256": sha256_file(root / SEED3_AUDIT_REL)},
        "seed4": {"path": str(SEED4_AUDIT_REL), "sha256": sha256_file(root / SEED4_AUDIT_REL)},
    }


def validate_producer_membership(
    root: Path, member: dict[str, Any], split: Any, split_sha: str
) -> dict[str, Any]:
    """Bind provenance fitting rows to the source-qualified campaign split."""
    run = str(member["run"])
    run_dir = root / "artifacts" / "runs" / run
    checkpoint = run_dir / "model_ep3.pt"
    provenance_path = run_dir / "provenance.json"
    checkpoint_sha = sha256_file(checkpoint)
    require(checkpoint_sha == member["checkpoint_sha256"], f"{run}: checkpoint hash mismatch")
    provenance = read_json(provenance_path)
    require(provenance.get("selected_epoch") == 3, f"{run}: selected epoch is not fixed at 3")
    require(provenance.get("seed") == (3 if "s3_" in run else 4), f"{run}: seed mismatch")
    require(provenance.get("split_manifest_sha256") == split_sha, f"{run}: split hash mismatch")
    args = provenance.get("args", {})
    expected_args = {
        "name": run,
        "device": "cuda",
        "arch": "gru",
        "hidden": 384,
        "layers": 2,
        "raw_only": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "train_source": "train+valid",
        "train_seqs": 10607,
        "valid_fold": 1,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "batch": 16,
        "chunk": 2000,
        "lr": 0.001,
        "wd": 0.01,
        "optim": "adamw",
        "sched": "onecycle",
        "precision": "fp16",
        "save_training_state": True,
    }
    mismatches = {key: (args.get(key), value) for key, value in expected_args.items() if args.get(key) != value}
    require(not mismatches, f"{run}: trainer args differ from the audited recipe: {mismatches}")
    require(
        provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") == checkpoint_sha,
        f"{run}: provenance checkpoint pin mismatch",
    )
    fit = provenance.get("fit_row_groups", {})
    train = list(map(int, fit.get("train", [])))
    valid = list(map(int, fit.get("valid", [])))
    evaluation = list(map(int, provenance.get("evaluation_row_groups", [])))
    require(len(train) == len(set(train)) and len(valid) == len(set(valid)), f"{run}: duplicate fit groups")
    require(len(evaluation) == len(set(evaluation)), f"{run}: duplicate evaluation groups")
    require(set(train) == set(split.train_groups), f"{run}: train membership differs from frozen split")
    require(set(valid) == set(split.valid_groups), f"{run}: validation fit membership differs from frozen split")
    require(set(evaluation) == set(split.evaluation_groups), f"{run}: evaluation membership differs from frozen split")
    split.assert_fit_groups(train, valid)
    component_of = {(source, group): component for source, group, component in split.components}
    fitted_components = {component_of[("train", group)] for group in train}
    fitted_components.update(component_of[("valid", group)] for group in valid)
    evaluated_components = {
        component_of[("valid", group)] for group in split.evaluation_groups
    }
    require(not fitted_components.intersection(evaluated_components), f"{run}: fit/evaluation component leakage")

    training_state = run_dir / "training_state.pt"
    provenance_summary = {
        "path": str(provenance_path.relative_to(root)),
        "sha256": sha256_file(provenance_path),
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": checkpoint_sha,
        "training_state": str(training_state.relative_to(root)),
        "training_state_sha256": sha256_file(training_state),
        "train_group_count": len(train),
        "valid_fit_group_count": len(valid),
        "evaluation_group_count": len(evaluation),
        "fit_evaluation_component_overlap": 0,
        "fit_membership_validation": "exact sets against source-qualified fold1 manifest",
        "resumed_from": provenance.get("resumed_from"),
    }
    return provenance_summary


def validate_prediction_artifact(
    root: Path, member: dict[str, Any]
) -> tuple[np.memmap, dict[str, Any]]:
    """Hash-bind one FP32 prediction map and its row-group sidecar."""
    prediction_path = root / member["prediction"]
    sidecar_path = prediction_path.with_suffix(".json")
    prediction_sha = sha256_file(prediction_path)
    sidecar_sha = sha256_file(sidecar_path)
    require(prediction_sha == member["prediction_sha256"], f"{member['run']}: prediction hash mismatch")
    require(sidecar_sha == member["sidecar_sha256"], f"{member['run']}: sidecar hash mismatch")
    sidecar = read_json(sidecar_path)
    require(sidecar.get("run") == member["run"], f"{member['run']}: sidecar run mismatch")
    require(sidecar.get("split") == "valid", f"{member['run']}: sidecar split mismatch")
    require(sidecar.get("epoch") == member["epoch"] == 3, f"{member['run']}: sidecar epoch mismatch")
    require(sidecar.get("checkpoint_sha256") == member["checkpoint_sha256"],
            f"{member['run']}: sidecar checkpoint hash mismatch")
    require(sidecar.get("prediction_sha256") == prediction_sha, f"{member['run']}: sidecar prediction hash mismatch")
    require(sidecar.get("split_manifest_sha256") == "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae",
            f"{member['run']}: sidecar split-manifest hash mismatch")
    groups = list(map(int, sidecar.get("row_groups", [])))
    require(groups == list(range(1873)), f"{member['run']}: prediction row-group order is not exact")
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    require(prediction.dtype == np.float32, f"{member['run']}: predictions are not FP32 arrays")
    require(prediction.shape == (1873, ROWS_PER_GROUP, 2), f"{member['run']}: unexpected prediction shape")
    return prediction, {
        "prediction_path": str(prediction_path.relative_to(root)),
        "prediction_sha256": prediction_sha,
        "prediction_bytes": prediction_path.stat().st_size,
        "sidecar_path": str(sidecar_path.relative_to(root)),
        "sidecar_sha256": sidecar_sha,
        "checkpoint_sha256": member["checkpoint_sha256"],
        "epoch": 3,
        "inference_precision": sidecar.get("inference_precision"),
        "dtype": str(prediction.dtype),
        "shape": list(prediction.shape),
        "row_group_order_sha256": hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest(),
    }


def grade(root: Path, spec_path: Path, expected_spec_sha256: str, deadline_seconds: float) -> dict[str, Any]:
    """Validate all bindings, stream scored labels, calculate frozen endpoints."""
    started = time.monotonic()
    deadline = started + deadline_seconds
    spec, spec_sha = validate_spec(root, spec_path)
    require(spec_sha == expected_spec_sha256 == SPEC_SHA256, "caller spec pin does not match frozen spec")
    split_path = root / spec["split"]
    split = load_campaign_split(split_path, root)
    require(split.sha256 == spec["split_sha256"], "loaded campaign split hash differs from frozen spec")
    require(split.fold == 1, "loaded campaign split is not fold1")
    groups = list(map(int, split.evaluation_groups))
    require(len(groups) == spec["evaluation_groups"] == 457, "fold1 evaluation group count mismatch")
    component_of = {
        (source, group): int(component) for source, group, component in split.components
    }
    components = np.asarray([component_of[("valid", group)] for group in groups], dtype=np.int64)
    require(len(np.unique(components)) == spec["evaluation_components"] == 149,
            "fold1 evaluation component count mismatch")

    prior_audits = validate_prior_audits(root, spec["members"])
    provenance_receipts = {
        member["run"]: validate_producer_membership(root, member, split, split.sha256)
        for member in spec["members"]
    }
    predictions_by_run: dict[str, np.memmap] = {}
    artifact_bindings: dict[str, dict[str, Any]] = {}
    for member in spec["members"]:
        mapped, identity = validate_prediction_artifact(root, member)
        predictions_by_run[member["run"]] = mapped
        artifact_bindings[member["run"]] = identity

    parquet_path = root / "datasets/valid.parquet"
    parquet = pq.ParquetFile(parquet_path)
    require(parquet.num_row_groups == 1873, "valid parquet row-group count changed")
    manifest = read_json(split_path)
    seq_by_group = {
        int(node["row_group"]): int(node["seq_ix"])
        for node in manifest["sequences"]["valid"]
    }
    require(len(seq_by_group) == 1873, "source-qualified valid sequence manifest is incomplete")
    require(set(groups).issubset(seq_by_group), "evaluation row group is absent from valid membership")

    run3 = "bd26_w384raw_s3_f1_resume_r3"
    run4 = "bd26_w384raw_s4_f1"
    scores = {name: GlobalAccumulator() for name in ("seed3", "seed4", "equal_pair")}
    blend_moments: list[np.ndarray] = []
    seed3_moments: list[np.ndarray] = []
    selected_rows_total = 0
    selected_groups = 0
    group_order_digest = hashlib.sha256()
    for group in groups:
        if time.monotonic() >= deadline:
            raise TimeoutError("grading deadline reached before all frozen evaluation groups were scored")
        table = parquet.read_row_group(
            group,
            columns=["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"],
            use_threads=False,
        )
        require(table.num_rows == ROWS_PER_GROUP, f"row group {group} has unexpected length")
        seq_ix = table.column("seq_ix").to_numpy(zero_copy_only=False)
        require(np.all(seq_ix == seq_by_group[group]), f"row group {group} seq_ix differs from source-qualified manifest")
        steps = table.column("step_in_seq").to_numpy(zero_copy_only=False)
        expected_steps = np.arange(ROWS_PER_GROUP, dtype=steps.dtype)
        require(np.array_equal(steps, expected_steps), f"row group {group} step order is not exact")
        need_prediction = table.column("need_prediction").to_numpy(zero_copy_only=False).astype(bool, copy=False)
        expected_need = np.arange(ROWS_PER_GROUP) >= WARMUP
        require(np.array_equal(need_prediction, expected_need), f"row group {group} prediction mask differs from warmup contract")
        is_scored = table.column("is_scored").to_numpy(zero_copy_only=False).astype(bool, copy=False)
        mask = need_prediction & is_scored
        targets = np.stack(
            (
                table.column("t0").to_numpy(zero_copy_only=False),
                table.column("t1").to_numpy(zero_copy_only=False),
            ),
            axis=-1,
        ).astype(np.float32, copy=False)

        p3 = np.asarray(predictions_by_run[run3][group], dtype=np.float32)
        p4 = np.asarray(predictions_by_run[run4][group], dtype=np.float32)
        equal_pair = ((p3.astype(np.float64) + p4.astype(np.float64)) * 0.5).astype(np.float32)
        scores["seed3"].add(targets[None, ...], p3[None, ...], mask[None, ...])
        scores["seed4"].add(targets[None, ...], p4[None, ...], mask[None, ...])
        scores["equal_pair"].add(targets[None, ...], equal_pair[None, ...], mask[None, ...])
        pair, reference = paired_component_moments(
            targets[None, ...], equal_pair[None, ...], p3[None, ...], mask[None, ...]
        )
        blend_moments.append(pair[0])
        seed3_moments.append(reference[0])
        selected_rows_total += int(mask.sum())
        selected_groups += 1
        group_order_digest.update(int(group).to_bytes(8, "little", signed=True))

    require(selected_groups == 457, "not every frozen group was scored")
    require(selected_rows_total > 0, "no official scored rows were selected")
    score_results = {name: accumulator.result() for name, accumulator in scores.items()}
    bootstrap_spec = spec["bootstrap"]
    comparison = paired_component_bootstrap(
        np.stack(blend_moments),
        np.stack(seed3_moments),
        components,
        n_boot=int(bootstrap_spec["replicates"]),
        seed=int(bootstrap_spec["seed"]),
        centered=True,
    )
    require(comparison["n_sequences"] == 457, "bootstrap sequence count differs")
    require(comparison["n_components"] == 149, "bootstrap component count differs")
    require(comparison["metric_id"] == METRIC_ID, "bootstrap metric differs")
    require(abs(float(comparison["candidate_wp"]) - float(score_results["equal_pair"]["wp"])) < 1e-12,
            "bootstrap equal-pair score disagrees with official streaming score")
    require(abs(float(comparison["reference_wp"]) - float(score_results["seed3"]["wp"])) < 1e-12,
            "bootstrap seed3 score disagrees with official streaming score")

    out_dir = root / "scratch/fold1_gru384_successor_20261009/grading_r1"
    moment_path = out_dir / "BASELINE_PAIR_R1.moments.npz"
    json_path = out_dir / "BASELINE_PAIR_R1.json"
    md_path = out_dir / "BASELINE_PAIR_R1.md"
    require(not any(path.exists() for path in (moment_path, json_path, md_path)),
            "refusing to overwrite a prior grade artifact")
    np.savez_compressed(
        moment_path,
        row_groups=np.asarray(groups, dtype=np.int64),
        seq_ix=np.asarray([seq_by_group[group] for group in groups], dtype=np.int64),
        components=components,
        equal_pair_moments=np.stack(blend_moments),
        seed3_moments=np.stack(seed3_moments),
    )
    source_paths = [
        Path(__file__),
        ROOT / "src/wnn/splits.py",
        ROOT / "src/wnn/gmetric.py",
        ROOT / "src/wnn/evaluation.py",
    ]
    source_hashes = {str(path.relative_to(root)): sha256_file(path) for path in source_paths}
    result = {
        "schema": "wnn.fold1-baseline-pair.grade.r1",
        "status": "evaluated",
        "created_utc": datetime.now(UTC).isoformat(),
        "grader": str(Path(__file__).relative_to(root)),
        "grader_sha256": source_hashes[str(Path(__file__).relative_to(root))],
        "frozen_spec": {"path": str(SPEC_REL), "sha256": spec_sha},
        "prior_terminal_audits": prior_audits,
        "source_hashes": source_hashes,
        "split": {
            "path": str(split_path.relative_to(root)),
            "sha256": split.sha256,
            "fold": split.fold,
            "evaluation_groups": selected_groups,
            "evaluation_components": int(len(np.unique(components))),
            "source_qualified_membership": "valid row_group + seq_ix + connected_component from frozen fold1 manifest",
            "group_order_sha256_little_endian_i64": group_order_digest.hexdigest(),
            "selected_scored_rows": selected_rows_total,
            "historical_exposure": spec["exposure"],
        },
        "producer_provenance": provenance_receipts,
        "prediction_artifacts": artifact_bindings,
        "endpoints": {
            "seed3": score_results["seed3"],
            "seed4": score_results["seed4"],
            "equal_raw_prediction_mean_fp64_cast_fp32_then_metric_clip": score_results["equal_pair"],
        },
        "primary_contrast": {
            "name": "equal_pair_minus_seed3",
            "method": "paired 2,000-resample whole-connected-component bootstrap over frozen fold1 components",
            **comparison,
        },
        "selection": {
            "endpoints": 3,
            "weight_search": False,
            "epoch_search": False,
            "blend": "raw FP32 predictions promoted to FP64, averaged 1:1, cast to FP32; official metric clips afterward",
        },
        "claim_limit": spec["claim_limit"],
        "expected_public_score": None,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "moments_artifact": {
            "path": str(moment_path.relative_to(root)),
            "sha256": sha256_file(moment_path),
            "bytes": moment_path.stat().st_size,
            "contains": "source-qualified row group, seq_ix, component IDs, centered moments for equal pair and seed3",
        },
    }
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(json_tmp, json_path)
    markdown = render_markdown(result)
    md_tmp = md_path.with_suffix(".md.tmp")
    md_tmp.write_text(markdown)
    os.replace(md_tmp, md_path)
    result["grade_json_path"] = str(json_path.relative_to(root))
    result["grade_json_sha256"] = sha256_file(json_path)
    return result


def render_markdown(result: dict[str, Any]) -> str:
    """Render concise, reviewable grade output."""
    endpoints = result["endpoints"]
    delta = result["primary_contrast"]
    ci_low, ci_high = delta["ci95"]
    lines = [
        "# Fold-1 seed3/seed4 baseline-pair grade R1",
        "",
        f"Frozen spec SHA-256: `{result['frozen_spec']['sha256']}`.",
        f"Metric: `{METRIC_ID}`; fold 1 has {result['split']['evaluation_groups']} exposed groups and "
        f"{result['split']['evaluation_components']} connected components ({result['split']['selected_scored_rows']:,} selected rows).",
        "",
        "| Endpoint | WP | t0 | t1 |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("seed3", "Seed 3"),
        ("seed4", "Seed 4"),
        ("equal_raw_prediction_mean_fp64_cast_fp32_then_metric_clip", "Raw equal mean (1:1)"),
    ):
        score = endpoints[name]
        lines.append(f"| {label} | {score['wp']:.9f} | {score['t0']:.9f} | {score['t1']:.9f} |")
    lines.extend(
        [
            "",
            "Primary contrast: equal mean minus seed 3.",
            f"Paired whole-component bootstrap: {delta['mean_diff']:+.9f} WP; 95% percentile CI "
            f"[{ci_low:+.9f}, {ci_high:+.9f}] across {delta['n_components']} components "
            f"({delta['n_boot']} replicates, seed {delta['seed']}).",
            "",
            "This is a baseline diagnostic on a historically inspected fold, not a fresh holdout, "
            "new-best claim, or public-score estimate. Component resampling does not include "
            "training-seed variation or public transfer.",
            "",
            f"Machine-readable result: `{result['grade_json_path']}` ({result['grade_json_sha256']}).",
            f"Centered moments: `{result['moments_artifact']['path']}` ({result['moments_artifact']['sha256']}).",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--spec", type=Path, default=ROOT / SPEC_REL)
    parser.add_argument("--expected-spec-sha256", default=SPEC_SHA256)
    parser.add_argument("--deadline-seconds", type=float, default=170.0)
    args = parser.parse_args()
    result = grade(args.root.resolve(), args.spec.resolve(), args.expected_spec_sha256, args.deadline_seconds)
    print(json.dumps({
        "grade_json_path": result["grade_json_path"],
        "grade_json_sha256": result["grade_json_sha256"],
        "scores": result["endpoints"],
        "primary_contrast": result["primary_contrast"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
