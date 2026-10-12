"""Validate the frozen F1 HFT export and recompute its predeclared blend check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011"
)
PACKET = OPS / "oldz4_f1_hft_export_r3"
REVIEW = Path(__file__).resolve().parent
PRED = MAIN / "cache/pred_oldz4_h230_hft_f1_s3_r2_valid.npy"
SIDECAR = PRED.with_suffix(".json")
PARTIAL = PRED.with_name(PRED.name + ".partial")
SEQSUMS = MAIN / "cache/seqsums_oldz4_h230_hft_f1_s3_r2.npz"
RESULT = PACKET / "CP0_EXPORT_RESULT_R3.json"
TERMINAL_GATE = PACKET / "ROOT_F1_HFT_TERMINAL_ADMISSION_R3.json"
EXPORT_GATE = PACKET / "ROOT_F1_HFT_EXPORT_ADMISSION_R3.json"
PRODUCER_REVIEW = PACKET / "ROOT_ACTUAL_PRODUCER_REVIEW_R3.json"
LAUNCH_RECEIPT = PACKET / "ROOT_CP0_EXPORT_DURABLE_LAUNCH_R3.json"
ATTEMPT = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/oldz4-r3-f1-s3-hft-export.json"
)
TARGETS = MAIN / "cache/valid_targets.npz"
SPLIT = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
SPLIT_SHA = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
BASELINE_EVIDENCE = OPS / "old_new_f1_fixed_half_20261012_r1/EVIDENCE_R1.json"
INTERACTION_CONTRACT = (
    OPS / "old_new_f1_fixed_half_20261012_r1/ROOT_EXTRA_SHARED_HEAD_INTERACTION_CONTRACT_R1.json"
)
COMMANDS = REVIEW / "COMMANDS_R1.txt"
CP_SHA = "41e7aa3fb27024f2612f8f6b8f4232ca446ba92022dbf7ba90893ffee2292410"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    receipt = json.loads(RESULT.read_text())
    terminal_gate = json.loads(TERMINAL_GATE.read_text())
    sidecar = json.loads(SIDECAR.read_text())
    pred = np.load(PRED, mmap_mode="r", allow_pickle=False)
    assert not PARTIAL.exists() and not SEQSUMS.exists()
    assert receipt["status"] == "PASS_ACTUAL_CP0_EXPORT"
    assert receipt["all_predictions_finite"] is True
    assert receipt["checkpoint_sha256"] == CP_SHA
    assert receipt["prediction_sha256"] == sha256(PRED)
    assert receipt["sidecar_sha256"] == sha256(SIDECAR)
    assert pred.shape == (1873, 20000, 2) and pred.dtype == np.dtype("float32")
    assert PRED.stat().st_size == 299_680_128
    assert sidecar["run"] == "oldz4_h230_hft_f1_s3_r2"
    assert sidecar["split"] == "valid" and sidecar["epoch"] == 0
    assert sidecar["checkpoint_sha256"] == CP_SHA
    assert sidecar["prediction_sha256"] == receipt["prediction_sha256"]
    assert sidecar["split_manifest_sha256"] == SPLIT_SHA
    assert sidecar["inference_precision"] == "fp16"
    with np.load(TARGETS, allow_pickle=False) as cache:
        target_groups = cache["row_group"]
        assert target_groups.dtype == np.dtype("int64")
        assert np.array_equal(target_groups, np.arange(1873, dtype=np.int64))
        y = np.asarray(cache["y"], dtype=np.float32)
        mask = np.asarray(cache["mask"], dtype=bool)
    assert np.array_equal(np.asarray(sidecar["row_groups"], dtype=np.int64), target_groups)
    for start in range(0, len(pred), 32):
        assert np.isfinite(pred[start : start + 32]).all()

    split = json.loads(SPLIT.read_text())
    groups = [int(group) for group in split["evaluation"]["valid"]]
    positions = {int(group): index for index, group in enumerate(target_groups)}
    index = np.asarray(sorted(positions[group] for group in groups), dtype=np.int64)
    ordered_groups = target_groups[index]
    components_by_group = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    components = np.asarray([components_by_group[int(group)] for group in ordered_groups])
    assert len(index) == 457 and len(np.unique(components)) == 149
    y_eval = y[index]
    mask_eval = mask[index] & (np.arange(y.shape[1]) >= 99)[None, :]

    names = {
        "old_student": MAIN / "cache/pred_oldz4_h230_student_f1_s3_r4_valid.npy",
        "hft": PRED,
        "strong_student": MAIN / "cache/pred_vt_f1_dsq_dm_t3_a9_e6_valid.npy",
    }
    input_sidecars = {
        key: json.loads(path.with_suffix(".json").read_text()) for key, path in names.items()
    }
    arrays = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in names.items()}
    for key, path in names.items():
        meta = input_sidecars[key]
        assert arrays[key].shape == (1873, 20000, 2) and arrays[key].dtype == np.dtype("float32")
        assert meta["run"] == path.name.removeprefix("pred_").removesuffix("_valid.npy")
        assert meta["split"] == "valid" and meta["split_manifest_sha256"] == SPLIT_SHA
        assert meta["row_groups"] == target_groups.tolist()
    old = np.asarray(arrays["old_student"][index], dtype=np.float32)
    hft = np.asarray(arrays["hft"][index], dtype=np.float32)
    strong = np.asarray(arrays["strong_student"][index], dtype=np.float32)
    candidate = (old + hft + 2.0 * strong) / 4.0
    reference = (old + strong) / 2.0

    import sys

    sys.path.insert(0, str(MAIN / "src"))
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments

    candidate_moments, reference_moments = paired_component_moments(
        y_eval, candidate, reference, mask_eval
    )
    bootstrap = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        components,
        n_boot=800,
        seed=271828,
        centered=True,
    )
    score_lines = (REVIEW / "BLEND_SCORE_STDOUT_R1.txt").read_text().splitlines()
    scores = {}
    for line in score_lines:
        fields = line.split()
        scores[fields[-1]] = {
            "wp": float(fields[0]),
            "t0": float(fields[2]),
            "t1": float(fields[4]),
        }
    old_name = "oldz4_h230_student_f1_s3_r4"
    hft_name = "oldz4_h230_hft_f1_s3_r2"
    strong_name = "vt_f1_dsq_dm_t3_a9_e6"
    expected_candidate = f"{old_name}:1+{hft_name}:1+{strong_name}:2"
    expected_reference = f"{old_name}:1+{strong_name}:1"
    assert abs(bootstrap["candidate_wp"] - scores[expected_candidate]["wp"]) < 1e-5
    assert abs(bootstrap["reference_wp"] - scores[expected_reference]["wp"]) < 1e-5
    attempt = json.loads(ATTEMPT.read_text())
    result = {
        "schema": "wnn.oldz4-f1-hft-export-review.r1",
        "status": "export_validated_and_predeclared_blend_graded",
        "claim": "Exposed fold-1 robustness evidence only; not a fresh holdout, cross-fold estimate, public score, or native-quantized grade.",
        "export": {
            "attempt_id": attempt["id"],
            "attempt_status": attempt["status"],
            "exit_status": attempt["exit_status"],
            "started_utc": attempt["started_utc"],
            "finished_utc": attempt["finished_utc"],
            "elapsed_seconds": attempt["elapsed_seconds"],
            "peak_worker_ram_bytes": attempt["peak_ram_bytes"],
            "peak_parent_ram_bytes": attempt["peak_parent_ram_bytes"],
            "peak_vram_bytes": attempt["peak_observed_vram_bytes"],
            "worker_memory_event_deltas": attempt["memory_event_deltas"],
            "prediction_path": str(PRED),
            "prediction_bytes": PRED.stat().st_size,
            "prediction_sha256": receipt["prediction_sha256"],
            "shape": list(pred.shape),
            "dtype": str(pred.dtype),
            "all_finite_checked": True,
            "sidecar_sha256": receipt["sidecar_sha256"],
            "sidecar_epoch": sidecar["epoch"],
            "checkpoint_sha256": sidecar["checkpoint_sha256"],
            "precision_scope": "local predictor inference_precision=fp16; stored prediction float32; no native Q8/Q4 claim",
            "row_groups": 1873,
            "row_group_sha256": receipt["row_groups_sha256"],
            "split_manifest_sha256": SPLIT_SHA,
            "prediction_receipt_sha256": sha256(RESULT),
            "attempt_receipt_sha256": sha256(ATTEMPT),
            "terminal_admission_receipt_sha256": sha256(TERMINAL_GATE),
            "export_admission_receipt_sha256": sha256(EXPORT_GATE),
            "producer_review_receipt_sha256": sha256(PRODUCER_REVIEW),
            "durable_launch_receipt_sha256": sha256(LAUNCH_RECEIPT),
            "producer_preflight_receipt_sha256": sha256(PACKET / "CP0_EXPORT_PREFLIGHT_R3.json"),
            "export_contract_sha256": "997cbe40016cd1acfe7af07b8e86bda990d6724d9bcbea1042336b42fd19c231",
            "export_runner_sha256": "677c7e81003660d543de91c356e768b9caf2fa0332ccc412c567b9a28db6bc2b",
            "terminal_gate_sha256": receipt["root_gates"]["terminal_gate_sha256"],
            "export_gate_sha256": receipt["root_gates"]["export_gate_sha256"],
            "profile_valid_for_throughput": False,
            "strict_profile_gate_passed": False,
        },
        "scoring": {
            "metric": "global_wp_clip2_v1",
            "official_scorer": "scripts/blend_score_remote.py",
            "review_driver": str(Path(__file__).resolve()),
            "review_driver_sha256": sha256(Path(__file__).resolve()),
            "scorer_sha256": sha256(MAIN / "scripts/blend_score_remote.py"),
            "metric_source_sha256": sha256(MAIN / "src/wnn/gmetric.py"),
            "split": "fold1; previously exposed outer evaluation groups",
            "n_sequences": len(index),
            "n_components": len(np.unique(components)),
            "scored_rows_start": 99,
            "target_mask": "cached official mask AND row index >= 99",
            "target_and_prediction_clip": [-2.0, 2.0],
            "weights": "absolute clipped target magnitude; two target correlations averaged",
            "stdout_path": str(REVIEW / "BLEND_SCORE_STDOUT_R1.txt"),
            "stdout_sha256": sha256(REVIEW / "BLEND_SCORE_STDOUT_R1.txt"),
            "exit_status": 0,
            "commands_record_sha256": sha256(COMMANDS),
            "scores_from_frozen_blend_scorer": scores,
            "fixed_half_baseline_evidence_sha256": sha256(BASELINE_EVIDENCE),
            "predeclared_interaction_contract_sha256": sha256(INTERACTION_CONTRACT),
            "prespecified_component_bootstrap": bootstrap,
            "bootstrap_contrast": "0.25 old student + 0.25 HFT + 0.50 strong alpha-0.9/e6 student minus 0.50 old student + 0.50 strong student",
            "bootstrap_utility_sha256": sha256(MAIN / "src/wnn/evaluation.py"),
            "bootstrap_seed": 271828,
            "bootstrap_resamples": 800,
        },
        "input_predictions": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "prediction_sha256_from_existing_bound_sidecar": input_sidecars[key][
                    "prediction_sha256"
                ],
                "sidecar_sha256": sha256(path.with_suffix(".json")),
                "checkpoint_sha256": input_sidecars[key]["checkpoint_sha256"],
                "epoch": input_sidecars[key]["epoch"],
                "inference_precision": input_sidecars[key]["inference_precision"],
            }
            for key, path in names.items()
        },
        "membership_and_ancestry": {
            "hft_fit_attempt_id": "oldz4-r3-f1-s3-hft-fit-resource",
            "hft_checkpoint_epoch": 0,
            "fit_group_counts": {"train": 0, "valid_fit": 1416, "outer_eval": 457, "overlap": 0},
            "actual_optimizer_counts": terminal_gate["optimizer_counts"],
            "strict_profile_gate_passed": terminal_gate["strict_profile_gate_passed"],
            "profile_valid_for_throughput": terminal_gate["profile_valid_for_throughput"],
            "fit_membership_source": "ROOT_F1_HFT_TERMINAL_ADMISSION_R3.json + frozen HFT export contract",
            "hft_parent": "oldz4_h230_student_f1_s3_r4@ep1 via audited core crop; not an independent lineage from that student",
            "eval_exposure": "Previously inspected and reused fold-1 evaluation; all results are robustness evidence only.",
            "no_new_fit_or_export": True,
        },
        "limitations": [
            "The local half-blend scores are not a native submission score and do not establish serving performance.",
            "The fixed interaction has one exposed fold and shared ancestry; its component bootstrap is conditional on these 149 complete components.",
            "Input prediction digests for old student and strong student come from their existing source-reviewed sidecars/producer receipts; this bounded run verified sidecar alignment but did not rehash those two 300 MB files.",
            "The old student and HFT share a direct student-to-HFT ancestry; old student and strong student also share source/teacher ancestry recorded in the earlier fixed-half evidence.",
            "All endpoints are local fp16 inference outputs stored as float32; no Q8/Q4 precision comparison is implied.",
        ],
    }
    (REVIEW / "EXPORT_GRADE_BOOTSTRAP_R1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"prediction_sha256": receipt["prediction_sha256"], "bootstrap": bootstrap}, indent=2
        )
    )


if __name__ == "__main__":
    main()
