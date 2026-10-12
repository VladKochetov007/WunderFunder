from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS = ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011"
RUN = "oldz4_h230_parent_f1_s3_r4"
PRED = ROOT / "cache" / f"pred_{RUN}_valid.npy"
SIDECAR = PRED.with_suffix(".json")
CP = ROOT / "artifacts/runs" / RUN / "model_ep3.pt"
PROVENANCE = ROOT / "artifacts/runs" / RUN / "provenance.json"
HISTORY = ROOT / "artifacts/runs" / RUN / "history.json"
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
INPUTS = OPS / "old_z4_f1_pipeline_successor_20261012_r4/INPUT_BINDINGS_R4.json"
SOURCE = OPS / "old_z4_f1_pipeline_successor_20261012_r4/SOURCE_BINDINGS_R4.json"
CP_REVIEW = OPS / "old_z4_f1_root_bound_exports_20261012_r3/ROOT_CP3_CONSUMER_REVIEW_R1.json"
ROOT_CONTRACT = OPS / "old_z4_f1_root_bound_exports_20261012_r3/ROOT_PARENT_CP3_BOUND_EXPORT_R3.json"
LAUNCH = OPS / "old_z4_f1_root_bound_exports_20261012_r3/ROOT_PARENT_EXPORT_LAUNCH_R3.json"
FIT_ATTEMPT = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/oldz4-r4-f1-s3-parent-fit.json"
EXPORT_ATTEMPT = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/oldz4-r4-f1-s3-parent-export-rootbind-r3.json"
SCORER = ROOT / "scripts/score_prediction_remote.py"
METRIC = ROOT / "src/wnn/gmetric.py"
TARGET_BUILDER = ROOT / "scripts/cache_targets.py"
TARGETS = ROOT / "cache/valid_targets.npz"
OUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    paths = [PRED, SIDECAR, CP, PROVENANCE, HISTORY, SPLIT, INPUTS, SOURCE, CP_REVIEW, ROOT_CONTRACT, LAUNCH, FIT_ATTEMPT, EXPORT_ATTEMPT, SCORER, METRIC, TARGET_BUILDER, TARGETS]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"required evidence paths missing: {missing}")

    meta = load_json(SIDECAR)
    split = load_json(SPLIT)
    inputs = load_json(INPUTS)
    source = load_json(SOURCE)
    fit_attempt = load_json(FIT_ATTEMPT)
    export_attempt = load_json(EXPORT_ATTEMPT)
    review = load_json(CP_REVIEW)
    contract = load_json(ROOT_CONTRACT)
    launch = load_json(LAUNCH)

    prediction_sha = sha256(PRED)
    sidecar_sha = sha256(SIDECAR)
    checkpoint_sha = sha256(CP)
    provenance_sha = sha256(PROVENANCE)
    history_sha = sha256(HISTORY)
    split_sha = sha256(SPLIT)
    inputs_sha = sha256(INPUTS)
    source_sha = sha256(SOURCE)
    fit_attempt_sha = sha256(FIT_ATTEMPT)
    export_attempt_sha = sha256(EXPORT_ATTEMPT)
    if (fit_attempt.get("status"), fit_attempt.get("exit_status")) != ("finished", 0):
        raise ValueError("fit attempt receipt is not terminal-success")
    if (export_attempt.get("status"), export_attempt.get("exit_status")) != ("finished", 0):
        raise ValueError("export attempt receipt is not terminal-success")

    expected_rows = 1873
    expected_shape = (expected_rows, 20_000, 2)
    if (meta.get("run"), meta.get("split"), meta.get("epoch")) != (RUN, "valid", 3):
        raise ValueError("prediction sidecar run/split/epoch does not match the admitted parent endpoint")
    if meta.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("prediction sidecar checkpoint SHA does not match model_ep3.pt")
    if meta.get("prediction_sha256") != prediction_sha:
        raise ValueError("prediction sidecar SHA does not match the prediction NPY")
    if meta.get("split_manifest_sha256") != split_sha:
        raise ValueError("prediction sidecar split SHA does not match fold1.json")
    if meta.get("inference_precision") != "fp16":
        raise ValueError("unexpected inference precision")
    if meta.get("row_groups") != list(range(expected_rows)):
        raise ValueError("prediction row_groups are not the complete canonical 0..1872 order")

    pred = np.load(PRED, mmap_mode="r", allow_pickle=False)
    if pred.shape != expected_shape or pred.dtype != np.dtype("float32"):
        raise ValueError(f"unexpected prediction array shape/dtype: {pred.shape}, {pred.dtype}")
    header_bytes = int(pred.offset)
    payload_bytes = int(pred.nbytes)
    if PRED.stat().st_size != header_bytes + payload_bytes:
        raise ValueError("prediction file has truncated or trailing payload bytes")
    finite_count = 0
    for start in range(0, expected_rows, 32):
        block = pred[start : start + 32]
        if not np.isfinite(block).all():
            raise ValueError(f"nonfinite prediction value in row-group block starting {start}")
        finite_count += int(block.size)
    del pred

    with np.load(TARGETS, allow_pickle=False) as cache:
        target_groups = np.asarray(cache["row_group"], dtype=np.int64)
    if target_groups.shape != (expected_rows,) or target_groups.tolist() != meta["row_groups"]:
        raise ValueError("prediction row-group order does not match official target-cache order")
    evaluation_groups = [int(g) for g in split["evaluation"]["valid"]]
    if len(evaluation_groups) != 457 or len(set(evaluation_groups)) != 457:
        raise ValueError("fold-1 evaluation membership count/uniqueness mismatch")
    if not set(evaluation_groups).issubset(set(target_groups.tolist())):
        raise ValueError("fold-1 evaluation groups are missing from official target-cache row groups")

    fold_membership = inputs["fold"]["source_qualified_fit_membership"]
    if inputs["fold"]["evaluation_group_count"] != 457:
        raise ValueError("input binding evaluation count differs from 457")
    if fold_membership["total_source_qualified_groups"] != 11_949:
        raise ValueError("input binding fit-member total differs from 11,949")
    if review.get("status") != "pass_actual_cp3_and_consumer":
        raise ValueError("root CP3 consumer review is not passing")
    if review.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("root CP3 consumer review checkpoint digest mismatch")
    if not review.get("checkpoint_matches_provenance") or not review.get("strict_consumer_state_load_passed"):
        raise ValueError("root CP3 state/provenance validation is incomplete")
    if contract.get("checkpoint_binding", {}).get("sha256") != checkpoint_sha:
        raise ValueError("root-bound export contract checkpoint digest mismatch")
    if (contract.get("run"), contract.get("fold"), contract.get("seed"), contract.get("checkpoint_epoch")) != (RUN, 1, 3, 3):
        raise ValueError("root-bound export contract endpoint identity mismatch")
    if contract.get("expected_outer_evaluation_groups") != 457:
        raise ValueError("root-bound export contract evaluation membership mismatch")
    if launch.get("returncode") != 0:
        raise ValueError("parent export launch receipt is not successful")

    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    command = [sys.executable, str(SCORER), "1", RUN]
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=100, check=False)
    stdout = completed.stdout
    stderr = completed.stderr
    (OUT / "OFFICIAL_SCORE_STDOUT_R1.txt").write_text(stdout)
    (OUT / "OFFICIAL_SCORE_STDERR_R1.txt").write_text(stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"official fold1 scorer failed ({completed.returncode}): {stderr[-1000:]}")
    match = re.fullmatch(rf"{re.escape(RUN)} ([0-9.]+) t0 ([0-9.]+) t1 ([0-9.]+)\s*", stdout)
    if match is None:
        raise ValueError(f"unexpected official scorer output: {stdout!r}")

    receipt = {
        "schema": "wnn.oldz4-f1-parent-export-official-grade.r1",
        "status": "payload_validated_and_official_grade_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "One source-qualified F1 seed-3 parent endpoint scored on the exact previously exposed 457-group fold-1 outer evaluation set; not a fresh holdout or a cross-fold conclusion.",
        "fit": {
            "run": RUN,
            "stage": "independently_fitted_old_z4_parent",
            "seed": 3,
            "fold": 1,
            "checkpoint_epoch": 3,
            "checkpoint_path": str(CP),
            "checkpoint_sha256": checkpoint_sha,
            "provenance_sha256": provenance_sha,
            "history_sha256": history_sha,
            "fit_membership": {
                "source_qualified_total": 11_949,
                "train_parquet": fold_membership["train.parquet"]["count"],
                "fold_eligible_valid_parquet": fold_membership["valid.parquet"]["count"],
                "outer_evaluation_excluded_count": 457,
                "outer_evaluation_fingerprint_sha256": fold_membership["evaluation_valid_parquet"]["fingerprint_sha256"],
            },
            "fit_attempt_id": "oldz4-r4-f1-s3-parent-fit",
            "fit_attempt_receipt_path": str(FIT_ATTEMPT),
            "fit_attempt_receipt_sha256": fit_attempt_sha,
            "fit_attempt_terminal_status": fit_attempt["status"],
            "fit_attempt_exit_status": fit_attempt["exit_status"],
            "fit_started_utc": fit_attempt["started_utc"],
            "fit_finished_utc": fit_attempt["finished_utc"],
            "fit_completion_review_sha256": sha256(CP_REVIEW),
            "strict_checkpoint_state_and_provenance_consumer_passed": True,
        },
        "prediction": {
            "path": str(PRED),
            "sidecar_path": str(SIDECAR),
            "bytes": PRED.stat().st_size,
            "npy_header_bytes": header_bytes,
            "raw_payload_bytes": payload_bytes,
            "shape": list(expected_shape),
            "dtype": "float32",
            "finite": True,
            "finite_values_checked": finite_count,
            "row_groups": expected_rows,
            "row_group_order": "identical to cache/valid_targets.npz row_group order",
            "sha256": prediction_sha,
            "sidecar_sha256": sidecar_sha,
            "sidecar_checkpoint_sha256": meta["checkpoint_sha256"],
            "sidecar_split_manifest_sha256": meta["split_manifest_sha256"],
            "inference_precision": meta["inference_precision"],
            "root_bound_export_contract_path": str(ROOT_CONTRACT),
            "root_bound_export_contract_sha256": sha256(ROOT_CONTRACT),
            "export_attempt_id": "oldz4-r4-f1-s3-parent-export-rootbind-r3",
            "export_attempt_receipt_path": str(EXPORT_ATTEMPT),
            "export_attempt_receipt_sha256": export_attempt_sha,
            "export_attempt_terminal_status": export_attempt["status"],
            "export_attempt_exit_status": export_attempt["exit_status"],
            "export_started_utc": export_attempt["started_utc"],
            "export_finished_utc": export_attempt["finished_utc"],
            "export_launch_receipt_sha256": sha256(LAUNCH),
        },
        "data_and_source_bindings": {
            "split_manifest_path": str(SPLIT),
            "split_manifest_sha256": split_sha,
            "input_bindings_path": str(INPUTS),
            "input_bindings_sha256": inputs_sha,
            "source_bindings_path": str(SOURCE),
            "source_bindings_sha256": source_sha,
            "official_target_archive_path": str(TARGETS),
            "official_target_archive_sha256": sha256(TARGETS),
            "target_mask_semantics": "cache_targets.py stores is_scored; official fold scorer additionally masks rows <99 (need_prediction warmup).",
            "target_cache_builder_sha256": sha256(TARGET_BUILDER),
            "source_snapshot_binding_sha256": contract["source_snapshot_bindings_sha256"],
            "trainer_sha256": contract["source_bindings"]["pinned_trainer_sha256"],
            "predictor_sha256": contract["source_bindings"]["predictor_sha256"],
            "root_binding_helper_sha256": contract["root_binding_helper"]["sha256"],
        },
        "scoring": {
            "official_metric_id": "global_wp_clip2_v1",
            "grader_driver_path": str(Path(__file__).resolve()),
            "grader_driver_sha256": sha256(Path(__file__).resolve()),
            "support_boundary": {"slice": "wnn-campaign.slice", "cpu_quota": "25%", "threads": 1, "memory_max_bytes": 1_500_000_000, "runtime_max_seconds": 120},
            "scorer_path": str(SCORER),
            "scorer_sha256": sha256(SCORER),
            "metric_implementation_path": str(METRIC),
            "metric_implementation_sha256": sha256(METRIC),
            "fold": 1,
            "evaluated_group_count": len(evaluation_groups),
            "evaluated_group_ids_sha256": hashlib.sha256(json.dumps(evaluation_groups, separators=(",", ":")).encode()).hexdigest(),
            "exposure": "previously inspected fold-1 outer evaluation; exposed robustness evidence",
            "warmup_row_excluded_below": 99,
            "prediction_mask": "official cached is_scored mask AND row >= 99",
            "target_and_prediction_clip": [-2.0, 2.0],
            "weighting": "absolute clipped target magnitude; two target correlations averaged",
            "command": command,
            "returncode": completed.returncode,
            "stdout_path": str(OUT / "OFFICIAL_SCORE_STDOUT_R1.txt"),
            "stderr_path": str(OUT / "OFFICIAL_SCORE_STDERR_R1.txt"),
            "stdout": stdout.strip(),
            "displayed_wp": match.group(1),
            "displayed_t0": match.group(2),
            "displayed_t1": match.group(3),
            "display_precision_decimal_places": 5,
        },
        "comparison_scope": {
            "n_distinct_seed_endpoints": 1,
            "same_fold_h382_matched_controls_scored": False,
            "control_dependency": "The bounded cache inventory found H382 clean outputs for folds 2/3, but no same-fold-1 H382 e3/e4 seed-3 prediction was located; do not compare across folds as a paired control.",
            "public_score_claim": False,
        },
    }
    (OUT / "F1_PARENT_EXPORT_GRADE_R1.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "stdout": stdout.strip(), "receipt": str(OUT / "F1_PARENT_EXPORT_GRADE_R1.json"), "prediction_sha256": prediction_sha, "finite_values_checked": finite_count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
