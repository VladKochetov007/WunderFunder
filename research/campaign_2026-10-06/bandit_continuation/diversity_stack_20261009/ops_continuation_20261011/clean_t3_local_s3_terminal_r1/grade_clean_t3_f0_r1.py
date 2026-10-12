"""Grade one admitted clean-T3 F0 export and its fixed half blend."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
HERE = Path(__file__).resolve().parent
RUN = "clean_t3_f0_muonall_h384_e3_s3_r5"
PREDICTION_NAME = f"{RUN}_local_full_valid_r1"
OLD_RUN = "oldz4_h230_student_f0_s3_r3"
TEACHER_PRED = ROOT / "cache" / f"pred_{PREDICTION_NAME}_valid.npy"
TEACHER_SIDECAR = TEACHER_PRED.with_suffix(".json")
OLD_PRED = ROOT / "cache" / f"pred_{OLD_RUN}_valid.npy"
OLD_SIDECAR = OLD_PRED.with_suffix(".json")
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGETS = ROOT / "cache/valid_targets.npz"
TEACHER_RUN_ROOT = ROOT / "artifacts/runs" / RUN
OLD_RUN_ROOT = ROOT / "artifacts/runs" / OLD_RUN
OPS = ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011"
ADMISSION = OPS / "clean_t3_local_s3_export_r1/ROOT_FIRST_F0_GRADE_ADMISSION_R1.json"
TERMINAL = HERE / "LOCAL_TERMINAL_HANDOFF_R1.json"
OLD_GRADE = OPS / "e529_cp2_fixed_blend_grade_20261012_r2/FIXED_BLEND_GRADE_R2.json"
OFFICIAL_STDOUT = HERE / "OFFICIAL_SCORE_STDOUT_R1.txt"
OFFICIAL_STDERR = HERE / "OFFICIAL_SCORE_STDERR_R1.txt"
OFFICIAL_SCORER = ROOT / "scripts/blend_score_remote.py"
N_BOOT = 800
BOOTSTRAP_SEED = 271828


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    admission = read_json(ADMISSION)
    terminal = read_json(TERMINAL)
    old_grade = read_json(OLD_GRADE)
    if admission.get("status") != "admitted_fixed_official_source_and_conditional_comparison":
        raise ValueError("root grade admission is absent or has wrong status")
    if admission.get("prediction_sha256") != terminal["output_validation"]["npy_sha256"]:
        raise ValueError("grade admission is bound to different teacher prediction bytes")
    if terminal.get("status") != "terminal_export_validated_grade_pending_root":
        raise ValueError("local terminal handoff is not a successful validated export")
    if old_grade.get("status") != "PASS_FIXED_EXPOSED_FOLD0_COMPARISON":
        raise ValueError("source-bound old-student control receipt is not successful")

    teacher_sidecar = read_json(TEACHER_SIDECAR)
    old_sidecar = read_json(OLD_SIDECAR)
    teacher_provenance = read_json(TEACHER_RUN_ROOT / "provenance.json")
    old_provenance = read_json(OLD_RUN_ROOT / "provenance.json")
    expected_old = old_grade["baseline"]
    observed_hashes = {
        "teacher_prediction_sha256": sha256(TEACHER_PRED),
        "teacher_sidecar_sha256": sha256(TEACHER_SIDECAR),
        "teacher_checkpoint_sha256": sha256(TEACHER_RUN_ROOT / "model_ep2.pt"),
        "teacher_provenance_sha256": sha256(TEACHER_RUN_ROOT / "provenance.json"),
        "old_prediction_sha256": sha256(OLD_PRED),
        "old_sidecar_sha256": sha256(OLD_SIDECAR),
        "old_checkpoint_sha256": sha256(OLD_RUN_ROOT / "model_ep1.pt"),
        "old_provenance_sha256": sha256(OLD_RUN_ROOT / "provenance.json"),
        "valid_targets_sha256": sha256(TARGETS),
        "fold0_split_sha256": sha256(SPLIT),
    }
    expected_teacher = terminal["output_validation"]
    if observed_hashes["teacher_prediction_sha256"] != expected_teacher["npy_sha256"]:
        raise ValueError("teacher prediction changed after terminal validation")
    if observed_hashes["teacher_sidecar_sha256"] != expected_teacher["sidecar_sha256"]:
        raise ValueError("teacher sidecar changed after terminal validation")
    if observed_hashes["teacher_checkpoint_sha256"] != admission["checkpoint_sha256"]:
        raise ValueError("teacher checkpoint differs from grade admission")
    for observed, expected in (
        ("old_prediction_sha256", expected_old["prediction_sha256"]),
        ("old_sidecar_sha256", expected_old["prediction_sidecar_sha256"]),
        ("old_checkpoint_sha256", expected_old["checkpoint_sha256"]),
        ("old_provenance_sha256", expected_old["provenance_sha256"]),
        ("valid_targets_sha256", old_grade["data_and_source_bindings"]["valid_targets_sha256"]),
        ("fold0_split_sha256", admission["split_sha256"]),
    ):
        if observed_hashes[observed] != expected:
            raise ValueError(f"{observed} differs from its source-bound receipt")

    split = read_json(SPLIT)
    if split["fold"] != 0:
        raise ValueError("expected source-qualified fold 0")
    evaluation_groups = list(map(int, split["evaluation"]["valid"]))
    eval_set = set(evaluation_groups)
    if len(eval_set) != 468:
        raise ValueError("fold-0 exposed evaluation group count changed")
    for run_name, provenance in ((RUN, teacher_provenance), (OLD_RUN, old_provenance)):
        fit = provenance.get("fit_row_groups")
        if not isinstance(fit, dict):
            raise ValueError(f"{run_name} provenance lacks source-qualified fit lists")
        if set(map(int, fit.get("train", []))) != set(map(int, split["fit"]["train"])):
            raise ValueError(f"{run_name} train-source fit membership differs from fold 0")
        if set(map(int, fit.get("valid", []))) != set(map(int, split["fit"]["valid"])):
            raise ValueError(f"{run_name} valid-source fit membership differs from fold 0")
        if set(map(int, fit["valid"])) & eval_set:
            raise ValueError(f"{run_name} valid-source fit overlaps evaluation groups")
        if set(map(int, provenance.get("evaluation_row_groups", []))) != eval_set:
            raise ValueError(f"{run_name} provenance evaluation membership differs from fold 0")
    if len(set(map(int, split["fit"]["valid"])) & eval_set) != 0:
        raise ValueError("fold-0 valid fit/evaluation membership overlaps")
    if sha256(SPLIT) != admission["split_sha256"]:
        raise ValueError("fold-0 split differs from grade admission")

    if (teacher_sidecar.get("run"), teacher_sidecar.get("split"), teacher_sidecar.get("epoch")) != (RUN, "valid", 2):
        raise ValueError("teacher prediction sidecar endpoint differs")
    if (old_sidecar.get("run"), old_sidecar.get("split"), old_sidecar.get("epoch")) != (OLD_RUN, "valid", 1):
        raise ValueError("old-student prediction sidecar endpoint differs")
    for meta, checkpoint_sha in (
        (teacher_sidecar, admission["checkpoint_sha256"]),
        (old_sidecar, expected_old["checkpoint_sha256"]),
    ):
        if meta.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("prediction sidecar checkpoint binding differs")
        if meta.get("split_manifest_sha256") != admission["split_sha256"]:
            raise ValueError("prediction sidecar split binding differs")
        if meta.get("row_groups") != list(range(1873)):
            raise ValueError("prediction sidecar row order is not canonical full-valid")

    from wnn.evaluation import paired_component_bootstrap, paired_component_moments
    from wnn.gmetric import GlobalAccumulator, METRIC_ID, WARMUP

    if METRIC_ID != "global_wp_clip2_v1" or WARMUP != 99:
        raise ValueError("official metric/warmup differs from admitted contract")
    teacher = np.load(TEACHER_PRED, mmap_mode="r", allow_pickle=False)
    old = np.load(OLD_PRED, mmap_mode="r", allow_pickle=False)
    expected_shape = (1873, 20000, 2)
    if teacher.shape != expected_shape or old.shape != expected_shape:
        raise ValueError("a prediction has an unexpected shape")
    if teacher.dtype != np.dtype("float32") or old.dtype != np.dtype("float32"):
        raise ValueError("a prediction has an unexpected dtype")

    target_cache = np.load(TARGETS, allow_pickle=False)
    try:
        target_groups = np.asarray(target_cache["row_group"], dtype=np.int64)
        if not np.array_equal(target_groups, np.arange(1873, dtype=np.int64)):
            raise ValueError("valid-target row-group order is not canonical")
        positions = {int(group): i for i, group in enumerate(target_groups)}
        indices = np.asarray(sorted(positions[group] for group in evaluation_groups), dtype=np.int64)
        targets = np.asarray(target_cache["y"][indices], dtype=np.float32)
        is_scored = np.asarray(target_cache["mask"][indices], dtype=bool)
    finally:
        target_cache.close()
    if targets.shape != (468, 20000, 2) or is_scored.shape != (468, 20000):
        raise ValueError("selected official fold-0 targets/mask have unexpected shape")
    mask = is_scored & (np.arange(20000) >= WARMUP)[None, :]

    component_map = {
        int(item["row_group"]): int(item["component"])
        for item in split["sequences"]["valid"]
    }
    components = np.asarray([component_map[group] for group in target_groups[indices]], dtype=np.int64)
    if len(np.unique(components)) != int(split["counts"]["evaluation_components"]):
        raise ValueError("fold-0 evaluation component count differs from split")

    teacher_acc = GlobalAccumulator()
    blend_acc = GlobalAccumulator()
    old_acc = GlobalAccumulator()
    candidate_moments = np.empty((len(indices), 2, 6), dtype=np.float64)
    reference_moments = np.empty_like(candidate_moments)
    for start in range(0, len(indices), 8):
        end = min(start + 8, len(indices))
        ix = indices[start:end]
        y = targets[start:end]
        m = mask[start:end]
        old_batch = np.asarray(old[ix], dtype=np.float32)
        teacher_batch = np.asarray(teacher[ix], dtype=np.float32)
        blend_batch = (1.0 * old_batch + 1.0 * teacher_batch) / 2.0
        teacher_acc.add(y, teacher_batch, m)
        blend_acc.add(y, blend_batch, m)
        old_acc.add(y, old_batch, m)
        candidate, reference = paired_component_moments(y, blend_batch, old_batch, m)
        candidate_moments[start:end] = candidate
        reference_moments[start:end] = reference
    scores = {
        "teacher": teacher_acc.result(),
        "fixed_half_old_student_teacher": blend_acc.result(),
        "old_student_control": old_acc.result(),
    }

    teacher_spec = PREDICTION_NAME
    blend_spec = f"{OLD_RUN}:1+{PREDICTION_NAME}:1"
    official_lines = OFFICIAL_STDOUT.read_text().splitlines()
    parsed: dict[str, dict[str, float]] = {}
    for line in official_lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            raise ValueError(f"malformed official scorer output line: {line!r}")
        parsed[fields[5]] = {"wp": float(fields[0]), "t0": float(fields[2]), "t1": float(fields[4])}
    expected_specs = {teacher_spec, blend_spec, OLD_RUN}
    if set(parsed) != expected_specs:
        raise ValueError("official scorer output names differ from admitted grade specs")
    for key, spec in (("teacher", teacher_spec), ("fixed_half_old_student_teacher", blend_spec), ("old_student_control", OLD_RUN)):
        if abs(scores[key]["wp"] - parsed[spec]["wp"]) > 0.0000051:
            raise ValueError(f"GlobalAccumulator result differs from official stdout for {spec}")

    bootstrap = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        components,
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    if abs(bootstrap["candidate_wp"] - scores["fixed_half_old_student_teacher"]["wp"]) > 1e-10:
        raise ValueError("paired candidate moments disagree with official blend score")
    if abs(bootstrap["reference_wp"] - scores["old_student_control"]["wp"]) > 1e-10:
        raise ValueError("paired reference moments disagree with official control score")

    result = {
        "schema": "wnn.clean-t3-local-f0-first-source-and-fixed-half-grade.r1",
        "status": "PASS_OFFICIAL_EXPOSED_FOLD0_SOURCE_AND_FIXED_BLEND_GRADE",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim": "One clean F0 teacher seed and one fixed 50/50 old-student/new-teacher output blend on the repeatedly exposed 468-group/141-component fold-0 development panel. This is not a fresh holdout, cross-seed estimate, native-precision grade, final all-data score, or public forecast.",
        "root_admission_sha256": sha256(ADMISSION),
        "terminal_handoff_sha256": sha256(TERMINAL),
        "old_control_prior_grade_sha256": sha256(OLD_GRADE),
        "source_and_artifact_hashes": observed_hashes,
        "metric": {
            "id": METRIC_ID,
            "source_sha256": sha256(ROOT / "src/wnn/gmetric.py"),
            "official_scorer_path": str(OFFICIAL_SCORER),
            "official_scorer_sha256": sha256(OFFICIAL_SCORER),
            "scorer_stdout_sha256": sha256(OFFICIAL_STDOUT),
            "scorer_stderr_sha256": sha256(OFFICIAL_STDERR),
            "warmup_row": WARMUP,
            "mask": "is_scored AND row >= 99",
            "clipping": [-2, 2],
            "weights": "original clipped-label magnitude weights",
            "selected_groups": len(indices),
            "components": int(len(np.unique(components))),
            "evaluation_group_sorted_int64_le_sha256": hashlib.sha256(
                np.asarray(sorted(eval_set), dtype="<i8").tobytes()
            ).hexdigest(),
            "valid_targets_sha256": observed_hashes["valid_targets_sha256"],
            "fold0_split_sha256": observed_hashes["fold0_split_sha256"],
        },
        "recipes": {
            "teacher": {
                "run": RUN,
                "seed": 3,
                "checkpoint_epoch": 2,
                "inference_precision": teacher_sidecar["inference_precision"],
                "prediction_sha256": observed_hashes["teacher_prediction_sha256"],
            },
            "old_student_control": {
                "run": OLD_RUN,
                "seed": 3,
                "checkpoint_epoch": 1,
                "inference_precision": old_sidecar["inference_precision"],
                "prediction_sha256": observed_hashes["old_prediction_sha256"],
            },
            "fixed_blend": {
                "old_student_weight": 0.5,
                "teacher_weight": 0.5,
                "blend_precision": "float32 arithmetic over stored float32 predictions; old member inference was fp16, teacher fp32",
                "weight_search": False,
            },
            "source_qualified_membership": {
                "train_fit_groups": 10535,
                "valid_fit_groups": 1405,
                "valid_evaluation_groups": 468,
                "excluded_train_groups_by_fold_manifest": int(split["counts"]["excluded_train"]),
                "valid_fit_evaluation_overlap": 0,
                "both_member_valid_fit_lists_match_fold_manifest": True,
                "ancestry_exclusion_gate": "Root-admitted source-qualified fit/propensity/teacher/initializer exclusion receipt; this grade only independently checked direct fit-list membership and artifact bindings.",
            },
        },
        "official_scores": scores,
        "official_scorer_stdout_parsed": parsed,
        "paired_fixed_blend_vs_old_student": {
            **bootstrap,
            "bootstrap_unit": "complete source-qualified validation component",
            "n_boot": N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "helper_source_sha256": sha256(ROOT / "src/wnn/evaluation.py"),
            "helper_imported_torch": "torch" in sys.modules,
            "weights_predeclared": [0.5, 0.5],
        },
        "execution": {
            "command": "WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=600s scripts/limited .venv/bin/python scripts/blend_score_remote.py 0 <teacher> <fixed-50-50> <old-student>",
            "grade_driver": str(Path(__file__).resolve()),
            "grade_driver_sha256": sha256(Path(__file__).resolve()),
            "no_gpu_or_fit": True,
            "new_spend_usd": 0,
        },
        "limitations": [
            "Fold 0 is an exposed development panel used repeatedly; uncertainty is conditional on this panel and these fitted endpoints.",
            "Only one seed is available for the new teacher and control recipe.",
            "The full-valid prediction artifact contains fit-valid as well as evaluation-valid inputs; only the split-selected 468 evaluation groups are scored here.",
            "New teacher inference is FP32 while the old student control predictions came from FP16 inference; this fixed blend is not a precision-matched comparison.",
        ],
    }
    out = HERE / "F0_FIRST_GRADE_R1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "official_scores": scores, "paired": bootstrap}, sort_keys=True))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
