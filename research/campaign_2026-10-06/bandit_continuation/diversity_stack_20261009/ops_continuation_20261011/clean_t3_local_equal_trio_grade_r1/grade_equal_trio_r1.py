"""Grade the admitted F0 distinct-seed teacher trio and fixed student blend."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
HERE = Path(__file__).resolve().parent
OPS = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011"
)
RUNS = [f"clean_t3_f0_muonall_h384_e3_s{seed}_r5" for seed in (3, 4, 5)]
PRED_NAMES = [f"{run}_local_full_valid_r1" for run in RUNS]
OLD_RUN = "oldz4_h230_student_f0_s3_r3"
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGETS = ROOT / "cache/valid_targets.npz"
ADMISSION = OPS / "clean_t3_local_s3_export_r1/ROOT_EQUAL_TEACHER_TRIO_GRADE_ADMISSION_R1.json"
S3_TERMINAL = OPS / "clean_t3_local_s3_terminal_r1/LOCAL_TERMINAL_HANDOFF_R1.json"
S3_GRADE = OPS / "clean_t3_local_s3_terminal_r1/F0_FIRST_GRADE_R1.json"
S4_S5_TERMINAL = (
    HERE.parent / "clean_t3_local_s4_s5_terminal_r1/LOCAL_EXPORT_TERMINAL_HANDOFF_R1.json"
)
ROOT_PREFLIGHT = OPS / "clean_t3_local_s3_export_r1/ROOT_S4_S5_ACTUAL_PRODUCER_PREFLIGHT_R1.json"
OFFICIAL_STDOUT = HERE / "OFFICIAL_SCORE_STDOUT_R1.txt"
OFFICIAL_STDERR = HERE / "OFFICIAL_SCORE_STDERR_R1.txt"
OFFICIAL_SCORER = ROOT / "scripts/blend_score_remote.py"
OLD_GRADE_SHA = "f6fb615b329d1d3acb513a8ac87693eeb87987af799ddd683329c1d9fa36fbae"
EXPECTED_ADMISSION_SHA = "4b48c41a9e583bd547e8f6d962ade35aa91675a9d7fd1f88849b02419968e5d1"
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


def parse_official(path: Path) -> dict[str, dict[str, float]]:
    parsed: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            raise ValueError(f"malformed official scorer line: {line!r}")
        parsed[fields[5]] = {"wp": float(fields[0]), "t0": float(fields[2]), "t1": float(fields[4])}
    return parsed


def main() -> None:
    admission = read_json(ADMISSION)
    s3_terminal = read_json(S3_TERMINAL)
    s3_grade = read_json(S3_GRADE)
    s4_s5_terminal = read_json(S4_S5_TERMINAL)
    root_preflight = read_json(ROOT_PREFLIGHT)
    if sha256(ADMISSION) != EXPECTED_ADMISSION_SHA:
        raise ValueError("root admission bytes differ from the reviewed grade authorization")
    if admission.get("status") != "admitted_frozen_distinct_seed_and_fixed_trio_comparisons":
        raise ValueError("root admission is absent or has an unexpected status")
    if admission.get("seeds") != [3, 4, 5] or admission.get("selected_checkpoint_epoch") != 2:
        raise ValueError("root admission endpoint differs")
    if (
        admission.get("metric") != "global_wp_clip2_v1"
        or admission.get("evaluation_groups") != 468
        or admission.get("components") != 141
    ):
        raise ValueError("root admission metric or fold panel differs")
    if admission.get("bootstrap") != {
        "unit": "completecomponent",
        "resamples": 800,
        "seed": 271828,
        "candidate": "fixed oldstudent/equaltrio half",
        "reference": "oldstudent",
    }:
        raise ValueError("bootstrap contract differs from admission")
    if s3_grade.get("status") != "PASS_OFFICIAL_EXPOSED_FOLD0_SOURCE_AND_FIXED_BLEND_GRADE":
        raise ValueError("the existing s3 official grade is not successful")
    if (
        s3_terminal.get("output_validation", {}).get("npy_sha256")
        != admission["prediction_sha256"]["s3"]
    ):
        raise ValueError("s3 terminal prediction differs from admission")
    if s4_s5_terminal.get("status") != "PASS_S4_S5_TERMINAL_FULL_VALID_EXPORTS":
        raise ValueError("s4/s5 terminal handoff is not successful")
    if (
        root_preflight.get("status")
        != "PASS_ACTUAL_CP_SOURCE_SPLIT_FULL_VALID_SELECTOR_BEFORE_CUDA"
        or len(root_preflight.get("successors", [])) != 2
        or root_preflight.get("source_predictor_sha256")
        != "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0"
    ):
        raise ValueError("root actual-producer preflight does not bind both reviewed seed exports")
    if sha256(S3_GRADE) != OLD_GRADE_SHA:
        raise ValueError("prior s3 grade changed after root admission")

    split = read_json(SPLIT)
    if split.get("fold") != 0:
        raise ValueError("expected fold 0")
    eval_groups = set(map(int, split["evaluation"]["valid"]))
    if len(eval_groups) != 468:
        raise ValueError("fold-0 evaluation membership count differs")

    pred_paths = [ROOT / "cache" / f"pred_{name}_valid.npy" for name in PRED_NAMES]
    sidecar_paths = [path.with_suffix(".json") for path in pred_paths]
    old_pred_path = ROOT / "cache" / f"pred_{OLD_RUN}_valid.npy"
    old_sidecar_path = old_pred_path.with_suffix(".json")
    old_run_root = ROOT / "artifacts/runs" / OLD_RUN
    old_prior = read_json(S3_GRADE)
    old_bindings = old_prior["source_and_artifact_hashes"]
    expected_pred_sha = admission["prediction_sha256"]
    expected_teacher_cp_sha = {
        "s3": "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
        "s4": "714ac9b04cd1be0f63fb5f9f60cce2059e5e9284848470cae9d7198794767df7",
        "s5": "763fac7f0aa24507167425e9905f1514a0a28ace48036b1c1c5819c7eb02c8d1",
    }
    expected_teacher_prov_sha = {
        "s3": "740b0d96ee311f215153db62272262e0153ce11f8f91560b03efae8e739f9ed0",
        "s4": "f738ef25f1b7d4244d6c900ebe4fba861efc24ddd106ff425fc60d7e3caccf11",
        "s5": "44728c0c60981166f63038f2a6696b6714ce57c52e039e304883ffe7449def18",
    }
    observed: dict[str, str] = {
        "root_admission_sha256": sha256(ADMISSION),
        "fold0_split_sha256": sha256(SPLIT),
        "valid_targets_sha256": sha256(TARGETS),
        "official_scorer_sha256": sha256(OFFICIAL_SCORER),
        "s3_terminal_handoff_sha256": sha256(S3_TERMINAL),
        "s3_grade_sha256": sha256(S3_GRADE),
        "s4_s5_terminal_handoff_sha256": sha256(S4_S5_TERMINAL),
        "root_s4_s5_producer_preflight_sha256": sha256(ROOT_PREFLIGHT),
        "old_student_prediction_sha256": sha256(old_pred_path),
        "old_student_sidecar_sha256": sha256(old_sidecar_path),
        "old_student_checkpoint_sha256": sha256(old_run_root / "model_ep1.pt"),
        "old_student_provenance_sha256": sha256(old_run_root / "provenance.json"),
    }
    if observed["old_student_prediction_sha256"] != old_bindings["old_prediction_sha256"]:
        raise ValueError("old-student prediction differs from its prior grade")
    if (
        observed["old_student_checkpoint_sha256"] != old_bindings["old_checkpoint_sha256"]
        or observed["old_student_provenance_sha256"] != old_bindings["old_provenance_sha256"]
    ):
        raise ValueError("old-student endpoint differs from its source-bound prior grade")

    prediction_arrays: list[np.ndarray] = []
    teacher_receipts: dict[str, dict] = {}
    split_sha = sha256(SPLIT)
    if split_sha != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d":
        raise ValueError("fold-0 split differs from the source-qualified run receipts")
    for seed, run, pred_path, sidecar_path in zip(
        (3, 4, 5), RUNS, pred_paths, sidecar_paths, strict=True
    ):
        key = f"s{seed}"
        prediction_sha = sha256(pred_path)
        sidecar_sha = sha256(sidecar_path)
        run_root = ROOT / "artifacts/runs" / run
        checkpoint_sha = sha256(run_root / "model_ep2.pt")
        provenance_sha = sha256(run_root / "provenance.json")
        if prediction_sha != expected_pred_sha[key]:
            raise ValueError(f"{key} prediction differs from root admission")
        if (
            checkpoint_sha != expected_teacher_cp_sha[key]
            or provenance_sha != expected_teacher_prov_sha[key]
        ):
            raise ValueError(
                f"{key} checkpoint/provenance differs from the collected terminal evidence"
            )
        array = np.load(pred_path, mmap_mode="r", allow_pickle=False)
        if array.shape != (1873, 20000, 2) or array.dtype != np.dtype("float32"):
            raise ValueError(
                f"{key} prediction header differs from the official full-valid contract"
            )
        sidecar = read_json(sidecar_path)
        if (
            sidecar.get("run"),
            sidecar.get("split"),
            sidecar.get("epoch"),
            sidecar.get("checkpoint_sha256"),
            sidecar.get("split_manifest_sha256"),
        ) != (run, "valid", 2, checkpoint_sha, split_sha):
            raise ValueError(f"{key} sidecar endpoint binding differs")
        if sidecar.get("inference_precision") != "fp32" or sidecar.get("row_groups") != list(
            range(1873)
        ):
            raise ValueError(f"{key} sidecar precision/order differs")
        provenance = read_json(run_root / "provenance.json")
        if (
            provenance.get("seed"),
            provenance.get("selected_epoch"),
            provenance.get("args", {}).get("seed"),
        ) != (seed, 2, seed):
            raise ValueError(f"{key} provenance endpoint differs")
        fit = provenance.get("fit_row_groups")
        if not isinstance(fit, dict):
            raise ValueError(f"{key} provenance lacks fit membership")
        fit_train = set(map(int, fit.get("train", [])))
        fit_valid = set(map(int, fit.get("valid", [])))
        if fit_train != set(map(int, split["fit"]["train"])) or fit_valid != set(
            map(int, split["fit"]["valid"])
        ):
            raise ValueError(f"{key} direct fit membership differs from the fold-0 source manifest")
        if (
            fit_valid & eval_groups
            or set(map(int, provenance.get("evaluation_row_groups", []))) != eval_groups
        ):
            raise ValueError(f"{key} direct valid/evaluation membership is inconsistent")
        prediction_arrays.append(array)
        teacher_receipts[key] = {
            "run": run,
            "seed": seed,
            "checkpoint_epoch": 2,
            "checkpoint_sha256": checkpoint_sha,
            "provenance_sha256": provenance_sha,
            "prediction_sha256": prediction_sha,
            "sidecar_sha256": sidecar_sha,
            "inference_precision": "fp32",
            "fit_train_groups": len(fit_train),
            "fit_valid_groups": len(fit_valid),
            "evaluation_valid_groups": len(eval_groups),
            "direct_valid_evaluation_overlap": 0,
            "ancestor_source_qualification": "bound to root-approved actual-producer preflight and terminal receipt; this grader rechecks direct membership and file identities",
        }
    observed["teacher_prediction_sha256"] = {
        f"s{seed}": teacher_receipts[f"s{seed}"]["prediction_sha256"] for seed in (3, 4, 5)
    }
    observed["teacher_sidecar_sha256"] = {
        f"s{seed}": teacher_receipts[f"s{seed}"]["sidecar_sha256"] for seed in (3, 4, 5)
    }
    observed["teacher_checkpoint_sha256"] = {
        f"s{seed}": teacher_receipts[f"s{seed}"]["checkpoint_sha256"] for seed in (3, 4, 5)
    }
    observed["teacher_provenance_sha256"] = {
        f"s{seed}": teacher_receipts[f"s{seed}"]["provenance_sha256"] for seed in (3, 4, 5)
    }

    old_array = np.load(old_pred_path, mmap_mode="r", allow_pickle=False)
    if old_array.shape != (1873, 20000, 2) or old_array.dtype != np.dtype("float32"):
        raise ValueError("old-student prediction header differs")
    old_sidecar = read_json(old_sidecar_path)
    old_provenance = read_json(old_run_root / "provenance.json")
    if (
        old_sidecar.get("checkpoint_sha256") != observed["old_student_checkpoint_sha256"]
        or old_sidecar.get("split_manifest_sha256") != split_sha
    ):
        raise ValueError("old-student sidecar source/split binding differs")
    old_fit = old_provenance.get("fit_row_groups", {})
    if set(map(int, old_fit.get("train", []))) != set(map(int, split["fit"]["train"])) or set(
        map(int, old_fit.get("valid", []))
    ) != set(map(int, split["fit"]["valid"])):
        raise ValueError("old-student direct fit membership differs")
    if set(map(int, old_fit.get("valid", []))) & eval_groups:
        raise ValueError("old-student direct valid fit overlaps evaluation")

    from wnn.evaluation import paired_component_bootstrap, paired_component_moments
    from wnn.gmetric import METRIC_ID, WARMUP, GlobalAccumulator

    if METRIC_ID != "global_wp_clip2_v1" or WARMUP != 99:
        raise ValueError("official metric implementation differs from admission")
    with np.load(TARGETS, allow_pickle=False) as target_cache:
        target_groups = np.asarray(target_cache["row_group"], dtype=np.int64)
        if not np.array_equal(target_groups, np.arange(1873, dtype=np.int64)):
            raise ValueError("valid targets are not in canonical rowgroup order")
        positions = {int(group): i for i, group in enumerate(target_groups)}
        indices = np.asarray(sorted(positions[group] for group in eval_groups), dtype=np.int64)
        targets = np.asarray(target_cache["y"][indices], dtype=np.float32)
        scored = np.asarray(target_cache["mask"][indices], dtype=bool)
    if targets.shape != (468, 20000, 2) or scored.shape != (468, 20000):
        raise ValueError("target/mask fold selection shape differs")
    mask = scored & (np.arange(20000) >= WARMUP)[None, :]
    component_by_group = {
        int(item["row_group"]): int(item["component"]) for item in split["sequences"]["valid"]
    }
    components = np.asarray(
        [component_by_group[int(target_groups[ix])] for ix in indices], dtype=np.int64
    )
    if len(np.unique(components)) != 141:
        raise ValueError("evaluation panel does not contain the admitted 141 components")

    accs = {
        name: GlobalAccumulator()
        for name in ("s3", "s4", "s5", "teacher_trio", "fixed_half_oldstudent_trio", "oldstudent")
    }
    candidate_moments = np.empty((len(indices), 2, 6), dtype=np.float64)
    reference_moments = np.empty_like(candidate_moments)
    for start in range(0, len(indices), 8):
        end = min(start + 8, len(indices))
        ix = indices[start:end]
        y = targets[start:end]
        m = mask[start:end]
        old_batch = np.asarray(old_array[ix], dtype=np.float32)
        batches = [np.asarray(array[ix], dtype=np.float32) for array in prediction_arrays]
        # Match blend_score_remote.py arithmetic and weights exactly.
        trio = sum(1.0 * batch for batch in batches) / 3.0
        fixed = sum((3.0 * old_batch, *batches)) / 6.0
        for name, pred in zip(("s3", "s4", "s5"), batches, strict=True):
            accs[name].add(y, pred, m)
        accs["teacher_trio"].add(y, trio, m)
        accs["fixed_half_oldstudent_trio"].add(y, fixed, m)
        accs["oldstudent"].add(y, old_batch, m)
        candidate, reference = paired_component_moments(y, fixed, old_batch, m)
        candidate_moments[start:end] = candidate
        reference_moments[start:end] = reference
    scores = {name: accumulator.result() for name, accumulator in accs.items()}
    official = parse_official(OFFICIAL_STDOUT)
    expected_specs = {
        "teacher_trio": "+".join(PRED_NAMES),
        "fixed_half_oldstudent_trio": f"{OLD_RUN}:3+" + "+".join(PRED_NAMES),
        "oldstudent": OLD_RUN,
        "s3": PRED_NAMES[0],
        "s4": PRED_NAMES[1],
        "s5": PRED_NAMES[2],
    }
    if set(official) != set(expected_specs.values()):
        raise ValueError("official scorer outputs differ from the six admitted specs")
    for name, spec in expected_specs.items():
        for metric in ("wp", "t0", "t1"):
            if abs(scores[name][metric] - official[spec][metric]) > 0.0000051:
                raise ValueError(
                    f"local accumulator and official scorer disagree for {name}/{metric}"
                )

    bootstrap = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        components,
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    if abs(bootstrap["candidate_wp"] - scores["fixed_half_oldstudent_trio"]["wp"]) > 1e-10:
        raise ValueError("bootstrap candidate does not equal fixed official blend")
    if abs(bootstrap["reference_wp"] - scores["oldstudent"]["wp"]) > 1e-10:
        raise ValueError("bootstrap reference does not equal old-student official score")

    seed_wps = [scores[f"s{seed}"]["wp"] for seed in (3, 4, 5)]
    result = {
        "schema": "wnn.clean-t3-local-f0-equal-teacher-trio-grade.r1",
        "status": "PASS_OFFICIAL_EXPOSED_FOLD0_DISTINCT_SEED_AND_FIXED_TRIO_GRADE",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim": "Three independently seeded clean F0 H384 Muon-all/prop2 teachers and one predeclared equal teacher trio / fixed half-student half-trio comparison on the repeatedly exposed 468-group/141-component development panel. This is not a fresh holdout, cross-fold result, native precision comparison, final all-data score, or public forecast.",
        "admission_sha256": sha256(ADMISSION),
        "s3_terminal_handoff_sha256": sha256(S3_TERMINAL),
        "s3_prior_grade_sha256": sha256(S3_GRADE),
        "s4_s5_terminal_handoff_sha256": sha256(S4_S5_TERMINAL),
        "root_s4_s5_producer_preflight_sha256": sha256(ROOT_PREFLIGHT),
        "input_hashes": observed,
        "metric": {
            "id": METRIC_ID,
            "official_scorer_sha256": sha256(OFFICIAL_SCORER),
            "scorer_stdout_sha256": sha256(OFFICIAL_STDOUT),
            "scorer_stderr_sha256": sha256(OFFICIAL_STDERR),
            "split_sha256": sha256(SPLIT),
            "valid_targets_sha256": sha256(TARGETS),
            "fold": 0,
            "selected_groups": len(indices),
            "components": int(len(np.unique(components))),
            "warmup_row": WARMUP,
            "mask": "is_scored AND row >= 99",
            "clipping": [-2, 2],
            "weights": "original clipped-label magnitude weights",
            "evaluation_group_sorted_int64_le_sha256": hashlib.sha256(
                np.asarray(sorted(eval_groups), dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "endpoints": {
            "teachers": teacher_receipts,
            "old_student_control": {
                "run": OLD_RUN,
                "seed": 3,
                "checkpoint_epoch": 1,
                "inference_precision": old_sidecar.get("inference_precision"),
                "prediction_sha256": observed["old_student_prediction_sha256"],
                "checkpoint_sha256": observed["old_student_checkpoint_sha256"],
                "provenance_sha256": observed["old_student_provenance_sha256"],
            },
            "precision_note": "new teacher predictions are FP32; old student control predictions are FP16; output blend uses float32 arithmetic. This predates native Q8 precision grading.",
            "direct_membership_note": "This script rechecks direct fit lists and evaluation exclusion. Full source-qualified ancestry is inherited from the bound root producer-preflight receipts.",
        },
        "scores": scores,
        "distinct_seed_summary": {
            "seed_order": [3, 4, 5],
            "per_seed_wp": seed_wps,
            "mean_wp": float(np.mean(seed_wps)),
            "sample_sd_wp": float(np.std(seed_wps, ddof=1)),
            "interpretation": "descriptive variation across three independently fitted seeds on one exposed fold; not a confidence interval or cross-fold estimate",
        },
        "predeclared_fixed_comparisons": {
            "teacher_trio": "equal 1/3 output mean of seeds 3,4,5",
            "fixed_mixture": "old student 1/2 plus each teacher 1/6; no weight search",
            "old_student_control": OLD_RUN,
            "paired_bootstrap_fixed_mix_vs_old": {
                **bootstrap,
                "unit": "complete source-qualified validation component",
                "n_boot": N_BOOT,
                "seed": BOOTSTRAP_SEED,
                "helper_sha256": sha256(ROOT / "src/wnn/evaluation.py"),
                "helper_imported_torch": "torch" in sys.modules,
                "weights": [0.5, 0.5],
            },
        },
        "execution": {
            "official_command_argv": [
                str(ROOT / "scripts/limited"),
                str(ROOT / ".venv/bin/python"),
                str(OFFICIAL_SCORER),
                "0",
                "+".join(PRED_NAMES),
                f"{OLD_RUN}:3+" + "+".join(PRED_NAMES),
                OLD_RUN,
                *PRED_NAMES,
            ],
            "official_command_environment": {
                "WNN_SLICE": "wnn-campaign.slice",
                "WNN_CPU_QUOTA": "50%",
                "WNN_THREADS": "1",
                "WNN_MEM_MAX": "1500000000",
                "WNN_RUNTIME_MAX": "600s",
            },
            "grade_driver_command_argv": [
                str(ROOT / ".venv/bin/python"),
                str(Path(__file__).resolve()),
            ],
            "limits": {
                "cpu_quota_percent": 50,
                "threads": 1,
                "memory_max_bytes": 1500000000,
                "wall_seconds": 600,
                "gpu_count": 0,
                "swap_bytes": 0,
            },
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": sha256(Path(__file__).resolve()),
            "no_fit_no_gpu_no_public_forecast": True,
        },
        "limitations": [
            "Fold 0 and its 468 evaluation groups are repeatedly exposed; bootstrap uncertainty is conditional on this panel and these fitted endpoints.",
            "The three seed results are not cross-fold validation and do not establish a pristine holdout.",
            "The full-valid prediction arrays include fit-valid inputs; only the split-selected 468 evaluation groups are scored.",
            "The old-student control uses FP16 inference while the new teachers use FP32.",
        ],
    }
    out = HERE / "F0_EQUAL_TEACHER_TRIO_GRADE_R1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "scores": scores,
                "seed_summary": result["distinct_seed_summary"],
                "bootstrap": bootstrap,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
