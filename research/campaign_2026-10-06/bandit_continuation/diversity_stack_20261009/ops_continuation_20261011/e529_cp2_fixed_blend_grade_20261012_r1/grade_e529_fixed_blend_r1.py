from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011"
)
EXPORT_PACKET = OPS / "e529_muon512_f0_cp2_export_r1"
EXPORT_NAME = "e529_muon512_f0_cp2_prefix4_retained_r1"
H512_RUN = "bd26_gru512_full_s9_f0_muon_seeded_b8_amp4096_capacityv5r2_fit_20261009"
OLD_RUN = "oldz4_h230_student_f0_s3_r3"
H512_PRED = MAIN / "cache" / f"pred_{EXPORT_NAME}_valid.npy"
H512_SIDECAR = H512_PRED.with_suffix(".json")
OLD_PRED = MAIN / "cache" / f"pred_{OLD_RUN}_valid.npy"
OLD_SIDECAR = OLD_PRED.with_suffix(".json")
H512_CP = MAIN / "artifacts/runs" / H512_RUN / "model_ep2.pt"
H512_PROVENANCE = MAIN / "artifacts/runs" / H512_RUN / "provenance.json"
OLD_CP = MAIN / "artifacts/runs" / OLD_RUN / "model_ep1.pt"
OLD_PROVENANCE = MAIN / "artifacts/runs" / OLD_RUN / "provenance.json"
E529_FOLD = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/gru512_768_muon_fit_v5r2_20261009/packet/fold0.json"
)
LEGACY_FOLD = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
INPUT_BINDING = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/gru512_768_muon_fit_v5r2_20261009/terminal_collect_r1/outputs/gru_capacity_v5r2_project/fit_input_binding.json"
)
TARGETS = MAIN / "cache/valid_targets.npz"
ADMISSION = EXPORT_PACKET / "ROOT_FIXED_BLEND_GRADE_ADMISSION_R1.json"
EXPORT_ATTEMPT = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/e529-muon512-f0-cp2-prefix4-retained-export-r1.json"
)
EXPORT_RELEASE = EXPORT_PACKET / "ROOT_EXPORT_RELEASE_R1.json"
DURABLE_LAUNCH = EXPORT_PACKET / "ROOT_DURABLE_LAUNCH_R1.json"
PRODUCER_REVIEW = EXPORT_PACKET / "ROOT_ACTUAL_PRODUCER_REVIEW_R1.json"
SCORER = MAIN / "scripts/blend_score_remote.py"
METRIC = MAIN / "src/wnn/gmetric.py"
EVALUATION = MAIN / "src/wnn/evaluation.py"
OUT = Path(__file__).resolve().parent
EXPECTED = {
    "admission_schema": "wnn.root.e529-cp2-fixed-conditional-blend-grade-admission.r1",
    "split_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    "valid_targets_sha256": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
    "h512_checkpoint_sha256": "9cbfa401c3432bc027dfb809be07d2d6ab76c82c5007e1fe99d413af566b684c",
    "h512_provenance_sha256": "909c0ff76e742cfd16342f9063364a36986d07aa2754330b59c183cd38bf4f71",
    "h512_prediction_sha256": "f60b535ec82b04a91388ffea5c078ea888dece07e42ace73af473b1f8ec3ebef",
    "h512_sidecar_sha256": "e682f51c0050119a949ee052096adc20b7e187083772cd16951a626e8944f0f9",
    "old_checkpoint_sha256": "664643b2653a85d192f74a976d76afa933ae56c02b2621387d2f576fd61695eb",
    "old_provenance_sha256": "5af59cf6dd5ed263e45560d164f99c6103301076908c297f1dab3c645e3df13c",
    "old_prediction_sha256": "1e8259b829dcde4e3e395ac6e56e2d187b5b0c3e0ba343a98f9cbf530893712f",
    "old_sidecar_sha256": "5310ee451377522b43eb0f217933ad73ad162b944093515107806a7ee1984f7a",
    "metric_sha256": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
    "scorer_sha256": "87cb00f8a2d9f4b2c0a9c9dc39dc5f5ffa627b45c0b2c14de950ebbbb259594f",
    "evaluation_sha256": "96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e",
}
SCORED_FROM_ROW = 99
N_BOOT = 800
BOOTSTRAP_SEED = 271828
PREDICTION_SPEC = f"{OLD_RUN}:3+{EXPORT_NAME}:1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def validate_admission() -> tuple[dict, dict, dict, dict]:
    admission = load_json(ADMISSION)
    if (
        admission.get("schema") != EXPECTED["admission_schema"]
        or admission.get("status") != "admitted_one_fixed_conditional_comparison"
    ):
        raise ValueError("fixed-blend grade admission is absent or has the wrong status")
    if admission.get("owner") != "root" or admission.get("operator") != "submission_packager":
        raise ValueError("grade admission owner/operator mismatch")
    if admission.get("candidate_weights") != [0.75, 0.25]:
        raise ValueError("grade admission does not bind the predeclared 0.75/0.25 weights")
    if admission.get("resources") != {
        "cpu_quota_percent": 50,
        "threads": 1,
        "memory_max_bytes": 700000000,
        "wall_seconds": 600,
        "gpu_count": 0,
        "swap_bytes": 0,
        "parent_current_bytes": admission["resources"].get("parent_current_bytes"),
    }:
        raise ValueError("grade resource contract differs from the bounded admission")
    if admission["resources"].get("parent_current_bytes", 10**30) >= 18_000_000_000:
        raise ValueError("recorded parent memory is above campaign memory-high")
    contract = load_json(EXPORT_PACKET / "EXPORT_CONTRACT_R1.json")
    release = load_json(EXPORT_RELEASE)
    attempt = load_json(EXPORT_ATTEMPT)
    if (
        release.get("status") != "authorized_one_export"
        or release.get("launch_authorized") is not True
    ):
        raise ValueError("root export release is not authorized")
    if (attempt.get("status"), attempt.get("exit_status")) != ("finished", 0):
        raise ValueError("export attempt is not terminal-success")
    if contract["producer"]["checkpoint_sha256"] != EXPECTED["h512_checkpoint_sha256"]:
        raise ValueError("export contract checkpoint pin differs from the fixed comparison")
    fixed_blend = contract["fixed_blend"]
    if (
        fixed_blend["old_student"]["run_id"] != OLD_RUN
        or fixed_blend["old_student"]["epoch"] != 1
        or fixed_blend["old_student"]["weight"] != 0.75
        or fixed_blend["h512_prefix"]["run_id"] != H512_RUN
        or fixed_blend["h512_prefix"]["checkpoint_epoch_argument"] != 2
        or fixed_blend["h512_prefix"]["weight"] != 0.25
    ):
        raise ValueError("export contract fixed-blend endpoint/weights mismatch")
    return admission, contract, release, attempt


def validate_member(
    run: str,
    epoch: int,
    prediction_path: Path,
    sidecar_path: Path,
    checkpoint_path: Path,
    provenance_path: Path,
    expected_checkpoint_sha: str,
    expected_provenance_sha: str,
    expected_prediction_sha: str,
    expected_sidecar_sha: str,
    split_sha: str,
    evaluation_groups: set[int],
) -> dict:
    checkpoint_sha = require_sha(checkpoint_path, expected_checkpoint_sha, f"{run} checkpoint")
    provenance_sha = require_sha(provenance_path, expected_provenance_sha, f"{run} provenance")
    sidecar_sha = require_sha(sidecar_path, expected_sidecar_sha, f"{run} sidecar")
    prediction_sha = require_sha(prediction_path, expected_prediction_sha, f"{run} prediction")
    sidecar = load_json(sidecar_path)
    provenance = load_json(provenance_path)
    if (sidecar.get("run"), sidecar.get("split"), sidecar.get("epoch")) != (run, "valid", epoch):
        raise ValueError(f"{run} sidecar endpoint identity mismatch")
    if (
        sidecar.get("checkpoint_sha256") != checkpoint_sha
        or sidecar.get("prediction_sha256") != prediction_sha
    ):
        raise ValueError(f"{run} sidecar checkpoint/prediction binding mismatch")
    if sidecar.get("split_manifest_sha256") != split_sha:
        raise ValueError(f"{run} prediction sidecar split-manifest SHA mismatch")
    if run == H512_RUN and sidecar.get("epoch") != 2:
        raise ValueError("H512 sidecar epoch does not equal the fixed CP2 prefix")
    if sidecar.get("row_groups") != list(range(1873)):
        raise ValueError(f"{run} prediction row order is not canonical")
    provenance_checkpoints = provenance.get("checkpoint_sha256")
    provenance_checkpoint = (
        provenance_checkpoints.get(checkpoint_path.name)
        if isinstance(provenance_checkpoints, dict)
        else provenance_checkpoints
    )
    if provenance_checkpoint != checkpoint_sha:
        raise ValueError(f"{run} provenance checkpoint binding mismatch")
    provenance_eval = set(map(int, provenance.get("evaluation_row_groups", [])))
    fit_valid = set(map(int, provenance.get("fit_row_groups", {}).get("valid", [])))
    if provenance_eval != evaluation_groups:
        raise ValueError(f"{run} provenance evaluation membership mismatch")
    if fit_valid & evaluation_groups:
        raise ValueError(f"{run} fitting membership overlaps the evaluated groups")
    array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (1873, 20000, 2) or array.dtype != np.dtype("float32"):
        raise ValueError(f"{run} prediction NPY header shape/dtype mismatch")
    if prediction_path.stat().st_size != 299_680_128:
        raise ValueError(f"{run} prediction file byte count mismatch")
    return {
        "run_id": run,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha,
        "provenance_sha256": provenance_sha,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "prediction_sidecar_sha256": sidecar_sha,
        "prediction_shape": list(array.shape),
        "prediction_dtype": str(array.dtype),
        "fit_valid_group_count": len(fit_valid),
        "evaluation_group_count": len(provenance_eval),
        "fit_valid_evaluation_overlap": 0,
        "source_membership_status": "direct provenance confirms evaluation groups excluded from fit-valid groups",
    }


def run_official_scorer() -> tuple[list[dict[str, float | str]], str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    command = [sys.executable, str(SCORER), "0", PREDICTION_SPEC, OLD_RUN]
    completed = subprocess.run(
        command, cwd=MAIN, env=env, capture_output=True, text=True, timeout=180, check=False
    )
    (OUT / "OFFICIAL_SCORE_STDOUT_R1.txt").write_text(completed.stdout)
    (OUT / "OFFICIAL_SCORE_STDERR_R1.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"official blend scorer returned {completed.returncode}: {completed.stderr[-1000:]}"
        )
    rows: list[dict[str, float | str]] = []
    for line in completed.stdout.splitlines():
        match = re.fullmatch(r"([0-9.]+) t0 ([0-9.]+) t1 ([0-9.]+)  (.+)", line.strip())
        if match is None:
            raise ValueError(f"unexpected official scorer line: {line!r}")
        rows.append(
            {
                "wp": float(match.group(1)),
                "t0": float(match.group(2)),
                "t1": float(match.group(3)),
                "spec": match.group(4),
            }
        )
    if [row["spec"] for row in rows] != [PREDICTION_SPEC, OLD_RUN]:
        raise ValueError("official scorer returned unexpected member specifications")
    return rows, completed.stdout, completed.stderr


def main() -> int:
    output_paths = [
        OUT / "FIXED_BLEND_GRADE_R1.json",
        OUT / "OFFICIAL_SCORE_STDOUT_R1.txt",
        OUT / "OFFICIAL_SCORE_STDERR_R1.txt",
    ]
    collisions = [str(path) for path in output_paths if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite prior grade outputs: {collisions}")
    admission, contract, release, attempt = validate_admission()
    metric_sha = require_sha(METRIC, EXPECTED["metric_sha256"], "official metric source")
    scorer_sha = require_sha(SCORER, EXPECTED["scorer_sha256"], "official blend scorer")
    evaluation_sha = require_sha(
        EVALUATION, EXPECTED["evaluation_sha256"], "component bootstrap helper"
    )
    split_sha = require_sha(
        E529_FOLD, EXPECTED["split_sha256"], "source-qualified E529 fold-0 manifest"
    )
    input_binding = load_json(INPUT_BINDING)
    if (
        input_binding.get("sources", {}).get("cache/valid_targets.npz", {}).get("sha256")
        != EXPECTED["valid_targets_sha256"]
    ):
        raise ValueError("E529 input binding does not pin valid_targets.npz to the expected SHA")
    e529_split = load_json(E529_FOLD)
    legacy_split = load_json(LEGACY_FOLD)
    e529_eval = list(map(int, e529_split["evaluation"]["valid"]))
    legacy_eval = list(map(int, legacy_split["evaluation"]["valid"]))
    if len(e529_eval) != 468 or len(set(e529_eval)) != 468 or set(e529_eval) != set(legacy_eval):
        raise ValueError("legacy scorer and source-qualified E529 evaluation IDs differ")
    evaluation_groups = set(e529_eval)
    member_results = {
        "old_student": validate_member(
            OLD_RUN,
            1,
            OLD_PRED,
            OLD_SIDECAR,
            OLD_CP,
            OLD_PROVENANCE,
            EXPECTED["old_checkpoint_sha256"],
            EXPECTED["old_provenance_sha256"],
            EXPECTED["old_prediction_sha256"],
            EXPECTED["old_sidecar_sha256"],
            split_sha,
            evaluation_groups,
        ),
        "h512_prefix": validate_member(
            H512_RUN,
            2,
            H512_PRED,
            H512_SIDECAR,
            H512_CP,
            H512_PROVENANCE,
            EXPECTED["h512_checkpoint_sha256"],
            EXPECTED["h512_provenance_sha256"],
            EXPECTED["h512_prediction_sha256"],
            EXPECTED["h512_sidecar_sha256"],
            split_sha,
            evaluation_groups,
        ),
    }
    scorer_rows, scorer_stdout, scorer_stderr = run_official_scorer()

    sys.path.insert(0, str(MAIN / "src"))
    sys.path.insert(0, str(MAIN / "scripts"))
    from evaluate_bandit_continuation import _official_mask

    from wnn.evaluation import paired_component_bootstrap, paired_component_moments

    target_sha = sha256(TARGETS)
    if target_sha != EXPECTED["valid_targets_sha256"]:
        raise ValueError("valid_targets.npz bytes differ from the pinned E529 input archive")
    with np.load(TARGETS, allow_pickle=False) as cache:
        row_groups = np.asarray(cache["row_group"], dtype=np.int64)
        if row_groups.shape != (1873,) or not np.array_equal(
            row_groups, np.arange(1873, dtype=np.int64)
        ):
            raise ValueError("official target-cache row-group order is not canonical 0..1872")
        position = {int(group): index for index, group in enumerate(row_groups)}
        indices = np.asarray(sorted(position[group] for group in e529_eval), dtype=np.int64)
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        is_scored = np.asarray(cache["mask"][indices], dtype=bool)
    if targets.shape != (468, 20000, 2) or is_scored.shape != (468, 20000):
        raise ValueError("selected targets or is_scored panel has unexpected shape")
    mask = _official_mask(is_scored)
    expected_mask = is_scored & (np.arange(20000) >= 99)[None, :]
    if not np.array_equal(mask, expected_mask):
        raise ValueError("official mask differs from is_scored AND need_prediction>=99")
    groups_in_target_order = row_groups[indices]
    component_by_group = {
        int(row["row_group"]): int(row["component"]) for row in legacy_split["sequences"]["valid"]
    }
    if not evaluation_groups.issubset(component_by_group):
        raise ValueError("fold-0 component map is missing evaluation groups")
    components = np.asarray(
        [component_by_group[int(group)] for group in groups_in_target_order], dtype=np.int64
    )
    if len(np.unique(components)) != 141:
        raise ValueError("fold-0 evaluation component count is not 141")
    old_all = np.load(OLD_PRED, mmap_mode="r", allow_pickle=False)
    h512_all = np.load(H512_PRED, mmap_mode="r", allow_pickle=False)
    old = np.asarray(old_all[indices], dtype=np.float32)
    h512 = np.asarray(h512_all[indices], dtype=np.float32)
    del old_all, h512_all
    candidate = (3.0 * old + h512) / 4.0
    candidate_moments, reference_moments = paired_component_moments(targets, candidate, old, mask)
    bootstrap = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        components,
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    if abs(bootstrap["candidate_wp"] - float(scorer_rows[0]["wp"])) > 1e-5:
        raise ValueError("component-moment candidate WP disagrees with official scorer output")
    if abs(bootstrap["reference_wp"] - float(scorer_rows[1]["wp"])) > 1e-5:
        raise ValueError("component-moment baseline WP disagrees with official scorer output")
    result = {
        "schema": "wnn.e529-h512-cp2-fixed-conditional-blend-grade.r1",
        "status": "PASS_FIXED_EXPOSED_FOLD0_COMPARISON",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim": "One predeclared fixed blend versus its old-student control on the 468-group/141-component exposed fold-0 development panel; not a fresh holdout, final four-epoch H512 endpoint, public score, or forecast.",
        "export": {
            "attempt_id": attempt.get("id"),
            "attempt_status": attempt.get("status"),
            "exit_status": attempt.get("exit_status"),
            "started_utc": attempt.get("started_utc"),
            "finished_utc": attempt.get("finished_utc"),
            "prediction_sha256": EXPECTED["h512_prediction_sha256"],
            "prediction_sidecar_sha256": EXPECTED["h512_sidecar_sha256"],
            "checkpoint_epoch": 2,
            "source_checkpoint_is_prefix_of_interrupted_four_epoch_run": True,
            "durable_launch_receipt_sha256": sha256(DURABLE_LAUNCH),
            "root_export_release_sha256": sha256(EXPORT_RELEASE),
            "producer_review_sha256": sha256(PRODUCER_REVIEW),
        },
        "baseline": member_results["old_student"],
        "candidate_member": member_results["h512_prefix"],
        "comparison": {
            "candidate": PREDICTION_SPEC,
            "candidate_weights": {OLD_RUN: 0.75, H512_RUN + "@ep2": 0.25},
            "control": OLD_RUN,
            "candidate_official_score": scorer_rows[0],
            "control_official_score": scorer_rows[1],
            "paired_component_bootstrap": bootstrap,
            "evaluation_groups": len(e529_eval),
            "evaluation_components": int(len(np.unique(components))),
            "evaluation_group_id_sha256": hashlib.sha256(
                np.asarray(sorted(evaluation_groups), dtype="<i8").tobytes()
            ).hexdigest(),
            "metric_id": "global_wp_clip2_v1",
            "mask": "is_scored AND need_prediction (row >= 99)",
            "metric_clip": [-2, 2],
            "weighting": "original clipped-label magnitude weights in official metric",
            "bootstrap_unit": "complete component",
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "exposure": "previously inspected fold-0 development panel; one source-qualified seed per member",
        },
        "data_and_source_bindings": {
            "source_qualified_fold0_manifest_sha256": split_sha,
            "legacy_scorer_fold0_manifest_sha256": sha256(LEGACY_FOLD),
            "evaluation_ids_exactly_equal": True,
            "valid_targets_sha256": target_sha,
            "official_metric_source_sha256": metric_sha,
            "official_scorer_source_sha256": scorer_sha,
            "component_bootstrap_source_sha256": evaluation_sha,
            "grade_admission_sha256": sha256(ADMISSION),
            "export_contract_sha256": sha256(EXPORT_PACKET / "EXPORT_CONTRACT_R1.json"),
        },
        "resources": admission["resources"],
        "official_scorer_stdout": scorer_stdout,
        "official_scorer_stderr": scorer_stderr,
        "not_claimed": [
            "public-score forecast",
            "fresh holdout",
            "H512 standalone score",
            "completed four-epoch H512 run",
            "weight search",
        ],
    }
    (OUT / "FIXED_BLEND_GRADE_R1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_wp": bootstrap["candidate_wp"],
                "control_wp": bootstrap["reference_wp"],
                "delta": bootstrap["mean_diff"],
                "ci95": bootstrap["ci95"],
                "n_components": bootstrap["n_components"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
