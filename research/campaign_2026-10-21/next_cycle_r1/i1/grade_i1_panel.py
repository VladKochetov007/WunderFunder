"""Grade the fixed A/B/C i1 panel on its complete F0/F2 evaluation groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grade_prediction_bank import aligned, score, sha256  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

PACKET = Path(__file__).resolve().parent
SPLITS = {
    0: ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    2: ROOT / "research/campaign_2026-09-29/artifacts/splits/fold2.json",
}
SUBSETS = {
    0: ROOT / "research/campaign_2026-10-20/plateau_review_r1/fit_f0_10pct_s1001.json",
    2: PACKET / "fit_f2_10pct_s1002.json",
}
SEEDS = {0: 1001, 2: 1002}
ARMS = ("A", "B", "C")
ROWS = 20_000
TIME_BINS = ((99, 500), (500, 2_000), (2_000, 10_000), (10_000, 20_000))
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 271828


def _npy_header(path: Path) -> dict[str, int]:
    """Parse the NPY header and verify its complete uncompressed float payload."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {version}")
        header_bytes = stream.tell()
    if fortran_order:
        raise ValueError("prediction NPY must use C-order layout")
    payload_bytes = math.prod(shape) * np.dtype(dtype).itemsize
    if path.stat().st_size != header_bytes + payload_bytes:
        raise ValueError("NPY byte count differs from its parsed header and payload")
    return {
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "file_bytes": path.stat().st_size,
    }


def _expected_membership(fold: int, split: dict[str, Any]) -> tuple[list[int], np.ndarray]:
    """Read exact evaluation order and its source-qualified component identities."""
    groups = [int(group) for group in split["evaluation"]["valid"]]
    component_of = {
        int(record["row_group"]): int(record["component"]) for record in split["sequences"]["valid"]
    }
    components = np.asarray([component_of[group] for group in groups], dtype=np.int64)
    expected = {0: (468, 141), 2: (443, 144)}[fold]
    if len(groups) != expected[0] or len(set(groups)) != len(groups):
        raise ValueError(f"F{fold} evaluation membership is not complete and unique")
    if len(np.unique(components)) != expected[1]:
        raise ValueError(f"F{fold} evaluation component count differs from the source manifest")
    return groups, components


def _verify_paired_fit_identities(identities: dict[str, dict[str, Any]], fold: int) -> None:
    """Require matched A/B/C seeded parameters and fitting-only normalization."""
    for identity in ("initial_parameter_sha256", "normalization_identity"):
        reference = identities.get("A", {}).get(identity)
        if reference is None or any(
            identities.get(arm, {}).get(identity) != reference for arm in ARMS[1:]
        ):
            raise ValueError(f"F{fold} arms do not share the same {identity}")


def _verify_run(
    fold: int,
    arm: str,
    run: str,
    groups: list[int],
    split_sha256: str,
    subset: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bind one cached prediction to the fixed fit, checkpoint and evaluation membership."""
    run_dir = ROOT / "artifacts/runs" / run
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    args = provenance.get("args", {})
    expected_control = {"A": "all", "B": "all", "C": "none"}[arm]
    expected_updates = {0: 4_620, 2: 4_830}[fold]
    updates_per_epoch = expected_updates // 3
    diagnostics = provenance.get("update_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 3:
        raise ValueError(f"{run}: expected exactly three epoch update records")
    for epoch_index, record in enumerate(diagnostics):
        epoch = record.get("epoch", {})
        if (
            record.get("epoch_index") != epoch_index
            or epoch.get("optimizer_attempts") != updates_per_epoch
            or epoch.get("optimizer_completed_updates") != updates_per_epoch
            or epoch.get("amp_skipped_updates") != 0
        ):
            raise ValueError(f"{run}: epoch {epoch_index} update evidence is incomplete")
    cumulative = diagnostics[-1].get("cumulative", {})
    if (
        cumulative.get("optimizer_attempts") != expected_updates
        or cumulative.get("optimizer_completed_updates") != expected_updates
        or cumulative.get("amp_skipped_updates") != 0
    ):
        raise ValueError(f"{run}: cumulative update evidence differs from the full fit budget")
    if (
        provenance.get("seed") != SEEDS[fold]
        or provenance.get("split_manifest_sha256") != split_sha256
        or provenance.get("selected_epoch") != 2
        or not provenance.get("initial_parameter_sha256")
        or not provenance.get("normalization_identity")
        or provenance.get("fit_subset_manifest_sha256") != file_sha256(SUBSETS[fold])
        or provenance.get("fit_subset_members_sha256") != subset["selected_members_sha256"]
        or provenance.get("fit_row_groups", {}).get("train") != subset["selected_groups"]["train"]
        or provenance.get("fit_row_groups", {}).get("valid") != subset["selected_groups"]["valid"]
        or args.get("name") != run
        or args.get("seed") != SEEDS[fold]
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != 2
        or args.get("arch") != "gru"
        or args.get("hidden") != 512
        or args.get("layers") != 1
        or args.get("batch") != 8
        or args.get("chunk") != 2000
        or args.get("epochs") != 3
        or args.get("no_eval_during_fit") is not True
        or args.get("loss") != "wmse"
        or args.get("loss_norm") != "chunk"
        or args.get("optim") != "adamw"
        or args.get("sched") != "onecycle"
        or args.get("lr") != 0.001
        or args.get("wd") != 0.01
        or args.get("beta2") != 0.999
        or args.get("clip") != 1.0
        or args.get("research_features") != "i1_history"
        or args.get("research_feature_control") != expected_control
        or bool(args.get("drop_i1")) != (arm == "A")
        or args.get("head_layers") != 1
        or args.get("precision") != "fp32"
        or args.get("train_source") != "train+valid"
        or args.get("valid_repeat") != 1
        or args.get("fit_subset_experiment_id") != subset["experiment_id"]
        or args.get("init")
        or args.get("distill")
        or args.get("distill_cache_root")
        or args.get("privileged_mode") != "none"
        or args.get("propensity")
    ):
        raise ValueError(f"{run}: fit provenance differs from the frozen A/B/C cell")

    checkpoint = run_dir / "model_ep2.pt"
    prediction_path = ROOT / "cache" / f"pred_{run}_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_groups_sha = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    if (
        sidecar.get("run") != run
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != 2
        or sidecar.get("checkpoint_sha256") != file_sha256(checkpoint)
        or sidecar.get("split_manifest_sha256") != split_sha256
        or sidecar.get("campaign_split_manifest_sha256") != split_sha256
        or sidecar.get("campaign_export_scope") != "evaluation"
        or sidecar.get("feature_only") is not True
        or sidecar.get("inference_precision") != "fp32"
        or sidecar.get("row_groups") != groups
        or sidecar.get("evaluation_membership_sha256") != expected_groups_sha
        or sidecar.get("prediction_sha256") != file_sha256(prediction_path)
    ):
        raise ValueError(f"{run}: prediction sidecar does not bind the exact evaluation fold")
    header = _npy_header(prediction_path)
    array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(groups), ROWS, 2) or array.dtype != np.float32:
        raise ValueError(f"{run}: prediction NPY header shape or dtype is wrong")
    mapped = aligned(
        {"prediction": prediction_path.relative_to(ROOT).as_posix()}, groups, split_sha256
    )
    if isinstance(mapped, str):
        raise ValueError(f"{run}: {mapped}")
    aligned_array, order = mapped
    if aligned_array.shape != array.shape or order != list(range(len(groups))):
        raise ValueError(f"{run}: evaluation row order is not exact")
    return array, {
        "run": run,
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "prediction_sha256": sidecar["prediction_sha256"],
        "sidecar_sha256": sha256(sidecar_path),
        "fit_subset_sha256": file_sha256(SUBSETS[fold]),
        "fit_subset_members_sha256": subset["selected_members_sha256"],
        "initial_parameter_sha256": provenance["initial_parameter_sha256"],
        "normalization_identity": provenance["normalization_identity"],
        "evaluation_membership_sha256": expected_groups_sha,
        "prediction_shape": list(array.shape),
        "prediction_dtype": str(array.dtype),
        **header,
    }


def _mixture(control: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Apply the frozen 90/10 control/candidate weights in float64, then store fp32."""
    return (
        0.9 * np.asarray(control, dtype=np.float64) + 0.1 * np.asarray(candidate, dtype=np.float64)
    ).astype(np.float32)


def _grade_fold(fold: int, run_names: dict[str, str]) -> dict[str, Any]:
    """Score three arms, two fixed mixtures and paired source-component deltas."""
    split_path, subset_path = SPLITS[fold], SUBSETS[fold]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    split_sha256 = file_sha256(split_path)
    if subset.get("split_manifest_sha256") != split_sha256:
        raise ValueError(f"F{fold} subset and evaluation split SHA differ")
    groups, components = _expected_membership(fold, split)

    predictions: dict[str, np.ndarray] = {}
    identities: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        predictions[arm], identities[arm] = _verify_run(
            fold, arm, run_names[arm], groups, split_sha256, subset
        )
    _verify_paired_fit_identities(identities, fold)

    target_path = ROOT / "cache/valid_targets.npz"
    with np.load(target_path, allow_pickle=False) as cache:
        positions = {int(group): index for index, group in enumerate(cache["row_group"])}
        if len(positions) != len(cache["row_group"]) or not set(groups).issubset(positions):
            raise ValueError("valid target cache has duplicate or missing evaluation groups")
        index = [positions[group] for group in groups]
        targets = np.asarray(cache["y"][index], dtype=np.float32)
        mask = np.asarray(cache["mask"][index], dtype=bool)
    mask &= np.arange(ROWS)[None, :] >= 99
    if targets.shape != (len(groups), ROWS, 2) or mask.shape != (len(groups), ROWS):
        raise ValueError(f"F{fold} target or official-mask shape is wrong")

    predictions["mix_90B_10A"] = _mixture(predictions["B"], predictions["A"])
    predictions["mix_90B_10C"] = _mixture(predictions["B"], predictions["C"])
    scores: dict[str, Any] = {
        name: {"overall": score(targets, mask, values)} for name, values in predictions.items()
    }
    for start, stop in TIME_BINS:
        bin_mask = mask.copy()
        bin_mask[:, :start] = False
        bin_mask[:, stop:] = False
        label = f"rows_{start}_{stop}"
        for name, values in predictions.items():
            scores[name][label] = score(targets, bin_mask, values)

    pairs = {
        "A_minus_B": (predictions["A"], predictions["B"]),
        "C_minus_B": (predictions["C"], predictions["B"]),
        "C_minus_A": (predictions["C"], predictions["A"]),
        "mix_90B_10A_minus_B": (predictions["mix_90B_10A"], predictions["B"]),
        "mix_90B_10C_minus_B": (predictions["mix_90B_10C"], predictions["B"]),
    }
    comparisons: dict[str, Any] = {}
    for label, (candidate, reference) in pairs.items():
        candidate_moments, reference_moments = paired_component_moments(
            targets, candidate, reference, mask
        )
        boot = paired_component_bootstrap(
            candidate_moments,
            reference_moments,
            components,
            n_boot=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        candidate_name, _, reference_name = label.partition("_minus_")
        comparisons[label] = {
            "wp_delta": scores[candidate_name]["overall"]["wp"]
            - scores[reference_name]["overall"]["wp"],
            "t0_delta": scores[candidate_name]["overall"]["t0"]
            - scores[reference_name]["overall"]["t0"],
            "t1_delta": scores[candidate_name]["overall"]["t1"]
            - scores[reference_name]["overall"]["t1"],
            "paired_component_bootstrap": boot,
            "position_bin_wp_deltas": {
                f"rows_{start}_{stop}": scores[candidate_name][f"rows_{start}_{stop}"]["wp"]
                - scores[reference_name][f"rows_{start}_{stop}"]["wp"]
                for start, stop in TIME_BINS
            },
        }
    return {
        "fold": fold,
        "split_manifest_sha256": split_sha256,
        "fit_subset_sha256": file_sha256(subset_path),
        "evaluation_groups": len(groups),
        "evaluation_components": int(len(np.unique(components))),
        "scored_rows": int(mask.sum()),
        "runs": run_names,
        "prediction_identity_and_npy_coverage": identities,
        "scores": scores,
        "comparisons": comparisons,
        "exposure": "previously inspected development fold; conditional on these two fits, with no training-seed or fold uncertainty",
    }


def main() -> None:
    """Write the fixed two-fold A/B/C grade without selecting mixture weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    target_path = ROOT / "cache/valid_targets.npz"
    target_sha = sha256(target_path)
    # Same immutable cache already pinned by the native and B16 endpoint plans.
    if target_sha != "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955":
        raise ValueError("official target cache differs from the frozen source bytes")
    names = {
        0: {"A": "i1_f0_s1001_a_r1", "B": "i1_f0_s1001_b_r1", "C": "i1_f0_s1001_c_r1"},
        2: {"A": "i1_f2_s1002_a_r1", "B": "i1_f2_s1002_b_r1", "C": "i1_f2_s1002_c_r1"},
    }
    report = {
        "schema": "wnn.i1-memory-panel-grade.v1",
        "experiment_id": "E-PLATEAU-I1-20261021-R1",
        "metric_id": "global_wp_clip2_v1",
        "fit_counts": {"fits_per_arm": 2, "total_new_fits": 6, "teacher_fits": 0},
        "mixtures": {"mix_90B_10A": {"B": 0.9, "A": 0.1}, "mix_90B_10C": {"B": 0.9, "C": 0.1}},
        "bootstrap": {
            "unit": "source-qualified connected evaluation component",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "source_sha256": {
            "valid_targets": target_sha,
            "grader": sha256(Path(__file__)),
            "scorer": sha256(ROOT / "scripts/grade_prediction_bank.py"),
            "metric": sha256(ROOT / "src/wnn/gmetric.py"),
            "evaluation": sha256(ROOT / "src/wnn/evaluation.py"),
        },
        "folds": {str(fold): _grade_fold(fold, names[fold]) for fold in (0, 2)},
    }
    if sha256(target_path) != target_sha:
        raise ValueError("official target cache changed during grading")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                fold: {name: values["scores"][name]["overall"]["wp"] for name in values["scores"]}
                for fold, values in report["folds"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
