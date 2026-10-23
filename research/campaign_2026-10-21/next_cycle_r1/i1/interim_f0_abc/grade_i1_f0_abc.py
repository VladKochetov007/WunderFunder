"""Grade the completed F0 A/B/C cell as an interim paired screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", Path(__file__).resolve().parents[5])).resolve()
PACKET = Path(__file__).resolve().parents[1]
AB_PACKET = PACKET / "interim_f0_ab"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(PACKET), str(AB_PACKET)]

import grade_i1_f0_ab as interim  # noqa: E402
import grade_i1_panel as panel  # noqa: E402
import grade_prediction_bank as bank  # noqa: E402

from wnn import evaluation as evaluation_module  # noqa: E402
from wnn import gmetric as metric_module  # noqa: E402

TARGET_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
SUBSET_SHA256 = "706181dbb9a096653b9c63af58f8f66b70ffea9b3f522a3467ff3fd6f720d313"
TRAINER_IMPLEMENTATION_SHA256 = "31b7fd2eb82a133cf73b6c7c6ca79bdd247afa724f60e98f3c42db27ae4b375f"
RUN_IDENTITY_SOURCE_SHA256 = "7033ea007b9c7dcf498d3ab1e31aaeacb5725b7009fe719e9071c05b99272a76"
SOURCE_PINS = {
    "panel_grader": "90603ae923db63897545a73166725bedf006a686c5b3271899f9b10914379812",
    "ab_interim_helper": "20673f737c3154e134228f178c103d5f3a9f5d3422e2f113c75ec81d5d382f78",
    "prediction_scorer": "fb7a85a5837f61fa6c65cd10a799768006067e87d4c18e92d4b813ba83b9d82e",
    "metric": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
    "paired_evaluation": "96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e",
}
RUNS = {"A": "i1_f0_s1001_a_r1", "B": "i1_f0_s1001_b_r1", "C": "i1_f0_s1001_c_r1"}
GROUP_CHUNK = interim.GROUP_CHUNK
BOOTSTRAP_DRAWS = interim.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = interim.BOOTSTRAP_SEED
MEMORY_LIMIT_BYTES = interim.MEMORY_LIMIT_BYTES

# The helpers were authored in the isolated branch, while predictions and run
# artifacts live in the canonical checkout when root integrates the packet.
panel.ROOT = ROOT
panel.SPLITS[0] = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
panel.SUBSETS[0] = ROOT / "research/campaign_2026-10-20/plateau_review_r1/fit_f0_10pct_s1001.json"
bank.ROOT = ROOT


def _source_paths() -> dict[str, Path]:
    """Return the exact code files used by this fixed grader."""
    return {
        "panel_grader": Path(panel.__file__).resolve(),
        "ab_interim_helper": Path(interim.__file__).resolve(),
        "prediction_scorer": Path(bank.__file__).resolve(),
        "metric": Path(metric_module.__file__).resolve(),
        "paired_evaluation": Path(evaluation_module.__file__).resolve(),
    }


def _verify_sources() -> dict[str, str]:
    """Require the prepared helper, official scorer, and metric source pins."""
    checked: dict[str, str] = {}
    for name, path in _source_paths().items():
        actual = panel.sha256(path)
        if actual != SOURCE_PINS[name]:
            raise ValueError(f"{name} source hash differs from the frozen grade: {actual}")
        checked[name] = actual
    checked["interim_abc_grader"] = panel.sha256(Path(__file__).resolve())
    return checked


def _verify_trainer_sources(run: str) -> dict[str, str]:
    """Bind each fit to the same frozen trainer and run-identity implementation."""
    provenance_path = ROOT / "artifacts/runs" / run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    implementation = provenance.get("trainer_implementation", {})
    if (
        implementation.get("sha256") != TRAINER_IMPLEMENTATION_SHA256
        or implementation.get("missing_dependency_files") != []
        or not implementation.get("files")
        or provenance.get("run_identity_source_sha256") != RUN_IDENTITY_SOURCE_SHA256
    ):
        raise ValueError(f"{run}: training source identity differs from the fitted panel")
    return {
        "trainer_implementation_sha256": implementation["sha256"],
        "run_identity_source_sha256": provenance["run_identity_source_sha256"],
    }


def _verify_source_cell(identities: dict[str, dict[str, Any]]) -> None:
    """Require the exact same implementation source set for all three fits."""
    source_ids = {
        (value["trainer_implementation_sha256"], value["run_identity_source_sha256"])
        for value in identities.values()
    }
    if len(source_ids) != 1:
        raise ValueError("F0 A/B/C do not share the same trainer source identities")


def _grade() -> dict[str, Any]:
    """Verify exact fit/export identities, then grade five fixed F0 contrasts."""
    source_hashes = _verify_sources()
    fold = 0
    split_path = panel.SPLITS[fold]
    subset_path = panel.SUBSETS[fold]
    target_path = ROOT / "cache/valid_targets.npz"
    target_sha = panel.sha256(target_path)
    if target_sha != TARGET_SHA256:
        raise ValueError("official target cache differs from the frozen source bytes")
    split_sha = panel.file_sha256(split_path)
    subset_sha = panel.file_sha256(subset_path)
    if split_sha != SPLIT_SHA256 or subset_sha != SUBSET_SHA256:
        raise ValueError("F0 split or fit-subset bytes differ from the frozen cell")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    if subset.get("split_manifest_sha256") != split_sha:
        raise ValueError("F0 subset and evaluation split SHA differ")
    groups, components = panel._expected_membership(fold, split)

    predictions: dict[str, np.ndarray] = {}
    identities: dict[str, dict[str, Any]] = {}
    for arm in ("A", "B", "C"):
        prediction, identity = panel._verify_run(fold, arm, RUNS[arm], groups, split_sha, subset)
        identity.update(_verify_trainer_sources(RUNS[arm]))
        predictions[arm] = prediction
        identities[arm] = identity
    panel._verify_paired_fit_identities(identities, fold)
    _verify_source_cell(identities)

    estimate = interim._memory_estimate_bytes(target_path, len(groups), identities)
    if estimate > MEMORY_LIMIT_BYTES:
        raise MemoryError(f"preflight estimate {estimate} exceeds {MEMORY_LIMIT_BYTES} bytes")
    targets, mask = interim._load_targets(target_path, groups)
    predictions["mix_90B_10A"] = interim.MixtureView(predictions["B"], predictions["A"])
    predictions["mix_90B_10C"] = interim.MixtureView(predictions["B"], predictions["C"])

    scores: dict[str, dict[str, Any]] = {
        name: {"overall": panel.score(targets, mask, prediction)}
        for name, prediction in predictions.items()
    }
    bin_masks: dict[str, np.ndarray] = {}
    for start, stop in panel.TIME_BINS:
        label = f"rows_{start}_{stop}"
        bin_masks[label] = interim._bin_mask(mask, start, stop)
        for name, prediction in predictions.items():
            scores[name][label] = panel.score(targets, bin_masks[label], prediction)

    pairs = (
        ("A_minus_B", "A", "B"),
        ("C_minus_B", "C", "B"),
        ("C_minus_A", "C", "A"),
        ("mix_90B_10A_minus_B", "mix_90B_10A", "B"),
        ("mix_90B_10C_minus_B", "mix_90B_10C", "B"),
    )
    comparisons: dict[str, Any] = {}
    for label, candidate_name, reference_name in pairs:
        candidate = predictions[candidate_name]
        reference = predictions[reference_name]
        comparisons[label] = {
            "overall_target_and_wp_deltas": interim._score_delta(
                scores[candidate_name]["overall"], scores[reference_name]["overall"]
            ),
            "paired_component_bootstrap": interim._paired_bootstrap(
                candidate, reference, targets, mask, components
            ),
            "position_bin_target_and_wp_deltas": {
                bin_label: interim._score_delta(
                    scores[candidate_name][bin_label], scores[reference_name][bin_label]
                )
                for bin_label in bin_masks
            },
        }

    if panel.sha256(target_path) != target_sha:
        raise ValueError("official target cache changed during grading")
    if (
        panel.file_sha256(split_path) != split_sha
        or panel.file_sha256(subset_path) != subset_sha
        or panel.sha256(Path(__file__).resolve()) != source_hashes["interim_abc_grader"]
    ):
        raise ValueError("pinned grade sources or F0 manifests changed during grading")
    for name, path in _source_paths().items():
        if panel.sha256(path) != SOURCE_PINS[name]:
            raise ValueError(f"{name} source bytes changed during grading")
    return {
        "schema": "wnn.i1-f0-abc-interim-grade.v1",
        "experiment_id": "E-PLATEAU-I1-20261021-R1",
        "metric_id": "global_wp_clip2_v1",
        "status": "interim_f0_abc_cell_only",
        "scope": {
            "fold": "F0",
            "seed": 1001,
            "arms": RUNS,
            "fits_scored": 3,
            "panel_total_new_fits": 6,
            "not_graded_here": "F2; the complete panel remains pending",
            "exposure": (
                "Previously inspected development fold; component intervals condition on these "
                "three fitted models and do not estimate fit-seed, fold, or public-transfer uncertainty."
            ),
            "interpretation_limit": (
                "C versus B is the matched history-channel contrast. A versus B compares the "
                "raw/event i1 inputs while both disable history. No standalone score selects a mechanism."
            ),
        },
        "mixtures": {
            "mix_90B_10A": {"B": 0.9, "A": 0.1},
            "mix_90B_10C": {"B": 0.9, "C": 0.1},
        },
        "bootstrap": {
            "unit": "source-qualified connected evaluation component",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "bin_intervals": "not computed; position-bin point scores and deltas are reported",
        },
        "source_sha256": {
            **source_hashes,
            "valid_targets": target_sha,
            "split": split_sha,
            "fit_subset": subset_sha,
        },
        "memory_estimate": {
            "estimated_peak_bytes": estimate,
            "admission_limit_bytes": MEMORY_LIMIT_BYTES,
            "basis": (
                "Uncompressed target NPZ ZIP-member sizes, all three full prediction mappings, "
                "selected F0 targets/mask, 128 MB chunk reserve, and 256 MB process reserve."
            ),
        },
        "evaluation": {
            "groups": len(groups),
            "components": int(len(np.unique(components))),
            "scored_rows": int(mask.sum()),
            "prediction_identity_and_npy_coverage": identities,
        },
        "scores": scores,
        "comparisons": comparisons,
        "target_cache_rechecked_sha256": panel.sha256(target_path),
    }


def main() -> None:
    """Write the create-only interim grade after a passing byte-level preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = _grade()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {name: values["overall"]["wp"] for name, values in report["scores"].items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
