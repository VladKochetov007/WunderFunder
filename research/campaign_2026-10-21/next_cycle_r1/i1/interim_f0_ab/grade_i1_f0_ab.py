"""Grade the completed F0 A/B i1 cell as an interim paired screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
PACKET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKET))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import grade_i1_panel as panel  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402

TARGET_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
RUNS = {"A": "i1_f0_s1001_a_r1", "B": "i1_f0_s1001_b_r1"}
GROUP_CHUNK = 8
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 271_828
MEMORY_LIMIT_BYTES = 1_400_000_000
PROCESS_RESERVE_BYTES = 256_000_000
CHUNK_RESERVE_BYTES = 128_000_000


class MixtureView:
    """Produce the fixed mixture per requested group chunk."""

    def __init__(self, control: np.ndarray, candidate: np.ndarray) -> None:
        self.control = control
        self.candidate = candidate

    def __len__(self) -> int:
        return len(self.control)

    def __getitem__(self, index: slice | np.ndarray) -> np.ndarray:
        return panel._mixture(self.control[index], self.candidate[index])


def _uncompressed_npz_bytes(path: Path) -> int:
    """Estimate decoded archive storage from ZIP metadata without loading arrays."""
    with ZipFile(path) as archive:
        return sum(info.file_size for info in archive.infolist())


def _memory_estimate_bytes(target_path: Path, groups: int, identities: dict[str, dict]) -> int:
    """Bound the NPZ, fold arrays, prediction pages, and fixed runtime reserves."""
    target_cache = _uncompressed_npz_bytes(target_path)
    selected_fold = groups * panel.ROWS * (2 * np.dtype(np.float32).itemsize + 1)
    prediction_pages = sum(int(identity["file_bytes"]) for identity in identities.values())
    return (
        target_cache
        + selected_fold
        + prediction_pages
        + CHUNK_RESERVE_BYTES
        + PROCESS_RESERVE_BYTES
    )


def _verify_paired_identity(identities: dict[str, dict[str, Any]]) -> None:
    """Require identical non-null initialization and fitting normalization."""
    for key in ("initial_parameter_sha256", "normalization_identity"):
        a_value = identities["A"].get(key)
        b_value = identities["B"].get(key)
        if not a_value or a_value != b_value:
            raise ValueError(f"F0 A/B do not share the same {key}")


def _load_targets(
    target_path: Path,
    groups: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load only the exact ordered F0 evaluation targets after source checks."""
    with np.load(target_path, allow_pickle=False) as cache:
        row_groups = np.asarray(cache["row_group"], dtype=np.int64)
        positions = {int(group): index for index, group in enumerate(row_groups)}
        if len(positions) != len(row_groups) or not set(groups).issubset(positions):
            raise ValueError("official target cache has duplicate or missing F0 groups")
        index = [positions[group] for group in groups]
        targets = np.asarray(cache["y"][index], dtype=np.float32)
        mask = np.asarray(cache["mask"][index], dtype=bool)
    mask &= np.arange(panel.ROWS)[None, :] >= 99
    if targets.shape != (len(groups), panel.ROWS, 2):
        raise ValueError("F0 targets do not cover every complete evaluation sequence")
    if mask.shape != (len(groups), panel.ROWS) or not mask.any():
        raise ValueError("F0 official mask has an invalid shape or no scored rows")
    return targets, mask


def _score_delta(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    """Return paired official WP and target deltas."""
    return {key: float(candidate[key] - reference[key]) for key in ("wp", "t0", "t1")}


def _paired_bootstrap(
    candidate: Any,
    reference: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> dict[str, Any]:
    """Bootstrap only sequences with scored rows, avoiding empty-bin moment calls."""
    candidate_moments: list[np.ndarray] = []
    reference_moments: list[np.ndarray] = []
    active_indices: list[np.ndarray] = []
    for start in range(0, len(targets), GROUP_CHUNK):
        stop = min(start + GROUP_CHUNK, len(targets))
        rows = np.flatnonzero(mask[start:stop].any(axis=1)) + start
        if rows.size == 0:
            continue
        left, right = paired_component_moments(
            targets[rows],
            candidate[rows],
            reference[rows],
            mask[rows],
        )
        candidate_moments.append(left)
        reference_moments.append(right)
        active_indices.append(rows)
    if not candidate_moments:
        raise ValueError("no evaluation sequence contributes to this paired bootstrap")
    active = np.concatenate(active_indices)
    return paired_component_bootstrap(
        np.concatenate(candidate_moments, axis=0),
        np.concatenate(reference_moments, axis=0),
        components[active],
        n_boot=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )


def _bin_mask(mask: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Select a fixed half-open time bin and reject an empty evaluation bin."""
    selected = mask.copy()
    selected[:, :start] = False
    selected[:, stop:] = False
    if not selected.any():
        raise ValueError(f"F0 position bin [{start}, {stop}) contains no scored rows")
    return selected


def _grade() -> dict[str, Any]:
    """Verify and score the first F0 A/B pair without changing the full panel grader."""
    fold = 0
    split_path = panel.SPLITS[fold]
    subset_path = panel.SUBSETS[fold]
    target_path = ROOT / "cache/valid_targets.npz"
    target_sha = panel.sha256(target_path)
    if target_sha != TARGET_SHA256:
        raise ValueError("official target cache differs from the frozen source bytes")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    split_sha = panel.file_sha256(split_path)
    if subset.get("split_manifest_sha256") != split_sha:
        raise ValueError("F0 subset and evaluation split SHA differ")
    groups, components = panel._expected_membership(fold, split)

    predictions: dict[str, np.ndarray] = {}
    identities: dict[str, dict[str, Any]] = {}
    for arm in ("A", "B"):
        predictions[arm], identities[arm] = panel._verify_run(
            fold, arm, RUNS[arm], groups, split_sha, subset
        )
    _verify_paired_identity(identities)

    estimate = _memory_estimate_bytes(target_path, len(groups), identities)
    if estimate > MEMORY_LIMIT_BYTES:
        raise MemoryError(f"preflight estimate {estimate} exceeds {MEMORY_LIMIT_BYTES} bytes")
    targets, mask = _load_targets(target_path, groups)
    predictions["mix_90B_10A"] = MixtureView(predictions["B"], predictions["A"])

    scores: dict[str, dict[str, Any]] = {
        name: {"overall": panel.score(targets, mask, prediction)}
        for name, prediction in predictions.items()
    }
    bin_masks: dict[str, np.ndarray] = {}
    for start, stop in panel.TIME_BINS:
        label = f"rows_{start}_{stop}"
        bin_masks[label] = _bin_mask(mask, start, stop)
        for name, prediction in predictions.items():
            scores[name][label] = panel.score(targets, bin_masks[label], prediction)

    comparisons: dict[str, Any] = {}
    for label, candidate_name in (
        ("A_minus_B", "A"),
        ("mix_90B_10A_minus_B", "mix_90B_10A"),
    ):
        candidate = predictions[candidate_name]
        comparisons[label] = {
            "overall_target_and_wp_deltas": _score_delta(
                scores[candidate_name]["overall"], scores["B"]["overall"]
            ),
            "paired_component_bootstrap": _paired_bootstrap(
                candidate, predictions["B"], targets, mask, components
            ),
            "position_bin_target_and_wp_deltas": {
                bin_label: _score_delta(scores[candidate_name][bin_label], scores["B"][bin_label])
                for bin_label in bin_masks
            },
        }

    if panel.sha256(target_path) != target_sha:
        raise ValueError("official target cache changed during grading")
    return {
        "schema": "wnn.i1-f0-ab-interim-grade.v1",
        "experiment_id": "E-PLATEAU-I1-20261021-R1",
        "metric_id": "global_wp_clip2_v1",
        "status": "interim_first_matched_cell_only",
        "scope": {
            "fold": "F0",
            "seed": 1001,
            "arms": RUNS,
            "remaining": "F2 and arm C pending",
            "exposure": (
                "Previously inspected development fold; bootstrap is conditional on these "
                "two fits and does not measure training-seed, fold, or public-transfer uncertainty."
            ),
            "interpretation_limit": (
                "A and B both disable the 156 history channels. A also masks raw and derived "
                "i1 inputs, so A-minus-B measures the raw/event i1 contribution in this cell; "
                "history value requires the pending C-minus-B contrast."
            ),
        },
        "fit_counts": {"fits_per_arm": 1, "total_new_fits_in_this_cell": 2},
        "mixture": {"id": "mix_90B_10A", "weights": {"B": 0.9, "A": 0.1}},
        "bootstrap": {
            "unit": "source-qualified connected evaluation component",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "source_sha256": {
            "valid_targets": target_sha,
            "split": split_sha,
            "fit_subset": panel.file_sha256(subset_path),
            "full_panel_grader": panel.sha256(Path(panel.__file__)),
            "scorer": panel.sha256(ROOT / "scripts/grade_prediction_bank.py"),
            "metric": panel.sha256(ROOT / "src/wnn/gmetric.py"),
            "evaluation": panel.sha256(ROOT / "src/wnn/evaluation.py"),
            "interim_grader": panel.sha256(Path(__file__)),
        },
        "memory_estimate": {
            "estimated_peak_bytes": estimate,
            "admission_limit_bytes": MEMORY_LIMIT_BYTES,
            "basis": (
                "Uncompressed NPZ ZIP-member sizes, full prediction file pages, selected F0 "
                "targets/mask, 128 MB chunk reserve, and 256 MB process reserve; metadata only."
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
        "source_cache_rechecked_sha256": panel.sha256(target_path),
    }


def main() -> None:
    """Write one no-search interim paired grade to a new JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = _grade()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {name: scores["overall"]["wp"] for name, scores in report["scores"].items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
