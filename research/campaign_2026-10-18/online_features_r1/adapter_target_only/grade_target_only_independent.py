from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PACKET = ROOT / "research/campaign_2026-10-18/online_features_r1/adapter_target_only"
SPLIT_REL = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGET_REL = "cache/valid_targets.npz"
SCORED_FROM_ROW = 99
GROUP_CHUNK = 8
BOOTSTRAP_COUNT = 2000
BOOTSTRAP_SEED = 20261018


def sha256_file(path: Path) -> str:
    """Hash a file incrementally to bind the grade to frozen artifact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def map_stored_npz(path: Path) -> dict[str, np.memmap]:
    """Memory-map uncompressed NPY members from the exact target-cache archive."""
    arrays: dict[str, np.memmap] = {}
    with zipfile.ZipFile(path) as archive, path.open("rb") as raw:
        for info in archive.infolist():
            if info.compress_type != zipfile.ZIP_STORED or not info.filename.endswith(".npy"):
                raise ValueError("target archive must contain only stored NPY members")
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError(f"unsupported NPY member version: {version}")
                header_bytes = member.tell()
            if dtype.hasobject:
                raise ValueError("object arrays are not permitted in target cache")
            raw.seek(info.header_offset)
            local = raw.read(30)
            if local[:4] != b"PK\x03\x04":
                raise ValueError("invalid ZIP local header")
            name_bytes, extra_bytes = struct.unpack("<HH", local[26:30])
            offset = info.header_offset + 30 + name_bytes + extra_bytes + header_bytes
            arrays[info.filename.removesuffix(".npy")] = np.memmap(
                path,
                dtype=dtype,
                shape=shape,
                offset=offset,
                mode="r",
                order="F" if fortran else "C",
            )
    required = {"y", "mask", "row_group"}
    if not required.issubset(arrays):
        raise ValueError(f"target cache lacks required arrays: {sorted(required - arrays.keys())}")
    return arrays


def assert_frozen_inputs(root: Path, frozen: dict[str, object]) -> None:
    """Recheck every pre-score hash before opening target or prediction values."""
    files = cast(dict[str, dict[str, object]], frozen["files"])
    runs = cast(dict[str, dict[str, object]], frozen["runs"])
    for relative, record in files.items():
        path = root / relative
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen source/data identity changed: {relative}")
    for label, record in runs.items():
        for field in ("checkpoint", "prediction", "sidecar", "provenance"):
            identity = cast(dict[str, object], record[field])
            path = root / cast(str, identity.get("path", identity.get("path")))
            if sha256_file(path) != identity["sha256"]:
                raise ValueError(f"frozen {label} {field} identity changed: {path}")
        provenance = cast(dict[str, object], record["provenance"])
        score_path = provenance.get("score_file")
        expected_score_hash = provenance.get("score_file_sha256")
        if (
            score_path is not None
            and sha256_file(root / cast(str, score_path)) != expected_score_hash
        ):
            raise ValueError(f"frozen {label} score.json changed")


def validate_fit_namespaces(root: Path, groups: list[int]) -> dict[str, object]:
    """Check validation holdout membership without mixing train/valid ID spaces."""
    result: dict[str, object] = {}
    memberships: dict[str, dict[str, list[int]]] = {}
    for label, run in (
        ("candidate", "geometry_adapter_f0_s3_target_only_r1"),
        ("r2_control", "geometry_adapter_f0_s3_r2"),
        ("parent", "warm_mechanism_f0_s3_c_r3"),
    ):
        provenance = json.loads((root / "artifacts/runs" / run / "provenance.json").read_text())
        fit = provenance.get("fit_row_groups", {})
        valid_groups = [int(group) for group in fit.get("valid", [])]
        overlap = sorted(set(valid_groups).intersection(groups))
        memberships[label] = {
            split: [int(group) for group in fit.get(split, [])] for split in ("train", "valid")
        }
        result[label] = {
            "fit_group_counts_by_source": {
                key: len(value) for key, value in memberships[label].items()
            },
            "valid_fit_f0_overlap_count": len(overlap),
            "valid_fit_f0_overlap_first20": overlap[:20],
        }
    candidate = memberships["candidate"]
    control = memberships["r2_control"]
    result["candidate_control_membership_equal_by_source"] = candidate == control
    result["all_valid_fit_memberships_disjoint_from_f0"] = all(
        cast(dict[str, object], result[label])["valid_fit_f0_overlap_count"] == 0
        for label in ("candidate", "r2_control", "parent")
    )
    result["namespace_note"] = (
        "The preliminary freeze script incorrectly unioned train and valid row-group integers. "
        "Those integers belong to separate parquet namespaces; only valid-fit IDs can be compared "
        "with evaluation.valid IDs. The preliminary overlap flag is an audit-script bug, not leakage."
    )
    return result


def score_official(
    y: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | str]:
    """Compute the unchanged official metric in eight-sequence chunks."""
    from wnn.gmetric import GlobalAccumulator

    accumulator = GlobalAccumulator()
    for start in range(0, len(y), GROUP_CHUNK):
        stop = min(start + GROUP_CHUNK, len(y))
        accumulator.add(y[start:stop], prediction[start:stop], mask[start:stop])
    return accumulator.result()


def paired_ci(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> dict[str, object]:
    """Bootstrap fixed-prediction WP deltas over the 141 exposed components."""
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments

    cand_moments, ref_moments = paired_component_moments(y, candidate, reference, mask)
    return paired_component_bootstrap(
        cand_moments,
        ref_moments,
        components,
        n_boot=BOOTSTRAP_COUNT,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )


def main() -> int:
    """Regrade one target-only adapter against its frozen control and parent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--frozen-inputs", type=Path, default=PACKET / "independent_grade_frozen_inputs.json"
    )
    parser.add_argument("--output", type=Path, default=PACKET / "independent_grade_result.json")
    args = parser.parse_args()

    root = args.repo_root.resolve(strict=True)
    frozen_path = args.frozen_inputs.resolve(strict=True)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert_frozen_inputs(root, frozen)

    split_path = root / SPLIT_REL
    split = json.loads(split_path.read_text(encoding="utf-8"))
    groups = [int(group) for group in split["evaluation"]["valid"]]
    components_by_group = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    components = np.asarray([components_by_group[group] for group in groups], dtype=np.int64)
    if len(groups) != 468 or len(set(components.tolist())) != 141:
        raise ValueError("fold0 panel differs from the frozen 468-group / 141-component panel")

    namespace_check = validate_fit_namespaces(root, groups)
    if not namespace_check["candidate_control_membership_equal_by_source"]:
        raise ValueError("candidate and matched R2 control fit memberships differ")

    target_path = root / TARGET_REL
    target_arrays = map_stored_npz(target_path)
    y_source = target_arrays["y"]
    mask_source = target_arrays["mask"]
    group_source = np.asarray(target_arrays["row_group"], dtype=np.int64)
    if y_source.shape != (len(group_source), 20_000, 2) or mask_source.shape != (
        len(group_source),
        20_000,
    ):
        raise ValueError("target-cache shapes do not match full sequence contract")
    if len(set(group_source.tolist())) != len(group_source):
        raise ValueError("duplicate target-cache row-group IDs")
    target_position = {int(group): index for index, group in enumerate(group_source)}
    missing_targets = [group for group in groups if group not in target_position]
    if missing_targets:
        raise ValueError(f"target cache misses {len(missing_targets)} F0 groups")

    names = {
        "candidate": "cache/pred_geometry_adapter_f0_s3_target_only_r1_valid.npy",
        "r2_control": "cache/pred_geometry_adapter_f0_s3_r2_valid.npy",
        "parent": "cache/pred_warm_mechanism_f0_s3_c_r3_valid.npy",
    }
    sidecars = {
        label: json.loads((root / relative.replace(".npy", ".json")).read_text(encoding="utf-8"))
        for label, relative in names.items()
    }
    prediction_sources: dict[str, np.ndarray] = {}
    positions: dict[str, dict[int, int]] = {}
    for label, relative in names.items():
        sidecar = sidecars[label]
        if sidecar.get("split_manifest_sha256") != frozen["fold0"]["sha256"]:
            raise ValueError(f"{label}: prediction split SHA differs from fold0")
        if sidecar.get("inference_precision") != "fp32":
            raise ValueError(f"{label}: expected FP32 prediction cache")
        row_groups = [int(value) for value in sidecar["row_groups"]]
        if label in ("candidate", "r2_control") and row_groups != groups:
            raise ValueError(f"{label}: row-group order differs from frozen F0")
        if not set(groups).issubset(row_groups):
            raise ValueError(f"{label}: prediction does not cover all F0 groups")
        array = np.load(root / relative, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(row_groups), 20_000, 2) or array.dtype != np.float32:
            raise ValueError(f"{label}: unexpected prediction shape or dtype")
        prediction_sources[label] = array
        positions[label] = {group: index for index, group in enumerate(row_groups)}

    y = np.empty((len(groups), 20_000, 2), dtype=np.float32)
    mask = np.empty((len(groups), 20_000), dtype=bool)
    predictions = {label: np.empty((len(groups), 20_000, 2), dtype=np.float32) for label in names}
    for index, group in enumerate(groups):
        target_index = target_position[group]
        y[index] = y_source[target_index]
        mask[index] = mask_source[target_index]
        for label, array in prediction_sources.items():
            predictions[label][index] = array[positions[label][group]]
    mask[:, :SCORED_FROM_ROW] = False

    finite_selected: dict[str, bool] = {}
    scores: dict[str, dict[str, float | int | str]] = {}
    for label, prediction in predictions.items():
        finite_selected[label] = bool(np.isfinite(prediction[mask]).all())
        if not finite_selected[label]:
            raise ValueError(f"{label}: nonfinite prediction on an officially selected row")
        scores[label] = score_official(y, prediction, mask)

    score_checks: dict[str, object] = {}
    for label in ("candidate", "r2_control"):
        score_path = root / "artifacts/runs" / str(frozen["runs"][label]["run"]) / "score.json"
        recorded = json.loads(score_path.read_text(encoding="utf-8"))
        score_checks[label] = {
            "recorded_wp": recorded["wp"],
            "independent_wp": scores[label]["wp"],
            "abs_diff": abs(float(recorded["wp"]) - float(scores[label]["wp"])),
            "matches_within_1e-12": abs(float(recorded["wp"]) - float(scores[label]["wp"]))
            <= 1e-12,
        }

    recipe = json.loads((PACKET / "recipe_target_only.json").read_text(encoding="utf-8"))
    expected_parent = float(recipe["comparison"]["parent_score"])
    deltas = {
        "candidate_minus_r2_control": float(scores["candidate"]["wp"])
        - float(scores["r2_control"]["wp"]),
        "candidate_minus_parent": float(scores["candidate"]["wp"]) - float(scores["parent"]["wp"]),
        "r2_control_minus_parent": float(scores["r2_control"]["wp"])
        - float(scores["parent"]["wp"]),
        "independently_graded_parent_minus_recipe_reference": float(scores["parent"]["wp"])
        - expected_parent,
    }
    intervals = {
        "candidate_minus_r2_control": paired_ci(
            y, predictions["candidate"], predictions["r2_control"], mask, components
        ),
        "candidate_minus_parent": paired_ci(
            y, predictions["candidate"], predictions["parent"], mask, components
        ),
    }
    result = {
        "schema": "wnn.adapter_target_only_independent_grade.v1",
        "grader_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_inputs_receipt": str(frozen_path.relative_to(root)),
        "frozen_inputs_receipt_sha256": sha256_file(frozen_path),
        "metric_id": "global_wp_clip2_v1",
        "metric_implementation": "canonical wnn.gmetric.GlobalAccumulator; clip2 and label-magnitude weights unchanged",
        "f0_panel": {
            "split_manifest_sha256": frozen["fold0"]["sha256"],
            "sequence_count": len(groups),
            "component_count": len(set(components.tolist())),
            "scored_from_row": SCORED_FROM_ROW,
            "selected_rows": int(mask.sum()),
            "evaluation_membership_sha256": sidecars["candidate"].get(
                "evaluation_membership_sha256"
            ),
            "exposure": "F0 is exposed and adaptively selected; this fixed-prediction interval is conditional on the exposed groups and these endpoints.",
        },
        "fit_membership": namespace_check,
        "artifacts": {
            label: {
                "run": frozen["runs"][label]["run"],
                "checkpoint_sha256": frozen["runs"][label]["checkpoint"]["sha256"],
                "prediction_sha256": frozen["runs"][label]["prediction"]["sha256"],
                "sidecar_sha256": frozen["runs"][label]["sidecar"]["sha256"],
                "sidecar_groups": frozen["runs"][label]["sidecar"]["row_group_count"],
                "fp32": sidecars[label].get("inference_precision") == "fp32",
                "finite_on_selected_rows": finite_selected[label],
            }
            for label in names
        },
        "scores": scores,
        "deltas": deltas,
        "conditional_component_bootstrap_95": intervals,
        "existing_score_reconciliation": score_checks,
        "matched_control": "R2 full-feature adapter: same source-qualified 468 F0 groups, checkpoint parent, fit membership by source, seed and frozen optimizer/sampler recipe; target-only objective is the intended treatment.",
        "parent_reference_reconciliation": {
            "recipe_parent_wp": expected_parent,
            "independent_parent_wp": scores["parent"]["wp"],
            "matches_within_1e-12": abs(float(scores["parent"]["wp"]) - expected_parent) <= 1e-12,
            "parent_cache_scope": sidecars["parent"].get("campaign_export_scope"),
            "parent_aligned_by_row_group_to_same_f0": True,
        },
        "old_head_invariance": {
            "status": "unverified_from_saved_artifacts",
            "basis": "objective_fidelity_receipt is AST/code-path only; integration_v2 fit_admitted is stale pre-fit metadata; train.log and run provenance contain no bitwise parent-head comparison. Per parent instruction no Torch/model checkpoint load was performed.",
        },
        "limitations": [
            "One seed, one exposed fold and adaptive mechanism/endpoint selection; bootstrap intervals condition on fixed predictions and do not cover seed, model-selection, transfer or public-score uncertainty.",
            "The inherited parent and its teacher/initializer ancestry are reused, not independent replications; lineage beyond the parent checkpoint identity was not re-audited here.",
            "The initial row-group overlap flag in independent_grade_frozen_inputs.json is invalid because it unions separate train and valid parquet ID spaces; the corrected source-specific check is recorded above.",
            "No public leaderboard score or gain forecast is implied.",
        ],
        "execution": {
            "interpreter": sys.executable,
            "scope": "wnn-campaign.slice",
            "cpu_quota": "20%",
            "memory_max_bytes": 1500000000,
            "swap_max_bytes": 0,
            "threads": 1,
            "torch_imported": False,
            "training_or_gpu_used": False,
            "custom_prior_grader_code_reused": False,
            "reused_components": [
                "canonical GlobalAccumulator",
                "canonical paired_component_moments/bootstrap",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scores": scores,
                "deltas": deltas,
                "score_checks": score_checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
