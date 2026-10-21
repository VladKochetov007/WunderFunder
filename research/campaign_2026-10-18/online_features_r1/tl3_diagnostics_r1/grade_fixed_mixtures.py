from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FOLD_REL = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGET_REL = "cache/valid_targets.npz"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_ROWS = 808_863
EXPECTED_GROUPS = 468
EXPECTED_COMPONENTS = 141
SCORED_FROM_ROW = 99
GROUP_CHUNK = 8
N_BOOT = 10_000
BOOT_SEED = 20261018

CANDIDATE_SPECS: dict[str, dict[str, str | None]] = {
    "parent": {
        "prediction": "cache/pred_warm_mechanism_f0_s3_c_r3_valid.npy",
        "sidecar": "cache/pred_warm_mechanism_f0_s3_c_r3_valid.json",
        "prediction_sha256": "c45c8d260d011fcc0c5900a4d11ee10d8261799716775ee6b26dd2cb6dd84482",
        "checkpoint_sha256": "78357eca2aec188e227544d46887935265720c35e99a1437269b952d01eb88c2",
        "zip": None,
        "zip_sha256": None,
    },
    "tl3": {
        "prediction": "cache/pred_tl3_unknown_ancestry_f0_valid.npy",
        "sidecar": "cache/pred_tl3_unknown_ancestry_f0_valid.json",
        "prediction_sha256": "3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3",
        "checkpoint_sha256": None,
        "zip": "submissions/team_tl3_original_20261018.zip",
        "zip_sha256": "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569",
    },
    "r3_native_q16x": {
        "prediction": "cache/pred_geometry_r3_reverse_half_q16x_20261018_v2_valid.npy",
        "sidecar": "cache/pred_geometry_r3_reverse_half_q16x_20261018_v2_valid.json",
        "prediction_sha256": "e82e1a9844b323b9886927adab4d3d579a6f278654eef322488de7784d4823e6",
        "checkpoint_sha256": None,
        "zip": "submissions/geometry_r3_reverse_half_q16x_20261018_v2.zip",
        "zip_sha256": "27f01f53ba32f7256ce1663e8b163b382e673fc96e7870fdebdf310d4886f98e",
    },
}

SOURCE_PATHS = (
    "src/wnn/gmetric.py",
    "src/wnn/evaluation.py",
    "src/wnn/splits.py",
    "src/wnn/metric.py",
    "scripts/grade_prediction_bank.py",
    "research/campaign_2026-09-30/audit/campaign_score.py",
)


def sha256_file(path: Path) -> str:
    """Hash an artifact incrementally so memory stays bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ints(values: list[int]) -> str:
    """Hash an ordered integer identity list using canonical JSON bytes."""
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    """Load a frozen helper module without invoking its command-line entry point."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CacheView:
    """Select target or mask rows from a memmapped stored-NPZ by evaluation order."""

    def __init__(self, array: np.memmap, row_indices: list[int], *, mask: bool) -> None:
        self.array = array
        self.row_indices = row_indices
        self.mask = mask

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, part: slice) -> np.ndarray:
        rows = [self.array[index] for index in self.row_indices[part]]
        result = np.stack(rows, axis=0)
        if self.mask:
            result = result.astype(bool, copy=False)
            result[:, :SCORED_FROM_ROW] = False
        return result


class WeightedPair:
    """Merge stable weighted covariance moments for a pair of row signals."""

    def __init__(self, merge: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self.moments = np.zeros(6, dtype=np.float64)
        self.merge = merge

    def add(self, left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> None:
        selected = weight > 0
        if not selected.any():
            return
        w = np.asarray(weight[selected], dtype=np.float64)
        x = np.asarray(left[selected], dtype=np.float64)
        y = np.asarray(right[selected], dtype=np.float64)
        total = float(w.sum())
        mean_x = float(np.dot(w, x) / total)
        mean_y = float(np.dot(w, y) / total)
        dx, dy = x - mean_x, y - mean_y
        part = np.array(
            [
                total,
                mean_x,
                mean_y,
                float(np.dot(w, dx * dx)),
                float(np.dot(w, dy * dy)),
                float(np.dot(w, dx * dy)),
            ],
            dtype=np.float64,
        )
        self.moments = self.merge(self.moments, part)

    def result(self) -> dict[str, float]:
        total, _, _, var_left, var_right, covariance = map(float, self.moments)
        correlation = (
            covariance / np.sqrt(var_left * var_right) if var_left > 0 and var_right > 0 else 0.0
        )
        return {
            "weight": total,
            "covariance": covariance / total if total else 0.0,
            "correlation": float(np.clip(correlation, -1.0, 1.0)),
        }


def load_candidates(
    root: Path,
    groups: list[int],
    split_sha: str,
    sequences_by_group: dict[int, dict[str, object]],
    grade: ModuleType,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Bind F0 prediction memmaps to their sidecars and reject nonfinite rows."""
    lazies: dict[str, object] = {}
    identities: dict[str, dict[str, object]] = {}
    for name, spec in CANDIDATE_SPECS.items():
        prediction = root / str(spec["prediction"])
        sidecar_path = root / str(spec["sidecar"])
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_groups = [int(value) for value in sidecar.get("row_groups", [])]
        if not sidecar_groups or len(set(sidecar_groups)) != len(sidecar_groups):
            raise ValueError(f"{name}: empty or duplicate sidecar row groups")
        if sidecar.get("split_manifest_sha256") != split_sha:
            raise ValueError(f"{name}: prediction split hash differs from fold0")
        if sidecar.get("prediction_sha256") != spec["prediction_sha256"]:
            raise ValueError(f"{name}: frozen prediction identity differs from recipe")
        prediction_sha = sha256_file(prediction)
        if prediction_sha != spec["prediction_sha256"]:
            raise ValueError(f"{name}: prediction bytes differ from frozen identity")
        if name == "parent" and sidecar.get("checkpoint_sha256") != spec["checkpoint_sha256"]:
            raise ValueError("parent checkpoint identity differs from its prediction sidecar")
        if name in ("tl3", "r3_native_q16x"):
            if sidecar_groups != groups:
                raise ValueError(f"{name}: F0 row-group order differs from the fixed split")
            expected_sequence_ids = [
                int(item["seq_ix"]) for group in groups for item in [sequences_by_group[group]]
            ]
            if [int(value) for value in sidecar.get("sequence_ids", [])] != expected_sequence_ids:
                raise ValueError(f"{name}: sequence identities differ from the split manifest")
        aligned = cast(Callable[..., object], grade.aligned)(
            {"prediction": str(prediction.relative_to(root))}, groups, split_sha
        )
        if isinstance(aligned, str):
            raise ValueError(f"{name}: {aligned}")
        array, order = cast(tuple[np.ndarray, list[int]], aligned)
        if array.shape[1:] != (20_000, 2) or array.dtype != np.float32:
            raise ValueError(f"{name}: prediction shape/dtype differs")
        lazy_type = grade.Lazy
        lazy = lazy_type(array, order)
        for start in range(0, EXPECTED_GROUPS, GROUP_CHUNK):
            if not np.isfinite(lazy[slice(start, start + GROUP_CHUNK)]).all():
                raise ValueError(f"{name}: nonfinite F0 prediction values")
        zip_path = spec["zip"]
        zip_hash = sha256_file(root / zip_path) if zip_path is not None else None
        if zip_hash != spec["zip_sha256"]:
            raise ValueError(f"{name}: frozen ZIP identity differs")
        if zip_path is not None and sidecar.get("zip_sha256") != zip_hash:
            raise ValueError(f"{name}: sidecar does not bind the exact ZIP")
        lazies[name] = lazy
        identities[name] = {
            "prediction": str(spec["prediction"]),
            "prediction_sha256": prediction_sha,
            "prediction_size_bytes": prediction.stat().st_size,
            "sidecar": str(spec["sidecar"]),
            "sidecar_sha256": sha256_file(sidecar_path),
            "sidecar_schema": sidecar.get("schema"),
            "row_group_count": len(sidecar_groups),
            "row_group_order_sha256": sha256_ints(sidecar_groups),
            "selected_f0_group_count": EXPECTED_GROUPS,
            "checkpoint_sha256": sidecar.get("checkpoint_sha256"),
            "archive": zip_path,
            "archive_sha256": zip_hash,
            "finite_selected_predictions": True,
        }
    return lazies, identities


def preflight(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Validate frozen identities and alignment without accessing label values."""
    root = root.resolve(strict=True)
    sys.path.insert(0, str(root / "src"))
    grade = load_module("fixed_mix_grade_bank", root / "scripts/grade_prediction_bank.py")
    campaign = load_module(
        "fixed_mix_campaign_score", root / "research/campaign_2026-09-30/audit/campaign_score.py"
    )
    from wnn.gmetric import _merge_moments

    split_path = root / FOLD_REL
    split_sha = sha256_file(split_path)
    if split_sha != SPLIT_SHA:
        raise ValueError("fold0 split manifest hash differs from the fixed recipe")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if int(manifest["fold"]) != 0:
        raise ValueError("the fixed split is not fold0")
    groups = [int(value) for value in manifest["evaluation"]["valid"]]
    if len(groups) != EXPECTED_GROUPS or len(set(groups)) != EXPECTED_GROUPS:
        raise ValueError("fold0 evaluation membership is not 468 unique groups")
    sequence_rows = {int(row["row_group"]): row for row in manifest["sequences"]["valid"]}
    if any(group not in sequence_rows for group in groups):
        raise ValueError("fold0 split lacks a source sequence record")
    components = [int(sequence_rows[group]["component"]) for group in groups]
    if len(set(components)) != EXPECTED_COMPONENTS:
        raise ValueError("fold0 evaluation component count differs from 141")

    target_path = root / TARGET_REL
    targets = cast(dict[str, np.memmap], campaign.mmap_stored_npz(target_path))
    target_groups = [int(value) for value in targets["row_group"]]
    target_position = {group: i for i, group in enumerate(target_groups)}
    if len(target_position) != len(target_groups) or any(
        group not in target_position for group in groups
    ):
        raise ValueError("target-cache row groups are not a unique superset of F0")
    target_indices = [target_position[group] for group in groups]
    lazies, candidate_identities = load_candidates(root, groups, split_sha, sequence_rows, grade)

    source_hashes = {relative: sha256_file(root / relative) for relative in SOURCE_PATHS}
    source_hashes["grade_fixed_mixtures.py"] = sha256_file(Path(__file__).resolve())
    fingerprint_payload = {
        "split_sha256": split_sha,
        "target_cache_sha256": sha256_file(target_path),
        "groups_sha256": sha256_ints(groups),
        "components_sha256": sha256_ints(components),
        "candidate_identities": candidate_identities,
        "source_hashes": source_hashes,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report: dict[str, object] = {
        "schema": "wnn.tl3-fixed-mixture-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint,
        "status": "pass",
        "fold": 0,
        "split_manifest": FOLD_REL,
        "split_manifest_sha256": split_sha,
        "evaluation_group_count": len(groups),
        "evaluation_group_order_sha256": sha256_ints(groups),
        "component_count": len(set(components)),
        "evaluation_component_order_sha256": sha256_ints(components),
        "target_cache": {
            "path": TARGET_REL,
            "size_bytes": target_path.stat().st_size,
            "sha256": fingerprint_payload["target_cache_sha256"],
            "mapped_y_shape": list(targets["y"].shape),
            "mapped_y_dtype": str(targets["y"].dtype),
            "mapped_mask_shape": list(targets["mask"].shape),
            "mapped_mask_dtype": str(targets["mask"].dtype),
            "row_group_count": len(target_groups),
            "label_values_accessed": False,
        },
        "candidates": candidate_identities,
        "source_hashes": source_hashes,
        "predeclared_blends": {
            "blend_75_tl3_25_r3": {"tl3": 0.75, "r3_native_q16x": 0.25},
            "blend_50_tl3_50_r3": {"tl3": 0.5, "r3_native_q16x": 0.5},
        },
    }
    state: dict[str, object] = {
        "groups": groups,
        "components": components,
        "target_indices": target_indices,
        "targets": targets,
        "lazies": lazies,
        "grade": grade,
        "merge_moments": _merge_moments,
    }
    return report, state


class FixedBlend:
    """Lazy fixed convex prediction blend that materializes only requested groups."""

    def __init__(self, left: object, right: object, left_weight: float) -> None:
        self.left, self.right, self.left_weight = left, right, left_weight

    def __len__(self) -> int:
        return len(self.left)

    def __getitem__(self, part: slice) -> np.ndarray:
        return np.asarray(
            self.left[part] * self.left_weight + self.right[part] * (1.0 - self.left_weight),
            dtype=np.float32,
        )


def score_recipe(
    root: Path,
    report: dict[str, object],
    state: dict[str, object],
    recipe: dict[str, object],
    recipe_path: Path,
) -> dict[str, object]:
    """Score the frozen endpoint set and conditional component comparisons."""
    if recipe.get("preflight_fingerprint") != report["fingerprint"]:
        raise ValueError("frozen recipe does not bind this exact preflight")
    targets = cast(dict[str, np.memmap], state["targets"])
    indices = cast(list[int], state["target_indices"])
    groups = cast(list[int], state["groups"])
    components = np.asarray(state["components"], dtype=np.int64)
    grade = cast(ModuleType, state["grade"])
    lazies = cast(dict[str, object], state["lazies"])
    prediction_map: dict[str, object] = dict(lazies)
    prediction_map["blend_75_tl3_25_r3"] = FixedBlend(lazies["tl3"], lazies["r3_native_q16x"], 0.75)
    prediction_map["blend_50_tl3_50_r3"] = FixedBlend(lazies["tl3"], lazies["r3_native_q16x"], 0.5)
    target_view = CacheView(targets["y"], indices, mask=False)
    mask_view = CacheView(targets["mask"], indices, mask=True)
    score_function = cast(Callable[[np.ndarray, np.ndarray, np.ndarray], dict], grade.score)
    scores: dict[str, dict[str, float]] = {}
    selected_rows = 0
    for start in range(0, EXPECTED_GROUPS, GROUP_CHUNK):
        selected_rows += int(mask_view[slice(start, start + GROUP_CHUNK)].sum())
    if selected_rows != EXPECTED_ROWS:
        raise ValueError(
            f"official target-cache mask selected {selected_rows}, expected {EXPECTED_ROWS}"
        )
    for name, prediction in prediction_map.items():
        scores[name] = score_function(target_view, mask_view, prediction)

    sys.path.insert(0, str(root / "src"))
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments

    merge_moments = cast(Callable[[np.ndarray, np.ndarray], np.ndarray], state["merge_moments"])

    sequence_stats = {
        name: np.zeros((EXPECTED_GROUPS, 2, 6), dtype=np.float64) for name in prediction_map
    }
    correction_pairs = {
        "corrections_tl3_vs_r3": [WeightedPair(merge_moments), WeightedPair(merge_moments)],
        "errors_tl3_vs_r3": [WeightedPair(merge_moments), WeightedPair(merge_moments)],
    }
    parent_lazy = cast(object, lazies["parent"])
    tl3_lazy = cast(object, lazies["tl3"])
    r3_lazy = cast(object, lazies["r3_native_q16x"])
    for index, target_index in enumerate(indices):
        target = np.asarray(targets["y"][target_index], dtype=np.float32)
        selection = np.asarray(targets["mask"][target_index], dtype=bool).copy()
        selection[:SCORED_FROM_ROW] = False
        raw_predictions = {
            name: prediction[slice(index, index + 1)][0]
            for name, prediction in prediction_map.items()
        }
        for name, prediction in raw_predictions.items():
            candidate_moments, _ = paired_component_moments(
                target[None], prediction[None], prediction[None], selection[None]
            )
            sequence_stats[name][index] = candidate_moments[0]
        label = np.clip(target, -2.0, 2.0)
        weight = np.abs(label) * selection[:, None]
        parent_prediction = np.clip(parent_lazy[slice(index, index + 1)][0], -2.0, 2.0)
        tl3_prediction = np.clip(tl3_lazy[slice(index, index + 1)][0], -2.0, 2.0)
        r3_prediction = np.clip(r3_lazy[slice(index, index + 1)][0], -2.0, 2.0)
        for target_index_in_pair in range(2):
            w = weight[:, target_index_in_pair]
            correction_pairs["corrections_tl3_vs_r3"][target_index_in_pair].add(
                tl3_prediction[:, target_index_in_pair]
                - parent_prediction[:, target_index_in_pair],
                r3_prediction[:, target_index_in_pair] - parent_prediction[:, target_index_in_pair],
                w,
            )
            correction_pairs["errors_tl3_vs_r3"][target_index_in_pair].add(
                tl3_prediction[:, target_index_in_pair] - label[:, target_index_in_pair],
                r3_prediction[:, target_index_in_pair] - label[:, target_index_in_pair],
                w,
            )

    comparisons = (
        ("tl3_vs_parent", "tl3", "parent"),
        ("r3_vs_parent", "r3_native_q16x", "parent"),
        ("blend_75_vs_parent", "blend_75_tl3_25_r3", "parent"),
        ("blend_50_vs_parent", "blend_50_tl3_50_r3", "parent"),
        ("blend_75_vs_tl3", "blend_75_tl3_25_r3", "tl3"),
        ("blend_75_vs_r3", "blend_75_tl3_25_r3", "r3_native_q16x"),
        ("blend_50_vs_tl3", "blend_50_tl3_50_r3", "tl3"),
        ("blend_50_vs_r3", "blend_50_tl3_50_r3", "r3_native_q16x"),
        ("tl3_vs_r3", "tl3", "r3_native_q16x"),
    )
    confidence = {
        key: paired_component_bootstrap(
            sequence_stats[candidate],
            sequence_stats[reference],
            components,
            n_boot=N_BOOT,
            seed=BOOT_SEED,
            centered=True,
        )
        for key, candidate, reference in comparisons
    }
    rounded_scores = {
        name: {key: float(value) for key, value in score.items()} for name, score in scores.items()
    }
    ranking = sorted(rounded_scores, key=lambda name: -rounded_scores[name]["wp"])
    correction_summary = {
        name: [pair.result() for pair in pairs] for name, pairs in correction_pairs.items()
    }
    result = {
        "schema": "wnn.tl3_fixed_mixture_grade.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "recipe_sha256": sha256_file(recipe_path),
        "preflight_fingerprint": report["fingerprint"],
        "metric_id": "global_wp_clip2_v1",
        "clip": [-2.0, 2.0],
        "mask": "official cached target mask AND row position >=99",
        "selected_rows": selected_rows,
        "evaluation_groups": len(groups),
        "evaluation_components": int(len(np.unique(components))),
        "exposure": "repeatedly exposed F0; TL3 exact training/teacher/initializer ancestry unknown and potentially fitted",
        "ancestry": {
            "parent": "fold0 parent checkpoint identity pinned by prediction sidecar",
            "r3_native_q16x": "exact served ZIP pinned by sidecar; inference-only reverse-half derivative of R2 checkpoint, not an independent fit",
            "tl3": "exact official-team ZIP pinned by sidecar; metadata calls parents all-data, exact training/teacher/initializer membership remains unknown",
        },
        "scores": rounded_scores,
        "point_ranking_descriptive_only": ranking,
        "fixed_weights": {
            "blend_75_tl3_25_r3": {"tl3": 0.75, "r3_native_q16x": 0.25},
            "blend_50_tl3_50_r3": {"tl3": 0.5, "r3_native_q16x": 0.5},
        },
        "component_bootstrap": {
            "scope": "conditional on these fixed prediction caches; resamples 141 F0 components, not training seeds or public outcomes",
            "replicates": N_BOOT,
            "seed": BOOT_SEED,
            "comparisons": confidence,
        },
        "weighted_complementarity": {
            "definition": "two-target label-magnitude-weighted covariance/correlation, using clipped candidate predictions and their errors on official selected rows",
            **correction_summary,
        },
        "expected_public_score": None,
        "claim_limit": "No clean-CV, training-ancestry, public-transfer, coefficient-selection, or causal-complementarity claim.",
    }
    return result


def main() -> None:
    """Run metadata-only preflight or the frozen fixed-mixture score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("preflight", "score"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report, state = preflight(args.repo_root)
    if args.mode == "preflight":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": report["status"], "fingerprint": report["fingerprint"]}))
        return
    if args.preflight_receipt is None or args.recipe is None:
        raise ValueError("score mode requires --preflight-receipt and --recipe")
    frozen_preflight = json.loads(args.preflight_receipt.read_text(encoding="utf-8"))
    if frozen_preflight.get("fingerprint") != report["fingerprint"]:
        raise ValueError("current inputs differ from the frozen pre-score preflight")
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if sha256_file(args.preflight_receipt) != recipe.get("preflight_receipt_sha256"):
        raise ValueError("recipe does not bind the frozen preflight receipt")
    if sha256_file(Path(__file__).resolve()) != recipe.get("grader_sha256"):
        raise ValueError("grader source differs from the frozen recipe")
    result = score_recipe(
        args.repo_root.resolve(strict=True), report, state, recipe, args.recipe.resolve()
    )
    result["preflight_receipt_sha256"] = sha256_file(args.preflight_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "scored",
                "selected_rows": result["selected_rows"],
                "ranking": result["point_ranking_descriptive_only"],
            }
        )
    )


if __name__ == "__main__":
    main()
