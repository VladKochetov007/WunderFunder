"""Grade frozen blend endpoints and paired connected-component uncertainty."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from grade_submissions import predictions as map_cached_predictions
from grade_submissions import score as checked_official_score

from wnn.gmetric import METRIC_ID, reference_global_wp, wp_from_sums, wp_sums

REPO = Path(__file__).resolve().parents[4]
AUDIT_DIR = Path(__file__).resolve().parent
CACHE = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
TL3C_DIR = REPO / "submissions/tl3c_pr49_20261029/blend-int8-full-v3"
SPLIT_PATH = REPO / "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261114
EXPECTED_GROUPS = 1873
EXPECTED_SCORED_ROWS = 3_528_273

CACHE_RUNS = {
    "kd50_s3": "l14_kd50_s3",
    "kd50_s4": "l14_kd50_s4",
    "kd90_s3": "l14_kd90_s3",
    "fbc1k_s5": "tt29rep_fbc1k_s5",
}
ENDPOINTS = (
    "kd50_pair_equal",
    "kd90_s3",
    "fbc1k_s5",
    "kd90_fbc1k_equal",
    "tl3c_int8_native",
)
CONTRASTS = (
    ("kd90_s3_minus_kd50_pair", "kd90_s3", "kd50_pair_equal"),
    ("fbc1k_s5_minus_kd50_pair", "fbc1k_s5", "kd50_pair_equal"),
    ("kd90_fbc1k_equal_minus_kd50_pair", "kd90_fbc1k_equal", "kd50_pair_equal"),
    ("tl3c_minus_kd50_pair", "tl3c_int8_native", "kd50_pair_equal"),
    ("kd90_fbc1k_equal_minus_kd90", "kd90_fbc1k_equal", "kd90_s3"),
    ("kd90_fbc1k_equal_minus_fbc1k", "kd90_fbc1k_equal", "fbc1k_s5"),
    ("kd90_fbc1k_equal_minus_tl3c", "kd90_fbc1k_equal", "tl3c_int8_native"),
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Abort closed when an artifact identity or metric invariant fails."""
    if not condition:
        raise RuntimeError(message)


def cached_artifact(run: str, groups: np.ndarray, seq_len: int) -> tuple[Path, Path]:
    """Verify cached prediction layout and full source row-group coverage."""
    sidecar_path = CACHE / f"pred_{run}_valid.json"
    prediction_path = CACHE / f"pred_{run}_valid.npy"
    sidecar = json.loads(sidecar_path.read_text())
    row_groups = [int(value) for value in sidecar["row_groups"]]
    require(len(row_groups) == len(set(row_groups)), f"duplicate row groups: {run}")
    require(set(row_groups) == set(map(int, groups)), f"row-group coverage mismatch: {run}")
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    require(
        prediction.shape == (len(row_groups), seq_len, 2),
        f"layout mismatch: {run}: {prediction.shape}",
    )
    require(prediction.dtype == np.float32, f"dtype mismatch: {run}: {prediction.dtype}")
    return sidecar_path, prediction_path


def component_layout(groups: np.ndarray) -> tuple[list[int], dict[int, int], dict[int, list[int]]]:
    """Map each source row group to its frozen chronological component."""
    sequence = json.loads(SPLIT_PATH.read_text())["sequences"]["valid"]
    row_to_component = {int(item["row_group"]): int(item["component"]) for item in sequence}
    require(len(row_to_component) == len(sequence), "duplicate row group in component map")
    require(
        set(row_to_component) == set(map(int, groups)),
        "component map does not cover valid row groups",
    )
    grouped: dict[int, list[int]] = {}
    for row_group, component in row_to_component.items():
        grouped.setdefault(component, []).append(row_group)
    return sorted(grouped), row_to_component, grouped


def score_from_sums(sums: np.ndarray) -> float:
    """Return the mean official target score from six raw sufficient sums."""
    return float(np.mean(wp_from_sums(sums)))


def main() -> None:
    """Run the frozen endpoint grade, parity checks, and paired bootstrap."""
    with np.load(CACHE / "valid_targets.npz", allow_pickle=False) as target_archive:
        y = target_archive["y"]
        mask = target_archive["mask"]
        groups = target_archive["row_group"].astype(np.int64)
    require(y.shape == (EXPECTED_GROUPS, 20_000, 2), f"target layout mismatch: {y.shape}")
    require(mask.shape == y.shape[:2], f"mask layout mismatch: {mask.shape}")
    require(len(set(map(int, groups))) == EXPECTED_GROUPS, "duplicate target row group")
    require(int(mask.sum()) == EXPECTED_SCORED_ROWS, "scored-row count mismatch")
    with np.load(CACHE / "valid_index.npz", allow_pickle=False) as index_archive:
        require(
            np.array_equal(groups, index_archive["row_group"]),
            "target/index row-group order mismatch",
        )

    selected_y = np.concatenate([y[i][mask[i]] for i in range(len(groups))], axis=0)
    require(selected_y.shape == (EXPECTED_SCORED_ROWS, 2), "selected target layout mismatch")
    predictions: dict[str, np.ndarray] = {}
    input_artifacts: dict[str, dict[str, str]] = {}
    source_metadata: dict[str, dict[str, object]] = {}
    for label, run in CACHE_RUNS.items():
        sidecar_path, prediction_path = cached_artifact(run, groups, y.shape[1])
        sidecar = json.loads(sidecar_path.read_text())
        predictions[label] = map_cached_predictions(run, groups, mask)
        input_artifacts[run] = {
            "prediction_sha256": sha256_file(prediction_path),
            "sidecar_sha256": sha256_file(sidecar_path),
        }
        source_metadata[run] = {
            "run": sidecar.get("run"),
            "campaign_split_manifest_sha256": sidecar.get("campaign_split_manifest_sha256"),
            "evaluation_membership_sha256": sidecar.get("evaluation_membership_sha256"),
            "campaign_export_scope": sidecar.get("campaign_export_scope"),
            "feature_only": sidecar.get("feature_only"),
        }
        require(
            predictions[label].shape == selected_y.shape, f"mapped prediction shape mismatch: {run}"
        )

    report_path = TL3C_DIR / "report.json"
    native_path = TL3C_DIR / "predictions.npy"
    native_report = json.loads(report_path.read_text())
    native_groups = [int(value) for value in native_report["row_groups"]]
    require(len(native_groups) == len(set(native_groups)), "duplicate TL3C row groups")
    require(set(native_groups) == set(map(int, groups)), "TL3C row-group coverage mismatch")
    native_array = np.load(native_path, mmap_mode="r", allow_pickle=False)
    require(
        native_array.shape == (len(native_groups), y.shape[1], 2),
        f"TL3C layout mismatch: {native_array.shape}",
    )
    require(native_array.dtype == np.float32, f"TL3C dtype mismatch: {native_array.dtype}")
    native_positions = {row_group: position for position, row_group in enumerate(native_groups)}
    predictions["tl3c_int8_native"] = np.concatenate(
        [
            native_array[native_positions[int(group)]][row_mask]
            for group, row_mask in zip(groups, mask, strict=True)
        ],
        axis=0,
    )
    input_artifacts["tl3c_int8_native"] = {
        "prediction_sha256": sha256_file(native_path),
        "report_sha256": sha256_file(report_path),
    }

    predictions["kd50_pair_equal"] = (predictions["kd50_s3"] + predictions["kd50_s4"]) * np.float32(
        0.5
    )
    predictions["kd90_fbc1k_equal"] = (
        predictions["kd90_s3"] + predictions["fbc1k_s5"]
    ) * np.float32(0.5)
    all_rows = np.ones(EXPECTED_SCORED_ROWS, dtype=bool)
    scores: dict[str, dict[str, float | int | str]] = {}
    references: dict[str, float] = {}
    for endpoint in ENDPOINTS:
        pred = predictions[endpoint]
        require(pred.shape == selected_y.shape, f"endpoint shape mismatch: {endpoint}")
        require(np.isfinite(pred).all(), f"nonfinite prediction: {endpoint}")
        result = checked_official_score(selected_y, pred)
        reference = reference_global_wp(selected_y, pred, all_rows)
        references[endpoint] = reference
        scores[endpoint] = {
            **result,
            "reference_wp": reference,
            "official_reference_abs_error": abs(float(result["wp"]) - reference),
        }

    component_ids, row_to_component, groups_by_component = component_layout(groups)
    component_index = {component: index for index, component in enumerate(component_ids)}
    raw = np.zeros((len(component_ids), len(ENDPOINTS), 2, 6), dtype=np.float64)
    scored_rows = np.zeros(len(component_ids), dtype=np.int64)
    offset = 0
    for seq_index, group_value in enumerate(groups):
        group = int(group_value)
        count = int(mask[seq_index].sum())
        component = component_index[row_to_component[group]]
        y_rows = y[seq_index][mask[seq_index]].T
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            p_rows = predictions[endpoint][offset : offset + count].T
            raw[component, endpoint_index] += wp_sums(y_rows, p_rows)
        scored_rows[component] += count
        offset += count
    require(offset == EXPECTED_SCORED_ROWS, "component aggregation row count mismatch")
    require(np.all(scored_rows > 0), "component with no scored rows")

    aggregate = raw.sum(axis=0)
    raw_scores: dict[str, float] = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        raw_score = score_from_sums(aggregate[endpoint_index])
        raw_scores[endpoint] = raw_score
        error = abs(raw_score - references[endpoint])
        scores[endpoint]["raw_sum_wp"] = raw_score
        scores[endpoint]["raw_sum_reference_abs_error"] = error
        require(error <= 1e-8, f"raw-sum/reference mismatch before bootstrap: {endpoint}: {error}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(
        0, len(component_ids), size=(BOOTSTRAPS, len(component_ids)), endpoint=False
    )
    paired: dict[str, dict[str, object]] = {}
    influence: dict[str, dict[str, object]] = {}
    for name, candidate, reference in CONTRASTS:
        candidate_index = ENDPOINTS.index(candidate)
        reference_index = ENDPOINTS.index(reference)
        point = raw_scores[candidate] - raw_scores[reference]
        boot_delta = np.empty(BOOTSTRAPS, dtype=np.float64)
        for replicate, draw in enumerate(draws):
            candidate_sum = raw[draw, candidate_index].sum(axis=0)
            reference_sum = raw[draw, reference_index].sum(axis=0)
            boot_delta[replicate] = score_from_sums(candidate_sum) - score_from_sums(reference_sum)
        low, high = np.percentile(boot_delta, [2.5, 97.5])
        paired[name] = {
            "candidate": candidate,
            "reference": reference,
            "delta_wp": point,
            "component_bootstrap_95pct_percentile_ci": [float(low), float(high)],
            "bootstrap_replicates": BOOTSTRAPS,
            "component_units": len(component_ids),
        }

        loo_delta = np.empty(len(component_ids), dtype=np.float64)
        total_candidate = aggregate[candidate_index]
        total_reference = aggregate[reference_index]
        for index in range(len(component_ids)):
            loo_delta[index] = score_from_sums(
                total_candidate - raw[index, candidate_index]
            ) - score_from_sums(total_reference - raw[index, reference_index])
        changes = loo_delta - point
        order_positive = np.argsort(changes)[-10:][::-1]
        order_negative = np.argsort(changes)[:10]

        def influence_rows(
            indices: np.ndarray, delta_changes: np.ndarray
        ) -> list[dict[str, float | int | list[int]]]:
            return [
                {
                    "component_id": int(component_ids[index]),
                    "row_groups": groups_by_component[component_ids[index]],
                    "scored_rows": int(scored_rows[index]),
                    "delta_change_when_omitted": float(delta_changes[index]),
                }
                for index in indices
            ]

        influence[name] = {
            "definition": "paired delta after omitting one whole component minus full paired delta; raw-sum score",
            "largest_positive_delta_changes": influence_rows(order_positive, changes),
            "largest_negative_delta_changes": influence_rows(order_negative, changes),
        }

    component_stats = []
    for index, component in enumerate(component_ids):
        endpoint_stats = {
            endpoint: {
                "t0": [float(value) for value in raw[index, endpoint_index, 0]],
                "t1": [float(value) for value in raw[index, endpoint_index, 1]],
            }
            for endpoint_index, endpoint in enumerate(ENDPOINTS)
        }
        component_stats.append(
            {
                "component_id": int(component),
                "row_groups": groups_by_component[component],
                "scored_rows": int(scored_rows[index]),
                "wp_sums_by_endpoint": endpoint_stats,
            }
        )

    result = {
        "schema": "blend_support_audit_r1",
        "metric_id": METRIC_ID,
        "evaluation": {
            "split": "full official valid",
            "exposure": "exposed, repeatedly used development data",
            "sequence_groups": EXPECTED_GROUPS,
            "scored_rows": EXPECTED_SCORED_ROWS,
            "connected_components": len(component_ids),
            "cluster_source": "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json#sequences.valid",
        },
        "frozen_procedure": {
            "equal_blend_weights": [0.5, 0.5],
            "bootstrap_replicates": BOOTSTRAPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_interval": "paired connected-component percentile 95%",
            "no_coefficient_search": True,
            "optional_kd90_tl3c_blend_scored": False,
            "optional_blend_exclusion": "existing TL3C full-valid native loop is 4436.4s versus 3600s competition runtime limit; added model has no demonstrated margin",
        },
        "input_artifacts": {
            **input_artifacts,
            "valid_targets_sha256": sha256_file(CACHE / "valid_targets.npz"),
            "valid_index_sha256": sha256_file(CACHE / "valid_index.npz"),
            "component_map_sha256": sha256_file(SPLIT_PATH),
            "grade_submissions_sha256": sha256_file(
                REPO / "research/campaign_2026-11-14/local_r1/grade_submissions.py"
            ),
            "official_metric_sha256": sha256_file(REPO / "src/wnn/gmetric.py"),
            "tl3c_report_metadata": {
                "split_sha256": native_report.get("split_sha256"),
                "data_sha256": native_report.get("data_sha256"),
                "score_scope": native_report.get("score_scope"),
                "artifact_sha256": native_report.get("artifact", {}).get("sha256"),
                "artifact_bytes": native_report.get("artifact", {}).get("bytes"),
            },
        },
        "endpoint_provenance": {
            "kd50_pair_equal": {
                "members": ["l14_kd50_s3", "l14_kd50_s4"],
                "training": "train-only, fixed KD50 recipe, seeds 3 and 4; frozen local same-count membership recipe",
                "prediction_sidecars": {
                    name: source_metadata[name] for name in ("l14_kd50_s3", "l14_kd50_s4")
                },
            },
            "kd90_s3": {
                "member": "l14_kd90_s3",
                "training": "train-only, fixed KD90 recipe, seed 3; frozen local same-count membership recipe",
                "prediction_sidecar": source_metadata["l14_kd90_s3"],
            },
            "fbc1k_s5": {
                "member": "tt29rep_fbc1k_s5",
                "training": "local train-only PR49 recipe reproduction; same-count membership; original PR49 byte membership not asserted",
                "prediction_sidecar": source_metadata["tt29rep_fbc1k_s5"],
            },
            "kd90_fbc1k_equal": {
                "members": ["l14_kd90_s3", "tt29rep_fbc1k_s5"],
                "weights": [0.5, 0.5],
            },
            "tl3c_int8_native": {
                "member_signature": native_report.get("member_signature"),
                "training": "historical train-only native INT8 blend; original split hash differs from the local PR49 reproduction, byte-level membership equality unverified",
            },
        },
        "endpoint_scores": scores,
        "paired_deltas": paired,
        "component_influence": influence,
        "component_sufficient_statistics": component_stats,
        "interpretation_limits": [
            "Intervals condition on fixed fitted models and measure only component resampling, not fitting-seed variation.",
            "The full valid set is exposed; these are not unseen-CV significance intervals and do not remove selection bias.",
            "TL3C's original split hash differs from the local reproduction; byte-level training-membership equality is not claimed.",
            "No public transfer result, serving package for new blends, or upload is established by this audit.",
        ],
    }
    output_path = AUDIT_DIR / "RESULTS_R1.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output_path), "endpoint_scores": scores, "paired_deltas": paired},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
