"""Grade the frozen Q14 R2 endpoints on full exposed official valid."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
OUTPUT = Path(__file__).resolve().parent / "ENDPOINT_GRADE_R2.json"
CONTRACT = ROOT / "research/campaign_2026-11-14/quality_r2/ENDPOINT_SELECTION_R2.json"
COLLECTION = CONTRACT.parent / "kaggle_r2/COLLECTION_R2.json"
METRIC = ROOT / "src/wnn/gmetric.py"
MAPPER = ROOT / "research/campaign_2026-11-14/local_r1/grade_submissions.py"
TL3C_DIR = ROOT / "submissions/tl3c_pr49_20261029/blend-int8-full-v3"

CONTRACT_SHA256 = "379fe822b2798cb3f1d01655ca9705221ee61415794ddd0515cb051c89c2b874"
METRIC_SHA256 = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
MAPPER_SHA256 = "b3fa753ba25f7194ed7e9b7841a04ceee5db51a3a7c8437fd0cc72441b16fff2"
SPLIT_SHA256 = "6d23a1b76f66d8001dac805d2b218a3b139e4eebe555f0c9b96fe1263ce186f0"
EVALUATION_SHA256 = "008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7"
VALID_SEQUENCES = 1873
SEQUENCE_LENGTH = 20_000
SCORED_ROWS = 3_528_273

ARMS: dict[str, dict[str, object]] = {
    "a50": {
        "alpha": 0.5,
        "boot_h": 0,
        "runs": {3: "q14s_a50_s3", 4: "q14s_a50_s4"},
    },
    "a90": {
        "alpha": 0.9,
        "boot_h": 0,
        "runs": {3: "q14s_a90_s3", 4: "q14s_a90_s4"},
    },
    "boot8": {
        "alpha": 0.9,
        "boot_h": 8,
        "runs": {3: "q14s_boot8_s3", 4: "q14s_boot8_s4"},
    },
}
OLD_RUNS = ("l14_kd50_s3", "l14_kd50_s4")


class GradeError(RuntimeError):
    """Raised when an input does not match the frozen endpoint contract."""


def sha256_file(path: Path) -> str:
    """Hash a file without loading its full contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    """Read an evidence object and reject malformed or non-object JSON."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GradeError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise GradeError(f"expected a JSON object in {path}")
    return value


def require_sha(path: Path, expected: str, label: str) -> str:
    """Require a frozen source file to retain its recorded digest."""
    actual = sha256_file(path)
    if actual != expected:
        raise GradeError(f"{label} SHA-256 changed: {actual}")
    return actual


def require_inputs() -> None:
    """Fail before grading if any fixed source, cache, or report is absent."""
    required = [
        CONTRACT,
        COLLECTION,
        METRIC,
        MAPPER,
        CACHE / "valid_targets.npz",
        CACHE / "valid_index.npz",
        TL3C_DIR / "predictions.npy",
        TL3C_DIR / "report.json",
    ]
    for run in (*OLD_RUNS, *(str(run) for arm in ARMS.values() for run in arm["runs"].values())):
        required.extend((CACHE / f"pred_{run}_valid.npy", CACHE / f"pred_{run}_valid.json"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"missing required grading inputs; no output written:\n{listing}")


def load_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the exact full-valid target, mask, and row-group order."""
    with np.load(CACHE / "valid_targets.npz", allow_pickle=False) as data:
        groups = np.asarray(data["row_group"], dtype=np.int64)
        mask = np.asarray(data["mask"], dtype=bool)
        y = np.asarray(data["y"], dtype=np.float32)
    with np.load(CACHE / "valid_index.npz", allow_pickle=False) as index:
        index_groups = np.asarray(index["row_group"], dtype=np.int64)
    expected_shape = (VALID_SEQUENCES, SEQUENCE_LENGTH)
    if (
        groups.shape != (VALID_SEQUENCES,)
        or len(np.unique(groups)) != VALID_SEQUENCES
        or not np.array_equal(groups, index_groups)
        or mask.shape != expected_shape
        or y.shape != (*expected_shape, 2)
        or int(mask.sum()) != SCORED_ROWS
        or not np.isfinite(y[mask]).all()
    ):
        raise GradeError("cached official valid targets, mask, or row-group identity changed")
    digest = hashlib.sha256(groups.astype("<i8", copy=False).tobytes()).hexdigest()
    if digest != EVALUATION_SHA256:
        raise GradeError(f"full-valid row-group membership hash changed: {digest}")
    return groups, mask, y[mask]


def validate_prediction_source(
    run: str, groups: np.ndarray, *, strict_source: bool
) -> dict[str, object]:
    """Check source identity, complete row coverage, layout, digest, and finite values."""
    path = CACHE / f"pred_{run}_valid.npy"
    metadata_path = path.with_suffix(".json")
    metadata = read_json(metadata_path)
    row_groups = metadata.get("row_groups")
    if (
        metadata.get("run") != run
        or metadata.get("split") != "valid"
        or metadata.get("epoch") != 3
        or metadata.get("split_manifest_sha256") != SPLIT_SHA256
        or not isinstance(row_groups, list)
        or len(row_groups) != VALID_SEQUENCES
        or any(type(value) is not int for value in row_groups)
        or len(set(row_groups)) != VALID_SEQUENCES
        or set(row_groups) != set(int(value) for value in groups)
    ):
        raise GradeError(f"{run}: source identity or complete valid row-group count mismatch")
    if strict_source:
        row_digest = hashlib.sha256(np.asarray(row_groups, dtype="<i8").tobytes()).hexdigest()
        if (
            metadata.get("feature_only") is not True
            or metadata.get("inference_precision") != "fp32"
            or metadata.get("quantization_preset") is not None
            or metadata.get("checkpoint_file") != "model_ep3.pt"
            or metadata.get("campaign_split_manifest_sha256") != SPLIT_SHA256
            or metadata.get("campaign_export_scope")
            not in {"full-valid-feature-only", "evaluation"}
            or metadata.get("evaluation_membership_sha256") != row_digest
            or not isinstance(metadata.get("checkpoint_sha256"), str)
            or len(str(metadata["checkpoint_sha256"])) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(metadata["checkpoint_sha256"])
            )
        ):
            raise GradeError(f"{run}: feature-only full-valid source qualification failed")
    prediction_hash = sha256_file(path)
    if metadata.get("prediction_sha256") != prediction_hash:
        raise GradeError(f"{run}: prediction digest differs from its sidecar")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (VALID_SEQUENCES, SEQUENCE_LENGTH, 2) or array.dtype != np.float32:
        raise GradeError(f"{run}: expected float32 predictions shaped (1873, 20000, 2)")
    for offset in range(0, VALID_SEQUENCES, 64):
        if not np.isfinite(array[offset : offset + 64]).all():
            raise GradeError(f"{run}: prediction array contains non-finite values")
    return {
        "run": run,
        "sidecar_sha256": sha256_file(metadata_path),
        "prediction_sha256": prediction_hash,
        "source_row_groups": len(row_groups),
        "expected_row_groups": VALID_SEQUENCES,
        "mapped_complete_valid": True,
        "feature_only": metadata.get("feature_only"),
        "campaign_export_scope": metadata.get("campaign_export_scope"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "split_manifest_sha256": metadata.get("split_manifest_sha256"),
    }


def verified_collection(runs: list[str]) -> dict:
    """Require completed worker verification and bind each exported checkpoint/provenance."""
    collection = read_json(COLLECTION)
    if collection.get("status") != "complete" or set(collection.get("runs", {})) != set(runs):
        raise GradeError("all six verified worker outputs are required before quality selection")
    groups = collection.get("optimizer_strict_fp32_group_counts", {})
    for run in runs:
        if not isinstance(groups.get(run), int) or groups[run] < 1:
            raise GradeError(f"{run}: completed strict optimizer verification missing")
        metadata = read_json(CACHE / f"pred_{run}_valid.json")
        files = collection["runs"][run]["artifact_files"]
        if (
            metadata.get("checkpoint_sha256")
            != files[f"artifacts/runs/{run}/model_ep3.pt"]["sha256"]
            or metadata.get("run_provenance_sha256")
            != files[f"artifacts/runs/{run}/provenance.json"]["sha256"]
        ):
            raise GradeError(f"{run}: export is not bound to the collected fixed endpoint")
    return collection


def json_score(
    score_fn: Callable[..., dict[str, object]], y: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
    """Use the unchanged shared grader, which checks its independent WP reference."""
    result = score_fn(y, prediction)
    return {
        key: int(value) if isinstance(value, (int, np.integer)) else float(value)
        for key, value in result.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def mean_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Form the preregistered equal-weight seed pair."""
    return (first + second) * np.float32(0.5)


def mean_triple(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    """Form the preregistered equal-weight three-member endpoint."""
    return (first + second + third) / np.float32(3.0)


def gate_record(deltas: list[float], threshold: float) -> dict[str, object]:
    """Apply the fixed positive-per-seed and mean-gain recipe gate."""
    average = float(np.mean(deltas))
    return {
        "paired_seed_deltas": {"s3": deltas[0], "s4": deltas[1]},
        "mean_delta": average,
        "minimum_mean_delta": threshold,
        "both_seeds_positive": all(delta > 0.0 for delta in deltas),
        "passed": all(delta > 0.0 for delta in deltas) and average >= threshold,
    }


def load_tl3c(groups: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Map the archived native TL3C blend by report row-group identity."""
    report_path = TL3C_DIR / "report.json"
    prediction_path = TL3C_DIR / "predictions.npy"
    report = read_json(report_path)
    row_groups = report.get("row_groups")
    prediction_record = report.get("predictions")
    metrics = report.get("metrics")
    if (
        not isinstance(row_groups, list)
        or len(row_groups) != VALID_SEQUENCES
        or any(type(value) is not int for value in row_groups)
        or len(set(row_groups)) != VALID_SEQUENCES
        or set(row_groups) != set(int(value) for value in groups)
        or not isinstance(prediction_record, dict)
        or not isinstance(metrics, dict)
        or report.get("score_scope") != "full_official_valid_exposed"
        or metrics.get("metric_id") != "global_wp_clip2_v1"
        or metrics.get("selected_rows") != SCORED_ROWS
        or not isinstance(metrics.get("wp"), (int, float))
    ):
        raise GradeError("TL3C report does not identify the complete official valid panel")
    prediction_hash = sha256_file(prediction_path)
    if prediction_record.get("sha256") != prediction_hash:
        raise GradeError("TL3C prediction file differs from its report digest")
    array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (VALID_SEQUENCES, SEQUENCE_LENGTH, 2) or array.dtype != np.float32:
        raise GradeError("TL3C prediction layout differs from (1873, 20000, 2) float32")
    positions = {int(group): index for index, group in enumerate(row_groups)}
    for offset in range(0, VALID_SEQUENCES, 64):
        if not np.isfinite(array[offset : offset + 64]).all():
            raise GradeError("TL3C prediction array contains non-finite values")
    prediction = np.concatenate(
        [
            array[positions[int(group)]][selected]
            for group, selected in zip(groups, mask, strict=True)
        ]
    )
    return prediction, {
        "prediction_sha256": prediction_hash,
        "report_sha256": sha256_file(report_path),
        "source_row_groups": len(row_groups),
        "score_scope": report.get("score_scope"),
        "recorded_wp": metrics.get("wp"),
        "split_sha256": report.get("split_sha256"),
        "member_signature": report.get("member_signature"),
        "artifact": report.get("artifact"),
        "ancestry_caveat": "original split hash differs; byte-level membership equivalence is unverified",
    }


def grade() -> dict[str, object]:
    """Score all fixed endpoints and construct the frozen final-fit decision."""
    require_sha(CONTRACT, CONTRACT_SHA256, "selection contract")
    metric_hash = require_sha(METRIC, METRIC_SHA256, "official metric")
    mapper_hash = require_sha(MAPPER, MAPPER_SHA256, "shared endpoint grader")
    contract = read_json(CONTRACT)
    if contract.get("source") != "48776fba8af26b16b7e6813def7ccf571f56c1ac":
        raise GradeError("selection contract source identity differs")

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(MAPPER.parent))
    from grade_submissions import predictions as mapped_predictions
    from grade_submissions import score as shared_score

    groups, mask, y = load_targets()
    all_runs = [str(run) for arm in ARMS.values() for run in arm["runs"].values()]
    collection = verified_collection(all_runs)
    source_records: dict[str, dict[str, object]] = {}
    for run in (*all_runs, *OLD_RUNS):
        source_records[run] = validate_prediction_source(run, groups, strict_source=run in all_runs)
    predictions = {run: mapped_predictions(run, groups, mask) for run in (*all_runs, *OLD_RUNS)}
    single_scores = {
        run: json_score(shared_score, y, prediction) for run, prediction in predictions.items()
    }
    for run in all_runs:
        if abs(float(single_scores[run]["wp"]) - float(collection["scores"][run])) > 1e-5:
            raise GradeError(f"{run}: exported score differs from verified worker history by >1e-5")
    old_pair_score = json_score(shared_score, y, mean_pair(*(predictions[run] for run in OLD_RUNS)))
    pair_predictions = {
        arm: mean_pair(*(predictions[run] for run in spec["runs"].values()))
        for arm, spec in ARMS.items()
    }
    pair_scores = {
        arm: json_score(shared_score, y, prediction) for arm, prediction in pair_predictions.items()
    }
    mixed_scores = {
        arm: json_score(
            shared_score,
            y,
            mean_triple(
                predictions[spec["runs"][3]],
                predictions[spec["runs"][4]],
                predictions[OLD_RUNS[0]],
            ),
        )
        for arm, spec in ARMS.items()
    }
    a50_pair_wp = float(pair_scores["a50"]["wp"])
    old_pair_wp = float(old_pair_score["wp"])
    gates: dict[str, dict[str, object]] = {
        "a90": gate_record(
            [
                float(single_scores[ARMS["a90"]["runs"][seed]]["wp"])
                - float(single_scores[ARMS["a50"]["runs"][seed]]["wp"])
                for seed in (3, 4)
            ],
            0.0005,
        ),
        "boot8_vs_a90": gate_record(
            [
                float(single_scores[ARMS["boot8"]["runs"][seed]]["wp"])
                - float(single_scores[ARMS["a90"]["runs"][seed]]["wp"])
                for seed in (3, 4)
            ],
            0.0003,
        ),
    }
    boot8_vs_a50 = [
        float(single_scores[ARMS["boot8"]["runs"][seed]]["wp"])
        - float(single_scores[ARMS["a50"]["runs"][seed]]["wp"])
        for seed in (3, 4)
    ]
    gates["boot8_vs_a50_mean"] = {
        "paired_seed_deltas": {"s3": boot8_vs_a50[0], "s4": boot8_vs_a50[1]},
        "mean_delta": float(np.mean(boot8_vs_a50)),
        "minimum_mean_delta": 0.0005,
        "passed": float(np.mean(boot8_vs_a50)) >= 0.0005,
    }
    arm_gate = {
        "a50": False,
        "a90": bool(gates["a90"]["passed"]),
        "boot8": bool(gates["boot8_vs_a90"]["passed"])
        and bool(gates["boot8_vs_a50_mean"]["passed"]),
    }

    endpoints: dict[str, dict[str, dict[str, object]]] = {}
    choices: list[tuple[float, str, str, list[str]]] = []
    for arm in ARMS:
        endpoints[arm] = {}
        for endpoint, score in (("pair", pair_scores[arm]), ("mixed3", mixed_scores[arm])):
            wp = float(score["wp"])
            delta_a50 = wp - a50_pair_wp
            delta_old = wp - old_pair_wp
            delta_pair = wp - float(pair_scores[arm]["wp"]) if endpoint == "mixed3" else None
            passes = arm_gate[arm] and delta_a50 > 0.0 and delta_old > 0.0
            if endpoint == "mixed3":
                passes = passes and delta_pair is not None and delta_pair > 0.0
            members = (
                list(ARMS[arm]["runs"].values())
                if endpoint == "pair"
                else [*ARMS[arm]["runs"].values(), OLD_RUNS[0]]
            )
            endpoints[arm][endpoint] = {
                "wp": wp,
                "scores": score,
                "delta_vs_a50_pair": delta_a50,
                "delta_vs_historical_kd50_pair": delta_old,
                "delta_vs_same_arm_pair": delta_pair,
                "recipe_gate_passed": arm_gate[arm],
                "endpoint_eligible": bool(passes),
                "members": members,
                "weights": [1 / len(members)] * len(members),
            }
            if passes:
                choices.append((wp, arm, endpoint, members))

    choices.sort(key=lambda item: (item[0], item[1] == "a90", item[2] == "pair"), reverse=True)
    selected = choices[0] if choices else None
    decision: dict[str, object] = {
        "admitted": selected is not None,
        "arm": selected[1] if selected else None,
        "alpha": float(ARMS[selected[1]]["alpha"]) if selected else None,
        "boot_h": int(ARMS[selected[1]]["boot_h"]) if selected else None,
        "endpoint": selected[2] if selected else None,
        "members": selected[3] if selected else [],
        "source": "q14s cloud train-only fixed-epoch valid prediction sidecars"
        if selected
        else None,
    }

    tl3c_prediction, tl3c_source = load_tl3c(groups, mask)
    tl3c_score = json_score(shared_score, y, tl3c_prediction)
    tl3c_comparisons = {
        f"{arm}_{endpoint}": float(tl3c_score["wp"]) - float(record["wp"])
        for arm, records in endpoints.items()
        for endpoint, record in records.items()
    }
    tl3c_comparisons["historical_kd50_pair"] = float(tl3c_score["wp"]) - old_pair_wp

    return {
        "schema": "endpoint-grade-r2-v1",
        "status": "graded_complete_exposed_valid_panel",
        "contract_sha256": CONTRACT_SHA256,
        "metric": "global_wp_clip2_v1",
        "metric_source_sha256": metric_hash,
        "shared_grader_sha256": mapper_hash,
        "evaluation": {
            "scope": "full official valid; exposed",
            "sequences": VALID_SEQUENCES,
            "sequence_length": SEQUENCE_LENGTH,
            "scored_rows": SCORED_ROWS,
            "fit_claim": "train-only endpoints; no fitted-data ranking",
            "uncertainty": "no unseen-CV significance, public forecast, or model-seed uncertainty claim",
        },
        "source_inputs": {
            "worker_collection_sha256": sha256_file(COLLECTION),
            "valid_targets_sha256": sha256_file(CACHE / "valid_targets.npz"),
            "valid_index_sha256": sha256_file(CACHE / "valid_index.npz"),
            "prediction_sources": source_records,
        },
        "singles": single_scores,
        "historical_kd50_pair": {
            "wp": float(old_pair_score["wp"]),
            "scores": old_pair_score,
            "members": list(OLD_RUNS),
        },
        "fixed_recipe_gates": gates,
        "arm_gate_eligible": arm_gate,
        "complete_endpoints": {
            "a50_pair_baseline_wp": a50_pair_wp,
            "historical_kd50_pair_wp": old_pair_wp,
            "by_arm": endpoints,
        },
        "tl3c_native_blend": {
            "scores": tl3c_score,
            "delta_vs_reported_wp": float(tl3c_score["wp"]) - float(tl3c_source["recorded_wp"]),
            "deltas_vs_complete_endpoints": tl3c_comparisons,
            "source": tl3c_source,
        },
        "final_decision": decision,
        "decision_limits": {
            "no_coefficient_search": True,
            "upstream_training_provenance": "checkpoint and provenance hashes bound to completed six-worker collection; strict optimizer verification required",
            "maximum_conditional_final_fits": 2 if decision["admitted"] else 0,
            "final_data_score_is_not_held_out": True,
            "serving_and_native_parity_remain_separate_checks": True,
        },
    }


def main() -> None:
    """Write the complete JSON result only after all source and score checks pass."""
    require_inputs()
    result = grade()
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(result["final_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, GradeError) as error:
        raise SystemExit(str(error)) from error
