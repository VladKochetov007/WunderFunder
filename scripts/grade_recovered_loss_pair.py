"""Grade frozen recovered loss-pair predictions without selecting blend weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import GlobalAccumulator, reference_global_wp  # noqa: E402

# Provenance and sidecars are preserved heterogeneous producer JSON objects.
Record = dict[str, Any]


def sha256(path: Path) -> str:
    """Hash an artifact in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_pinned(root: Path, binding: Record) -> Record:
    """Read a JSON artifact only after checking its frozen identity."""
    path = root / binding["path"]
    if sha256(path) != binding["sha256"]:
        raise ValueError(f"artifact identity changed: {path}")
    return json.loads(path.read_text())


def validate_sidecar(
    sidecar: Record, arm: Record, spec: Record, groups: list[int], prediction_sha: str
) -> None:
    """Reject changed checkpoints, precision, membership, order and payload."""
    expected = {
        "run": arm["run"],
        "split": "valid",
        "epoch": 1,
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "prediction_sha256": prediction_sha,
        "split_manifest_sha256": spec["split"]["sha256"],
        "inference_precision": "fp32",
        "row_groups": groups,
        "evaluation_membership_sha256": hashlib.sha256(
            np.asarray(groups, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    for name, value in expected.items():
        if sidecar.get(name) != value:
            raise ValueError(f"prediction binding differs: {name}")
    cache = sidecar.get("parsed_sequence_cache", {})
    for name, value in spec["cache_identity"].items():
        if cache.get(name) != value:
            raise ValueError(f"prediction cache binding differs: {name}")
    # A verified relocation legitimately changes this receipt; content identities
    # above must stay fixed. The independent export review audits both receipts.
    receipt_sha = cache.get("validation_receipt_sha256", "")
    if len(receipt_sha) != 64 or any(c not in "0123456789abcdef" for c in receipt_sha):
        raise ValueError("prediction has no cache validation receipt identity")


def validate_export_binding(receipt: Record, label: str, sidecar: Record, sidecar_sha: str) -> None:
    """Bind predictions to the separately reviewed, hash-pinned export receipt."""
    if (
        receipt.get("schema") != "wnn.loss-pair.export-recovery-run-receipt.v1"
        or receipt.get("state") != "complete_exports_pending_independent_grade"
    ):
        raise ValueError("export receipt is not a completed recovery")
    producer = receipt.get("prediction_results", {}).get(label, {})
    identity = producer.get("sidecar_identity", {})
    if producer.get("returncode") != 0 or producer.get("guard_stop"):
        raise ValueError("export process did not complete successfully")
    expected = {
        "sidecar_sha256": sidecar_sha,
        "prediction_sha256": sidecar["prediction_sha256"],
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "checkpoint_epoch": sidecar["epoch"],
        "cache_validation_receipt_sha256": sidecar["parsed_sequence_cache"][
            "validation_receipt_sha256"
        ],
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("export receipt does not bind the actual prediction sidecar")
    if receipt.get("cache_validation_sha256") != expected["cache_validation_receipt_sha256"]:
        raise ValueError("export receipt cache validation differs")
    model = receipt.get("model_receipts", {}).get(label, {})
    if model.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
        raise ValueError("export receipt model lineage differs")


def grade(
    root: Path,
    spec_path: Path,
    expected_spec_sha256: str,
    export_receipt: Path,
    expected_export_receipt_sha256: str,
    paths: dict[str, Path],
    output: Path,
) -> None:
    """Evaluate WMSE, Huber and their fixed equal average on an exposed panel."""
    if output.exists():
        raise ValueError("result path already exists; preserve prior results")
    spec = read_pinned(root, {"path": str(spec_path), "sha256": expected_spec_sha256})
    receipt = read_pinned(
        root, {"path": str(export_receipt), "sha256": expected_export_receipt_sha256}
    )
    if spec["status"] != "frozen_before_recovered_prediction_scoring":
        raise ValueError("unfrozen grader specification")
    for relative, expected in spec["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"grader source changed: {relative}")
    split = read_pinned(root, spec["split"])
    groups = list(map(int, split["evaluation"]["valid"]))
    if len(groups) != 468 or len(set(groups)) != 468:
        raise ValueError("expected exactly 468 unique evaluation groups")
    component_map = {
        (source, int(row["row_group"])): int(row["component"])
        for source, rows in split["sequences"].items()
        for row in rows
    }
    components = np.asarray([component_map["valid", g] for g in groups])
    if len(np.unique(components)) != 141:
        raise ValueError("expected 141 evaluation components")
    predictions, bindings = {}, {}
    for label, arm in spec["arms"].items():
        provenance = read_pinned(root, arm["provenance"])
        fit_components = {
            component_map[source, int(group)]
            for source, fitted in provenance["fit_row_groups"].items()
            for group in fitted
        }
        if fit_components.intersection(components.tolist()):
            raise ValueError(f"fit/evaluation component leakage: {label}")
        if provenance["args"].get("init") or provenance["args"].get("distill"):
            raise ValueError("expected independent scratch initialization without teacher")
        path = paths[label]
        sidecar_path = path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        digest = sha256(path)
        validate_sidecar(sidecar, arm, spec, groups, digest)
        validate_export_binding(receipt, label, sidecar, sha256(sidecar_path))
        prediction = np.load(path, mmap_mode="r", allow_pickle=False)
        if prediction.dtype != np.float32 or prediction.shape != (468, 20_000, 2):
            raise ValueError(f"wrong prediction shape/dtype: {label}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"nonfinite predictions: {label}")
        predictions[label] = prediction
        bindings[label] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "sidecar_sha256": sha256(sidecar_path),
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "validation_receipt_sha256": sidecar["parsed_sequence_cache"][
                "validation_receipt_sha256"
            ],
        }
    targets_path = root / spec["targets"]["path"]
    if sha256(targets_path) != spec["targets"]["sha256"]:
        raise ValueError("target cache changed")
    with np.load(targets_path, allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        positions = {group: index for index, group in enumerate(row_groups)}
        if len(positions) != len(row_groups) or not set(groups).issubset(positions):
            raise ValueError("duplicate or missing target row groups")
        indices = [positions[group] for group in groups]
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        mask = np.asarray(cache["mask"][indices], dtype=bool)
    mask &= (np.arange(20_000) >= 99)[None, :]
    predictions["equal_blend"] = (
        (predictions["wmse"].astype(np.float64) + predictions["huber_d1"]) * 0.5
    ).astype(np.float32)
    scores, moments = {}, {}
    for label, prediction in predictions.items():
        accumulator = GlobalAccumulator()
        for start in range(0, len(groups), 4):
            selection = slice(start, start + 4)
            accumulator.add(targets[selection], prediction[selection], mask[selection])
        scores[label] = accumulator.result()
        direct = reference_global_wp(targets, prediction, mask)
        if abs(float(scores[label]["wp"]) - direct) > 1e-12:
            raise ValueError(f"streamed versus centered formula disagreement: {label}")
        scores[label]["independent_centered_formula_wp"] = direct
        moments[label], _ = paired_component_moments(targets, prediction, prediction, mask)
    comparisons = {}
    for candidate, reference in (
        ("huber_d1", "wmse"),
        ("equal_blend", "wmse"),
        ("equal_blend", "huber_d1"),
    ):
        key = f"{candidate}_minus_{reference}"
        comparisons[key] = paired_component_bootstrap(
            moments[candidate],
            moments[reference],
            components,
            n_boot=2000,
            seed=271828,
            centered=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output.with_suffix(".moments.npz"),
        row_groups=np.asarray(groups),
        components=components,
        **moments,
    )
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "spec_sha256": sha256(spec_path),
        "reviewed_export_receipt_sha256": expected_export_receipt_sha256,
        "bindings": bindings,
        "scores": scores,
        "comparisons": comparisons,
        "evaluation_groups": 468,
        "components": 141,
        "metric_id": "global_wp_clip2_v1",
        "selection": "three declared endpoints; no searched weights or selected epochs",
        "uncertainty": "Paired component resampling of fixed predictions; excludes training-seed variation and public transfer. Exposed development panel.",
        "expected_public_score": None,
        "moments_sha256": sha256(output.with_suffix(".moments.npz")),
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"scores": scores, "comparisons": comparisons}))


def main() -> None:
    """Read explicit recovered prediction paths and a frozen grading contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--expected-export-receipt-sha256", required=True)
    parser.add_argument("--wmse", type=Path, required=True)
    parser.add_argument("--huber", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grade(
        args.root,
        args.spec,
        args.expected_spec_sha256,
        args.export_receipt,
        args.expected_export_receipt_sha256,
        {"wmse": args.wmse, "huber_d1": args.huber},
        args.output,
    )


if __name__ == "__main__":
    main()
