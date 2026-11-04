"""Grade fixed endpoints, paired seeds and predeclared equal-seed TL3 mixtures."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

Q = Path("/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1")
H = Path("/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1")


def read(path: Path) -> dict[str, object]:
    """Read declared local scientific metadata."""
    return json.loads(path.read_bytes())


def main() -> None:
    """Keep missing cells explicit; do not optimize weights or select epochs."""
    sys.path.insert(0, str(H / "source6-eval/src"))
    sys.path.append(str(H / "cpu-deps-r1"))
    from wnn.gmetric import global_wp

    helper = H / "requests/verify-primary-independent-r1.py"
    assert (
        hashlib.sha256(helper.read_bytes()).hexdigest()
        == "5df00bb5e209ba723b32d04ba8c8748380b3207d07c4108f35d0543f572a9963"
    )
    spec = importlib.util.spec_from_file_location("independent_primary", helper)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    split = read(H / "data/chronological_primary.json")
    targets, mask, source = verifier.read_primary_labels(H / "data", split)
    output = Q / "reports/final-comparison-r1"
    output.mkdir(exist_ok=False)
    request = read(Q / "reports/final-scores-r1/score_request.json")
    original = read(H / "reports/full-primary-refits-r1/full_primary_refit_report.json")
    base = read(Q / "reports/fixed-tl3-mixtures-r1/REPORT.json")
    panel_path = Q / "inputs/final-comparison-panel-r1.json"
    panel = read(panel_path)
    assert panel["analyzer_sha256"] == verifier.sha256(Path(__file__))
    assert panel["seed_weights"] == [0.5, 0.5] and panel["candidate_weights"] == [0.05, 0.10]
    assert panel["primary_valid_used_for_weight_learning"] is False
    records, arrays = [], {}
    baseline = next(row for row in base["records"] if row["name"] == "replayed_TL3")
    strong = next(row for row in base["records"] if row["name"] == "TL3_plus_lstm_0.10")

    def checked_array(receipt: dict[str, object]) -> np.ndarray:
        return verifier._array(Path(receipt["path"]), receipt["sha256"])

    tl3 = checked_array(baseline["prediction"])
    assert baseline["prediction"]["sha256"] == panel["TL3_prediction_sha256"]

    def add(name: str, prediction: np.ndarray, kind: str, evidence: object) -> None:
        metrics = global_wp(targets, prediction, np.ones(len(targets), dtype=bool))
        independent = verifier.independent_metrics(targets, prediction)
        error = max(abs(metrics[key] - independent[key]) for key in ["wp", "t0", "t1"])
        assert error < 1e-10
        records.append(
            {
                "name": name,
                "kind": kind,
                **metrics,
                "wmse": independent["wmse"],
                "delta_vs_TL3": metrics["wp"] - baseline["metrics"]["wp"],
                "delta_vs_known_TL3_LSTM": metrics["wp"] - strong["metrics"]["wp"],
                "independent_max_absolute_error": error,
                "evidence": evidence,
            }
        )

    add("TL3_replayed", tl3, "frozen_reference", baseline)
    add(
        "TL3_plus_previous_LSTM_0.10",
        checked_array(strong["prediction"]),
        "known_exposed_reference",
        strong,
    )
    cached = next(
        row["score"]
        for row in original["singleton_scores"]
        if row["score"]["fit_id"] == panel["reused_control"]["fit_id"]
    )
    assert cached["checkpoint"]["sha256"] == panel["reused_control"]["model_sha256"]
    assert all(cached["predictions"][key] == mask[key] for key in verifier.MASK_KEYS)
    arrays["A1301"] = checked_array(cached["predictions"])
    add("A1301", arrays["A1301"], "reused_exact_control", cached)
    for member in request["members"]:
        receipt_path = Path(member["receipt_path"])
        if not receipt_path.exists():
            continue
        receipt = read(receipt_path)
        assert receipt["status"] == "complete"
        assert all(receipt["predictions"][key] == mask[key] for key in verifier.MASK_KEYS)
        name = receipt["fit_id"]
        arrays[name] = checked_array(receipt["predictions"])
        add(name, arrays[name], "fresh_configured_final", receipt)
        assert abs(records[-1]["wp"] - receipt["metrics"]["wp"]) < 1e-10
    for arm in panel["arms"]:
        if not all(arm + str(seed) in arrays for seed in [1301, 1302]):
            continue
        members = [arrays[arm + str(seed)] for seed in [1301, 1302]]
        mean = verifier.reconstruct_block(members, np.asarray([0.5, 0.5], dtype=np.float64))
        path = output / (arm + "_seed_mean.npy")
        np.save(path, mean, allow_pickle=False)
        reference = {
            "members": [arm + "1301", arm + "1302"],
            "weights": [0.5, 0.5],
            "prediction_path": str(path),
            "prediction_sha256": verifier.sha256(path),
        }
        add(arm + "_seed_mean", mean, "frozen_equal_seed_mean", reference)
        for alpha in panel["candidate_weights"]:
            prediction = verifier.reconstruct_block(
                [tl3, mean], np.asarray([1 - alpha, alpha], dtype=np.float64)
            )
            name = f"TL3_plus_{arm}_seed_mean_{alpha:.2f}"
            path = output / (name + ".npy")
            np.save(path, prediction, allow_pickle=False)
            add(
                name,
                prediction,
                "predeclared_fixed_mixture",
                {
                    "candidate": reference,
                    "candidate_weight": alpha,
                    "prediction_path": str(path),
                    "prediction_sha256": verifier.sha256(path),
                },
            )
    scores = {row["name"]: row["wp"] for row in records}
    paired = {
        comparison: [
            {"seed": seed, "delta": scores[left + str(seed)] - scores[right + str(seed)]}
            for seed in [1301, 1302]
            if left + str(seed) in scores and right + str(seed) in scores
        ]
        for comparison, left, right in [("B_minus_A", "B", "A"), ("C_minus_B", "C", "B")]
    }
    report = {
        "schema": "wnn.quality-horizon-final-comparison.v1",
        "status": "complete",
        "created_epoch": time.time(),
        "records": records,
        "paired_deltas": paired,
        "observed_complete_fresh_models": sum(
            row["kind"] == "fresh_configured_final" for row in records
        ),
        "expected_fresh_fits": 5,
        "reused_controls": 1,
        "missing_cells": sorted(set(["A1302", "B1301", "B1302", "C1301", "C1302"]) - set(arrays)),
        "primary_split_sha256": verifier.PRIMARY,
        "scored_rows": verifier.ROWS,
        "mask_and_order": mask,
        "raw_validation_source": source,
        "weights_predeclared": True,
        "comparison_panel": {
            "path": str(panel_path),
            "sha256": verifier.sha256(panel_path),
            "frozen_epoch": panel["frozen_epoch"],
            "exposure": panel["exposure"],
        },
        "missing_composite_cells": sorted(set(panel["composite_names"]) - set(scores)),
        "primary_valid_used_for_weight_learning": False,
        "exposure": "development-exposed full official valid, two matched seeds; no significance/public-score claim",
        "budget_comparison": "A/B changes batch and update frequency jointly at four epochs; B/C matches updates and row windows",
        "fitting_members": {"train": 10325, "valid": 0},
        "process_receipts": str(Q / "receipts/final-scoring.completed.json"),
    }
    with (output / "REPORT.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    with (output / "COMPARISON.csv").open("x", newline="") as stream:
        fields = [
            "name",
            "kind",
            "wp",
            "t0",
            "t1",
            "wmse",
            "delta_vs_TL3",
            "delta_vs_known_TL3_LSTM",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    for index, name in enumerate(["A1302", "B1301", "B1302", "C1301", "C1302"]):
        path = Q / "runs" / name / "monitor_evaluations.jsonl"
        events = (
            [json.loads(line) for line in path.read_text().splitlines() if line]
            if path.exists()
            else []
        )
        for column, key in enumerate(["loss", "wp"]):
            ax = axes[index, column]
            for cohort in ["train_eval", "valid"]:
                ax.plot(
                    [event["epoch"] for event in events],
                    [event["metrics"][cohort + "/" + key] for event in events],
                    marker="o",
                    label=cohort,
                )
            ax.set_title(name + " " + key)
            ax.set_xlabel("completed epoch")
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle(
        "Fixed four-epoch screen. Train diagnostic=16/all rows; official valid=1873/is_scored. Masks differ.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ["png", "svg"]:
        fig.savefig(output / ("train_valid_curves." + ext), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(
        "QUALITY_FINAL_COMPARISON",
        json.dumps({"scores": scores, "paired_deltas": paired}),
        flush=True,
    )


if __name__ == "__main__":
    main()
