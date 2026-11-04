"""Export singleton/ensemble/status comparisons without converting partial fits to final."""

from __future__ import annotations

import csv
from pathlib import Path

from report_multistage_primary_ensemble_worker import TL3_WP, _read_json, _sha256, _write_new


def _plot(rows: list[dict[str, object]], output: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [row for row in rows if isinstance(row.get("wp"), (int, float))]
    fig, axis = plt.subplots(figsize=(11, 5))
    positions = list(range(len(scored)))
    axis.bar(
        positions,
        [row["wp"] for row in scored],
        color=[
            "gray"
            if row["kind"] == "reference"
            else "seagreen"
            if row["kind"] == "ensemble"
            else "steelblue"
            for row in scored
        ],
    )
    axis.set_xticks(positions, [row["name"] for row in scored], rotation=25, ha="right")
    axis.set_ylabel("Full official primary-valid WP")
    axis.set_title("Frozen full-primary refits: configured-final endpoints only")
    axis.grid(axis="y", alpha=0.25)
    skipped = [row["name"] for row in rows if row["kind"] == "singleton" and row.get("wp") is None]
    fig.text(
        0.01,
        0.01,
        "Development-exposed reporting; TL3 fixed reference was not rescored.\n"
        + (
            "Unscored/partial: " + ", ".join(skipped)
            if skipped
            else "Both ensemble vectors were frozen before the four fits."
        ),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    paths = [output / "primary_refit_comparison.png", output / "primary_refit_comparison.svg"]
    for path in paths:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [{"path": str(path), "sha256": _sha256(path)} for path in paths]


def export_report(
    request: dict[str, object],
    request_path: Path,
    request_sha: str,
    outcomes: dict[str, object],
    blend_execution: dict[str, object],
) -> dict[str, object]:
    output = Path(request["output_dir"])
    rows, scores = [], []
    for member in request["members"]:
        result = outcomes.get(str(member["index"]), {"status": "not_eligible_partial_fit"})
        receipt = (
            _read_json(Path(member["receipt_path"]), "singleton score")
            if result.get("status") == "completed"
            else None
        )
        if receipt is not None and (
            receipt.get("status") != "complete"
            or receipt.get("fit_id") != member["fit_id"]
            or receipt.get("checkpoint") != member["checkpoint"]
        ):
            raise ValueError(
                "singleton report receipt differs from its exact configured-final member"
            )
        metrics = receipt["metrics"] if receipt else {}
        scores.append({"fit_id": member["fit_id"], "execution": result, "score": receipt})
        rows.append(
            {
                "name": member["fit_id"],
                "kind": "singleton",
                "family": member["family"],
                "fit_status": member["child_status"],
                "score_status": result["status"],
                "configured_final_complete": member["configured_final_complete"],
                "configured_epochs": member["config"]["epochs"],
                "native_completed_epochs": member["native_terminal"].get("completed_epochs"),
                "native_epoch_index_completed_epochs": member.get(
                    "native_epoch_index_completed_epochs"
                ),
                "trained_train_sequences": 10325,
                "trained_valid_sequences": 0,
                **{
                    key: metrics.get(key)
                    for key in ("wp", "t0", "t1", "wmse", "wmse_t0", "wmse_t1")
                },
                "delta_wp_vs_fixed_TL3": metrics["wp"] - TL3_WP if metrics else None,
            }
        )
    blends = None
    if blend_execution.get("status") == "completed":
        blends = _read_json(output / "fixed_ensemble_scores.json", "fixed ensemble scores")
        if (
            not all(member["configured_final_complete"] for member in request["members"])
            or blends.get("status") != "complete"
        ):
            raise ValueError("fixed-four ensemble receipt cannot admit a partial member")
        for name, spec in blends["ensembles"].items():
            if spec.get("weights") != request["weights"].get(name):
                raise ValueError("reported blend weights differ from the frozen vector")
            row = dict.fromkeys(rows[0])
            row.update(
                name=name,
                kind="ensemble",
                family="fixed four-member blend",
                fit_status="all_four_configured_final_complete",
                score_status="complete",
                configured_final_complete=True,
                trained_train_sequences=10325,
                trained_valid_sequences=0,
                **{
                    key: spec["metrics"][key]
                    for key in ("wp", "t0", "t1", "wmse", "wmse_t0", "wmse_t1")
                },
                delta_wp_vs_fixed_TL3=spec["metrics"]["wp"] - TL3_WP,
            )
            rows.append(row)
    reference = dict.fromkeys(rows[0])
    reference.update(
        name="canonical fresh TL3",
        kind="reference",
        family="TL3",
        score_status="fixed reference; not rescored",
        trained_train_sequences=10325,
        trained_valid_sequences=0,
        wp=TL3_WP,
        delta_wp_vs_fixed_TL3=0,
    )
    rows.append(reference)
    csv_path = output / "primary_refit_comparison.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plots = _plot(rows, output)
    report = {
        "schema": "wnn.full-primary-refit-final-report.v1",
        "status": "complete" if blends is not None else "partial_or_singletons_only",
        "report_type": "primary_development_exposed_report_only",
        "objective_eligible": False,
        "checkpoint_selection_allowed": False,
        "primary_valid_used_for_weights": False,
        "primary_split": request["primary_split"],
        "evaluation_mask": "is_scored AND row_index>=99",
        "source6": request["source6"],
        "score_request": {"path": str(request_path), "sha256": request_sha},
        "frozen_selection": request["selection"],
        "frozen_weights": request["frozen_weights"],
        "weights": request["weights"],
        "fit_statuses": request["members"],
        "singleton_scores": scores,
        "fixed_ensemble_execution": blend_execution,
        "fixed_ensemble_scores": blends,
        "ensemble_policy": request["ensemble_policy"],
        "comparison_rows": rows,
        "fixed_TL3_reference": {
            "wp": TL3_WP,
            "measured_here": False,
            "source": "research/experiment_drafts/heterogeneous-tl3-20261030/results/R3/REPORT.json",
        },
        "comparison_csv": {"path": str(csv_path), "sha256": _sha256(csv_path)},
        "plots": plots,
    }
    path = output / "full_primary_refit_report.json"
    _write_new(path, report)
    manifest = {
        "schema": "wnn.full-primary-refit-report-artifacts.v1",
        "files": [
            {
                "path": str(item.relative_to(output)),
                "bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
            for item in sorted(output.rglob("*"))
            if item.is_file() and not item.is_symlink()
        ],
    }
    _write_new(output / "ARTIFACT_MANIFEST.json", manifest)
    return {
        "status": report["status"],
        "report_path": str(path),
        "report_sha256": _sha256(path),
        "comparison_rows": rows,
    }
