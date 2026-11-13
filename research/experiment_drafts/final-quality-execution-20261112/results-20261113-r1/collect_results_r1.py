"""Grade the sixteen frozen completed endpoints from retained canonical monitor evidence."""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT.parent / "final-quality-wave-20261112-r1/F3/live-dashboard/native"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def read(path: Path) -> dict:
    """Read a canonical producer document retained by the bounded mirror."""
    return json.loads(path.read_bytes())


def sha(path: Path) -> str:
    """Bind the exact retained native receipt rather than a chat observation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require complete fixed endpoints and official valid coverage before comparison."""
    rows = []
    for progress_path in sorted(NATIVE.glob("*-fit/monitor_progress.json")):
        run = progress_path.parent
        terminal = read(run / "terminal_status.json")
        meta = read(run / "run_metadata.json")
        index = read(run / "epoch_index.json")
        evaluations = [
            json.loads(line)
            for line in (run / "monitor_evaluations.jsonl").read_bytes().splitlines()
        ]
        config = meta["config"]
        endpoint = int(config["epochs"])
        selected = [event for event in evaluations if event["epoch"] == endpoint]
        assert len(selected) == 1, run
        event = selected[0]
        metrics, context = event["metrics"], event["context"]
        data = meta["provenance"]["data"]
        assert terminal["status"] == "completed"
        assert terminal["completed_epochs"] == endpoint == index["completed_epochs"]
        assert context["split_sha256"] == SPLIT == data["split_sha256"]
        assert metrics["valid/sequence_count"] == 1873
        assert metrics["valid/selected_rows"] == 3_528_273
        assert context["metric_id"] == "global_wp_clip2_v1"
        assert len(data["fit_members"]) == 10325
        assert all(member[0] == "train" for member in data["fit_members"])
        assert len(data["evaluation_members"]) == 1873
        checkpoint = index["entries"][-1]
        row = {
            "run": run.name,
            "seed": config["seed"],
            "epochs": endpoint,
            "global_updates": terminal["global_updates"],
            "wp": metrics["valid/wp"],
            "t0": metrics["valid/wp/t0"],
            "t1": metrics["valid/wp/t1"],
            "valid_loss": metrics["valid/loss"],
            "evaluation_at": event["wall_time"],
            "source_sha256": meta["source"]["sha256"],
            "config_sha256": terminal["identity"]["config_sha256"],
            "checkpoint": checkpoint,
            "terminal_receipt_sha256": sha(run / "terminal_status.json"),
            "evaluation_journal_sha256": sha(run / "monitor_evaluations.jsonl"),
            "since_weight": config.get("since_weight", 0),
            "kd_weight": config["kd_weight"],
            "teacher_manifest_sha256": (data.get("teacher") or {}).get("manifest_sha256"),
            "endpoint_kind": "full official-valid epoch monitor; fixed final endpoint",
            "evidence_limit": "retained remote journal, not a fresh native file/process check",
        }
        previous = [item for item in evaluations if item["epoch"] == endpoint - 1]
        if previous:
            row["previous_epoch_wp"] = previous[0]["metrics"]["valid/wp"]
            row["last_epoch_delta"] = row["wp"] - row["previous_epoch_wp"]
        rows.append(row)
    assert len(rows) == 16

    def values(prefix: str) -> list[float]:
        """Order each two-seed arm consistently without mixing scientific panels."""
        selected = sorted(
            (row for row in rows if row["run"].startswith(prefix)), key=lambda row: row["seed"]
        )
        assert len(selected) == 2, prefix
        return [row["wp"] for row in selected]

    arms = {
        "zero_since": values("l4-zero_since_control_"),
        "since32": values("l4-label_since32_"),
        "since8": values("l4-age8_"),
        "since128": values("l4-age128_"),
        "kd_zero": values("l4-standalone_kd0_k16_"),
        "kd020": values("l4-standalone_kd0.2_"),
        "linear": values("b300-be_input_linear_"),
        "bilinear": values("b300-be_input_bilinear_"),
    }
    canonical = read(ROOT / "CANONICAL_B300_R1.json")
    for cell in canonical["cells"]:
        assert cell["n"] == 1873 and cell["score"]["selected_rows"] == 3_528_273
        name = "b300-" + cell["cell"] + "-fit"
        row = next(row for row in rows if row["run"] == name)
        row["canonical_export_wp"] = cell["score"]["wp"]
        row["canonical_export_checkpoint_sha256"] = cell["ckp"]
        row["canonical_grade_sha256"] = cell["sha"]
        row["canonical_minus_monitor_wp"] = cell["score"]["wp"] - row["wp"]
    for arm, prefix in (("linear", "be_input_linear_"), ("bilinear", "be_input_bilinear_")):
        selected = sorted(
            (cell for cell in canonical["cells"] if cell["cell"].startswith(prefix)),
            key=lambda cell: cell["cell"],
        )
        arms[arm] = [cell["score"]["wp"] for cell in selected]
    means = {name: sum(values) / 2 for name, values in arms.items()}
    comparisons = []
    for control, treatment in (
        ("zero_since", "since32"),
        ("since32", "since8"),
        ("since32", "since128"),
        ("kd_zero", "kd020"),
        ("linear", "bilinear"),
    ):
        deltas = [new - old for old, new in zip(arms[control], arms[treatment], strict=True)]
        comparisons.append(
            {
                "control": control,
                "treatment": treatment,
                "control_mean": means[control],
                "treatment_mean": means[treatment],
                "paired_deltas": deltas,
                "mean_delta": sum(deltas) / 2,
            }
        )
    report = {
        "schema": "wnn.final-quality-results.r1",
        "collected_at": datetime.now(UTC).isoformat(),
        "scientific_fits": 16,
        "completed_fits": 16,
        "validation": {
            "split_sha256": SPLIT,
            "fit_train_sequences": 10325,
            "fit_valid_sequences": 0,
            "valid_sequences": 1873,
            "scored_rows": 3528273,
            "metric": "global_wp_clip2_v1",
            "exposure": "repeatedly exposed development validation",
        },
        "runs": rows,
        "arms": arms,
        "means": means,
        "comparisons": comparisons,
        "interpretation": "Two seed-paired descriptive comparisons, not significance or public-score forecasts. KD and since panels use different seeds; their combination is untested.",
        "grade_basis": "GRU/KD fixed-endpoint full-valid epoch monitor; Transformer canonical independent exports/scorer, freshly read on B300. Corresponding Transformer monitor results are retained separately.",
        "platform_observation": {
            "method": "signed-in Edge screenshots of official submit and public leaderboard pages on2026-11-13",
            "team": "kotiki",
            "best_public": 0.66832,
            "public_rank": 61,
            "total_submissions": 108,
            "submissions_today": 0,
            "latest_submission_date": "2026-11-04",
            "latest_submission_id": "1H7KP3OQ",
            "latest_submission_file": "tabdata-w18-final-20261103-r1.zip",
            "latest_submission_score": 0.66453,
            "latest_duration": "14m32s",
            "new_fullmix_uploaded": False,
        },
    }
    archive = ROOT / "fixed-endpoint-monitor-evidence-r1.tar.gz"
    with tarfile.open(archive, "x:gz") as handle:
        for row in rows:
            for name in (
                "run_metadata.json",
                "terminal_status.json",
                "epoch_index.json",
                "monitor_evaluations.jsonl",
            ):
                path = NATIVE / row["run"] / name
                handle.add(path, arcname=row["run"] + "/" + name, recursive=False)
    report["raw_monitor_evidence"] = {
        "file": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha(archive),
        "files": len(rows) * 4,
        "contains_raw_dataset_or_weights": False,
    }
    (ROOT / "RESULTS_R1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"completed": len(rows), "means": means, "comparisons": comparisons}, indent=2)
    )


if __name__ == "__main__":
    main()
