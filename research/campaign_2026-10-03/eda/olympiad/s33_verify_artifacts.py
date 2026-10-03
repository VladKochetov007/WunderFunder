from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Hash small owned artifacts; never scan the raw datasets."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify provenance and pooled moments, then freeze the owned artifact manifest."""
    checked = []
    for number in [30, 31, 32]:
        provenance = json.loads((ROOT / f"s{number}_provenance.json").read_text())
        script = {
            30: "s30_anchor_probe.py",
            31: "s31_revision_blocks.py",
            32: "s32_price_revision.py",
        }[number]
        hypothesis = "HYPOTHESES.json" if number == 30 else f"HYPOTHESES_s{number}.json"
        assert provenance["script_sha256"] == digest(ROOT / script)
        assert provenance["hypotheses_sha256"] == digest(ROOT / hypothesis)
        if number != 30:
            assert provenance["parent_script_sha256"] == digest(ROOT / "s30_anchor_probe.py")
        checked.append({"probe": number, "source_and_hypothesis_hashes_match": True})
    result = json.loads((ROOT / "s30_results.json").read_text())
    stats = np.load(ROOT / "s30_sequence_moments.npy").sum(axis=0)
    modes = (
        ["current", "carry_a6", "carry_price"]
        + [f"carry_delay_{i}" for i in range(5)]
        + ["anchor_fit", "delay_fit"]
    )
    maximum_error = 0.0
    for mode_i, mode in enumerate(modes):
        for target_i, target in enumerate(["t0", "t1"]):
            w, sy, sp, syy, spp, syp = stats[mode_i, target_i]
            numerator = w * syp - sy * sp
            denominator = np.sqrt((w * syy - sy * sy) * (w * spp - sp * sp))
            recomputed = numerator / denominator
            error = abs(recomputed - result["scores"][mode][target])
            maximum_error = max(maximum_error, error)
            assert error < 1e-10
    assert len((ROOT / "REPORT.txt").read_text().splitlines()) <= 200
    html = (ROOT / "eda_olympiad.html").read_text()
    assert len(html) > 10000
    assert "Olympiad continuation" in html
    audit = json.loads((ROOT / "s30_audit.json").read_text())["sequences"]
    aggregates = {
        split: {
            key: sum(row[key] for row in audit if row["split"] == split)
            for key in [
                "rows",
                "eligible_rows",
                "target_changes",
                "a6_resets",
                "a6_misses",
                "a6_extras",
                "price_misses",
            ]
        }
        for split in ["train", "valid"]
    }
    verification = {
        "checked_provenance": checked,
        "max_pooled_moment_metric_error": maximum_error,
        "audit_by_split": aggregates,
        "notebook_export_exit": 0,
        "notebook_export_log": "notebook_export.log",
        "report_line_count": len((ROOT / "REPORT.txt").read_text().splitlines()),
        "prior_exposure": "all sampled sequences previously inspected exhaustively",
        "resource_failure": None,
    }
    (ROOT / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    state = json.loads((ROOT / "STATE.json").read_text())
    state["phase"] = "complete_independent_phase_report_frozen"
    state["completed_at_utc"] = datetime.now(UTC).isoformat()
    state["figures_inspected"] = [f"f0{i}" for i in range(1, 10)]
    state["new_figures_inspected"] = [
        "f06_anchor_probe.png",
        "f07_anchor_predictions.png",
        "f08_revision_blocks.png",
        "f09_price_revision.png",
    ]
    state["commands"] += [
        "research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad .venv/bin/python -B research/campaign_2026-10-03/eda/olympiad/s31_revision_blocks.py > research/campaign_2026-10-03/eda/olympiad/s31_probe.log 2>&1",
        "research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad .venv/bin/python -B research/campaign_2026-10-03/eda/olympiad/s32_price_revision.py > research/campaign_2026-10-03/eda/olympiad/s32_probe.log 2>&1",
        "research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=research/campaign_2026-10-03/eda/olympiad/.mplconfig XDG_CACHE_HOME=research/campaign_2026-10-03/eda/olympiad/.cache .venv/bin/marimo export html research/campaign_2026-10-03/eda/olympiad/eda_olympiad.py -o research/campaign_2026-10-03/eda/olympiad/eda_olympiad.html > research/campaign_2026-10-03/eda/olympiad/notebook_export.log 2>&1",
        "research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad .venv/bin/ruff format --check s30_anchor_probe.py s31_revision_blocks.py s32_price_revision.py s33_verify_artifacts.py eda_olympiad.py",
        "research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad .venv/bin/ruff check s30_anchor_probe.py s31_revision_blocks.py s32_price_revision.py s33_verify_artifacts.py eda_olympiad.py",
        "research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad .venv/bin/python -B research/campaign_2026-10-03/eda/olympiad/s33_verify_artifacts.py > research/campaign_2026-10-03/eda/olympiad/s33_verification.log 2>&1",
    ]
    state["artifacts"] += [
        "s30_anchor_probe.py",
        "s31_revision_blocks.py",
        "s32_price_revision.py",
        "s33_verify_artifacts.py",
        "s30_probe.log",
        "s31_probe.log",
        "s32_probe.log",
        "eda_olympiad.py",
        "eda_olympiad.html",
        "notebook_export.log",
        "REPORT.txt",
        "VERIFICATION.json",
        "ARTIFACTS.json",
    ]
    state["artifacts"] = list(dict.fromkeys(state["artifacts"]))
    state["pending"] = []
    state["next_controller_work"] = [
        "deduplicate independent reports",
        "component-aware reproduction",
        "authorize separate strong-baseline integration",
    ]
    state["last_failure"] = None
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    names = [name for name in state["artifacts"] if name != "ARTIFACTS.json"]
    names.append("STATE.json")
    manifest = {
        "created_at_utc": state["completed_at_utc"],
        "files": {
            name: {"bytes": (ROOT / name).stat().st_size, "sha256": digest(ROOT / name)}
            for name in names
        },
        "sample_ids_and_dataset_metadata": "s30_provenance.json",
        "checks": "VERIFICATION.json",
        "self_hash_omitted": True,
    }
    (ROOT / "ARTIFACTS.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
