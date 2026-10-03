from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Record exact sample lists, preserved failures, and artifact hashes."""
    manifest = json.loads((HERE / "clock_probe_manifest.json").read_text())
    lines = ["All IDs previously exposed to catalogue; not independent validation."]
    for split in ("train", "valid"):
        for field in ("rg", "seq_ix"):
            lines.append(
                f"{split} {field}: " + ", ".join(str(s[field]) for s in manifest[split])
            )
    (HERE / "clock_sample_ids.txt").write_text("\n".join(lines) + "\n")
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = "bounded_pass_complete_frozen_adoption_rule_failed"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    state["pending_work"] = [
        "Controller integration review of regime-dependent clock reversal",
        "Future frozen input-only gate and same-group clock ablation; no job running",
    ]
    state["failure_history"] = [
        {
            "exit_code": 75,
            "log": "clock_probe.log",
            "reason": "Initial admission barrier; explicit controller retry authorization received",
        },
        {
            "exit_code": 1,
            "log": "clock_probe_retry.log",
            "reason": "Numba float(bool) compilation failure before fitting; equivalent conditional fixed it",
        },
    ]
    state["last_failure"] = {**state["failure_history"][-1], "resolved": True}
    state["inspected_sequence_ids"].pop("planned", None)
    state["commands"] = [
        {
            "command": "run_cpu.sh eda-signal .venv/bin/python -B research/campaign_2026-10-03/eda/signal/clock_probe.py",
            "attempts": [
                {"log": "clock_probe.log", "exit": 75},
                {"log": "clock_probe_retry.log", "exit": 1},
                {"log": "clock_probe_run.log", "exit": 0},
            ],
        },
        {
            "command": "run_cpu.sh eda-signal .venv/bin/python -B research/campaign_2026-10-03/eda/signal/plot_clock_probe.py",
            "log": "plot_clock_probe.log",
            "exit": 0,
        },
        {
            "command": "run_cpu.sh eda-signal .venv/bin/python -B research/campaign_2026-10-03/eda/signal/diagnose_clock_probe.py",
            "log": "diagnose_clock_probe.log",
            "exit": 0,
        },
        {
            "command": "Ruff format and check under run_cpu.sh; final checks isolated and cache disabled",
            "logs": [
                "format_clock_probe.log",
                "lint_clock_probe.log",
                "lint_scripts.log",
                "final_lint.log",
            ],
        },
        {
            "command": "run_cpu.sh eda-signal .venv/bin/marimo export html research/campaign_2026-10-03/eda/signal/eda_signal.py -o research/campaign_2026-10-03/eda/signal/eda_signal_continuation.html",
            "logs": ["notebook_export.log", "notebook_export_final.log"],
        },
    ]
    state["inspected_figures"] = ["fig_clock_probe.png", "fig_clock_activity.png"]
    artifacts = [
        "RECOVERY_SUMMARY.txt",
        "HYPOTHESES.json",
        "REPORT.txt",
        "REPORT_admission_blocked.txt",
        "STATE.json",
        "clock_probe.py",
        "clock_probe_v1_executed.py",
        "diagnose_clock_probe.py",
        "plot_clock_probe.py",
        "finalize_clock_probe.py",
        "clock_probe_manifest.json",
        "clock_probe_results.json",
        "clock_probe_diagnostics.json",
        "clock_probe_stats.npz",
        "clock_probe_weights.npz",
        "clock_sample_ids.txt",
        "fig_clock_probe.png",
        "fig_clock_activity.png",
        "eda_signal.py",
        "eda_signal_continuation.html",
        "clock_probe.log",
        "clock_probe_retry.log",
        "clock_probe_run.log",
        "plot_clock_probe.log",
        "diagnose_clock_probe.log",
        "notebook_export.log",
        "notebook_export_final.log",
        "format_clock_probe.log",
        "lint_clock_probe.log",
        "lint_scripts.log",
        "final_lint.log",
    ]
    state["artifacts"] = artifacts + ["artifact_provenance.json"]
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    provenance = {
        "created_utc": state["finished_utc"],
        "claim_class": "Exploratory exposed-sequence mechanism probe; adoption rule failed",
        "frozen_hypotheses_hash": manifest["hypotheses_sha256"],
        "executed_source_hash": manifest["script_sha256"],
        "files": {
            name: {
                "sha256": hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
                "bytes": (HERE / name).stat().st_size,
            }
            for name in artifacts
            if (HERE / name).exists()
        },
        "hash_note": "Executed source is clock_probe_v1_executed.py; clock_probe.py was subsequently formatted without changing method. No whole-dataset hashing or extra data census.",
    }
    (HERE / "artifact_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
