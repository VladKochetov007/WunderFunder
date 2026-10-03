from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = Path(__file__).resolve().parent
CAMPAIGN = STATE.parent


def write_json(path: Path, value: object) -> None:
    """Persist readable controller metadata."""
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity(path: Path) -> dict[str, str]:
    """Record the exact reviewed file independently of runtime logs."""
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    """Finalize evidence without claiming the public goal or completed queue."""
    now = datetime.now().astimezone()
    registry_path = STATE / "agents.json"
    registry = json.loads(registry_path.read_text())
    for agent in registry["agents"]:
        agent["status"] = "completed_closed"
        role = agent["role"]
        if role.startswith("eda-"):
            path = CAMPAIGN / "eda" / role.removeprefix("eda-").replace("-", "_") / "REPORT.txt"
        elif role == "opt2-builder":
            path = CAMPAIGN / "opt2/REPORT.txt"
        else:
            path = STATE / "reproduction/REPORT.txt"
        agent["report"] = identity(path)
    for agent in registry["runtime_workers"]:
        agent.update(status="completed_closed", model="gpt-6.1-sol", reasoning_effort="medium")
    registry["independent_optimizer_critic"] = {
        "agent_id": "01a101f9-30d6-78d3-9a2b-ab61367428ad",
        "nickname": "Carver",
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "status": "completed_closed",
        "report": identity(STATE / "opt2_review/REVIEW.txt"),
    }
    registry["updated_at"] = now.isoformat()
    write_json(registry_path, registry)
    write_json(
        STATE / "STATE.json",
        {
            "phase": "independent_passes_complete_queue_active",
            "updated_at": now.isoformat(),
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
            "original_claude_session": "7ea48a59-c833-4cf5-b1fe-467c6d7c1ac5",
            "independent_eda_completed": 10,
            "optimizer_builder_completed": True,
            "strong_baseline_falsifier_completed": True,
            "independent_critic_completed": True,
            "resource_limits_verified": True,
            "runtime_checks_passed": 20,
            "loader_checks_passed": 7,
            "archive_checks_passed": 11,
            "optimizer_builder_checks_passed": 81,
            "corrected_optimizer_checks_passed": 39,
            "optimizer_patch_applied": False,
            "public_best_reported": 0.6641,
            "target_achieved": False,
            "monitor_service": "wnn-codex-research-watch.service",
            "monitor_interval_seconds": 60,
            "active_training": "c20261003_w320_s3_f0",
            "next": "review width320 pair and frozen combo folds2/3 after automatic scoring; GPU parity before optimizer integration",
        },
    )
    changed = [
        CAMPAIGN / "opt2" / name
        for name in (
            "sampling.py",
            "test_training_dynamics.py",
            "test_patch.py",
            "trainer.patch",
            "trainer_candidate.py",
            "PLAN.txt",
            "FLAGS.txt",
            "CONTROLLER_CORRECTIONS.txt",
        )
    ]
    (CAMPAIGN / "opt2/SOURCE_HASHES_CONTROLLER.txt").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in changed
        )
    )
    opt_path = CAMPAIGN / "opt2/STATE.json"
    opt_state = json.loads(opt_path.read_text())
    opt_state.update(
        controller_review="two issues corrected;39 focused CPU tests pass;patch unapplied",
        controller_corrections="CONTROLLER_CORRECTIONS.txt",
    )
    write_json(opt_path, opt_state)
    ledger = ROOT / "research/experiments.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    kind = "Codex puzzle continuations and corrected unapplied optimizer prototype"
    if not any(row.get("kind") == kind for row in rows):
        number = (
            max(
                int(row["id"][2:])
                for row in rows
                if row.get("id", "").startswith("E-") and row["id"][2:].isdigit()
            )
            + 1
        )
        record = {
            "id": f"E-{number}",
            "date": now.isoformat(),
            "kind": kind,
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
            "reports": [
                identity(CAMPAIGN / relative)
                for relative in (
                    "eda/puzzle_a/REPORT.txt",
                    "eda/puzzle_b/REPORT.txt",
                    "opt2/REPORT.txt",
                    "opt2/CONTROLLER_CORRECTIONS.txt",
                    "continuation/opt2_review/REVIEW.txt",
                )
            ],
            "result": {
                "puzzle_a": "timeless price-code sum lookup falsified;unchanged-price label constancy retained on subset",
                "puzzle_b": "period6/phase fail;lag4 gain concentrated in two sequences;consistency fails",
                "optimizer": "81 builder CPU tests and6 synthetic fits;review finds two failures;controller fixes both and39 focused tests pass",
            },
            "verdict": "No forecasting efficacy established;patch unapplied pending GPU/fused/EMA parity;continue admitted width320/additionalfold combo tests",
        }
        with ledger.open("a") as output:
            output.write(json.dumps(record) + "\n")
        bandit_path = ROOT / "research/bandit/arms.json"
        bandit = json.loads(bandit_path.read_text())
        bandit["continuation_evidence"]["all_independent_passes"] = {
            "ledger": record["id"],
            "eda_completed": 10,
            "optimizer": "corrected CPU prototype;unapplied",
            "synthesis": str((STATE / "RESEARCH_SYNTHESIS.md").relative_to(ROOT)),
        }
        write_json(bandit_path, bandit)
        print(record["id"])
    chain_dir = ROOT / ".claude/chains/wnn-connectome"
    chain_dir.mkdir(parents=True, exist_ok=True)
    chain_path = chain_dir / (now.strftime("%Y-%m-%d-%H%M") + "-reviewed-research-handoff.md")
    sections = {
        "Primary Request and Intent": "Continue independent Claude research as active Codex R&D, save progress, monitor crashes;40GB RAM/5-6GB VRAM/12CPU limits.",
        "Key Technical Concepts": "Fold/seed-matched pooled clipped weighted Pearson; component uncertainty; resource admission; paused restore.",
        "Work Completed": "Ten EDA passes, optimizer prototype, falsifier, reviews, runtime repairs and saved evidence.",
        "Decisions and Rationale": "Combo replicates but needs broader confirmation; state calibration closes; optimizer patch stays unapplied pending GPU parity.",
        "Files and Code Changes": "See durable handoff below and agents.json; runtime edits:campaign_job,resources,train_rnn,predict_run.",
        "Unresolved Issues and Blockers": "Public target unproven; GPU optimizer parity and queued fits outstanding.",
        "Pending Tasks": "Review width/additional-fold scores, update ledger/bandit, select narrow next arm.",
        "Current Work": "Healthy width320 training; persistent sequential queue,watcher and scoring.",
        "Next Step": "Read continuation/HANDOFF.md,watcher status,ETA and latest snapshot;reconcile live units.",
    }
    chain_path.write_text(
        "# Reviewed research continuation\n\n"
        + "".join(f"## {title}\n\n{body}\n\n" for title, body in sections.items())
        + (STATE / "HANDOFF.md").read_text()
    )
    print(str(chain_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
