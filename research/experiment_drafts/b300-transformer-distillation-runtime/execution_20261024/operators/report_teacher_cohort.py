"""Update the authorized PR32 progress comment hourly and on evaluated endpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
MARKER = "<!-- tabdata-pr32-teachers-20261024 -->"


def tail(path: Path, limit: int = 65536) -> list[dict]:
    """Read bounded complete records without scanning growing epoch logs."""
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - limit))
        content = stream.read(limit)
    lines = content.splitlines()
    if size > limit:
        lines = lines[1:]
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return records


def atomic(path: Path, value: dict) -> None:
    """Keep publication identity and last completed local report atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    """Publish only verified model progress, distinguishing profiles and evaluated fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    run = Path(plan["run_root"])
    if ROOT / "runs" not in run.resolve().parents:
        raise ValueError("progress root outside tabdata runs")
    if not (run / "STATE.json").is_file():
        return
    stamp = run / "GITHUB_PROGRESS.json"
    previous = json.loads(stamp.read_text()) if stamp.exists() else {}
    if previous.get("terminal"):
        return
    state = json.loads((run / "STATE.json").read_text())
    if state.get("status") == "running":
        try:
            stat = Path(f"/proc/{int(state['controller_pid'])}/stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            alive = int(fields[19]) == int(state["controller_start_ticks"])
        except (FileNotFoundError, KeyError, ValueError):
            alive = False
        if not alive:
            state["status"] = "controller_not_running"
    rows = []
    total_updates = completed_updates = evaluated = 0
    for job in plan["jobs"]:
        directory = run / job["name"]
        config = json.loads(Path(job["config"]).read_text())
        profile = json.loads(Path(job["profile_receipt"]).read_text())
        budget = int(profile["identity"]["optimizer"]["total_updates"])
        records = tail(directory / "fit/rank0.jsonl")
        last = records[-1] if records else {}
        step = min(int(last.get("step", 0)), budget)
        score_path = directory / "fit/score.json"
        score = (
            json.loads(score_path.read_text())["score"]
            if score_path.is_file()
            else None
        )
        evaluated += int(score is not None)
        total_updates += budget
        completed_updates += step
        epochs = (
            f"{int(last['epoch']) + 1}/{config['epochs']}"
            if "epoch" in last
            else f"0/{config['epochs']}"
        )
        loss = f"{float(last['total_loss']):.6f}" if "total_loss" in last else "—"
        score_text = (
            json.dumps(score, separators=(",", ":"))
            if score is not None
            else "not evaluated"
        )
        rows.append(
            f"| {job['name']} | {epochs} | {step:,}/{budget:,} | {100 * step / budget:.2f}% | {loss} | {score_text} |"
        )
    terminal = state.get("status") in (
        "complete",
        "partial",
        "completed",
        "failed",
        "stopped",
        "interrupted",
        "timed_out",
        "controller_not_running",
    )
    changed = evaluated != previous.get("evaluated", -1) or terminal
    if (
        previous
        and not changed
        and time.time() - float(previous.get("published_unix", 0)) < 3600
    ):
        return
    body = "\n".join(
        [
            MARKER,
            "PR32 full-teacher execution progress (UTC "
            + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            + ").",
            "",
            f"Source: `{plan['source']['commit']}`. Status: `{state.get('status', 'running')}`.",
            f"Attempt: `{run.name}`; revision `{plan.get('revision', 'unspecified')}`.",
            "A previous startup completed zero updates and was stopped by exact profile admission because optimizer metadata used tuples before JSON serialization. The metadata-only repair is included in the current source; original attempt receipts are retained."
            if plan.get("prior_attempt")
            else "",
            f"Evaluated scientific fits: **{evaluated}/4**. Completed optimizer updates: **{completed_updates:,}/{total_updates:,} ({100 * completed_updates / total_updates:.2f}%)**.",
            "",
            "| Cell | Current epoch | Optimizer updates | Update progress | Last training loss | Official held-out score |",
            "|---|---:|---:|---:|---:|---|",
            *rows,
            "",
            "Each cell is one 30-epoch teacher fit. Engineering profiles, warmups and failed starts are not completed experiments. Update progress is not an evaluated quality result. Scores, when present, use the frozen exposed F0/F2 endpoints; no public score is inferred.",
            "",
            "Runtime precision, source/config/data identities, resource allocation and deadlines are bound in the execution plan and approved profile receipts. The operator releases each GPU group after its fit and evaluation; it touches only this cohort’s processes.",
        ]
    )
    helper = ROOT / "runtime/github-reporting-v1"
    sys.path.insert(0, str(helper / "code"))
    from github_api import GitHub, read_token

    settings = json.loads((helper / "CONFIG.json").read_text())
    api = GitHub(read_token(Path(settings["token_file"])), request_limit=12)
    assert api.request("GET", "/user")["login"].lower() == "semyonepanov"
    comment_id = previous.get("comment_id")
    if comment_id is None:
        matches = [
            c
            for c in api.comments(32)
            if MARKER in c.get("body", "")
            and c.get("user", {}).get("login", "").lower() == "semyonepanov"
        ]
        if len(matches) > 1:
            raise RuntimeError("ambiguous progress-comment identity")
        if matches:
            comment_id = matches[0]["id"]
    result = (
        api.patch(int(comment_id), body)
        if comment_id is not None
        else api.post(32, body)
    )
    receipt = {
        "comment_id": result["id"],
        "url": result["html_url"],
        "published_unix": time.time(),
        "evaluated": evaluated,
        "completed_updates": completed_updates,
        "terminal": terminal,
    }
    atomic(stamp, receipt)
    with (run / "github_progress_history.jsonl").open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
