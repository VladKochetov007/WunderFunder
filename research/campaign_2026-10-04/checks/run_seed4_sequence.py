"""Queue the fixed seed-4 replication after the admitted architecture sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "checks/e450_sequence_v1.json"
PREDECESSOR = "wnn-e424-sequence-v5.service"
PARENT = "c20261004_combo_s4_final"
HEAD = "c20261004_combo_head_s4_final"
ARMS = (
    "h093-learned",
    "h093-gru-budget",
    "h094-conditional",
    "h094-gru-budget",
    "h093-hash",
    "h093-universal",
    "h094-unconditional",
    "h094-diagonal",
)


def read_json(path: Path) -> dict:
    """Read an authoritative record; missing or malformed records fail closed."""
    return json.loads(path.read_text())


def save(status: str, completed: list[str], **details: object) -> None:
    """Replace only this queue's current snapshot, preserving runner raw records."""
    OUTPUT.write_text(
        json.dumps({"status": status, "completed": completed, **details}, indent=2) + "\n"
    )


def guard() -> None:
    """Keep existing campaign resource stops authoritative."""
    if (RUNNER / "PAUSE.json").exists():
        raise RuntimeError("Campaign resource pause requires review")
    if not (CAMPAIGN / "CONTINUATION_AUTHORIZATION_20261004.md").exists():
        raise FileNotFoundError("Missing continuation authorization record")


def properties(unit: str) -> dict[str, str]:
    """Observe the exact systemd handle, not merely a queue lock or snapshot."""
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "ExecMainStatus"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def require_success(attempt: str) -> dict:
    """Verify a finished job's exit and teardown before consuming its artifacts."""
    record = read_json(RUNNER / f"attempts/{attempt}.json")
    if (
        record["status"] != "finished"
        or record.get("exit_code") != 0
        or not record.get("teardown_verified")
    ):
        raise RuntimeError(f"Unsuccessful required attempt: {attempt}")
    return record


def wait_predecessor(completed: list[str]) -> None:
    """Wait only while the observed E424 controller handle remains live."""
    while True:
        guard()
        observed = properties(PREDECESSOR)
        sequence = read_json(CAMPAIGN / "checks/e424_sequence_v2.json")
        if sequence["status"] == "completed":
            if observed.get("ActiveState") in ("active", "activating", "deactivating"):
                time.sleep(5)
                continue
            for arm in ARMS:
                require_success(f"e424-{arm}-v2")
            require_success("e424-score-all-v3")
            return
        if sequence["status"] != "running":
            raise RuntimeError(f"E424 controller stopped: {sequence}")
        if observed.get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"E424 handle is no longer live: {observed}")
        save("waiting_for_e424", completed, predecessor=PREDECESSOR, observed=observed)
        time.sleep(15)


def launch(draft: Path, attempt: str, completed: list[str]) -> None:
    """Admit one immutable job and reuse only an identical successful attempt."""
    guard()
    command = [sys.executable, "scripts/run_campaign.py"]
    sealed = draft.with_name(draft.name.replace(".draft.json", ".sealed.json"))
    if not sealed.exists():
        subprocess.run([*command, "seal", str(draft), str(sealed)], cwd=ROOT, check=True)
    expected = read_json(sealed)
    record_path = RUNNER / f"attempts/{attempt}.json"
    if record_path.exists():
        record = read_json(record_path)
        if record["spec"] != expected:
            raise ValueError(f"Existing attempt identity differs: {attempt}")
    else:
        subprocess.run(
            [*command, "launch", str(sealed), "--attempt", attempt], cwd=ROOT, check=True
        )
    deadline = time.monotonic() + int(expected["limits"]["timeout_seconds"]) + 240
    while True:
        guard()
        record = read_json(record_path)
        if record["status"] == "finished":
            require_success(attempt)
            break
        if record["status"] not in ("running", "reserved"):
            raise RuntimeError(f"Attempt stopped: {attempt}: {record['status']}")
        observed = properties(record["unit"])
        if observed.get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"Attempt handle disappeared: {attempt}: {observed}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Observation deadline: {attempt}; do not relaunch")
        save("running", completed, active_attempt=attempt, observed=observed)
        time.sleep(5)
    if attempt not in completed:
        completed.append(attempt)
    save("running", completed)


def write_draft(name: str, draft: dict) -> Path:
    """Keep generated recipe identity immutable across controller restarts."""
    path = CAMPAIGN / f"specs/{name}.draft.json"
    if path.exists():
        if read_json(path) != draft:
            raise ValueError(f"Recipe changed: {path}")
    else:
        path.write_text(json.dumps(draft, indent=2) + "\n")
    return path


def main() -> None:
    """Run parent, fixed head procedure, then common-FP32 prediction exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.resume:
        raise FileExistsError(OUTPUT)
    completed = read_json(OUTPUT)["completed"] if OUTPUT.exists() else []
    sys.path.insert(0, str(ROOT / "scripts"))
    from continuation_recipes import head_args, input_paths, prediction_recipe, recipe

    try:
        wait_predecessor(completed)
        launch(
            CAMPAIGN / "specs/e450_combo_seed4_final_v1.draft.json",
            "e450-combo-seed4-final-v1",
            completed,
        )
        head = recipe("combo_head", None, 3, "local", root=ROOT)
        head["name"] = "e450_combo_head_seed4_final_v1"
        head["argv"] = head_args(HEAD, f"{PARENT}@ep3", None)
        head["inputs"] = input_paths(None, parent=f"{PARENT}@ep3")
        head["artifacts"] = [
            f"artifacts/runs/{HEAD}/{name}"
            for name in ("model_ep0.pt", "provenance.json", "history.json", "training_state.pt")
        ]
        head["limits"]["timeout_seconds"] = 3600
        launch(write_draft(head["name"], head), "e450-combo-head-seed4-final-v1", completed)
        for run, epoch in ((PARENT, 3), (HEAD, 0)):
            draft = prediction_recipe(run, epoch, root=ROOT)
            export = f"{run}_fp32"
            draft["name"] = f"e450_predict_{export}_v1"
            draft["argv"] += ["--inference-precision", "fp32", "--output-name", export]
            draft["artifacts"] = [f"cache/pred_{export}_valid.{ext}" for ext in ("npy", "json")]
            draft["limits"]["ram_bytes"] = 24 * 1024**3
            draft["limits"]["timeout_seconds"] = 1800
            launch(
                write_draft(draft["name"], draft),
                f"e450-predict-{run.replace('_', '-')}-v1",
                completed,
            )
        save("fits_and_exports_complete", completed, paired_score="pending")
    except Exception as error:
        save("stopped", completed, reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
