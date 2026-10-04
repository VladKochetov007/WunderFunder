"""Finish the preregistered EMA comparison, stopping on any guard incident."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "checks/ema_sequence_v1.json"


def wait_finished(attempt: str) -> None:
    """Require success; never acknowledge or retry an incident automatically."""
    deadline = time.monotonic() + 45_000
    while time.monotonic() < deadline:
        if (RUNNER / "PAUSE.json").exists():
            raise RuntimeError("Resource guard paused")
        record = json.loads((RUNNER / f"attempts/{attempt}.json").read_text())
        if record["status"] == "finished":
            time.sleep(5)
            return
        if record["status"] not in ("reserved", "running"):
            raise RuntimeError(f"Execution stopped: {attempt}: {record['status']}")
        time.sleep(10)
    raise TimeoutError(attempt)


def launch(draft: Path, attempt: str, completed: list[str]) -> None:
    """Seal each future artifact dependency once available and admit one job."""
    command = [sys.executable, "scripts/run_campaign.py"]
    sealed = draft.with_name(draft.name.replace(".draft.json", ".sealed.json"))
    if not sealed.exists():
        subprocess.run([*command, "seal", str(draft), str(sealed)], check=True, cwd=ROOT)
    subprocess.run([*command, "launch", str(sealed), "--attempt", attempt], check=True, cwd=ROOT)
    wait_finished(attempt)
    completed.append(attempt)
    OUTPUT.write_text(json.dumps({"status": "running", "completed": completed}, indent=2) + "\n")


def main() -> None:
    """Run the fixed same-source control and common-precision exports, then score."""
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    completed: list[str] = []
    OUTPUT.write_text(json.dumps({"status": "running", "completed": completed}) + "\n")
    try:
        wait_finished("e423-ema-rows-f0-v1")
        completed.append("e423-ema-rows-f0-v1")
        launch(CAMPAIGN / "specs/e423_combo_control_v1.draft.json", "e423-combo-control-f0-v1", completed)
        sys.path.insert(0, str(ROOT / "scripts"))
        from continuation_recipes import prediction_recipe

        for run in ("c20261004_ema_rows_s3_f0_v2", "c20261004_e423_combo_control_s3_f0_v1"):
            draft = prediction_recipe(run, 3)
            export = run + "_common_fp32"
            draft["name"] = "predict_" + export
            draft["argv"] += ["--inference-precision", "fp32", "--output-name", export]
            draft["code"] = sorted(set(draft["code"]) | {str(p.relative_to(ROOT)) for p in (ROOT / "src/wnn").glob("*.py")})
            draft["artifacts"] = [f"cache/pred_{export}_valid.{ext}" for ext in ("npy", "json")]
            path = CAMPAIGN / f"specs/predict_{export}.draft.json"
            with path.open("x") as stream:
                stream.write(json.dumps(draft, indent=2) + "\n")
            launch(path, "predict-" + export.replace("_", "-") + "-v1", completed)
        launch(CAMPAIGN / "specs/e423_score_v1.draft.json", "e423-score-v1", completed)
        OUTPUT.write_text(json.dumps({"status": "completed", "completed": completed}, indent=2) + "\n")
    except Exception as error:
        OUTPUT.write_text(json.dumps({"status": "stopped", "completed": completed,
                                      "reason": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
