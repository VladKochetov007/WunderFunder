"""Advance fixed precision controls only after reviewed jobs finish successfully."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "checks/precision_sequence_v1.json"


def wait_finished(attempt: str) -> None:
    deadline = time.monotonic() + 8000
    while time.monotonic() < deadline:
        if (RUNNER / "PAUSE.json").exists():
            raise RuntimeError("Resource guard paused; no automatic acknowledgement")
        record = json.loads((RUNNER / f"attempts/{attempt}.json").read_text())
        if record["status"] == "finished":
            time.sleep(5)  # Let the successful worker unit release its GPU admission.
            return
        if record["status"] not in ("reserved", "running"):
            raise RuntimeError(f"Reviewed execution stopped: {attempt}")
        time.sleep(5)
    raise TimeoutError(attempt)


def launch(draft: Path, attempt: str, completed: list[str]) -> None:
    command = [sys.executable, "scripts/run_campaign.py"]
    sealed = draft.with_name(draft.name.replace(".draft.json", ".sealed.json"))
    if not sealed.exists():
        subprocess.run([*command, "seal", str(draft), str(sealed)], check=True)
    subprocess.run([*command, "launch", str(sealed), "--attempt", attempt], check=True)
    wait_finished(attempt)
    completed.append(attempt)
    OUTPUT.write_text(json.dumps({"status": "running", "completed": completed}, indent=2) + "\n")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    completed: list[str] = []
    try:
        wait_finished("e419-float-fp16-f0-v3")
        completed.append("e419-float-fp16-f0-v3")
        launch(CAMPAIGN / "specs/e419_float_fp32_f0.draft.json", "e419-float-fp32-f0-v1", completed)
        sys.path.insert(0, str(ROOT / "scripts"))
        from continuation_recipes import prediction_recipe
        for run in ("c20261004_e419_float_fp16_s3_f0_r2", "c20261004_e419_float_fp32_s3_f0"):
            draft = prediction_recipe(run, 0)
            export = run + "_common_fp32"
            draft["name"] = "predict_" + export
            draft["argv"] += ["--inference-precision", "fp32", "--output-name", export]
            draft["artifacts"] = [f"cache/pred_{export}_valid.{ext}" for ext in ("npy", "json")]
            path = CAMPAIGN / f"specs/predict_{export}.draft.json"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(draft, indent=2) + "\n")
            launch(path, "predict-" + export.replace("_", "-") + "-v1", completed)
        for part in ("a", "b"):
            launch(CAMPAIGN / f"specs/e422_gain_cache_f0_{part}_gpu_v2.draft.json",
                   f"e422-gain-cache-f0-{part}-gpu-v2", completed)
        OUTPUT.write_text(json.dumps({"status": "completed", "completed": completed}, indent=2) + "\n")
    except Exception as error:
        OUTPUT.write_text(json.dumps({"status": "stopped", "completed": completed,
                                     "reason": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
