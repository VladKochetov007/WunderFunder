"""Run the sixteen prespecified feature screens through campaign admission."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    root = Path.cwd()
    campaign = root / "research/campaign_2026-10-04"
    state = campaign / "runner"
    output = campaign / "checks/e418_sequence_v1.json"
    if output.exists():
        raise FileExistsError(output)
    for name in ("e418-histories-f0-a-v1", "e418-histories-f0-b-v1"):
        if json.loads((state / f"attempts/{name}.json").read_text())["status"] != "finished":
            raise ValueError("Both frozen history caches must be complete")
    drafts = sorted((campaign / "specs").glob("e418_exact_f0_d*_cpu.draft.json"))
    if len(drafts) != 16:
        raise ValueError("Require the complete fixed eight-family/two-direction inventory")
    completed = []
    try:
        for draft in drafts:
            if (state / "PAUSE.json").exists():
                raise RuntimeError("Campaign resource pause; no automatic acknowledgement")
            name = json.loads(draft.read_text())["name"]
            attempt = "e418-" + name.replace("_", "-") + "-v1"
            sealed = draft.with_name(draft.name.replace(".draft.json", ".sealed.json"))
            command = [sys.executable, "scripts/run_campaign.py"]
            subprocess.run([*command, "seal", str(draft.relative_to(root)),
                            str(sealed.relative_to(root))], check=True)
            subprocess.run([*command, "launch", str(sealed.relative_to(root)),
                            "--attempt", attempt], check=True)
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                record = json.loads((state / f"attempts/{attempt}.json").read_text())
                if record["status"] == "finished":
                    completed.append({"attempt": attempt, "specification": str(sealed.relative_to(root)),
                                      "elapsed_seconds": record["elapsed_seconds"]})
                    break
                if record["status"] not in ("reserved", "running"):
                    raise RuntimeError(f"Contained job stopped: {attempt}; inspect its preserved record")
                time.sleep(2)
            else:
                raise TimeoutError("Sequence monitor deadline; no automatic cancellation or retry")
            output.write_text(json.dumps({"status": "running", "completed": completed}, indent=2) + "\n")
        output.write_text(json.dumps({"status": "completed", "completed": completed}, indent=2) + "\n")
    except Exception as error:
        output.write_text(json.dumps({"status": "stopped", "completed": completed,
                                     "reason": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
