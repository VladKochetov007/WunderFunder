"""Save the bounded synthetic loss-pooling discriminator's exact output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "research/campaign_2026-10-04/E430_optimizer_cadence/check_pooling.py"
OUTPUT = ROOT / "research/campaign_2026-10-04/checks/e430_pooling_discriminator_v2.json"


def main() -> None:
    """Record the tiny numerical example without importing training or data."""
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = subprocess.run(
        [sys.executable, str(SOURCE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    evidence = json.loads(result.stdout)
    evidence["scope"] = "Synthetic arithmetic discriminator; not a trainer-gradient test"
    with OUTPUT.open("x") as target:
        target.write(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
