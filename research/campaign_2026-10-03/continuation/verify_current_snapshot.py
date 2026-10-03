"""Actually restore the current source and each numerical part into paused staging."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from restore_progress_artifacts import restore
from save_progress_artifacts import ROOT, STATE
from save_state import restore_dry_run


def main() -> None:
    """Verify fresh source and every companion part without starting restored code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot = ROOT / (STATE / "LATEST_SNAPSHOT").read_text().strip()
    checksums = {}
    for line in (snapshot / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        if name in checksums:
            raise ValueError("duplicate checksum entry")
        checksums[name] = expected
    source = restore_dry_run(
        snapshot / "research-session.tar", args.restore_dir, checksums["research-session.tar"]
    )
    parts = json.loads((snapshot / "PROGRESS_PARTS.json").read_text())
    results = []
    for part in parts:
        name = part["archive"]
        if checksums[name] != part["sha256"]:
            raise ValueError("part checksum metadata differs")
        result = restore(snapshot / name, checksums[name], args.restore_dir)
        if result["members"] != part["members"] or result["payload_bytes"] != part["payload_bytes"]:
            raise ValueError("restored part metadata differs")
        results.append({"archive": name, **result})
        (args.restore_dir / f"_recovery/{name}.verification.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    report = {
        "status": "PASS",
        "verified_at": datetime.now().astimezone().isoformat(),
        "source_restore": source,
        "progress_restore": results,
        "progress_members": sum(result["members"] for result in results),
        "progress_payload_bytes": sum(result["payload_bytes"] for result in results),
        "staging": str(args.restore_dir.relative_to(ROOT)),
        "paused": True,
        "services_started": 0,
        "standalone_training_restore": False,
        "context": "Separate explicit context extraction still required.",
    }
    (snapshot / "SOURCE_PROGRESS_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
