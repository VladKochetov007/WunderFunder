from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
LEGACY = OWN.parent / "capture_review_v2"
HISTORY = Path(
    "/home/vlad/.codex/sessions/2026/10/03/rollout-2026-10-03T14-54-35-01a1019d-b660-75c1-bcfe-56fc8a15f897.jsonl"
)


def main() -> None:
    """Recover only exact inverse edit bytes verified against the immutable ledger."""
    with (ROOT / "research/experiments.jsonl").open() as handle:
        entry = next(json.loads(line) for line in handle if '"E-385"' in line)
    changes = None
    with HISTORY.open() as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("ordinal") == 1006:
                changes = item["payload"]["item"]["changes"]
                break
    if changes is None:
        raise RuntimeError("Own exact edit footprint not found")
    recovered = {}
    for record in entry["reports"]:
        path = ROOT / record["path"]
        text = path.read_text()
        diff = changes[str(path)]["unified_diff"]
        for hunk in reversed(re.split(r"(?m)^@@[^\n]*\n", diff)[1:]):
            old, new = [], []
            for line in hunk.splitlines(keepends=True):
                if line.startswith((" ", "-")):
                    old.append(line[1:])
                if line.startswith((" ", "+")):
                    new.append(line[1:])
            before, after = "".join(old), "".join(new)
            if text.count(after) != 1:
                raise RuntimeError(f"Inverse hunk ambiguous: {path.name}")
            text = text.replace(after, before, 1)
        content = text.encode()
        actual = hashlib.sha256(content).hexdigest()
        if actual != record["sha256"]:
            raise RuntimeError(f"Recovered bytes do not match E385: {path.name} {actual}")
        recovered[path.name] = (content, actual)
    version = LEGACY / "versions/E385"
    version.mkdir(parents=True, exist_ok=False)
    for name, (content, _) in recovered.items():
        (version / name).write_bytes(content)
    proof = {
        "ledger_id": entry["id"],
        "own_history_ordinal": 1006,
        "own_history_timestamp": "2026-10-03T17:32:15.401Z",
        "method": "exact inverse applied unified diffs; independently matched immutable ledger SHA256 before writing",
        "recovered_sha256": {name: value[1] for name, value in recovered.items()},
        "final_reports_unchanged": True,
        "ledger_unchanged": True,
        "historical_STATE_recovered": False,
    }
    (version / "RECOVERY.json").write_text(json.dumps(proof, indent=2) + "\n")
    (OWN / "legacy_recovery.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
