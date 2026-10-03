"""Append terminal indexed evidence after checking its frozen small-file bindings."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    index_path = HERE / "landed_evidence_index.json"
    index = json.loads(index_path.read_text())
    manifest = json.loads((HERE / "HASHES.json").read_text())
    for name, declared in manifest["outputs"].items():
        path = ROOT / name
        if path.stat().st_size != declared["bytes"] or sha(path) != declared["sha256"]:
            raise ValueError(f"Indexed artifact changed: {name}")
    for name, declared in index["consumed_metadata"].items():
        path = ROOT / name
        if path.stat().st_size != declared["bytes"] or sha(path) != declared["sha256"]:
            raise ValueError(f"Consumed metadata changed: {name}")
    ledger = ROOT / "research/experiments.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    if any(row.get("landed_index_sha256") == sha(index_path) for row in rows):
        raise ValueError("Already indexed; refusing duplicate records")
    first = max(int(row["id"].removeprefix("E-")) for row in rows if row.get("id", "").startswith("E-")) + 1
    additions = []
    for offset, entry in enumerate(index["entries"]):
        additions.append({
            "id": f"E-{first + offset}", "date": datetime.now().astimezone().isoformat(),
            "kind": entry["id"], "model": "gpt-6.1-sol", "reasoning_effort": "medium",
            "result": entry.get("reported_measurements", {}),
            "terminal_interpretation": entry["terminal_interpretation"],
            "missing_gates": entry["missing_gates"],
            "landed_index": str(index_path.relative_to(ROOT)),
            "landed_index_sha256": sha(index_path), "claim_limits": index["claim_limits"],
        })
    with ledger.open("a") as output:
        for row in additions:
            output.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({"records": [row["id"] for row in additions], "small_bindings_checked": len(index["consumed_metadata"])}))


if __name__ == "__main__":
    main()
