"""Freeze eight metadata-only cases before checkpoint/input/output access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PANEL = ROOT / "research/campaign_2026-10-03/continuation/round3/matched_usage"


def key(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def main() -> None:
    if (HERE / "selection.json").exists():
        raise RuntimeError("selection already frozen")
    panel = json.loads((PANEL / "samples.json").read_text())
    ranked = sorted(
        enumerate(panel["samples"]),
        key=lambda pair: key(f"state-forgetting-v1:{pair[1]['component']}:{pair[1]['seq_ix']}"),
    )
    cases = []
    for index, sample in ranked[:8]:
        query = min(
            sample["queries"],
            key=lambda t: key(
                f"state-forgetting-query-v1:{sample['component']}:{sample['seq_ix']}:{t}"
            ),
        )
        cases.append(
            {
                "panel_index": index,
                "component": sample["component"],
                "seq_ix": sample["seq_ix"],
                "row_group": sample["row_group"],
                "query": query,
                "start": query - 288,
                "end": query,
                "direction_seed": 614407 + index,
                "exposure": "inherited frozen mask-selected queries on previously exposed validation components; no new mask/target/value selection",
            }
        )
    assert len({case["component"] for case in cases}) == 8
    result = {
        "cases": cases,
        "prereg_sha256": hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest(),
        "panel_samples_sha256": hashlib.sha256((PANEL / "samples.json").read_bytes()).hexdigest(),
        "checkpoint": panel["control"]["checkpoint"],
        "checkpoint_sha256": panel["control"]["checkpoint_sha256"],
        "source_sha256": panel["control"]["source_sha256"],
        "access": "only samples metadata and PREREG before freezing; no array values/model/targets/masks",
    }
    (HERE / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
