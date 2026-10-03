"""Compact prior contrasts and observed optimization-clock metadata."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    prior = {}
    for report, data in result["prior_comparisons"].items():
        prior[report] = {
            "pooled": {
                name: {k: value[k] for k in ("mean_diff", "ci95")}
                for name, value in data["pooled"].items()
            },
            "folds": {
                fold: {name: value["mean_diff"] for name, value in comparisons.items()}
                for fold, comparisons in data["folds"].items()
            },
            "class3": {name: value["wp"] for name, value in data["pooled_class3"].items()},
        }
    members = {}
    for name, data in result["metadata_members"].items():
        members[name] = {
            "args": data["args"],
            "cumulative_updates": data["cumulative_updates"],
            "first_epoch_diagnostic": data["update_diagnostics"][0],
            "teacher_metadata_entries": len(data["teacher_metadata"]),
        }
    output = {"prior": prior, "members": members}
    (HERE / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
