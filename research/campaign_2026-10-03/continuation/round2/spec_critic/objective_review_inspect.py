from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OWN = Path(__file__).resolve().parent
ACTOR = OWN.parent / "objective"


def main() -> None:
    source = ACTOR / "chunk_summaries.npz"
    with np.load(source) as summaries:
        schema = {
            name: {"shape": list(summaries[name].shape), "dtype": str(summaries[name].dtype)}
            for name in summaries.files
        }
    result = json.loads((ACTOR / "results.json").read_text())
    metadata = {
        "chunk_summaries_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "schema": schema,
        "results_top_keys": list(result),
        "results_nonlarge_metadata": {
            k: v for k, v in result.items() if k not in {"analyses", "sham", "synthetic", "actual"}
        },
    }
    (OWN / "objective_review_schema.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"schema": schema, "results_top_keys": list(result)}, indent=2))


if __name__ == "__main__":
    main()
