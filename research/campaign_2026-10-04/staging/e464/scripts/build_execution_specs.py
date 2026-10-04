"""Emit schema-only runner drafts from the richer E464 review designs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STAGING = ROOT / "research/campaign_2026-10-04/staging/e464"
REQUIRED_FIELDS = {
    "version",
    "name",
    "kind",
    "argv",
    "code",
    "inputs",
    "artifacts",
    "limits",
    "environment",
}


def main() -> None:
    """Create exact runner-spec-shaped drafts without dropping design records."""
    output_dir = STAGING / "execution_specs"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted((STAGING / "specs").glob("c20261004_e464_*_head_s3_final.draft.json"))
    for source in sources:
        design = json.loads(source.read_text(encoding="utf-8"))
        code = design.get("code")
        if not isinstance(code, list) or any(not isinstance(item, dict) for item in code):
            raise ValueError(f"design draft lacks code content descriptors: {source}")
        execution = {
            "version": design["version"],
            "name": design["name"],
            "kind": design["kind"],
            "argv": design["argv"],
            "code": [item["path"] for item in code],
            "inputs": design["inputs"],
            "artifacts": design["artifacts"],
            "limits": design["limits"],
            "environment": design["environment"],
        }
        if set(execution) != REQUIRED_FIELDS:
            raise ValueError(f"runner draft has noncanonical fields: {source}")
        destination = output_dir / source.name
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(
            json.dumps(execution, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {len(sources)} E464 execution drafts")


if __name__ == "__main__":
    main()
