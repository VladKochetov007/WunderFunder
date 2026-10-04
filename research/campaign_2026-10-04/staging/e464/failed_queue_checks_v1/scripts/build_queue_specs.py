"""Emit exact-schema path-only runtime drafts from reviewed E464 designs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
STAGING = ROOT / "research/campaign_2026-10-04/staging/e464"
FIELDS = {
    "version", "name", "kind", "argv", "code", "inputs", "artifacts", "limits", "environment"
}


def _paths(values: object, label: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    result = []
    for item in values:
        if isinstance(item, str):
            path = item
        elif isinstance(item, dict) and isinstance(item.get("path"), str):
            path = item["path"]
        else:
            raise ValueError(f"{label} entries must be paths or path descriptors")
        pure = Path(path)
        if not path or pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path:
            raise ValueError(f"{label} path is not normalized and root-relative: {path!r}")
        result.append(path)
    return sorted(set(result))


def canonicalize_design_spec(design: dict[str, Any]) -> dict[str, Any]:
    """Drop review-only metadata, retaining runner schema and path-only manifests."""
    result = {
        "version": design["version"],
        "name": design["name"],
        "kind": design["kind"],
        "argv": list(design["argv"]),
        "code": _paths(design["code"], "code"),
        "inputs": _paths(design["inputs"], "inputs"),
        "artifacts": list(design["artifacts"]),
        "limits": dict(design["limits"]),
        "environment": dict(design["environment"]),
    }
    if set(result) != FIELDS:
        raise ValueError("canonical runtime spec schema changed")
    if set(result["artifacts"]) & (set(result["code"]) | set(result["inputs"])):
        raise ValueError("runtime spec output collides with a source or input")
    return result


def _write(source: Path, destination: Path) -> None:
    design = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise ValueError(f"design must be a JSON object: {source}")
    canonical = canonicalize_design_spec(design)
    encoded = json.dumps(canonical, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace changed runtime draft: {destination}")
        return
    destination.write_text(encoded, encoding="utf-8")


def main() -> None:
    """Create eight export and one score draft with current code/future input paths."""
    execution = STAGING / "execution_specs"
    for source in sorted((STAGING / "specs").glob("e464_predict_*_fp32_v1.draft.json")):
        destination = execution / source.name
        _write(source, destination)
    _write(
        STAGING / "specs/e464_score_all_parent_head_fp32_v1.draft.json",
        execution / "e464_score_all_parent_head_fp32_v1.draft.json",
    )
    runtime_drafts = sorted(execution.glob("e464_predict_*_fp32_v1.draft.json"))
    if len(runtime_drafts) != 8:
        raise RuntimeError(f"expected exactly eight runtime export drafts, found {len(runtime_drafts)}")
    print(json.dumps({"status": "path_only_runtime_drafts_ready", "exports": len(runtime_drafts),
                      "score": "e464_score_all_parent_head_fp32_v1.draft.json"}, sort_keys=True))


if __name__ == "__main__":
    main()
