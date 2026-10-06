"""Inventory prepared research jobs without admitting or launching any process."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from wnn.campaign import PREFLIGHT_AND_TEARDOWN_SECONDS
from wnn.campaign_spec import digest, validate_spec


def inspect_draft(root: Path, path: Path) -> dict[str, object]:
    """Validate a draft's schema without reading large scientific input contents.

    Placeholder digests are used only for schema validation. The campaign runner
    must independently seal real bytes before admission; this inventory is never
    an executable sealed specification.
    """
    draft = json.loads(path.read_text())
    candidate = dict(draft)
    for section in ("code", "inputs"):
        candidate[section] = [
            {"path": entry, "size": 0, "sha256": "0" * 64} for entry in draft[section]
        ]
    candidate["code"] = sorted(candidate["code"], key=lambda entry: entry["path"])
    candidate["code_sha256"] = digest(candidate["code"])
    validate_spec(candidate)
    missing_code = [entry for entry in draft["code"] if not (root / entry).is_file()]
    missing_inputs = [entry for entry in draft["inputs"] if not (root / entry).is_file()]
    existing_outputs = [entry for entry in draft["artifacts"] if (root / entry).exists()]
    limits = draft["limits"]
    is_local_shape = (
        draft["kind"] != "tpu"
        and limits["cpus"] <= 10
        and limits["ram_bytes"] <= 40_000_000_000 - 4_294_967_296
        and limits["gpu_stop_bytes"] <= 5_000_000_000
    )
    return {
        "spec": str(path.relative_to(root)),
        "draft_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "name": draft["name"],
        "kind": draft["kind"],
        "limits": limits,
        "declared_outputs": draft["artifacts"],
        "missing_code": missing_code,
        "missing_inputs": missing_inputs,
        "existing_outputs": existing_outputs,
        "fits_local_shape_limits": is_local_shape,
        "reservation_ceiling_device_seconds": (
            limits["timeout_seconds"] + PREFLIGHT_AND_TEARDOWN_SECONDS
            if draft["kind"] == "gpu"
            else 0
        ),
        "requires_live_admission": True,
        "requires_independent_review": True,
    }


def dependency_order(edges: dict[str, set[str]]) -> tuple[list[str], list[str]]:
    """Order all acyclic jobs; return unresolved cycles separately."""
    pending = {name: set(parents) for name, parents in edges.items()}
    ordered: list[str] = []
    while pending:
        ready = sorted(name for name, parents in pending.items() if not parents)
        if not ready:
            break
        ordered.extend(ready)
        for name in ready:
            del pending[name]
        for parents in pending.values():
            parents.difference_update(ready)
    return ordered, sorted(pending)


def inventory(root: Path, directories: list[Path]) -> dict[str, object]:
    """Expose dependencies, ambiguous producers and blockers across workstreams."""
    paths = sorted({path for directory in directories for path in directory.rglob("*.draft.json")})
    jobs: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            jobs.append(inspect_draft(root, path))
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append({"spec": str(path.relative_to(root)), "error": str(error)})
    producers: dict[str, list[str]] = {}
    for job in jobs:
        for output in job["declared_outputs"]:
            producers.setdefault(output, []).append(job["spec"])
    ambiguous = {output: values for output, values in producers.items() if len(values) > 1}
    edges: dict[str, set[str]] = {}
    for job in jobs:
        dependencies = set()
        external_missing = []
        for entry in job["missing_inputs"]:
            matches = producers.get(entry, [])
            if len(matches) == 1:
                dependencies.add(matches[0])
            else:
                external_missing.append(entry)
        job["dependencies"] = sorted(dependencies)
        job["missing_external_or_ambiguous_inputs"] = external_missing
        job["ambiguous_outputs"] = [
            output for output in job["declared_outputs"] if output in ambiguous
        ]
        job["preparation_status"] = (
            "blocked_missing_code"
            if job["missing_code"]
            else "blocked_ambiguous_output_producer"
            if job["ambiguous_outputs"]
            else "output_exists_choose_new_attempt"
            if job["existing_outputs"]
            else "blocked_external_dependency"
            if external_missing
            else "waiting_for_predecessor"
            if dependencies
            else "inputs_available_review_and_live_admission_required"
        )
        edges[job["spec"]] = dependencies
    ordered, cycles = dependency_order(edges)
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "inventory_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": "inventory only; not a sealed spec, approval, admission or scientific result",
        "jobs": jobs,
        "schema_errors": errors,
        "ambiguous_output_producers": ambiguous,
        "dependency_order": ordered,
        "cyclic_or_cycle_dependent_specs": cycles,
    }


def main() -> None:
    """Write a reviewable JSON queue for explicitly supplied workstream directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directories = [path.resolve() for path in args.directories]
    if any(not path.is_dir() or not path.is_relative_to(root) for path in directories):
        parser.error("workstream directories must exist inside this repository")
    result = inventory(root, directories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    if (
        result["schema_errors"]
        or result["ambiguous_output_producers"]
        or result["cyclic_or_cycle_dependent_specs"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
