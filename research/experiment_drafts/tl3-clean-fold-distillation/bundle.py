"""Separate content-addressed scientific evidence and weights; never copy supplied data."""

from __future__ import annotations

import argparse
import tarfile
import uuid
from pathlib import Path

from commands import ROLES, run_name
from common import digest, load_assets, output_root, preflight, read_json, write_new


def archive(path: Path, root: Path, members: list[Path]) -> dict[str, object]:
    """Stream a create-only archive of explicit regular files and return its byte identity."""
    if path.exists() or path.with_suffix(".partial").exists():
        raise FileExistsError(path)
    partial = path.with_suffix(".partial")
    with tarfile.open(partial, "w:gz") as destination:
        for member in sorted(set(members)):
            if (
                member.is_symlink()
                or not member.is_file()
                or not member.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError(f"unsafe archive member: {member}")
            destination.add(member, arcname=str(member.relative_to(root)), recursive=False)
    partial.rename(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def main() -> None:
    """Bundle every completed cell, including negative results, only after a full valid grade."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    assets = load_assets(args.assets)
    binding = preflight(assets)
    root = Path(assets["project_root"])
    output = output_root(assets)
    grade = output / "grade.json"
    graded = read_json(grade)
    expected = {run_name(cell, role) for cell in ("F0", "F2") for role in ROLES}
    inputs = graded.get("inputs", {})
    if (
        graded.get("status") != "complete_two_cell_screen"
        or set(graded.get("folds", {})) != {"F0", "F2"}
        or set(inputs.get("runs", {})) != expected
        or any(inputs.get(key) != value for key, value in binding.items())
    ):
        raise ValueError("bundle requires a complete eight-fit grade bound to current inputs")
    for record in inputs["runs"].values():
        for kind in ("checkpoint", "provenance", "prediction", "sidecar", "audit", "terminal"):
            if digest(Path(record[f"{kind}_path"])) != record[f"{kind}_sha256"]:
                raise ValueError(f"graded {kind} artifact changed before bundling")
    evidence = [grade]
    for directory in ("start", "terminal", "audit", "targets", "stages", "logs", "attempts"):
        evidence.extend((output / directory).glob("*"))
    weights = []
    for cell in ("F0", "F2"):
        for role in ROLES:
            name = run_name(cell, role)
            run = root / "artifacts/runs" / name
            weights.extend(run.glob("*.pt"))
            evidence.extend(run.glob("*.json"))
            prediction = root / "cache" / f"pred_{name}_eval_valid.npy"
            evidence += [prediction, prediction.with_suffix(".json")]
    inputs_manifest = {
        **binding,
        "grade_sha256": digest(grade),
        "files": [
            {
                "path": str(path.relative_to(root)),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(set(evidence + weights))
        ],
    }
    inputs_path = output / "ARCHIVE_INPUTS.json"
    if inputs_path.exists():
        if read_json(inputs_path) != inputs_manifest:
            raise ValueError("archive inputs changed since the previous packaging attempt")
    else:
        write_new(inputs_path, inputs_manifest)
    evidence.append(inputs_path)
    final = output / "bundles"
    if final.exists():
        previous = read_json(final / "MANIFEST.json")
        if previous.get("archive_inputs_sha256") != digest(inputs_path):
            raise ValueError("existing archive manifest belongs to different inputs")
        for name in ("evidence", "weights"):
            if digest(final / f"{name}.tar.gz") != previous[name]["sha256"]:
                raise ValueError("published archive bytes changed")
        return
    # Failed generations remain available for diagnosis; a retry uses new staging paths.
    generation = output / f".bundle_generation_{uuid.uuid4().hex}"
    generation.mkdir()
    result = {
        "evidence": archive(generation / "evidence.tar.gz", root, evidence),
        "weights": archive(generation / "weights.tar.gz", root, weights),
    }
    for entry in inputs_manifest["files"]:
        if digest(root / entry["path"]) != entry["sha256"]:
            raise ValueError("source artifact changed while archiving")
    for name, record in result.items():
        record["path"] = str(final / f"{name}.tar.gz")
    write_new(
        generation / "MANIFEST.json",
        {
            **binding,
            **result,
            "archive_inputs_sha256": digest(inputs_path),
            "note": "No supplied datasets or submission ZIP. Retain sealed assets and published source commit.",
        },
    )
    generation.rename(final)


if __name__ == "__main__":
    main()
