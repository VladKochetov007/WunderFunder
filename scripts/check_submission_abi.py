from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


def inspect_library(path: Path) -> dict[str, object]:
    """Reject Python-library linkage and symbols newer than Debian bookworm.

    This static gate does not establish importability or callback correctness;
    those require the actual ZIP in the offline Python 3.11 scorer environment.
    """
    dynamic = subprocess.run(
        ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
    ).stdout
    versions = subprocess.run(
        ["readelf", "--version-info", str(path)], capture_output=True, text=True, check=True
    ).stdout
    needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)
    glibc = sorted(
        {tuple(map(int, v.split("."))) for v in re.findall(r"GLIBC_(\d+\.\d+)", versions)}
    )
    errors = [
        f"version-specific Python dependency: {name}"
        for name in needed
        if name.startswith("libpython")
    ]
    if glibc and glibc[-1] > (2, 36):
        errors.append(f"requires GLIBC_{'.'.join(map(str, glibc[-1]))}; scorer provides 2.36")
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "needed": needed,
        "maximum_glibc": ".".join(map(str, glibc[-1])) if glibc else None,
        "errors": errors,
        "passed": not errors,
    }


def inspect_artifact(path: Path) -> dict[str, object]:
    """Inspect native libraries in a ZIP or build directory without importing them."""
    libraries: dict[str, dict[str, object]] = {}
    if path.is_dir():
        for library in sorted(path.rglob("*.so")):
            libraries[str(library.relative_to(path))] = inspect_library(library)
        artifact_hash = None
    else:
        artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as temporary:
            if archive.testzip() is not None:
                raise ValueError(f"invalid ZIP: {path}")
            for index, member in enumerate(archive.infolist()):
                if member.filename.endswith(".so"):
                    library = Path(temporary) / f"native_{index}.so"
                    library.write_bytes(archive.read(member))
                    libraries[member.filename] = inspect_library(library)
    if not libraries:
        raise ValueError(f"no native libraries found: {path}")
    return {
        "artifact": str(path),
        "artifact_sha256": artifact_hash,
        "libraries": libraries,
        "passed": all(result["passed"] for result in libraries.values()),
        "scope": "static ELF dependencies only; offline Python 3.11 callback test also required",
    }


def main() -> None:
    """Emit a static compatibility receipt and exit nonzero for a rejected artifact."""
    parser = argparse.ArgumentParser(description="Check native submission ELF dependencies")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_artifact(args.artifact)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
