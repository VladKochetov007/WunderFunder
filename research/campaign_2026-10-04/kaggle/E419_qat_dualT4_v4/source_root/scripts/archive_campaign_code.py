"""Preserve the exact small source files named by a sealed experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def archive(root: Path, spec_path: Path, destination: Path) -> dict:
    """Recover matching committed bytes when current working sources have advanced."""
    spec = json.loads(spec_path.read_text())
    output = destination / spec["code_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    missing = []
    for item in spec["code"]:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source path escapes archive")
        target = output / path
        candidates = [target, root / path]
        data = None
        for candidate in candidates:
            if candidate.is_file():
                value = candidate.read_bytes()
                if hashlib.sha256(value).hexdigest() == item["sha256"]:
                    data = value
                    break
        if data is None:
            commits = subprocess.check_output(
                ["git", "log", "--all", "--format=%H", "--", path.as_posix()], cwd=root, text=True
            ).splitlines()
            for commit in commits:
                value = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
                if hashlib.sha256(value).hexdigest() == item["sha256"]:
                    data = value
                    break
        if data is None:
            missing.append(path.as_posix())
            continue
        if len(data) != item["size"]:
            raise ValueError("source length differs from sealed identity")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
        elif target.read_bytes() != data:
            raise ValueError("existing archive content changed")
    report = {"code_sha256": spec["code_sha256"], "files": spec["code"], "missing": missing}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report = archive(args.root, args.spec, args.destination)
    print(json.dumps({"code_sha256": report["code_sha256"], "missing": report["missing"]}))
    if report["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
