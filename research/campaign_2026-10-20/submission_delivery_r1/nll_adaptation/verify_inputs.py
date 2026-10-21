from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def digest(path: Path, offset: int = 0) -> str:
    """Hash a file or its declared suffix without materializing it."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Verify frozen source, local input identities and small model artifacts."""
    directory = Path(__file__).resolve().parent
    recipe_path = directory / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    source = recipe["source"]
    worktree = Path(source["worktree"])
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts", "pyproject.toml"],
        cwd=worktree,
        check=True,
    )
    for name, expected in source["files_sha256"].items():
        if digest(worktree / name) != expected:
            raise ValueError(f"source digest mismatch: {name}")
    records = []
    for item in recipe["preflight_actual_byte_checks"]:
        path = Path(item["path"])
        before = path.stat()
        identity = item.get("frozen_identity")
        record = {"path": str(path), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
        if identity:
            if (before.st_size, before.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
                raise ValueError(f"frozen source metadata mismatch: {path}")
            observed = digest(path, before.st_size - identity["footer_bytes"] - 8)
            if observed != identity["footer_sha256"]:
                raise ValueError(f"Parquet footer mismatch: {path}")
            record["footer_sha256"] = observed
        else:
            observed = digest(path)
            expected = item.get("sha256", "")
            if len(expected) == 64 and observed != expected:
                raise ValueError(f"artifact digest mismatch: {path}")
            if "bytes" in item and before.st_size != item["bytes"]:
                raise ValueError(f"artifact size mismatch: {path}")
            record["sha256"] = observed
            record["matched_prior_digest"] = len(expected) == 64
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"artifact changed during identity check: {path}")
        records.append(record)
    result = {
        "schema": "wnn.overnight.input-verification.v1",
        "recipe_sha256": digest(recipe_path),
        "source_commit": source["commit"],
        "source_module_digests_match": True,
        "inputs": records,
        "passed": True,
        "scope": "No tensor/data array loading, training or quality evaluation.",
    }
    with (directory / "input_verification.json").open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "input_count": len(records)}), flush=True)


if __name__ == "__main__":
    main()
