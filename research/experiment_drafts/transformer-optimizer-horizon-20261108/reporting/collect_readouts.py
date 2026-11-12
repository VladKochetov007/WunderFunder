from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "5d73bc2b10ae20e607e0d4b4533da85e76e27bca"
MANIFEST_SHA256 = "a1e16d24ab53032643807edb5471b8bc65b06bdd96da516944bcf36736339937"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
PLAN_SHA256 = "b40ba65c7c2495f1a08a42331800ce8090bf9bbe63e7cb5514af401e784cf1c0"
EXPECTED_CELLS = (
    "C0_muon256_a4_s203101",
    "C1_adamw256_a4_s203101",
    "C2_muon512_a2_s203101",
    "C0_muon256_a4_s203102",
    "C1_adamw256_a4_s203102",
    "C2_muon512_a2_s203102",
)
MAX_FILE_BYTES = 4 * 1_048_576
MAX_TOTAL_BYTES = 16 * 1_048_576
MAX_PLAN_BYTES = 2 * 1_048_576


def _strict_json(payload: bytes) -> dict[str, Any]:
    """Parse only finite JSON objects from the small ROOT readout payloads."""
    value = json.loads(
        payload,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"invalid JSON {item}")),
    )
    if not isinstance(value, dict):
        raise ValueError("readout artifact must be a JSON object")
    return value


def _read_object(path: Path, root: Path, remaining: int) -> dict[str, Any]:
    """Read one no-follow regular file under runtime root within per-file/total limits."""
    if path.is_symlink():
        raise ValueError("readout artifact cannot be a symlink")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or not stat.S_ISREG(resolved.lstat().st_mode):
        raise ValueError("readout artifact escaped the runtime or is not regular")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > MAX_FILE_BYTES or before.st_size > remaining:
            raise ValueError("readout collector byte bound exceeded")
        payload = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        if (
            len(payload) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("readout artifact changed during bounded read")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "value": _strict_json(payload),
    }


def _read_plan(path: Path, root: Path) -> bytes:
    """Read the immutable native plan through a no-follow descriptor under2MiB."""
    if path.is_symlink():
        raise ValueError("receiver plan cannot be a symlink")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError("receiver plan escaped the declared runtime")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_PLAN_BYTES:
            raise ValueError("receiver plan is not a bounded regular file")
        payload = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        if (
            len(payload) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("receiver plan changed during read")
    return payload


def collect_readouts(runtime: Path, plan_path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    """Read only six frozen cells' terminal/score receipts and the ROOT comparison."""
    root = runtime.resolve(strict=True)
    if plan_path.is_symlink():
        raise ValueError("receiver plan cannot be a symlink")
    plan_real = plan_path.resolve(strict=True)
    if root not in plan_real.parents:
        raise ValueError("receiver plan is outside the declared runtime or is a symlink")
    plan_payload = _read_plan(plan_real, root)
    if hashlib.sha256(plan_payload).hexdigest() != expected_plan_sha256:
        raise ValueError("receiver plan byte digest differs")
    plan = _strict_json(plan_payload)
    cells = plan.get("cells", [])
    if (
        expected_plan_sha256 != PLAN_SHA256
        or plan.get("schema") != "wnn.transformer-optimizer-horizon-panel.v1"
        or plan.get("source_commit") != SOURCE_COMMIT
        or plan.get("source_manifest_sha256") != MANIFEST_SHA256
        or plan.get("split_sha256") != SPLIT_SHA256
        or Path(plan.get("runtime", "")).resolve() != root
        or [cell.get("cell") for cell in cells] != list(EXPECTED_CELLS)
    ):
        raise ValueError("collector only accepts the exact ROOT-frozen six-cell panel")
    files: dict[str, dict[str, Any]] = {}
    total = 0
    paths = [
        (f"runs/{name}/fit/{filename}", root / "runs" / name / "fit" / filename)
        for name in EXPECTED_CELLS
        for filename in ("terminal_status.json", "score.json")
    ]
    paths.append(("comparison.json", root / "comparison.json"))
    for relative, path in paths:
        try:
            artifact = _read_object(path, root, MAX_TOTAL_BYTES - total)
        except FileNotFoundError:
            files[relative] = {"status": "missing"}
            continue
        except (OSError, ValueError) as error:
            files[relative] = {"status": "invalid", "reason": str(error)[:160]}
            continue
        total += artifact["bytes"]
        files[relative] = {"status": "read", **artifact}
    return {
        "schema": "wnn.transformer-optimizer-horizon-readouts.v1",
        "plan_sha256": expected_plan_sha256,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": MANIFEST_SHA256,
        "split_sha256": SPLIT_SHA256,
        "collected_at_unix": time.time(),
        "total_bytes": total,
        "files": files,
    }


def main() -> None:
    """Emit a single bounded JSON result for ROOT's existing read-only file RPC."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    payload = collect_readouts(args.runtime, args.plan, args.plan_sha256)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
