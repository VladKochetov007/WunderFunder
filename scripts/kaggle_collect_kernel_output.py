"""Collect one Kaggle kernel's outputs once it reaches a terminal status.

Designed for an unattended systemd timer: each invocation polls the kernel
status once.  A non-terminal status records a poll row and exits 0.  A terminal
status downloads all outputs into ``<dest>/outputs``, writes a SHA-256 manifest
and a ``COLLECTED.json`` marker; later invocations see the marker and exit 0
without contacting Kaggle.  Disk and download-size limits are checked before
and after the download; a violation leaves no marker so the operator can
inspect it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

TERMINAL = ("COMPLETE", "ERROR", "CANCEL")


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cli: str, *args: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cli, *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def append(log: Path, row: dict) -> None:
    with log.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    parser.add_argument("--max-bytes", type=int, default=12_000_000_000)
    parser.add_argument("--min-free-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "COLLECTED.json"
    log = dest / "poll_log.jsonl"
    if marker.exists():
        return 0

    status = run(args.kaggle, "kernels", "status", args.kernel, timeout=120)
    text = (status.stdout + status.stderr).strip()
    row = {"observed_utc": now(), "kernel": args.kernel, "rc": status.returncode, "status": text[-400:]}
    terminal = status.returncode == 0 and any(f"KernelWorkerStatus.{s}" in text for s in TERMINAL)
    row["terminal"] = terminal
    append(log, row)
    if not terminal:
        return 0

    free = shutil.disk_usage(dest).free
    if free < args.min_free_bytes + args.max_bytes:
        append(log, {"observed_utc": now(), "error": "insufficient_free_space", "free_bytes": free})
        return 1

    staging = dest / "outputs.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    download = run(
        args.kaggle, "kernels", "output", args.kernel, "-p", str(staging), timeout=3000
    )
    (dest / "download_stdout.txt").write_text((download.stdout + download.stderr)[-20000:])
    if download.returncode != 0:
        append(log, {"observed_utc": now(), "error": "download_failed", "rc": download.returncode})
        return 1

    files = sorted(p for p in staging.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    if total > args.max_bytes:
        append(log, {"observed_utc": now(), "error": "download_exceeds_max_bytes", "bytes": total})
        return 1
    manifest = {
        str(p.relative_to(staging)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files
    }
    final = dest / "outputs"
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    marker.write_text(
        json.dumps(
            {
                "kernel": args.kernel,
                "terminal_status": text[-400:],
                "collected_utc": now(),
                "file_count": len(files),
                "total_bytes": total,
                "manifest_sha256": sha256(dest / "MANIFEST.json"),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
