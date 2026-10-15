"""Collect the completed V8 bridge with bounded streaming and private URLs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests

KERNEL = "vladkochetov07/e463-tpu-realdata-reload-v8-20261013"
PREFIX = "cloud_preflight_tpu_v8"
MAX_BYTES = 128_000_000
HELPER_SHA = "a6e258be44c92d52642046f3fb1ffb45787a2e2e3b66a8f71bcb8f37ad7b1899"


def collect(root: Path) -> dict[str, object]:
    """Download one exact terminal output set without consuming unbounded bodies."""
    packet = root / "research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8"
    receipt = packet / "TERMINAL_COLLECTION_R1.json"
    output = root / "artifacts/kaggle_tpu_v8_terminal_20261015_r1"
    if receipt.exists() or output.exists():
        raise FileExistsError("terminal collection already exists; inspect before any retry")
    if shutil.disk_usage(root).free < 4_000_000_000:
        raise OSError("four GB physical disk headroom required")
    helper_path = root / (
        "research/campaign_2026-10-14/cpu_student_frontier_r1/"
        "kaggle_profile_r4/collect_terminal_r4.py"
    )
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError("output-listing helper differs from its frozen hash")
    spec = importlib.util.spec_from_file_location("tpu_collection_support", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load output-listing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.KERNEL_ID = KERNEL
    helper.OWNER, helper.KERNEL_SLUG = KERNEL.split("/", 1)
    api = helper.KaggleApi()
    api.authenticate()
    status, failure = helper.terminal_status(api)
    files, log = helper.output_listing(api)
    log_bytes = log.encode()
    total = len(log_bytes) + 4_000_000
    selected = []
    seen = set()
    for item in files:
        name = str(item.file_name)
        relative = PurePosixPath(name)
        if "__pycache__" in relative.parts:
            continue
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or "\x00" in name
            or not relative.parts
            or relative.parts[0] != PREFIX
            or name in seen
        ):
            raise ValueError("provider output contains an unexpected or repeated path")
        seen.add(name)
        with requests.get(str(item.url), stream=True, timeout=(15, 90)) as response:
            value = response.headers.get("Content-Length", "")
            if response.status_code != 200 or not value.isdecimal():
                raise RuntimeError(f"output header probe failed: HTTP{response.status_code}")
            size = int(value)
        total += size
        if total > MAX_BYTES:
            raise ValueError("declared collection exceeds128MB before download")
        selected.append((item, relative, size))
    if not selected:
        raise ValueError("no bridge output files")
    output.mkdir(parents=True)
    (output / "provider.log").write_bytes(log_bytes)
    rows = [{"path": "provider.log", "bytes": len(log_bytes),
             "sha256": hashlib.sha256(log_bytes).hexdigest()}]
    for item, relative, expected in selected:
        destination = output.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        digest = hashlib.sha256()
        with requests.get(str(item.url), stream=True, timeout=(15, 90)) as response:
            if response.status_code != 200:
                raise RuntimeError(f"output download failed: HTTP{response.status_code}")
            with destination.open("xb") as stream:
                for block in response.iter_content(1 << 20):
                    count += len(block)
                    if count > expected:
                        raise ValueError("download exceeds declared length")
                    stream.write(block)
                    digest.update(block)
        if count != expected:
            raise ValueError("download length differs from declared length")
        rows.append({"path": relative.as_posix(), "bytes": count,
                     "sha256": digest.hexdigest()})
    result = {"utc": datetime.now(UTC).isoformat(), "kernel": KERNEL,
              "provider_status": status, "provider_failure": failure,
              "output": str(output.relative_to(root)), "maximum_bytes": MAX_BYTES,
              "files": rows, "downloaded_bytes": sum(row["bytes"] for row in rows),
              "helper_sha256": HELPER_SHA,
              "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "terminal collection; actual update/reload/causality/export checks pending"}
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    result = collect(parser.parse_args().root.resolve(strict=True))
    print(json.dumps({key: result[key] for key in ("provider_status", "downloaded_bytes")}))
