from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

import requests

PACKET = Path("research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5")
HELPER = Path(
    "research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/collect_terminal_r4.py"
)
HELPER_SHA256 = "a6e258be44c92d52642046f3fb1ffb45787a2e2e3b66a8f71bcb8f37ad7b1899"
KERNEL = "vladkochetov07/wnn-f3-teacher-s3-20261014-r5-runner"
MAX_BYTES = 1_000_000_000
MIN_FREE_BYTES = 4_000_000_000
MANIFEST_RESERVE_BYTES = 4_000_000


def load_helper(root: Path) -> ModuleType:
    """Reuse the hash-pinned streaming collector without editing its source."""
    path = root / HELPER
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA256:
        raise ValueError("shared collector source differs from the reviewed pin")
    spec = importlib.util.spec_from_file_location("f3_r5_collection_support", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the pinned collection support")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.KERNEL_ID = KERNEL
    module.OWNER, module.KERNEL_SLUG = KERNEL.split("/", 1)
    return module


def declared_length(url: str) -> int:
    """Read GET headers without consuming the body or logging its signed URL."""
    if urlparse(url).scheme != "https":
        raise ValueError("provider returned a non-HTTPS output URL")
    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=(15, 90)) as response:
            if response.status_code != 200:
                raise RuntimeError(f"output inventory GET headers returned HTTP {response.status_code}")
            value = response.headers.get("Content-Length")
    except requests.RequestException:
        raise RuntimeError("output inventory GET headers failed; no payload downloaded") from None
    if value is None or not value.isdecimal():
        raise ValueError("output inventory requires an explicit nonnegative length")
    return int(value)


def inventory(files: list[object], log_bytes: int, helper: ModuleType) -> list[dict]:
    """Validate all selected paths and declared sizes before the first download."""
    rows: list[dict] = []
    seen: set[str] = set()
    total = log_bytes + MANIFEST_RESERVE_BYTES
    for item in files:
        name = str(getattr(item, "file_name", ""))
        if not name.startswith("results/"):
            continue
        helper.relative_result_path(name)
        if name in seen:
            raise ValueError("duplicate result filename in provider inventory")
        seen.add(name)
        size = declared_length(str(getattr(item, "url", "")))
        total += size
        if total > MAX_BYTES:
            raise ValueError("declared results exceed the one-GB collection bound")
        rows.append({"path": name, "size_bytes": size})
    if not rows:
        raise ValueError("provider has no selected result files")
    return rows


def collect(root: Path) -> dict:
    """Preserve one terminal fit attempt; scientific grading remains separate."""
    root = root.resolve(strict=True)
    packet = root / PACKET
    output = packet / "collected/attempt_001_http_get_r6"
    terminal_path = packet / "TERMINAL_R6.json"
    if output.exists() or terminal_path.exists():
        raise FileExistsError("collection already exists; no automatic retry or overwrite")
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(packet)], text=True
    ).strip()
    if filesystem in {"tmpfs", "ramfs"} or not filesystem:
        raise ValueError("collection requires a physical filesystem")
    if shutil.disk_usage(packet).free < MIN_FREE_BYTES:
        raise OSError("collection requires at least four GB free physical disk")
    helper = load_helper(root)
    admission = packet / "ROOT_ADMISSION_R5.json"
    push = packet / "KERNEL_PUSH_R5.json"
    binding = json.loads((packet / "ENTRYPOINT_BINDING_R5.json").read_text())
    prerequisite_hashes = {p.name: helper.sha256_file(p) for p in (admission, push)}
    api = helper.KaggleApi()
    api.authenticate()
    status, failure = helper.terminal_status(api)
    quota_before = helper.quota_snapshot(api)
    files, log = helper.output_listing(api)
    encoded_log = log.encode("utf-8")
    rows = inventory(files, len(encoded_log), helper)
    selected = {str(getattr(item, "file_name", "")): item for item in files}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    helper.write_json(
        output / "remote_inventory.json", {"files": rows, "log_bytes": len(encoded_log)}
    )
    log_path = output / "kernel.log"
    with log_path.open("xb") as stream:
        stream.write(encoded_log)
    manifest = [
        {
            "path": "kernel.log",
            "size_bytes": len(encoded_log),
            "sha256": hashlib.sha256(encoded_log).hexdigest(),
        }
    ]
    downloaded = len(encoded_log)
    for row in rows:
        record = helper.download_result_file(selected[row["path"]], output, row["size_bytes"])
        if record["size_bytes"] != row["size_bytes"]:
            raise ValueError("download length differs from the pre-download inventory")
        downloaded += record["size_bytes"]
        manifest.append(record)
    collection = {
        "schema": "wnn.f3-teacher-r5.terminal-collection.v1",
        "recorded_utc": helper.utc_now(),
        "kernel": KERNEL,
        "provider_status": status,
        "provider_failure_message": failure,
        "source_binding": binding,
        "prerequisite_hashes": prerequisite_hashes,
        "helper_sha256": HELPER_SHA256,
        "maximum_collection_bytes": MAX_BYTES,
        "downloaded_payload_bytes": downloaded,
        "files": manifest,
        "quota_before": quota_before,
        "quota_after": helper.quota_snapshot(api),
        "fit_complete": "pending checkpoint/provenance/counter verification",
        "prediction_exports": "all three kinds pending; checkpoint is not a teacher cache",
        "automatic_retry": False,
        "inventory_method": "GET stream headers only; body not consumed until bounded download",
        "collector_predecessor_sha256": "14952f4abf06bc9963df540119131eaee76828d3aa7f53d35abd72d4a98cc63f",
        "collector_source_sha256": helper.sha256_file(Path(__file__)),
    }
    helper.write_json(output / "collection_manifest.json", collection)
    actual_size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if actual_size > MAX_BYTES:
        raise ValueError("collection metadata exceeded the reserved four MB margin")
    collection["bytes_including_metadata"] = actual_size
    helper.write_json(terminal_path, collection)
    return collection


def main() -> None:
    """Collect only the fixed private kernel's single terminal attempt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.repo_root)
    print(
        json.dumps(
            {key: result[key] for key in ("kernel", "provider_status", "bytes_including_metadata")}
        )
    )


if __name__ == "__main__":
    main()
