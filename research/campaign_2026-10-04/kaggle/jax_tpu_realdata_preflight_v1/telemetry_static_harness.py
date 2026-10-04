"""Extract the embedded worker milestone writer and verify real JSONL framing.

Standard-library-only: this does not import JAX, Torch, NumPy, or campaign data.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL = ROOT / "kernel_push/kernel.py"
source = KERNEL.read_text(encoding="utf-8")
tree = ast.parse(source)
assignment = next(
    node
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "WORKER_SOURCE" for target in node.targets)
)
worker_source = ast.literal_eval(assignment.value)
worker_tree = ast.parse(worker_source)
writer = next(node for node in worker_tree.body if isinstance(node, ast.FunctionDef) and node.name == "milestone")

with tempfile.TemporaryDirectory(prefix="wnn-tpu-milestone-check-") as temporary_directory:
    output_path = Path(temporary_directory) / "phase_milestones.jsonl"
    scope = {
        "Path": Path,
        "json": json,
        "os": os,
        "resource": resource,
        "time": time,
        "started": time.monotonic(),
        "MILESTONE_PATH": output_path,
    }
    exec(compile(ast.Module(body=[writer], type_ignores=[]), str(KERNEL), "exec"), scope)
    scope["milestone"]("first", check=1)
    scope["milestone"]("second", check=2)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, f"expected two JSONL records, found {len(lines)}"
    records = [json.loads(line) for line in lines]
    assert [record["phase"] for record in records] == ["first", "second"]
    assert [record["check"] for record in records] == [1, 2]

print(json.dumps({
    "status": "pass",
    "records_written": len(lines),
    "each_splitline_json_decodes": True,
    "kernel_sha256": hashlib.sha256(KERNEL.read_bytes()).hexdigest(),
}, sort_keys=True))

# Exercise the actual preflight memory writer concurrently with its periodic
# sampler using fake structured device stats. This tests framing only; it does
# not import JAX or allocate accelerator/data arrays.
import concurrent.futures
import threading

PREFLIGHT = ROOT / "source_bundle/scripts/realdata_preflight.py"
preflight_tree = ast.parse(PREFLIGHT.read_text(encoding="utf-8"))
run_node = next(node for node in preflight_tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
sample_node = next(node for node in run_node.body if isinstance(node, ast.FunctionDef) and node.name == "sample_tpu_memory")
sampler_node = next(node for node in run_node.body if isinstance(node, ast.FunctionDef) and node.name == "memory_sampler")


class FakeDevice:
    def __init__(self, device_id: int) -> None:
        self.id = device_id
        self.platform = "tpu"
        self.device_kind = "TPU v5 lite"

    def memory_stats(self) -> dict[str, int]:
        return {"bytes_limit": 1_000_000, "bytes_in_use": 100, "peak_bytes_in_use": 150}

    def __str__(self) -> str:
        return f"TPU v5 lite:{self.id}"


def device_record(device: FakeDevice) -> dict[str, object]:
    return {"platform": device.platform, "device_kind": device.device_kind, "id": device.id, "repr": str(device)}


with tempfile.TemporaryDirectory(prefix="wnn-tpu-memory-check-") as temporary_directory:
    memory_path = Path(temporary_directory) / "tpu_memory_samples.jsonl"
    stop = threading.Event()
    memory_scope = {
        "Path": Path,
        "json": json,
        "os": os,
        "time": time,
        "devices": [FakeDevice(index) for index in range(8)],
        "memory_path": memory_path,
        "memory_started": time.monotonic(),
        "stop_memory_sampler": stop,
        "_device_record": device_record,
    }
    exec(compile(ast.Module(body=[sample_node, sampler_node], type_ignores=[]), str(PREFLIGHT), "exec"), memory_scope)
    background = threading.Thread(target=memory_scope["memory_sampler"], daemon=True)
    background.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: memory_scope["sample_tpu_memory"](f"phase-{index}"), range(12)))
    stop.set()
    background.join(timeout=2)
    memory_lines = memory_path.read_text(encoding="utf-8").splitlines()
    memory_records = [json.loads(line) for line in memory_lines]
    assert len(memory_records) >= 13, f"expected concurrent + periodic samples, got {len(memory_records)}"
    assert all(len(record["devices"]) == 8 for record in memory_records)
    assert len(memory_records) == len(memory_lines)

print(json.dumps({
    "status": "pass",
    "memory_samples_written_concurrently": len(memory_records),
    "each_memory_splitline_json_decodes": True,
    "devices_per_sample": 8,
}, sort_keys=True))
