"""Synthetic byte-grammar controls only; never load capture tensors or actor modules."""
from __future__ import annotations

import ast
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
R6 = HERE.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    helper = R6 / "prepare_capture_metadata.py"
    actor = R6 / "state_drift_v2/protocol.py"
    ns = {"torch": torch, "json": json, "hashlib": hashlib}
    identity = extract(helper, {"identity"}, ns.copy())["identity"]
    candidate = extract(actor, {"tensor_identity"}, ns.copy())["tensor_identity"]
    cases = {
        "fp32_matrix": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "fp16_vector": torch.arange(9, dtype=torch.float16),
        "bf16_normalization_shape": torch.arange(9, dtype=torch.bfloat16).reshape(1, 1, 9),
        "stream_boundary_fp32": torch.arange(262145, dtype=torch.float32),
    }
    checks = {}
    for name, value in cases.items():
        legacy = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}).encode()
                                + value.view(torch.uint8).numpy().tobytes()).hexdigest()
        checks[name] = identity(value) == candidate(value) == legacy
    plus_zero, minus_zero = torch.tensor([0.0]), torch.tensor([-0.0])
    checks["signed_zero_distinguished"] = identity(plus_zero) != identity(minus_zero)
    same_bytes = cases["fp32_matrix"].reshape(-1)
    checks["shape_header_distinguished"] = identity(same_bytes) != identity(cases["fp32_matrix"])
    try:
        identity(cases["fp32_matrix"].transpose(0, 1))
    except ValueError:
        checks["noncontiguous_helper_rejected"] = True
    else:
        checks["noncontiguous_helper_rejected"] = False
    assert all(checks.values())
    result = {
        "checks": checks,
        "status": "PASS_SYNTHETIC_GRAMMAR_ONLY",
        "scope": "same ordered default-JSON shape/dtype header plus raw contiguous uint8 bytes",
        "helper_sha256": sha(helper), "draft_protocol_sha256": sha(actor),
        "seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "actual_tensor_reads": 0, "capture_loads": 0, "GRU_forwards": 0, "GPU_calls": 0,
    }
    (HERE / "identity_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
