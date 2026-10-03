from __future__ import annotations

import ast
import copy
import json
import logging
import pickle
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any  # Actual source-bound nested Torch capture has heterogeneous state.

import numpy as np
import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
BUILDER = OWN.parent / "live_capture_v3"
sys.path[:0] = [str(BUILDER), str(ROOT / "src")]
import baselines
import live_capture as capture
import runner

CHECKS: dict[str, dict[str, Any]] = {}


def reject(name: str, function: Any, *args: Any) -> None:
    """Record required rejection without treating unexpected exceptions as success."""
    try:
        function(*args)
    except (AssertionError, ValueError, TypeError) as error:
        CHECKS[name] = {"pass": True, "reason": str(error)}
    else:
        CHECKS[name] = {"pass": False, "reason": "accepted unsupported state"}


def mutate_rng(pre: dict[str, Any], name: str, key: str, value: Any, module: Any) -> None:
    """Require all three pre-forward boundaries to reject without global RNG mutation."""
    altered = copy.deepcopy(pre)
    altered["rng"][key] = value
    before = capture.get_rng("cpu")
    reject(name + ".validate", capture.validate, altered)
    reject(name + ".restore", capture.restore, altered, module)
    reject(name + ".set", capture.set_rng, altered["rng"], "cpu")
    capture.compare(before, capture.get_rng("cpu"))
    CHECKS[name + ".no_global_mutation"] = {"pass": True}


class EraseObservations(ast.NodeTransformer):
    """Independently remove observer statements, retaining actual source math."""

    def visit_If(self, node: ast.If) -> ast.AST | None:
        if ast.unparse(node.test) == "_capture.enabled":
            return None
        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        if ast.unparse(node.test) == "_capture.enabled":
            return self.visit(node.orelse)
        return self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func).startswith(
            "_baseline."
        ):
            if ast.unparse(node.value.func) == "_baseline.record_clip":
                return ast.Expr(value=self.visit(node.value.args[0]))
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if ast.unparse(node.func) == "_baseline.component":
            return self.visit(node.args[1])
        return self.generic_visit(node)


def main() -> None:
    """Run bounded independent typed-state falsifiers and one pristine probe."""
    logging.basicConfig(level=logging.INFO)
    start = time.monotonic()
    torch.set_num_threads(1)
    capture.verify_generated()
    output = OWN / "cpu_reproduction"
    output.mkdir(exist_ok=False)
    smoke = runner.run_probe(output, "cpu")
    (output / "results.json").write_text(json.dumps(smoke, indent=2) + "\n")
    runner.output_hashes(output)
    pre = torch.load(output / "normal/pre.pt", weights_only=False)
    module = capture.trainer_module()
    capture.validate(pre)
    restored = capture.restore(pre, module)
    capture.compare(pre["optimizer"], capture.cloned(restored["opt"].state_dict()))
    capture.compare(pre["rng"], capture.get_rng("cpu"))
    capture.compare(pre["rng"], pickle.loads(pickle.dumps(pre["rng"])))
    CHECKS["pristine_named_moments_rng_serialization_restore"] = {"pass": True}
    del restored
    for seed in (17, 814):
        np.random.seed(seed)
        np.random.normal(size=3)
        random.seed(seed)
        random.gauss(0, 1)
        torch.manual_seed(seed)
        torch.rand(7)
        state = capture.get_rng("cpu")
        capture.validate_rng(state, "cpu")
        capture.set_rng(state, "cpu")
        capture.compare(state, capture.get_rng("cpu"))
        CHECKS[f"seed_{seed}_advanced_cached_roundtrip"] = {"pass": True}
    original_np = pre["rng"]["numpy"]
    endian = ">u4" if np.little_endian else "<u4"
    for dtype in (np.uint64, np.float64, endian):
        state = list(copy.deepcopy(original_np))
        state[1] = state[1].astype(dtype)
        name = "numpy_dtype_" + str(np.dtype(dtype))
        mutate_rng(pre, name, "numpy", tuple(state), module)
        reject(name + ".compare", capture.compare, original_np, tuple(state))
    for name, index, value in (
        ("position_bool", 2, True),
        ("position_float", 2, float(original_np[2])),
        ("position_numpy_int", 2, np.int64(original_np[2])),
        ("position_oob", 2, 625),
        ("cache_flag_bool", 3, False),
        ("cache_flag_float", 3, 0.0),
        ("cached_numpy_scalar", 4, np.float64(original_np[4])),
        ("cached_int", 4, 0),
        ("cached_nonfinite", 4, float("nan")),
        ("algorithm_alias", 0, "OTHER"),
        ("wrong_shape", 1, original_np[1][:-1]),
        ("strided_storage", 1, np.repeat(original_np[1], 2)[::2]),
    ):
        state = list(copy.deepcopy(original_np))
        state[index] = value
        mutate_rng(pre, "numpy_grammar_" + name, "numpy", tuple(state), module)
    mutate_rng(pre, "numpy_list", "numpy", list(original_np), module)
    state = list(copy.deepcopy(pre["rng"]["python"]))
    words = list(state[1])
    words[0] = True
    state[1] = tuple(words)
    mutate_rng(pre, "python_word_bool", "python", tuple(state), module)
    state = list(copy.deepcopy(pre["rng"]["python"]))
    state[0] = float(state[0])
    mutate_rng(pre, "python_version_float", "python", tuple(state), module)
    mutate_rng(pre, "torch_rng_int16", "torch_cpu", pre["rng"]["torch_cpu"].to(torch.int16), module)
    mutate_rng(pre, "CPU_cuda_nonempty", "cuda_all", [pre["rng"]["torch_cpu"]], module)
    for tag, a, b in (
        ("array_bool_int", np.array([True]), np.array([1], dtype=np.uint32)),
        ("array_int_float", np.array([1], dtype=np.uint32), np.array([1.0])),
        ("array_signed_zero", np.array([0], dtype=np.float32), np.array([-0.0], dtype=np.float32)),
        ("array_one_word", np.array([1], dtype=np.uint32), np.array([2], dtype=np.uint32)),
        ("scalar_signed_zero", 0.0, -0.0),
        ("scalar_bool_int", True, 1),
        ("tensor_clock", torch.tensor(100000), torch.tensor(100001)),
        ("tensor_bool", torch.tensor(True), torch.tensor(False)),
    ):
        reject(tag, capture.compare, a, b)
    for field in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
        identifier = next(i for i, state in pre["optimizer"]["state"].items() if field in state)
        original = pre["optimizer"]["state"][identifier][field]
        for tag, replacement in (
            ("float64", original.double()),
            ("attached", original.clone().requires_grad_()),
            ("wrong_shape", original.reshape(-1)[:1]),
            ("meta_device", torch.empty_like(original, device="meta")),
            ("nonfinite", torch.full_like(original, float("inf"))),
        ):
            altered = copy.deepcopy(pre)
            altered["optimizer"]["state"][identifier][field] = replacement
            reject(field + "." + tag + ".validate", capture.validate, altered)
            reject(field + "." + tag + ".restore", capture.restore, altered, module)
    for forged in (False, True):
        altered = copy.deepcopy(pre)
        ids = altered["optimizer"]["param_groups"][1]["params"]
        a, b = ids[:2]
        assert (
            pre["student"][pre["optimizer_id_names"][a]].shape
            == pre["student"][pre["optimizer_id_names"][b]].shape
        )
        ids[0], ids[1] = b, a
        if forged:
            mapping = altered["optimizer_id_names"]
            mapping[a], mapping[b] = mapping[b], mapping[a]
        reject(f"same_shape_id_swap_forged_{forged}", capture.restore, altered, module)
    for key, value in (("step_count", True), ("ema_updates", 1.0), ("skips", torch.tensor(False))):
        altered = copy.deepcopy(pre)
        altered["clock"][key] = value
        reject("clock_" + key, capture.restore, altered, module)
    altered = copy.deepcopy(pre)
    altered["execution"]["device"] = "unsupported"
    reject("backend_mismatch", capture.restore, altered, module)
    reject("pending_receipt", runner.verify_review, BUILDER / "review_request.json")
    reject("runtime_outside_GPU_chain", runner.runtime_guard)
    original = ast.dump(
        ast.Module(body=capture.chunk_nodes(), type_ignores=[]), include_attributes=False
    )
    for name, body in (
        ("online", capture.instrument_body(capture.chunk_nodes())),
        ("original_baseline", baselines.independent_original_nodes(False)),
    ):
        tree = EraseObservations().visit(ast.Module(body=body, type_ignores=[]))
        CHECKS["AST_" + name] = {"pass": ast.dump(tree, include_attributes=False) == original}
    result = {
        "checks": CHECKS,
        "all_pass": all(item["pass"] for item in CHECKS.values()),
        "count": len(CHECKS),
        "CPU_reproduction": smoke,
        "wall_seconds": time.monotonic() - start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "CUDA_evidence": False,
    }
    (OWN / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))
    if not result["all_pass"]:
        raise AssertionError("Independent adversary or pristine control failed")


if __name__ == "__main__":
    main()
