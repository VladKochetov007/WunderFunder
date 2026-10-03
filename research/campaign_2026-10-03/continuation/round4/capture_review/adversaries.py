"""Independent targeted corruption tests against frozen capture implementation."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import resource
import sys
import time
from pathlib import Path

import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
BUILDER = OWN.parent / "live_capture"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BUILDER))
import live_capture as capture
import runner

LOGGER = logging.getLogger(__name__)


def byte_differences(left: object, right: object, path: str = "") -> list[str]:
    if isinstance(left, torch.Tensor):
        same = isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
        if same:
            same = torch.equal(left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8))
        return [] if same else [path]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [path + "/keys"]
        return [p for key in left for p in byte_differences(left[key], right[key], f"{path}/{key}")]
    if isinstance(left, (list, tuple)):
        return [p for index, (a, b) in enumerate(zip(left, right, strict=True)) for p in byte_differences(a, b, f"{path}/{index}")]
    # RNG numpy array has exact equality independent of tensor tolerance.
    if hasattr(left, "shape"):
        return [] if (left == right).all() else [path]
    return [] if left == right else [path]


class RemoveHooks(ast.NodeTransformer):
    """Independently recover original AST from generated instrumentation."""

    def visit_If(self, node: ast.If):
        if ast.unparse(node.test) == "_capture.enabled":
            return None
        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp):
        if ast.unparse(node.test) == "_capture.enabled":
            return self.visit(node.orelse)
        return self.generic_visit(node)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(90241)
    env = runner.synthetic_environment("cpu")
    observed = runner.capture_step(env, OWN / "cpu_capture")
    payload = observed.pre
    module = capture.trainer_module()
    pristine = capture.replay(payload, module)
    result = {"pristine_byte_differences": byte_differences(observed.post, pristine)}
    original = ast.Module(body=capture.chunk_nodes(), type_ignores=[])
    stripped = RemoveHooks().visit(ast.Module(body=capture.instrument_body(capture.chunk_nodes()), type_ignores=[]))
    result["disabled_instrumentation_AST_exact"] = ast.dump(original, include_attributes=False) == ast.dump(stripped, include_attributes=False)
    # Actual recurrent ih/hh weights of the first layer have identical shapes.
    corrupt = copy.deepcopy(payload)
    ids = corrupt["optimizer"]["param_groups"][1]["params"]
    ids[0], ids[1] = ids[1], ids[0]
    result["changed_optimizer_ids"] = ids
    try:
        capture.validate(corrupt)
        other = capture.replay(corrupt, module)
        result["optimizer_id_swap"] = {
            "schema_and_restore_accepted": True,
            "post_byte_differences": byte_differences(pristine, other),
            "preclip_gradients_identical": not byte_differences(pristine["preclip"], other["preclip"]),
            "clocks_identical": not byte_differences(pristine["clock"], other["clock"]),
        }
    except (ValueError, AssertionError) as error:
        result["optimizer_id_swap"] = {"schema_and_restore_accepted": False, "reason": str(error)}
    # Test the actual fixed CUDA comparison envelope WITHOUT CUDA execution.
    changed = copy.deepcopy(pristine)
    name = next(iter(changed["parameter_delta"]))
    changed["parameter_delta"][name].flatten()[0].add_(5e-6)
    try:
        deltas = capture.compare(pristine, changed, 1e-5, 1e-4)
        result["sub_tolerance_delta_corruption"] = {"accepted": True, "measured_max": max(deltas.values()), "byte_differences": byte_differences(pristine, changed)}
    except AssertionError:
        result["sub_tolerance_delta_corruption"] = {"accepted": False}
    for value in (4, 100000):
        left, right = torch.tensor(value, dtype=torch.int64), torch.tensor(value + 1, dtype=torch.int64)
        try:
            capture.compare(left, right, 1e-5, 1e-4)
            accepted = True
        except AssertionError:
            accepted = False
        result[f"integer_clock_{value}_plus1_accepted"] = accepted
    # Pending author request must remain unusable; this is not an approval write.
    try:
        runner.verify_review(BUILDER / "review_request.json")
        result["pending_receipt_rejected"] = False
    except ValueError:
        result["pending_receipt_rejected"] = True
    result["capture_post_missing_predictors"] = [key for key in ("out", "out_ema", "clip_norm") if key not in observed.post]
    result["cost"] = {"wall_seconds": time.monotonic() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OWN / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("RESULTS %s", json.dumps(result))


if __name__ == "__main__":
    main()
