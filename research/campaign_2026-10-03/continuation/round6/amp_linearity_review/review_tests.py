from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import resource
import time
from pathlib import Path
from typing import Any  # Heterogeneous tiny synthetic approval/capture fixtures.

import torch

OWN = Path(__file__).resolve().parent
AUTHOR = OWN.parent / "amp_linearity_audit"


def fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create a tiny independent fixture, never load real captures."""
    names = ["inp.weight", "head.weight"]
    identity = {name: {"shape": [3], "dtype": "torch.float32"} for name in names}
    pre = {
        "version": 4,
        "phase": "pre_forward_pre_zero_grad",
        "force_skip": False,
        "source": {"source": "tiny_fixture"},
        "execution": {"device": "cuda", "fixture_only": True},
        "metadata": {"evidence_kind": "real_fit_capture", "s": 4000},
        "clock": {"attempts": 4, "skips": 0},
        "optimizer_id_names": {0: names[0], 1: names[1]},
        "optimizer_names": [[names[0]], [names[1]]],
        "optimizer": {
            "param_groups": [{"params": [0], "muon": True}, {"params": [1], "muon": False}]
        },
        "student": {name: torch.zeros(3) for name in names},
    }
    case = {
        "source": pre["source"],
        "execution": pre["execution"],
        "metadata": copy.deepcopy(pre["metadata"]),
        "optimizer_id_names": {str(key): value for key, value in pre["optimizer_id_names"].items()},
        "optimizer_groups": [
            {"ids": [0], "names": [names[0]], "muon": True},
            {"ids": [1], "names": [names[1]], "muon": False},
        ],
        "tensor_identity": identity,
    }
    components = {
        "component_gradients": {
            term: {name: None for name in names} for term in ("main", "ema", "struct", "next")
        }
    }
    components["component_gradients"]["main"][names[0]] = torch.tensor([2.0, 3.0, -1.0])
    components["component_gradients"]["ema"][names[0]] = torch.tensor([-1.0, 0.0, 4.0])
    components["component_gradients"]["struct"][names[0]] = torch.tensor([0.0, -1.0, 0.0])
    components["component_gradients"]["next"][names[0]] = torch.tensor([1.0, 0.0, -2.0])
    post = {
        "clock": {"attempts": 5, "skips": 0},
        "preclip": {names[0]: torch.tensor([1.0, 2.0, 2.0]), names[1]: None},
    }
    return pre, components, post, case


def main() -> None:
    """Run only small CPU algebra and actual-source-schema boundary fixtures."""
    started = time.monotonic()
    torch.set_num_threads(1)
    module_spec = importlib.util.spec_from_file_location(
        "author_linearity_review_only", AUTHOR / "audit.py"
    )
    assert module_spec is not None and module_spec.loader is not None
    audit = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(audit)
    pre, components, post, case = fixture()
    positive = audit.discrepancy(pre, components, post, case)
    total = positive["scopes"]["total"]
    # Independent Python arithmetic: fixed FP64 order sums to[2,2,1], reference[1,2,2].
    errors = [1.0, 0.0, -1.0]
    reference = [1.0, 2.0, 2.0]
    oracle = {
        "l2": math.sqrt(sum(value * value for value in errors)),
        "maxabs": max(abs(value) for value in errors),
        "rms": math.sqrt(sum(value * value for value in errors) / 6),
        "relative": math.sqrt(sum(value * value for value in errors))
        / math.sqrt(sum(value * value for value in reference)),
    }
    assert total["error_l2"] == oracle["l2"] and total["error_max_abs"] == oracle["maxabs"]
    assert math.isclose(total["error_rms"], oracle["rms"], rel_tol=1e-15)
    assert total["relative_error_l2"] == oracle["relative"]
    assert total["elements"] == 6 and total["active_parameters"] == total["unused_parameters"] == 1
    assert positive["scopes"]["adam"]["relative_error_l2"] is None
    assert (
        positive["acceptance_bound_inferred"] is False and positive["calibrated_allowance"] is None
    )
    assert positive["scientific_conflict_or_score_claim"] is False
    # Approval JSON roundtrip succeeds with the scalar-only synthetic metadata control.
    json_case = json.loads(json.dumps(case))
    audit.discrepancy(pre, components, post, json_case)
    tensor_pre = copy.deepcopy(pre)
    tensor_pre["metadata"]["cached_propensity"] = torch.tensor([[0.5, 0.7]])
    tensor_pre["metadata"]["scored_mask"] = torch.tensor([[True, False]])
    serialization_failure = None
    try:
        json.dumps(tensor_pre["metadata"])
    except TypeError as error:
        serialization_failure = str(error)
    assert serialization_failure is not None
    approval_case = copy.deepcopy(case)
    approval_case["metadata"]["cached_propensity"] = [[0.5, 0.7]]
    approval_case["metadata"]["scored_mask"] = [[True, False]]
    approval_case = json.loads(json.dumps(approval_case))
    boundary_failure = None
    try:
        audit.discrepancy(tensor_pre, components, post, approval_case)
    except (ValueError, RuntimeError, TypeError) as error:
        boundary_failure = f"{type(error).__name__}: {error}"
    assert boundary_failure is not None
    # Confirm source compatibility problem without opening any real pre.pt.
    adapter = OWN.parents[1] / "round4/live_capture_v4/fit_panel.py"
    tree = ast.parse(adapter.read_text())
    tensor_metadata_fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in {
                    "cached_propensity",
                    "scored_mask",
                }:
                    tensor_metadata_fields.append(
                        {"name": key.value, "expression": ast.unparse(value), "line": value.lineno}
                    )
    assert {row["name"] for row in tensor_metadata_fields} == {"cached_propensity", "scored_mask"}
    # Approved-but-large residual must remain descriptive, not magnitude-gated.
    large_post = copy.deepcopy(post)
    large_post["preclip"]["inp.weight"] = torch.tensor([1e10, 2.0, 2.0])
    large = audit.discrepancy(pre, components, large_post, case)
    assert large["status"] == "DESCRIPTIVE_DIFFERENT_OPERATION_DISCREPANCY_ONLY"
    assert large["acceptance_bound_inferred"] is False
    results = {
        "algebra_oracle_pass": True,
        "oracle": oracle,
        "tiny_metrics": total,
        "JSON_scalar_metadata_positive_control_pass": True,
        "real_source_schema_tiny_witness": {
            "serialization_failure": serialization_failure,
            "approved_JSON_case_rejected": boundary_failure,
            "source_tensor_metadata_fields": tensor_metadata_fields,
        },
        "large_residual_remains_descriptive": True,
        "conservative_workspace_max_bytes": positive["max_converted_workspace_bytes"],
        "actual_capture_reads": 0,
        "GPU_calls": 0,
        "wall_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (OWN / "tests.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
