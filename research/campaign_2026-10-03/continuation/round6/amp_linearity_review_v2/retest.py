from __future__ import annotations

import copy
import importlib.util
import json
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any  # Tiny heterogeneous fixtures and metadata-only Tensor dispatch.

import torch

OWN = Path(__file__).resolve().parent
AUTHOR = OWN.parent / "amp_linearity_audit_v2"
sys.path.insert(0, str(AUTHOR))
import audit
import metadata


class MetadataOnlyTensor(torch.Tensor):
    """Reject any numeric tensor operation, permitting only metadata properties."""

    @classmethod
    def __torch_function__(
        cls, function: Callable, types: tuple, args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        if function.__name__ != "__get__":
            raise AssertionError("Numerical metadata access: " + function.__name__)
        return super().__torch_function__(function, types, args, kwargs or {})


def main() -> None:
    """Retest only the corrected metadata boundary using independent tiny inputs."""
    started = time.monotonic()
    torch.set_num_threads(1)
    prior_spec = importlib.util.spec_from_file_location(
        "prior_independent_fixture", OWN.parent / "amp_linearity_review/review_tests.py"
    )
    assert prior_spec is not None and prior_spec.loader is not None
    prior = importlib.util.module_from_spec(prior_spec)
    prior_spec.loader.exec_module(prior)
    pre, components, post, case = prior.fixture()
    pre["metadata"]["cached_propensity"] = torch.tensor([[0.2, 0.6]])
    pre["metadata"]["scored_mask"] = torch.tensor([[True, False]])
    pre["metadata"]["additional_provenance"] = {"rows": [1, 2], "enabled": True}
    case.pop("metadata")
    case["metadata_identity"] = metadata.canonical_metadata(pre["metadata"])
    case = json.loads(json.dumps(case))
    result = audit.discrepancy(pre, components, post, case)
    assert result["scopes"]["total"]["error_max_abs"] == 1
    assert result["scopes"]["total"]["reference_l2"] == 3
    changed_values = copy.deepcopy(pre["metadata"])
    changed_values["cached_propensity"] = torch.tensor([[99.0, -3.0]])
    assert metadata.canonical_metadata(changed_values) == case["metadata_identity"]
    sentinel = MetadataOnlyTensor._make_subclass(MetadataOnlyTensor, torch.zeros((1, 2)), False)
    sentinel_projection = metadata.canonical_metadata(
        {"cached_propensity": sentinel, "scored_mask": sentinel}
    )
    assert json.loads(json.dumps(sentinel_projection)) == sentinel_projection
    failures = {}

    def reject(name: str, callback: Callable[[], Any]) -> None:
        try:
            callback()
        except ValueError as error:
            failures[name] = str(error)
        else:
            raise AssertionError("Accepted: " + name)

    for name, replacement in (
        ("shape", torch.zeros((1, 3))),
        ("dtype", torch.zeros((1, 2), dtype=torch.float64)),
        ("attachedness", torch.zeros((1, 2), requires_grad=True)),
    ):
        altered = copy.deepcopy(pre)
        altered["metadata"]["cached_propensity"] = replacement
        reject(name, lambda altered=altered: audit.discrepancy(altered, components, post, case))
    for name in ("scalar_change", "extra_provenance"):
        altered = copy.deepcopy(pre)
        if name == "scalar_change":
            altered["metadata"]["s"] += 1
        else:
            altered["metadata"]["extra"] = "must not disappear"
        reject(name, lambda altered=altered: audit.discrepancy(altered, components, post, case))
    reject(
        "unexpected_tensor_field", lambda: metadata.canonical_metadata({"rogue": torch.zeros(2)})
    )
    cycle = {}
    cycle["loop"] = cycle
    reject("cycle", lambda: metadata.canonical_metadata(cycle))
    assert metadata.canonical_metadata({"flag": True}) != metadata.canonical_metadata({"flag": 1})
    helper_spec = importlib.util.spec_from_file_location(
        "controller_helper_review_only", OWN.parent / "prepare_capture_metadata_v2.py"
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    assert helper.canonical_metadata is audit.canonical_metadata is metadata.canonical_metadata
    assert Path(metadata.__file__).resolve() == (AUTHOR / "metadata.py").resolve()
    assert helper.canonical_metadata(pre["metadata"]) == case["metadata_identity"]
    # Metadata-only helper output remains unapproved and cannot be run as approval.
    prepared = json.loads((OWN.parent / "fixed_case_metadata_v2.json").read_text())
    reject("prepared_metadata_not_authorization", lambda: audit.approval_guard(prepared))
    # Tiny guard fixture is not an actual approval or permission to read captures.
    case_guard = dict(
        case,
        default_ordinary_passed=True,
        forced_skip=False,
        evidence_kind="synthetic_cuda_actual_architecture",
        sha256={key: "a" * 64 for key in ("pre", "post", "components")},
    )
    tiny_guard = {
        "schema_version": 4,
        "independent_measurement_review_passed": True,
        "controller_analysis_authorized": True,
        "independent_measurement_review_receipt_sha256": "a" * 64,
        "cases": {name: dict(case_guard, evidence_kind=kind) for name, kind in audit.CASES.items()},
    }
    audit.approval_guard(tiny_guard)
    legacy_guard = copy.deepcopy(tiny_guard)
    legacy_guard["cases"]["real_panel_normal"]["metadata"] = {"old": "forbidden"}
    reject("raw_metadata_forbidden", lambda: audit.approval_guard(legacy_guard))
    invalid_version = copy.deepcopy(tiny_guard)
    invalid_version["cases"]["real_panel_normal"]["metadata_identity"]["projection_version"] = "old"
    reject("projection_version", lambda: audit.approval_guard(invalid_version))
    output = {
        "passed": True,
        "tiny_residual_arithmetic_unchanged": True,
        "Tensor_metadata_JSON_roundtrip_pass": True,
        "metadata_numerical_access_sentinel_pass": True,
        "same_descriptor_different_values_control": "same projection intentionally; independent pre-file hash must bind contents",
        "typed_bool_int_distinct": True,
        "shared_projection_function_identity": True,
        "rejections": failures,
        "actual_capture_tensor_reads": 0,
        "GPU_calls": 0,
        "actual_diagnostic_executed": False,
        "wall_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (OWN / "tests.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
