from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable

import torch
from audit import CASES, CHUNK, ROOT, TERMS, approval_guard, discrepancy, sha
from metadata import canonical_metadata


def fixture() -> tuple[dict, dict, dict, dict]:
    """Construct synthetic metadata and tiny gradients, never capture files."""
    names = ["inp.weight", "gru.weight_ih_l0", "head.2.weight"]
    pre = dict(
        version=4,
        phase="pre_forward_pre_zero_grad",
        force_skip=False,
        clock=dict(attempts=10, skips=0),
        source=dict(hash="synthetic"),
        execution=dict(device="cuda", precision="synthetic_only"),
        metadata=dict(evidence_kind="synthetic_cuda_actual_architecture"),
        optimizer_id_names={str(i): name for i, name in enumerate(names)},
        optimizer_names=[[names[0], names[2]], [names[1]]],
        optimizer=dict(param_groups=[dict(params=[0, 2], muon=False), dict(params=[1], muon=True)]),
        student={name: torch.zeros(2) for name in names},
    )
    case = dict(
        default_ordinary_passed=True,
        forced_skip=False,
        evidence_kind="synthetic_cuda_actual_architecture",
        sha256={kind: "a" * 64 for kind in ["pre", "post", "components"]},
        source=pre["source"],
        execution=pre["execution"],
        metadata_identity=canonical_metadata(pre["metadata"]),
        optimizer_id_names=pre["optimizer_id_names"],
        optimizer_groups=[
            dict(ids=[0, 2], names=[names[0], names[2]], muon=False),
            dict(ids=[1], names=[names[1]], muon=True),
        ],
        tensor_identity={name: dict(shape=[2], dtype="torch.float32") for name in names},
    )
    gradients = {t: {name: None for name in names} for t in TERMS}
    gradients["main"][names[0]] = torch.tensor([1.0, 2.0])
    gradients["ema"][names[0]] = torch.tensor([2.0, 1.0])
    gradients["main"][names[1]] = torch.tensor([10.0, -10.0])
    gradients["ema"][names[1]] = torch.tensor([-10.0, 10.0])
    components = dict(component_gradients=gradients)
    post = dict(
        clock=dict(attempts=11, skips=0),
        preclip={names[0]: torch.tensor([3.0, 3.0]), names[1]: torch.zeros(2), names[2]: None},
    )
    return pre, components, post, case


class GuardedPre(dict):
    """Fail loudly if an input/hidden/EMA array container is inspected."""

    def __getitem__(self, key: str) -> object:
        if key in {"batch", "ema", "h", "h_ema"}:
            raise AssertionError(f"Forbidden pre values access: {key}")
        return super().__getitem__(key)


class MetadataOnlyTensor(torch.Tensor):
    """Permit tensor metadata methods only; every numerical operation fails."""

    @classmethod
    def __torch_function__(
        cls, func: Callable, types: tuple, args: tuple = (), kwargs: dict | None = None
    ) -> object:
        if func.__name__ not in {"__get__", "is_floating_point", "is_contiguous"}:
            raise AssertionError(f"Forbidden pre tensor operation: {func.__name__}")
        return super().__torch_function__(func, types, args, kwargs or {})


def main() -> None:
    """Execute numerical nulls and fail-closed controls before any actual reads."""
    started = time.monotonic()
    for name, expected in json.loads((ROOT / "FROZEN_SPEC.json").read_text()).items():
        assert sha(ROOT / name) == expected
    torch.set_num_threads(1)
    pre, components, post, case = fixture()
    linear = discrepancy(pre, components, post, case)
    assert linear["scopes"]["total"]["error_l2"] == 0
    assert linear["scopes"]["total"]["relative_error_l2"] == 0
    assert linear["scopes"]["muon"]["zero_reference"]
    assert linear["scopes"]["muon"]["relative_error_l2"] is None
    assert linear["scopes"]["total"]["elements"] == 6
    assert linear["scopes"]["total"]["unused_parameters"] == 1
    poisoned = GuardedPre(pre)
    poisoned["student"] = {
        name: torch.Tensor._make_subclass(MetadataOnlyTensor, value, False)
        for name, value in pre["student"].items()
    }
    assert discrepancy(poisoned, components, post, case) == linear
    null_components = copy.deepcopy(components)
    null_post = copy.deepcopy(post)
    for name in pre["student"]:
        null_post["preclip"][name] = None
        for term in TERMS:
            null_components["component_gradients"][term][name] = None
    null = discrepancy(pre, null_components, null_post, case)
    assert null["scopes"]["total"]["unused_parameters"] == 3
    assert null["scopes"]["total"]["error_rms"] == 0
    assert null["scopes"]["total"]["relative_error_l2"] is None
    changed_post = copy.deepcopy(post)
    changed_post["preclip"]["inp.weight"][0] = 4
    known = discrepancy(pre, components, changed_post, case)["scopes"]["total"]
    assert known["error_l2"] == known["error_max_abs"] == 1
    assert abs(known["error_rms"] - 1 / math.sqrt(6)) < 1e-12
    assert known["relative_error_l2"] == 0.2
    rejects = []

    def reject(label: str, callback: Callable[[], object]) -> None:
        try:
            callback()
        except ValueError as error:
            rejects.append(dict(case=label, reason=str(error)))
        else:
            raise AssertionError(label)

    for label in [
        "nan_component",
        "inf_reference",
        "missing_key",
        "extra_key",
        "shape",
        "noncontiguous",
        "unused_mismatch",
    ]:
        p, c, q, binding = fixture()
        if label == "nan_component":
            c["component_gradients"]["main"]["inp.weight"][0] = float("nan")
        elif label == "inf_reference":
            q["preclip"]["inp.weight"][0] = float("inf")
        elif label == "missing_key":
            c["component_gradients"]["main"].pop("inp.weight")
        elif label == "extra_key":
            c["component_gradients"]["main"]["rogue"] = None
        elif label == "shape":
            q["preclip"]["inp.weight"] = torch.zeros(3)
        elif label == "noncontiguous":
            c["component_gradients"]["main"]["inp.weight"] = torch.zeros(4)[::2]
        else:
            q["preclip"]["inp.weight"] = None
        reject(label, lambda p=p, c=c, q=q, binding=binding: discrepancy(p, c, q, binding))
    for label in ["version", "forced_skip", "post_skip", "skip_clock", "mapping", "group_flag"]:
        p, c, q, binding = fixture()
        if label == "version":
            p["version"] = 3
        elif label == "forced_skip":
            p["force_skip"] = True
        elif label == "post_skip":
            q["skipped"] = True
        elif label == "skip_clock":
            q["clock"]["skips"] = 1
        elif label == "mapping":
            p["optimizer_names"][0].reverse()
        else:
            p["optimizer"]["param_groups"][0]["muon"] = True
        reject(label, lambda p=p, c=c, q=q, binding=binding: discrepancy(p, c, q, binding))
    cases = {name: dict(case, evidence_kind=kind) for name, kind in CASES.items()}
    approval = dict(
        schema_version=4,
        independent_measurement_review_passed=True,
        controller_analysis_authorized=True,
        independent_measurement_review_receipt_sha256="a" * 64,
        cases=cases,
    )
    approval_guard(approval)
    for label in [
        "no_review",
        "no_controller",
        "default_failed",
        "case_replaced",
        "no_receipt",
        "approval_old_version",
        "approved_force_skip",
    ]:
        changed = copy.deepcopy(approval)
        if label == "no_review":
            changed["independent_measurement_review_passed"] = False
        elif label == "no_controller":
            changed["controller_analysis_authorized"] = False
        elif label == "default_failed":
            changed["cases"]["real_panel_normal"]["default_ordinary_passed"] = False
        elif label == "case_replaced":
            changed["cases"].pop("real_panel_normal")
        elif label == "no_receipt":
            changed["independent_measurement_review_receipt_sha256"] = None
        elif label == "approval_old_version":
            changed["schema_version"] = 2
        else:
            changed["cases"]["real_panel_normal"]["forced_skip"] = True
        reject(label, lambda changed=changed: approval_guard(changed))
    p, c, q, binding = fixture()
    n = CHUNK + 1
    name = "inp.weight"
    binding["tensor_identity"][name]["shape"] = [n]
    p["student"][name] = torch.zeros(n)
    c["component_gradients"]["main"][name] = torch.ones(n)
    c["component_gradients"]["ema"][name] = None
    q["preclip"][name] = torch.ones(n)
    chunked = discrepancy(p, c, q, binding)
    assert chunked["scopes"]["total"]["error_l2"] == 0
    assert chunked["max_converted_workspace_bytes"] <= 32 * 1024**2
    p, c, q, binding = fixture()
    p["metadata"].update(
        cached_propensity=torch.zeros(1, 8),
        scored_mask=torch.zeros(1, 8, dtype=torch.bool),
        panel_sha256="a" * 64,
        row_ids=[dict(source="TRAIN", row_group=1)],
        teacher_binding=dict(hash="b" * 64),
        departure="synthetic fixture",
        s=4000,
    )
    binding["metadata_identity"] = canonical_metadata(p["metadata"])
    decoded = json.loads(json.dumps(binding, allow_nan=False))
    assert discrepancy(p, c, q, decoded) == linear
    original_identity = canonical_metadata(p["metadata"])
    # Descriptor equality deliberately cannot establish numerical tensor identity.
    p["metadata"]["cached_propensity"] = torch.ones(1, 8)
    assert canonical_metadata(p["metadata"]) == original_identity
    sentinel = torch.Tensor._make_subclass(MetadataOnlyTensor, torch.zeros(1, 8), False)
    projected = canonical_metadata(dict(cached_propensity=sentinel))
    json.dumps(projected, allow_nan=False)
    for label in ["metadata_shape", "metadata_dtype", "metadata_detachedness"]:
        changed = copy.deepcopy(decoded)
        for key, value in changed["metadata_identity"]["identity"]["entries"]:
            if key == "cached_propensity":
                if label == "metadata_shape":
                    value["shape"] = [1, 7]
                elif label == "metadata_dtype":
                    value["dtype"] = "torch.float16"
                else:
                    value["detached"] = False
        reject(label, lambda changed=changed: discrepancy(p, c, q, changed))
    changed = copy.deepcopy(decoded)
    changed["metadata_identity"]["identity"]["entries"].append(
        ["unapproved_extra", dict(kind="scalar", scalar_type="int", value=1)]
    )
    reject("metadata_extra_field", lambda: discrepancy(p, c, q, changed))
    changed = copy.deepcopy(decoded)
    for key, value in changed["metadata_identity"]["identity"]["entries"]:
        if key == "s":
            value["value"] = 5000
    reject("metadata_scalar_provenance", lambda: discrepancy(p, c, q, changed))
    reject(
        "unexpected_metadata_tensor_field", lambda: canonical_metadata(dict(unknown=torch.zeros(1)))
    )
    reject("unexpected_metadata_type", lambda: canonical_metadata(dict(unexpected=set())))
    cycle = []
    cycle.append(cycle)
    reject("cyclic_metadata", lambda: canonical_metadata(cycle))
    deep = {}
    for _ in range(20):
        deep = dict(child=deep)
    reject("metadata_depth_budget", lambda: canonical_metadata(deep))
    changed = copy.deepcopy(approval)
    changed["cases"]["real_panel_normal"].pop("metadata_identity")
    changed["cases"]["real_panel_normal"]["metadata"] = {}
    reject("old_raw_metadata_approval", lambda: approval_guard(changed))
    results = dict(
        passed=True,
        tiny_fixture_only=True,
        actual_gradient_reads=0,
        exact_linear=linear,
        all_unused_null=null,
        known_error=known,
        metadata_only_sentinel_passed=True,
        two_chunk_passed=True,
        active_zero_and_cancellation_passed=True,
        rejection_cases=rejects,
        acceptance_bound_inferred=False,
        metadata_json_scalar_and_tensor_roundtrip=True,
        metadata_numerical_access_sentinel_passed=True,
        same_descriptor_different_values_not_content_identity=True,
        seconds=time.monotonic() - started,
    )
    (ROOT / "tests.json").write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
