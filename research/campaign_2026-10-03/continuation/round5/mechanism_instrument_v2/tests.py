from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable

import torch
from analyze import OPERATION, ROOT, TERMS, accepted_bound, geometry, sha


def fixture() -> tuple[dict, dict, dict, dict, dict, dict]:
    """Construct fictitious v4 payloads without capture files or real approval."""
    names = ["inp.weight", "gru.weight_ih_l0", "head.2.weight"]
    canonical = {str(i): name for i, name in enumerate(names)}
    groups = [
        dict(params=[0, 2], muon=False, betas=(0.9, 0.99), eps=1e-8, lr=0.01, weight_decay=0.01),
        dict(params=[1], muon=True, lr=0.01, weight_decay=0.01),
    ]
    pre = dict(
        version=4,
        phase="pre_forward_pre_zero_grad",
        force_skip=False,
        student={n: torch.ones(2, dtype=torch.float64) for n in names},
        optimizer_names=[[names[0], names[2]], [names[1]]],
        optimizer_id_names=canonical,
        optimizer=dict(
            param_groups=groups,
            state={
                0: dict(step=3, exp_avg=torch.ones(2), exp_avg_sq=torch.ones(2)),
                1: dict(momentum_buffer=torch.ones(2)),
                2: dict(step=3, exp_avg=torch.ones(2), exp_avg_sq=torch.ones(2)),
            },
        ),
        clock=dict(attempts=10, skips=0),
        h=torch.ones(2, 1, 2),
        h_ema=torch.zeros(2, 1, 2),
    )
    gradients = {term: {n: None for n in names} for term in TERMS}
    for name in names[:2]:
        gradients["main"][name] = torch.tensor([1.0, 0.0], dtype=torch.float64)
        gradients["ema"][name] = torch.tensor([-0.5, 0.0], dtype=torch.float64)
    components = dict(
        component_gradients=gradients, losses={term: torch.tensor(1.0) for term in TERMS}
    )
    post = dict(
        skipped=False, clock=dict(attempts=11, skips=0), preclip={}, clipped={}, parameter_delta={}
    )
    for i, name in enumerate(names):
        g = torch.tensor([0.5, 0.0], dtype=torch.float64) if i < 2 else None
        post["preclip"][name] = g
        post["clipped"][name] = g.clone() if g is not None else None
        post["parameter_delta"][name] = (
            -0.01 * g - 0.0001 * pre["student"][name]
            if g is not None
            else torch.zeros(2, dtype=torch.float64)
        )
    hashes = {key: str(i) * 64 for i, key in enumerate(["pre", "components", "post"])}
    context = {
        key: "synthetic-only fixture"
        for key in [
            "source",
            "backend_precision",
            "graph_reduction",
            "loss_coefficients",
            "scale_unscale_stage",
            "tensor_identity",
        ]
    }
    binding = dict(
        post_update_binding=dict(
            independently_accepted=True,
            optimizer_update_executed=True,
            parameter_participation={n: i < 2 for i, n in enumerate(names)},
        ),
        independently_accepted_decomposition_context=context,
        decomposition_policy=dict(
            operation=OPERATION,
            independently_accepted=True,
            bound_origin="independent_operation_validation",
            max_abs_bound=0.0,
            capture_sha256=hashes,
            context=context,
            independent_review_receipt_sha256="a" * 64,
        ),
    )
    return pre, components, post, canonical, binding, hashes


def main() -> None:
    """Exercise guard failures and a synthetic-only certified-zero arithmetic case."""
    started = time.monotonic()
    for name, expected in json.loads((ROOT / "FROZEN_SPEC.json").read_text()).items():
        assert sha(ROOT / name) == expected
    torch.set_num_threads(1)
    pre, components, post, canonical, binding, hashes = fixture()
    raw = geometry(pre, components, canonical)
    assert raw["scientific_update_interpretation"] is False
    assert raw["first_order_loss_changes"] is None
    assert raw["geometry"]["shared"]["scientific_conflict_witness"] is None
    assert raw["geometry"]["shared"]["component_selfgradient_antialignment_uncertified"] is True
    assert raw["decomposition_status"] == "UNAVAILABLE_NO_CERTIFIED_DECOMPOSITION"
    certified = geometry(pre, components, canonical, post, binding, hashes, True)
    assert certified["decomposition_max_abs"] == 0
    assert certified["decay_corrected_first_order"][0] < 0
    assert certified["unused_parameters"] == ["head.2.weight"]
    assert "head.2.weight" not in certified["decay_removed_parameters"]
    assert certified["structural_denominator"].startswith("UNAVAILABLE")
    rejects = []

    def reject(label: str, call: Callable[[], object]) -> None:
        try:
            call()
        except ValueError as error:
            rejects.append(dict(case=label, reason=str(error)))
        else:
            raise AssertionError(label)

    for version in [2, 3]:
        changed = copy.deepcopy(pre)
        changed["version"] = version
        reject(
            f"version{version}", lambda changed=changed: geometry(changed, components, canonical)
        )
    reject(
        "missing_bound",
        lambda: geometry(
            pre,
            components,
            canonical,
            post,
            {"post_update_binding": binding["post_update_binding"]},
            hashes,
            True,
        ),
    )
    reject(
        "missing_post",
        lambda: geometry(
            pre, components, canonical, binding=binding, hashes=hashes, scientific_update=True
        ),
    )
    for value in [float("nan"), float("inf"), -1.0, True]:
        changed = copy.deepcopy(binding)
        changed["decomposition_policy"]["max_abs_bound"] = value
        reject(f"invalid_bound_{value}", lambda changed=changed: accepted_bound(changed, hashes))
    policy_changes = {
        "default_operation": ("operation", "six_default_repeats"),
        "desired_difference": ("bound_origin", "desired_difference_fit"),
        "not_accepted": ("independently_accepted", False),
        "wrong_hash": ("capture_sha256", {}),
        "wrong_context": ("context", {}),
        "default_floor_transfer": ("default_six_repeat_floor", 0.0),
    }
    for label, (key, value) in policy_changes.items():
        changed = copy.deepcopy(binding)
        changed["decomposition_policy"][key] = value
        reject(label, lambda changed=changed: accepted_bound(changed, hashes))
    for field in ["preclip", "clipped", "parameter_delta"]:
        changed = copy.deepcopy(post)
        changed[field]["inp.weight"][0] = float("nan")
        reject(
            f"nonfinite_{field}",
            lambda changed=changed: geometry(
                pre, components, canonical, changed, binding, hashes, True
            ),
        )
    for label in [
        "skipped",
        "skip_clock",
        "active_none",
        "unused_moved",
        "wrong_shape",
        "sum_mismatch",
    ]:
        changed = copy.deepcopy(post)
        if label == "skipped":
            changed["skipped"] = True
        elif label == "skip_clock":
            changed["clock"]["skips"] = 1
        elif label == "active_none":
            changed["preclip"]["inp.weight"] = None
        elif label == "unused_moved":
            changed["parameter_delta"]["head.2.weight"][0] = 0.001
        elif label == "wrong_shape":
            changed["parameter_delta"]["inp.weight"] = torch.zeros(3)
        else:
            changed["preclip"]["inp.weight"][0] += 0.001
        reject(
            label,
            lambda changed=changed: geometry(
                pre, components, canonical, changed, binding, hashes, True
            ),
        )
    for label in [
        "nonfinite_gradient",
        "nonfinite_loss",
        "missing_key",
        "unused_component_participation",
    ]:
        changed = copy.deepcopy(components)
        if label == "nonfinite_gradient":
            changed["component_gradients"]["main"]["inp.weight"][0] = float("inf")
        elif label == "nonfinite_loss":
            changed["losses"]["main"] = torch.tensor(float("nan"))
        elif label == "missing_key":
            changed["component_gradients"]["main"].pop("inp.weight")
        else:
            changed["component_gradients"]["main"]["head.2.weight"] = torch.zeros(2)
        reject(
            label,
            lambda changed=changed: geometry(pre, changed, canonical, post, binding, hashes, True),
        )
    for label in ["nonfinite_pre", "nonfinite_hidden", "missing_moment", "swapped_names"]:
        changed = copy.deepcopy(pre)
        if label == "nonfinite_pre":
            changed["student"]["inp.weight"][0] = float("nan")
        elif label == "nonfinite_hidden":
            changed["h"][0, 0, 0] = float("inf")
        elif label == "missing_moment":
            changed["optimizer"]["state"][0].pop("exp_avg")
        else:
            changed["optimizer_names"][0].reverse()
        reject(label, lambda changed=changed: geometry(changed, components, canonical))
    reject(
        "unknown_participation",
        lambda: geometry(pre, components, canonical, post, {}, hashes, True),
    )
    # Distinct fp16 backprop operations: repeat floor zero is NOT a sum bound.
    totals = []
    for _ in range(6):
        p = torch.tensor(1.0, requires_grad=True)
        q = p.half().float()
        terms = [1.0003 * q, -q, 0 * q, 0 * q]
        totals.append(float(torch.autograd.grad(128 * sum(terms), p)[0] / 128))
    p = torch.tensor(1.0, requires_grad=True)
    q = p.half().float()
    values = [
        float(torch.autograd.grad(128 * t, p, retain_graph=True)[0] / 128)
        for t in [1.0003 * q, -q, 0 * q, 0 * q]
    ]
    assert max(totals) - min(totals) == 0 and abs(totals[0] - sum(values)) > 0
    result = dict(
        passed=True,
        evidence="tiny fictitious v4 dictionaries only; no actual approval or captures",
        rejection_cases=rejects,
        descriptive_status=raw["decomposition_status"],
        scientific_witness_available=False,
        fictitious_zero_bound_pass=True,
        unused_decay_zero=True,
        default_repeat_floor=max(totals) - min(totals),
        component_total_discrepancy=abs(totals[0] - sum(values)),
        decomposition_bound_inferred=False,
        actual_capture_analysis_executed=False,
        seconds=time.monotonic() - started,
    )
    (ROOT / "tests.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
