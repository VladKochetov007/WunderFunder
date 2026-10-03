from __future__ import annotations

import copy
import json
import time

import torch
from analyze import ROOT, TERMS, geometry, sha


def fixture() -> tuple[dict, dict, dict[str, str]]:
    """Build a tiny declared synthetic capture, not actual training evidence."""
    names = ["inp.weight", "gru.weight_ih_l0", "head.2.weight"]
    ids = [10, 99, 55]
    groups = [
        dict(params=[10, 55], muon=False, betas=(0.9, 0.99), eps=1e-8, lr=0.01, weight_decay=0.01),
        dict(params=[99], muon=True, lr=0.01, weight_decay=0.01),
    ]
    state = {
        k: dict(momentum_buffer=torch.ones(2))
        if k == 99
        else dict(step=3, exp_avg=torch.ones(2), exp_avg_sq=torch.ones(2))
        for k in ids
    }
    canonical = {str(k): n for k, n in zip(ids, names, strict=True)}
    pre = dict(
        version=2,
        phase="pre_forward_pre_zero_grad",
        force_skip=False,
        student={n: torch.ones(2, dtype=torch.float64) for n in names},
        optimizer=dict(param_groups=groups, state=state),
        optimizer_names=[[names[0], names[2]], [names[1]]],
        optimizer_id_names=canonical,
        h=torch.ones(2, 1, 2),
        h_ema=torch.zeros(2, 1, 2),
        metadata=dict(s=1),
        args=dict(chunk=2, ema_teacher=1, aux_struct=0.3, aux_next=0.3),
        batch=dict(y=torch.ones(1, 8, 2), main_valid=torch.ones(1, 8), nw=torch.ones(1, 8)),
    )
    gradients = {t: {n: None for n in names} for t in TERMS}
    for n in names[:2]:
        gradients["main"][n] = torch.tensor([1.0, 0.0], dtype=torch.float64)
        gradients["ema"][n] = torch.tensor([-0.5, 0.0], dtype=torch.float64)
    return pre, dict(component_gradients=gradients), canonical


def main() -> None:
    """Execute frozen mechanism-discrimination fixtures with no fits."""
    started = time.monotonic()
    frozen = json.loads((ROOT / "FROZEN_SPEC.json").read_text())
    for name, expected in frozen.items():
        assert sha(ROOT / name) == expected, name
    torch.set_num_threads(1)
    pre, components, canonical = fixture()
    conflict = geometry(pre, components, canonical)
    assert conflict["geometry"]["shared"]["local_conflict_witness"]
    assert conflict["geometry"]["shared"]["cosine"][0][1] == -1
    assert conflict["geometry"]["adam_frozen_metric"]["local_conflict_witness"]
    assert conflict == geometry(copy.deepcopy(pre), copy.deepcopy(components), canonical)
    aligned = copy.deepcopy(components)
    for name in list(pre["student"])[:2]:
        aligned["component_gradients"]["ema"][name] *= -1
    amplitude = []
    for divisor in [0.1, 1, 10]:
        scaled = copy.deepcopy(aligned)
        for term in TERMS:
            for gradient in scaled["component_gradients"][term].values():
                if gradient is not None:
                    gradient.div_(divisor)
        result = geometry(pre, scaled, canonical)["geometry"]["shared"]
        assert result["cosine"][0][1] == 1
        assert not result["local_conflict_witness"]
        amplitude.append(
            dict(divisor=divisor, main_norm=result["norms"][0], cosine=result["cosine"][0][1])
        )
    for row in amplitude:
        assert abs(row["main_norm"] * row["divisor"] - 2**0.5) < 1e-12
    heads = copy.deepcopy(components)
    for t in TERMS:
        heads["component_gradients"][t] = {name: None for name in pre["student"]}
    heads["component_gradients"]["main"]["head.2.weight"] = torch.tensor([1.0, 0.0])
    heads["component_gradients"]["ema"]["head.2.weight"] = torch.tensor([0.0, -1.0])
    isolated = geometry(pre, heads, canonical)
    assert not isolated["geometry"]["shared"]["local_conflict_witness"]
    assert isolated["geometry"]["shared"]["cosine"][0][1] is None
    rejected = []
    wrong = copy.deepcopy(canonical)
    wrong["10"], wrong["99"] = wrong["99"], wrong["10"]
    for label, input_pre, mapping in [
        ("canonical_swap", pre, wrong),
        ("missing_moments", copy.deepcopy(pre), canonical),
    ]:
        if label == "missing_moments":
            input_pre["optimizer"]["state"][10].pop("exp_avg_sq")
        try:
            geometry(input_pre, components, mapping)
        except ValueError as error:
            rejected.append(dict(control=label, reason=str(error)))
        else:
            raise AssertionError(label)
    post = dict(preclip={}, parameter_delta={})
    for name in pre["student"]:
        summed = sum(
            (
                components["component_gradients"][t][name]
                for t in TERMS
                if components["component_gradients"][t][name] is not None
            ),
            torch.zeros(2, dtype=torch.float64),
        )
        post["preclip"][name] = summed
        post["parameter_delta"][name] = -0.01 * summed - 0.0001 * pre["student"][name]
    update = geometry(pre, components, canonical, post)
    assert update["decomposition_max_abs"] == 0
    assert update["decay_corrected_first_order"][0] < 0
    assert update["decay_corrected_first_order"][1] > 0
    assert update["state_drift_identification"].startswith("UNIDENTIFIED")
    assert update["denominators"]["main"]["raw"] == 4
    assert update["denominators"]["ema"]["raw"] == 2
    result = dict(
        passed=True,
        evidence="tiny synthetic algebra/schema controls only",
        no_actual_captures_read=True,
        planted_conflict=conflict["geometry"]["shared"],
        aligned_divisor_controls=amplitude,
        sham_head_isolation=True,
        identity_exact=True,
        unused_cosines_undefined=True,
        drift_not_misidentified=True,
        rejects=rejected,
        actual_toy_step=update,
        seconds=time.monotonic() - started,
    )
    (ROOT / "controls.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
