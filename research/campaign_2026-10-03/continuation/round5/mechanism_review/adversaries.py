from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import resource
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

OWN = Path(__file__).resolve().parent
REPO = OWN.parents[4]
ACTOR = OWN.parent / "mechanism_prereg"
V3 = OWN.parent.parent / "round4/live_capture_v3"
LOG = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_analyzer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "readonly_mechanism_analyzer", ACTOR / "analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture() -> tuple[dict, dict, dict]:
    """Tiny fictitious schema tensors, not serialized captures or a model."""
    names = ["inp.weight", "head.2.bias"]
    mapping = {"7": names[0], "19": names[1]}
    pre = {
        "version": 2,
        "phase": "pre_forward_pre_zero_grad",
        "force_skip": False,
        "student": {n: torch.ones(2, dtype=torch.float64) for n in names},
        "optimizer": {
            "param_groups": [
                {
                    "params": [7, 19],
                    "muon": False,
                    "betas": (0.9, 0.99),
                    "eps": 1e-8,
                    "lr": 0.01,
                    "weight_decay": 0.01,
                }
            ],
            "state": {
                i: {"step": 3, "exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)}
                for i in (7, 19)
            },
        },
        "optimizer_names": [names],
        "optimizer_id_names": mapping,
        "h": torch.ones(1, 1, 2),
        "h_ema": torch.zeros(1, 1, 2),
        "metadata": {"s": 0},
        "args": {"chunk": 2, "ema_teacher": 0.7, "aux_struct": 0.3, "aux_next": 0.3},
        "batch": {
            "y": torch.tensor([[[1.0, 2.0], [0.0, -3.0]]]),
            "main_valid": torch.ones(1, 2),
            "nw": torch.ones(1, 2),
        },
    }
    components = {
        "component_gradients": {
            t: {n: None for n in names} for t in ("main", "ema", "struct", "next")
        }
    }
    components["component_gradients"]["main"][names[0]] = torch.tensor(
        [1.0, 0.0], dtype=torch.float64
    )
    components["component_gradients"]["ema"][names[0]] = torch.tensor(
        [-0.5, 0.0], dtype=torch.float64
    )
    return pre, components, mapping


def decomposition_gate(policy: dict | None, residual: float) -> str:
    """Independent reviewer gate; unknown proof cannot authorize interpretation."""
    if policy is None:
        return "UNKNOWN_NO_DECOMPOSITION_BOUND"
    if policy.get("operation") != "separate_scaled_component_sum_vs_total_unscaled":
        return "BLOCK_WRONG_OPERATION"
    if (
        policy.get("independently_accepted") is not True
        or policy.get("desired_difference_used") is not False
    ):
        return "BLOCK_UNACCEPTED_OR_DESIRED_CALIBRATION"
    bound = policy.get("bound")
    if (
        not isinstance(bound, (int, float))
        or not math.isfinite(bound)
        or bound < 0
        or not math.isfinite(residual)
    ):
        return "BLOCK_NONFINITE_OR_INVALID"
    if residual > bound:
        return "BLOCK_DECOMPOSITION_MISMATCH"
    return "PASS_DECOMPOSITION_ONLY"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    begin, cpu = time.monotonic(), time.process_time()
    inputs = [
        ACTOR / name
        for name in (
            "PREREG.md",
            "REPORT.md",
            "analyze.py",
            "controls.py",
            "FROZEN_SPEC.json",
            "HASHES.json",
        )
    ]
    inputs += [
        REPO / name
        for name in ("scripts/train_rnn.py", "src/wnn/optim.py", "src/wnn/spread_aux.py")
    ]
    inputs += [V3 / name for name in ("live_capture.py", "baselines.py")]
    inputs += [OWN / "PREREG.json", OWN / "adversaries.py"]
    before = {str(p): sha(p) for p in inputs}
    for name, digest in json.loads((ACTOR / "FROZEN_SPEC.json").read_text()).items():
        assert before[str(ACTOR / name)] == digest
    analyzer = load_analyzer()
    pre, components, mapping = fixture()
    base = analyzer.geometry(pre, components, mapping)
    assert base["geometry"]["shared"]["local_conflict_witness"]
    assert base["decomposition_max_abs"] is None
    default = base["denominators"]
    assert (
        default["main"]["raw"] == 5 and default["ema"]["raw"] == 2 and default["next"]["raw"] == 2
    )
    assert default["coefficients_already_in_gradients"]["ema"] == 0.7
    divisor_controls = []
    for divisor in (0.1, 1.0, 10.0):
        scaled = copy.deepcopy(components)
        for term in scaled["component_gradients"].values():
            for gradient in term.values():
                if gradient is not None:
                    gradient.div_(divisor)
        summary = analyzer.geometry(pre, scaled, mapping)["geometry"]["shared"]
        assert abs(summary["cosine"][0][1] + 1) < 1e-12
        assert abs(summary["norms"][0] * divisor - 1) < 1e-12
        divisor_controls.append(
            {
                "divisor": divisor,
                "main_norm": summary["norms"][0],
                "cosine": summary["cosine"][0][1],
            }
        )
    metric_pre = copy.deepcopy(pre)
    metric_pre["args"]["ema_metric_weight"] = True
    metric = analyzer.geometry(metric_pre, components, mapping)["denominators"]
    assert metric["ema"]["raw"] == 5 and metric["coefficients_already_in_gradients"]["ema"] == 1.4
    rejects = []
    for case in ("v3", "map_swap", "missing_moments"):
        p, g, n = copy.deepcopy(pre), copy.deepcopy(components), copy.deepcopy(mapping)
        if case == "v3":
            p["version"] = 3
        elif case == "map_swap":
            n["7"], n["19"] = n["19"], n["7"]
        else:
            p["optimizer"]["state"][7].pop("exp_avg_sq")
        try:
            analyzer.geometry(p, g, n)
        except (ValueError, KeyError) as error:
            rejects.append({"case": case, "rejected": True, "reason": str(error)})
        else:
            raise AssertionError(case)
    # One fp32 leaf, deterministic fp16 cast backward. No neural model or CUDA.
    p = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    q = p.to(torch.float16).float()
    terms = (1.0003 * q, -q, 0 * q, 0 * q)
    scale = 128.0
    defaults = []
    for _ in range(6):
        defaults.append(torch.autograd.grad(scale * sum(terms), p, retain_graph=True)[0] / scale)
    separate = [
        torch.autograd.grad(scale * term, p, retain_graph=True)[0] / scale for term in terms
    ]
    default_floor = max(float((a - b).abs().max()) for a in defaults for b in defaults)
    sum_separate = sum(separate)
    residual = float((sum_separate - defaults[0]).abs().max())
    assert default_floor == 0 and residual > 0
    zero_policy = {
        "operation": "separate_scaled_component_sum_vs_total_unscaled",
        "independently_accepted": True,
        "desired_difference_used": False,
        "bound": 0.0,
    }
    policies = {
        "missing": None,
        "default_only": {
            "operation": "original_default_vs_original_default",
            "bound": default_floor,
        },
        "desired_fit": {**zero_policy, "desired_difference_used": True, "bound": residual},
        "NaN": {**zero_policy, "bound": float("nan")},
        "Inf": {**zero_policy, "bound": float("inf")},
        "accepted_zero": zero_policy,
    }
    policy_outcomes = {
        name: decomposition_gate(policy, residual) for name, policy in policies.items()
    }
    assert all(not outcome.startswith("PASS") for outcome in policy_outcomes.values())
    assert decomposition_gate(zero_policy, 0) == "PASS_DECOMPOSITION_ONLY"
    # Reproduce CLI's scalar comparison without touching payload files.
    cli_nan_rejects = bool(residual > float("nan"))
    cli_inf_rejects = bool(residual > float("inf"))
    assert not cli_nan_rejects and not cli_inf_rejects
    names = list(pre["student"])
    post = {"preclip": {}, "parameter_delta": {}}
    for name in names:
        total = sum(
            (
                components["component_gradients"][term][name]
                for term in analyzer.TERMS
                if components["component_gradients"][term][name] is not None
            ),
            torch.zeros(2, dtype=torch.float64),
        )
        post["preclip"][name] = total
        post["parameter_delta"][name] = -0.01 * total - 0.0001 * pre["student"][name]
    good = analyzer.geometry(pre, components, mapping, post)
    assert good["decomposition_max_abs"] == 0
    assert good["decay_corrected_first_order"][0] < 0
    nan_post = copy.deepcopy(post)
    nan_post["preclip"][names[0]][0] = float("nan")
    nan_result = analyzer.geometry(pre, components, mapping, nan_post)
    assert nan_result["decomposition_max_abs"] == 0
    cancel = copy.deepcopy(components)
    cancel["component_gradients"]["ema"][names[0]] = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    skipped = {
        "preclip": {name: None for name in names},
        "parameter_delta": {name: torch.zeros(2, dtype=torch.float64) for name in names},
        "skipped": True,
    }
    unsupported = analyzer.geometry(pre, cancel, mapping, skipped)
    assert (
        unsupported["decomposition_max_abs"] == 0
        and unsupported["decay_corrected_first_order"][0] > 0
    )
    # Pure vector clipping and active decoupled decay algebra, not optimizer replay.
    vector = np.array([3.0, 4.0])
    factor = min(1.0, 1.0 / (np.linalg.norm(vector) + 1e-6))
    clipped = vector * factor
    theta = np.array([2.0, -1.0])
    data_step = np.array([-0.04, 0.03])
    lr, wd = 0.01, 0.02
    delta = data_step - lr * wd * theta
    assert np.allclose(delta + lr * wd * theta, data_step, atol=1e-15, rtol=0)
    m, v, g, b1, b2, step, eps = 0.2, 0.4, 0.5, 0.9, 0.99, 3, 1e-8
    frozen_metric = 1 / (math.sqrt(v / (1 - b2**step)) + eps)
    next_m = b1 * m + (1 - b1) * g
    next_v = b2 * v + (1 - b2) * g * g
    actual_adam = (
        -lr * (next_m / (1 - b1 ** (step + 1))) / (math.sqrt(next_v / (1 - b2 ** (step + 1))) + eps)
    )
    surrogate = -lr * frozen_metric * g
    assert abs(actual_adam - surrogate) > 1e-6
    assert before == {str(p): sha(p) for p in inputs}
    result = {
        "verdict": "BLOCK_ACTUAL_CAPTURE_GEOMETRY_PENDING_VERSION_AND_DECOMPOSITION_BINDING",
        "actual_captures_read": 0,
        "GPU": False,
        "GRU_forwards": 0,
        "fits": 0,
        "denominator_controls": {
            "default_main": default["main"]["raw"],
            "default_EMA": default["ema"]["raw"],
            "metric_EMA": metric["ema"]["raw"],
            "default_EMA_coefficient": 0.7,
            "metric_EMA_coefficient": 1.4,
            "structural": base["structural_denominator"],
        },
        "warm_schema_rejections": rejects,
        "AMP_rounding_adversary": {
            "scale": scale,
            "dtype_path": "fp32 parameter ->fp16 cast->fp32 loss; CPU synthetic arithmetic only",
            "six_default_repeat_floor": default_floor,
            "separate_unscaled_components": [float(x.item()) for x in separate],
            "total_default_unscaled": float(defaults[0].item()),
            "separate_sum": float(sum_separate.item()),
            "residual": residual,
            "accepted_decomposition_bound": None,
            "review_policy_outcomes": policy_outcomes,
            "zero_bound_exact_positive_control": True,
        },
        "adapter_gaps": {
            "without_post_emits_conflict_witness": True,
            "without_post_decomposition_status": None,
            "NaN_binding_bound_comparison_rejects": cli_nan_rejects,
            "Inf_binding_bound_comparison_rejects": cli_inf_rejects,
            "NaN_preclip_reports_residual": nan_result["decomposition_max_abs"],
            "skip_or_unused_parameter_decay_correction_unsuppressed": unsupported[
                "decay_corrected_first_order"
            ],
            "all_toy_only_not_upstream_accepted_captures": True,
        },
        "conditional_update_algebra": {
            "positive_divisor_controls": divisor_controls,
            "clip_coefficient": float(factor),
            "clipped_vector": clipped.tolist(),
            "active_supported_decay_correction_exact": True,
            "incoming_Adam_metric_surrogate_step": surrogate,
            "next_Adam_formula_step": actual_adam,
            "not_equal_as_expected": True,
        },
        "input_sha256": before,
        "cost": {
            "wall_seconds": time.monotonic() - begin,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (OWN / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (OWN / "STATE.json").write_text(
        json.dumps(
            {
                "phase": "review_tested",
                "verdict": result["verdict"],
                "actual_captures_read": 0,
                "promotion": False,
            },
            indent=2,
        )
        + "\n"
    )
    LOG.info(
        "RESULT %s",
        json.dumps({key: value for key, value in result.items() if key != "input_sha256"}),
    )


if __name__ == "__main__":
    main()
