from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Capture payloads deliberately contain heterogeneous tensors, mappings and clocks.
Payload = dict[str, Any]
ROOT = Path(__file__).resolve().parent
TERMS = ("main", "ema", "struct", "next")


def sha(path: Path) -> str:
    """Hash a file without loading its payload into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    pre: Payload, components: Payload, canonical: dict[str, str]
) -> list[tuple[str, int, Payload]]:
    """Require name-bound warm optimizer state; never infer canonical IDs."""
    if pre["version"] != 4 or pre["phase"] != "pre_forward_pre_zero_grad":
        raise ValueError("Unsupported capture phase/version")
    if pre["force_skip"]:
        raise ValueError("Forced-skip capture excluded from scientific geometry")
    if set(components["component_gradients"]) != set(TERMS):
        raise ValueError("Exactly four actual weighted loss terms required")
    if set(components.get("losses", {})) != set(TERMS):
        raise ValueError("Four finite component loss scalars required")
    for value in components["losses"].values():
        finite_tensor(value, torch.Size([]), "component loss")
    declared = {str(k): v for k, v in pre["optimizer_id_names"].items()}
    if declared != canonical:
        raise ValueError("Canonical independently accepted ID/name map differs")
    entries = []
    for group, names in zip(pre["optimizer"]["param_groups"], pre["optimizer_names"], strict=True):
        for index, name in zip(group["params"], names, strict=True):
            if canonical[str(index)] != name:
                raise ValueError("Serialized ID/name association differs")
            parameter = pre["student"][name]
            finite_tensor(parameter, parameter.shape, "student parameter")
            if parameter.numel() * 64 > 32 * 1024**2:
                raise ValueError("Parameter exceeds bounded float64 workspace")
            state = pre["optimizer"]["state"][index]
            fields = (
                {"momentum_buffer"}
                if group.get("muon", False)
                else {"step", "exp_avg", "exp_avg_sq"}
            )
            if set(state) != fields:
                raise ValueError("Warm optimizer fields missing or unsupported")
            if "step" in fields and (type(state["step"]) is not int or state["step"] <= 0):
                raise ValueError("Warm integer Adam step required")
            for field in fields - {"step"}:
                if state[field].shape != parameter.shape or not torch.isfinite(state[field]).all():
                    raise ValueError("Invalid warm moment tensor")
            for term in TERMS:
                gradients = components["component_gradients"][term]
                if name not in gradients:
                    raise ValueError("Missing parameter key; unused must explicitly be None")
                gradient = gradients[name]
                if gradient is not None and (
                    gradient.shape != parameter.shape or not torch.isfinite(gradient).all()
                ):
                    raise ValueError("Invalid component gradient")
            entries.append((name, index, group))
    if len(entries) != len(canonical) or len({e[0] for e in entries}) != len(entries):
        raise ValueError("Duplicate or incomplete parameter membership")
    for term in TERMS:
        if set(components["component_gradients"][term]) != set(canonical.values()):
            raise ValueError("Extra or missing component parameter keys")
    for key in ["h", "h_ema"]:
        finite_tensor(pre[key], pre["h"].shape, key)
    for group in pre["optimizer"]["param_groups"]:
        for key in ["lr", "weight_decay"]:
            if not scalar_finite(group[key]) or group[key] < 0:
                raise ValueError("Invalid optimizer hyperparameter")
        if not group.get("muon", False):
            if not scalar_finite(group["eps"]) or group["eps"] <= 0:
                raise ValueError("Invalid Adam epsilon")
            if any(not scalar_finite(b) or not 0 <= b < 1 for b in group["betas"]):
                raise ValueError("Invalid Adam betas")
    return entries


def scalar_finite(value: object) -> bool:
    """Reject booleans and nonfinite numeric bounds before comparisons."""
    return type(value) in (int, float) and bool(np.isfinite(value))


def finite_tensor(value: object, shape: torch.Size, label: str) -> None:
    """Require finite CPU floating tensors before any residual reduction."""
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or not value.is_floating_point()
        or value.shape != shape
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"Invalid {label} tensor")


OPERATION = "separate_scaled_component_autograd_sum_vs_unscaled_total_preclip"


def accepted_bound(binding: Payload, hashes: dict[str, str]) -> float:
    """Accept only explicit independent operation-specific provenance, never floors."""
    policy = binding.get("decomposition_policy")
    if not isinstance(policy, dict):
        raise ValueError("UNAVAILABLE_NO_INDEPENDENT_DECOMPOSITION_BOUND")
    if policy.get("operation") != OPERATION:
        raise ValueError("Wrong decomposition operation; default floors forbidden")
    if set(policy) & {
        "default_repeat_floor",
        "default_six_repeat_floor",
        "desired_difference",
        "fit_to_observed_difference",
    }:
        raise ValueError("Default-floor or desired-difference transfer forbidden")
    if (
        policy.get("independently_accepted") is not True
        or policy.get("bound_origin") != "independent_operation_validation"
    ):
        raise ValueError("Unaccepted or desired-difference/default-floor bound")
    bound = policy.get("max_abs_bound")
    if not scalar_finite(bound) or bound < 0:
        raise ValueError("Nonfinite or invalid decomposition bound")
    if policy.get("capture_sha256") != hashes:
        raise ValueError("Decomposition policy capture hashes differ")
    context = policy.get("context")
    required = {
        "source",
        "backend_precision",
        "graph_reduction",
        "loss_coefficients",
        "scale_unscale_stage",
        "tensor_identity",
    }
    if (
        not isinstance(context, dict)
        or set(context) != required
        or any(not isinstance(v, str) or not v for v in context.values())
    ):
        raise ValueError("Incomplete operation context")
    if context != binding.get("independently_accepted_decomposition_context"):
        raise ValueError("Independently accepted operation context differs")
    receipt = policy.get("independent_review_receipt_sha256")
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(c not in "0123456789abcdef" for c in receipt)
    ):
        raise ValueError("Independent decomposition review receipt unavailable")
    return float(bound)


def post_participation(
    pre: Payload, post: Payload, binding: Payload, names: set[str]
) -> dict[str, bool]:
    """Require independently accepted executed update and exact parameter support."""
    update = binding.get("post_update_binding")
    if (
        not isinstance(update, dict)
        or update.get("independently_accepted") is not True
        or update.get("optimizer_update_executed") is not True
    ):
        raise ValueError("UNAVAILABLE_EXECUTED_UPDATE_OR_PARTICIPATION")
    if post.get("skipped", False) is not False:
        raise ValueError("Skipped attempt has no scientific update interpretation")
    before, after = pre["clock"], post["clock"]
    if (
        int(after["skips"]) != int(before["skips"])
        or int(after["attempts"]) != int(before["attempts"]) + 1
    ):
        raise ValueError("Post skip/attempt clocks inconsistent")
    participation = update.get("parameter_participation")
    if (
        not isinstance(participation, dict)
        or set(participation) != names
        or any(type(v) is not bool for v in participation.values())
    ):
        raise ValueError("Complete boolean parameter participation required")
    for field in ["preclip", "clipped", "parameter_delta"]:
        if set(post[field]) != names:
            raise ValueError("Incomplete or extra post parameter keys")
    for name in names:
        shape = pre["student"][name].shape
        finite_tensor(post["parameter_delta"][name], shape, "parameter delta")
        for field in ["preclip", "clipped"]:
            value = post[field][name]
            if value is None:
                if participation[name]:
                    raise ValueError("Participating parameter has unused post gradient")
            else:
                finite_tensor(value, shape, field)
                if not participation[name]:
                    raise ValueError("Nonparticipating parameter has gradient")
        if not participation[name] and torch.count_nonzero(post["parameter_delta"][name]):
            raise ValueError("Unused parameter moved")
    return participation


def summarize(gram: np.ndarray) -> Payload:
    """Summarize exact additive force geometry; zero cosine is undefined."""
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    denominator = np.outer(norms, norms)
    cosine = np.divide(gram, denominator, out=np.full((4, 4), np.nan), where=denominator > 0)
    total = float(np.sqrt(max(float(gram.sum()), 0)))
    cosine_json = [[None if not np.isfinite(v) else float(v) for v in row] for row in cosine]
    combined_aux_norm = np.sqrt(max(float(gram[2:, 2:].sum()), 0))
    aux_cos = (
        float(gram[0, 2:].sum() / (norms[0] * combined_aux_norm))
        if norms[0] * combined_aux_norm > 0
        else None
    )
    meaningful = 1e-6 * norms.sum()
    witness = bool(
        norms[0] >= meaningful
        and norms[1] >= meaningful
        and norms[0] > 0
        and norms[1] > 0
        and cosine[0, 1] <= -0.1
    )
    witness |= bool(
        norms[0] >= meaningful
        and combined_aux_norm >= meaningful
        and aux_cos is not None
        and aux_cos <= -0.1
    )
    return dict(
        norms=norms.tolist(),
        gram=gram.tolist(),
        cosine=cosine_json,
        total_norm=total,
        cancellation=1 - total / norms.sum() if norms.sum() else None,
        main_combined_aux_cosine=aux_cos,
        local_conflict_witness=witness,
    )


def denominators(pre: Payload, norms: list[float]) -> Payload:
    """Do not transfer old source/branch assumptions to unapproved v4."""
    return dict(
        coefficients="UNAVAILABLE: independent v4 branch binding absent; gradients never reweighted",
        all_loss_denominators="UNAVAILABLE: no approved v4 source normalization binding",
        structural="UNAVAILABLE",
    )


def geometry(
    pre: Payload,
    components: Payload,
    canonical: dict[str, str],
    post: Payload | None = None,
    binding: Payload | None = None,
    hashes: dict[str, str] | None = None,
    scientific_update: bool = False,
) -> Payload:
    """Analyze captured tensors only; no forward, optimizer step or CUDA call."""
    entries = validate(pre, components, canonical)
    participation = None
    bound = None
    if post is not None:
        participation = post_participation(pre, post, binding or {}, set(canonical.values()))
    if scientific_update:
        if post is None:
            raise ValueError("UNAVAILABLE_NO_POST_UPDATE")
        bound = accepted_bound(binding or {}, hashes or {})
    matrices = {
        name: np.zeros((4, 4))
        for name in ["total", "shared", "readout", "other", "muon", "adam", "adam_frozen_metric"]
    }
    update = np.zeros(4)
    corrected_update = np.zeros(4)
    residual = 0.0
    moments = []
    for name, index, group in entries:
        parameter = pre["student"][name]
        block = (
            "shared"
            if name.startswith(("gru.", "inp."))
            else "readout"
            if name.startswith("head.")
            else "other"
        )
        family = "muon" if group.get("muon", False) else "adam"
        gradients = [components["component_gradients"][t][name] for t in TERMS]
        if participation is not None and participation[name] != any(
            g is not None for g in gradients
        ):
            raise ValueError("Component and executed update participation differ")
        x = np.stack(
            [
                np.zeros(parameter.numel())
                if g is None
                else g.detach().double().numpy().reshape(-1)
                for g in gradients
            ]
        )
        gram = x @ x.T
        if not np.isfinite(gram).all():
            raise ValueError("Nonfinite component Gram matrix")
        for label in ["total", block, family]:
            matrices[label] += gram
        state = pre["optimizer"]["state"][index]
        if family == "adam":
            beta2 = group["betas"][1]
            v = state["exp_avg_sq"].double().numpy().reshape(-1)
            if np.any(v < 0):
                raise ValueError("Negative Adam second moment")
            diagonal = 1 / (np.sqrt(v / (1 - beta2 ** state["step"])) + group["eps"])
            matrices["adam_frozen_metric"] += (x * diagonal) @ x.T
            memory = state["exp_avg"].double().numpy().reshape(-1)
        else:
            memory = state["momentum_buffer"].double().numpy().reshape(-1)
        moments.append(
            dict(
                name=name,
                family=family,
                moment_norm=float(np.linalg.norm(memory)),
                gradient_dot_memory=(x @ memory).tolist(),
                unused_terms=[TERMS[k] for k, g in enumerate(gradients) if g is None],
            )
        )
        if scientific_update:
            observed = post["preclip"][name]
            original = (
                np.zeros(parameter.numel())
                if observed is None
                else observed.double().numpy().reshape(-1)
            )
            residual = max(residual, float(np.abs(x.sum(0) - original).max()))
            delta = post["parameter_delta"][name].double().numpy().reshape(-1)
            update += x @ delta
            if observed is None and any(g is not None for g in gradients):
                raise ValueError("Unused total parameter has participating component")
            corrected = delta + (
                group["lr"] * group["weight_decay"] * parameter.double().numpy().reshape(-1)
                if participation[name]
                else 0
            )
            corrected_update += x @ corrected
    if scientific_update and (not np.isfinite(residual) or residual > bound):
        raise ValueError("BLOCK_DECOMPOSITION_MISMATCH")
    if (
        any(not np.isfinite(matrix).all() for matrix in matrices.values())
        or not np.isfinite(update).all()
        or not np.isfinite(corrected_update).all()
    ):
        raise ValueError("Nonfinite geometry or update diagnostic")
    h_difference = pre["h"].double() - pre["h_ema"].double()
    summary = {k: summarize(v) for k, v in matrices.items()}
    for row in summary.values():
        row["component_selfgradient_antialignment_uncertified"] = row.pop("local_conflict_witness")
        row["component_selfgradient_sum_norm_uncertified"] = row.pop("total_norm")
        row["component_selfgradient_cancellation_uncertified"] = row.pop("cancellation")
        row["scientific_conflict_witness"] = None
    return dict(
        terms=list(TERMS),
        geometry=summary,
        denominators=denominators(pre, summary["total"]["norms"]),
        moment_diagnostics=moments,
        paired_hidden_rms=float(h_difference.square().mean().sqrt()),
        state_drift_identification="UNIDENTIFIED: different student/EMA weights and histories; same-weight prefix reference absent",
        structural_denominator="UNAVAILABLE: actual subtask normalization must be source-bound separately",
        first_order_loss_changes=update.tolist() if scientific_update else None,
        decay_corrected_first_order=corrected_update.tolist() if scientific_update else None,
        decomposition_max_abs=residual if scientific_update else None,
        decomposition_status="accepted_operation_bound_checked"
        if scientific_update
        else "UNAVAILABLE_NO_CERTIFIED_DECOMPOSITION",
        scientific_update_interpretation=scientific_update,
        decay_removed_parameters=[name for name, active in (participation or {}).items() if active]
        if scientific_update
        else None,
        unused_parameters=[name for name, active in (participation or {}).items() if not active]
        if participation is not None
        else None,
        shared_adam_metric_survival="UNAVAILABLE: existing metric aggregates all Adam parameters",
        actual_nonlinear_component_steps="NOT COMPUTED or attributed",
        evidence="UNCERTIFIED descriptive component self-gradients; not decomposition of actual full update; no scientific conflict witness",
    )


def main() -> None:
    """Require explicit accepted binding before loading future capture files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--post", type=Path)
    parser.add_argument("--accepted-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scientific-update", action="store_true")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Output must stay within own directory")
    binding = json.loads(args.accepted_binding.read_text())
    if (
        binding.get("independent_capture_v4_accepted") is not True
        or binding.get("capture_schema_version") != 4
        or binding["controller_analysis_authorized"] is not True
    ):
        raise ValueError("Independent acceptance and controller authorization required")
    if args.scientific_update:
        accepted_bound(binding, binding["sha256"])
    torch.set_num_threads(1)
    loaded = {}
    hashes = {}
    for name in ["pre", "components", "post"]:
        path = getattr(args, name)
        if path is None:
            continue
        if path.stat().st_size > 900_000_000 or sha(path) != binding["sha256"][name]:
            raise ValueError("Capture size or accepted hash differs")
        hashes[name] = sha(path)
        # These are trusted, explicitly accepted local PyTorch capture payloads.
        loaded[name] = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    result = geometry(
        loaded["pre"],
        loaded["components"],
        binding["canonical_optimizer_id_names"],
        loaded.get("post"),
        binding=binding,
        hashes=hashes,
        scientific_update=args.scientific_update,
    )
    result.update(
        capture_sha256=hashes,
        accepted_binding_sha256=sha(args.accepted_binding),
        prereg_sha256=sha(ROOT / "PREREG.md"),
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
