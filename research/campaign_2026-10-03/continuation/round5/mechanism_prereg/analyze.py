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
    if pre["version"] != 2 or pre["phase"] != "pre_forward_pre_zero_grad":
        raise ValueError("Unsupported capture phase/version")
    if pre["force_skip"]:
        raise ValueError("Forced-skip capture excluded from scientific geometry")
    if set(components["component_gradients"]) != set(TERMS):
        raise ValueError("Exactly four actual weighted loss terms required")
    declared = {str(k): v for k, v in pre["optimizer_id_names"].items()}
    if declared != canonical:
        raise ValueError("Canonical independently accepted ID/name map differs")
    entries = []
    for group, names in zip(pre["optimizer"]["param_groups"], pre["optimizer_names"], strict=True):
        for index, name in zip(group["params"], names, strict=True):
            if canonical[str(index)] != name:
                raise ValueError("Serialized ID/name association differs")
            parameter = pre["student"][name]
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
    return entries


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
    """Recover three exact positive-divisor branches, not structural subtasks."""
    start = pre["metadata"]["s"]
    window = slice(start, start + pre["args"]["chunk"])
    batch = pre["batch"]
    y = batch["y"][:, window].float().clamp(-2, 2)
    valid = batch["main_valid"][:, window]
    main = y.abs() * valid[..., None]
    ema = valid.float()[..., None]
    if pre["args"].get("ema_metric_weight", False):
        ema = ema * y.abs()
    weights = {"main": main, "ema": ema, "next": batch["nw"][:, window]}
    output = {}
    for term, weight in weights.items():
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Invalid loss weights")
        raw = float(weight.sum())
        divisor = float(weight.sum().clamp_min(1e-8))
        squares = float(weight.double().square().sum())
        output[term] = dict(
            raw=raw,
            clamped=divisor,
            weighted_effective_count=raw**2 / squares if squares else 0,
            numerator_equivalent_gradient_norm=divisor * norms[TERMS.index(term)],
        )
    output["struct"] = "UNAVAILABLE: independently bind structural subtask normalizers"
    output["coefficients_already_in_gradients"] = dict(
        main=1,
        ema=pre["args"]["ema_teacher"] * (2 if pre["args"].get("ema_metric_weight", False) else 1),
        struct=pre["args"]["aux_struct"],
        next=pre["args"]["aux_next"],
    )
    return output


def geometry(
    pre: Payload, components: Payload, canonical: dict[str, str], post: Payload | None = None
) -> Payload:
    """Analyze captured tensors only; no forward, optimizer step or CUDA call."""
    entries = validate(pre, components, canonical)
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
        x = np.stack(
            [
                np.zeros(parameter.numel())
                if g is None
                else g.detach().double().numpy().reshape(-1)
                for g in gradients
            ]
        )
        gram = x @ x.T
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
        if post is not None:
            observed = post["preclip"][name]
            original = (
                np.zeros(parameter.numel())
                if observed is None
                else observed.double().numpy().reshape(-1)
            )
            residual = max(residual, float(np.abs(x.sum(0) - original).max()))
            delta = post["parameter_delta"][name].double().numpy().reshape(-1)
            update += x @ delta
            corrected = delta + group["lr"] * group.get(
                "weight_decay", 0
            ) * parameter.double().numpy().reshape(-1)
            corrected_update += x @ corrected
    h_difference = pre["h"].double() - pre["h_ema"].double()
    summary = {k: summarize(v) for k, v in matrices.items()}
    return dict(
        terms=list(TERMS),
        geometry=summary,
        denominators=denominators(pre, summary["total"]["norms"]),
        moment_diagnostics=moments,
        paired_hidden_rms=float(h_difference.square().mean().sqrt()),
        state_drift_identification="UNIDENTIFIED: different student/EMA weights and histories; same-weight prefix reference absent",
        structural_denominator="UNAVAILABLE: actual subtask normalization must be source-bound separately",
        first_order_loss_changes=update.tolist() if post is not None else None,
        decay_corrected_first_order=corrected_update.tolist() if post is not None else None,
        decomposition_max_abs=residual if post is not None else None,
        actual_nonlinear_component_steps="NOT COMPUTED or attributed",
        evidence="local descriptive captured geometry only; no score or stochastic-causality inference",
    )


def main() -> None:
    """Require explicit accepted binding before loading future capture files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--post", type=Path)
    parser.add_argument("--accepted-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Output must stay within own directory")
    binding = json.loads(args.accepted_binding.read_text())
    if (
        binding["independent_capture_v2_accepted"] is not True
        or binding["controller_analysis_authorized"] is not True
    ):
        raise ValueError("Independent acceptance and controller authorization required")
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
    )
    if (
        args.post is not None
        and result["decomposition_max_abs"] > binding["decomposition_max_abs_bound"]
    ):
        raise ValueError("Component-sum bound failed; scientific interpretation blocked")
    result.update(
        capture_sha256=hashes,
        accepted_binding_sha256=sha(args.accepted_binding),
        prereg_sha256=sha(ROOT / "PREREG.md"),
    )
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
