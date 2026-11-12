"""Small server-only behavior and export gates for PR81 GRU cards08–10."""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

from impact81.receiver_gru_support import (
    ATOL,
    assert_close,
    assert_head_gradients,
    build_probe_model,
    copy_state_dict,
    synthetic_inputs,
    synthetic_targets,
)
from impact81.receiver_gru_support import (
    common_function_and_export_checks as _common_function_checks,
)
from impact81.receiver_gru_support import (
    compare_sequence_state as _same_state,
)
from impact81.receiver_gru_support import (
    initialization_rng_checks as _initial_rng_checks,
)

SEED = 204101
MODEL_CONFIG: dict[str, Any] = {
    "architecture": "gru",
    "hidden": 256,
    "layers": 2,
    "heads": 12,
    "context_rows": 512,
    "dropout": 0.0,
    "input_stem_kernel": 16,
    "hint_dim": 0,
    "precision_mode": "fp32",
}
VARIANTS = ("perturbation_control", "state_perturbation", "nll_transfer", "clip_event")


def _build(variant: str, artifact: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Construct one deterministic probe arm using the frozen native GRU defaults."""
    return build_probe_model(MODEL_CONFIG, variant, SEED, artifact, device)


def _state_perturbation_checks(
    artifact: dict[str, Any], device: torch.device, plain: torch.nn.Module, sample: torch.Tensor
) -> dict[str, object]:
    """Check matched chunk splits, clean carried state, separate RNG restore and eval bypass."""
    from impact81.gru_models import drop_gru_auxiliary_heads
    from impact81.gru_objectives import gru_scaled_loss, perturbation_objective_sums
    from wnn.training.sequence_predict import predict_sequence

    control = _build("perturbation_control", artifact, device).train()
    treatment = _build("state_perturbation", artifact, device).train()
    inputs = synthetic_inputs(2, 4000, device, SEED + 17)
    chunks = inputs.split(1000, dim=1)
    control_state = treatment_state = None
    max_clean_gap = max_state_gap = 0.0
    active_discrepancy = None
    for chunk_index, chunk in enumerate(chunks):
        control_prediction, control_auxiliary, control_state = control(chunk, control_state)
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        treatment_prediction, treatment_auxiliary, treatment_state = treatment(
            chunk, treatment_state
        )
        if not torch.equal(torch.get_rng_state(), cpu_rng_before) or any(
            not torch.equal(actual, expected)
            for actual, expected in zip(
                torch.cuda.get_rng_state_all(), cuda_rng_before, strict=True
            )
        ):
            raise AssertionError("card08 branch consumed the shared CPU or CUDA RNG")
        max_clean_gap = max(
            max_clean_gap,
            assert_close(treatment_prediction, control_prediction, "card08 clean chunk prediction"),
        )
        max_state_gap = max(
            max_state_gap,
            _same_state(treatment_state, control_state, "card08 clean carried state"),
        )
        if control_auxiliary.shape != (2, 0, 2):
            raise AssertionError(
                "card08 perturbation control must return an empty auxiliary tensor"
            )
        if chunk_index < 3:
            if treatment_auxiliary.shape != (2, 0, 2):
                raise AssertionError("card08 noise branch activated before the fourth chunk")
            if float(treatment._perturbation_rng_counter.item()) != 0.0:
                raise AssertionError("card08 private RNG advanced before an eligible chunk")
        else:
            if treatment_auxiliary.shape != (2, 32, 2):
                raise AssertionError("card08 fourth chunk must return exactly 32 comparison rows")
            if float(treatment._perturbation_rng_counter.item()) != 1.0:
                raise AssertionError(
                    "card08 private RNG counter must advance once per active branch"
                )
            active_discrepancy = treatment_auxiliary
    if active_discrepancy is None:
        raise AssertionError("card08 did not execute its first declared perturbation branch")

    mask_suffix = torch.ones((2, 32), dtype=torch.bool, device=device)
    sums = perturbation_objective_sums(active_discrepancy, mask_suffix)
    loss = gru_scaled_loss(sums, sums["count"], 1, "state_perturbation")
    loss.backward()
    if (
        not bool(torch.isfinite(loss.detach()))
        or float(active_discrepancy.detach().abs().max().item()) <= 0
    ):
        raise AssertionError("card08 branch loss or prediction difference is not meaningful")

    saved_state = copy_state_dict(treatment)
    reloaded = _build("state_perturbation", artifact, device)
    reloaded.load_state_dict(saved_state, strict=True)
    reloaded.train()
    treatment_state = reloaded_state = None
    reload_gap = 0.0
    for chunk_index, chunk in enumerate(chunks):
        treatment_prediction, treatment_auxiliary, treatment_state = treatment(
            chunk, treatment_state
        )
        reloaded_prediction, reloaded_auxiliary, reloaded_state = reloaded(chunk, reloaded_state)
        reload_gap = max(
            reload_gap,
            assert_close(treatment_prediction, reloaded_prediction, "card08 RNG-resume prediction"),
        )
        reload_gap = max(
            reload_gap,
            assert_close(treatment_auxiliary, reloaded_auxiliary, "card08 RNG-resume discrepancy"),
        )
        if chunk_index == 3 and float(reloaded._perturbation_rng_counter.item()) != 2.0:
            raise AssertionError("card08 strict reload did not continue the private RNG counter")

    reloaded.eval()
    counter_before_eval = reloaded._perturbation_rng_counter.clone()
    cpu_rng_before_eval = torch.get_rng_state().clone()
    cuda_rng_before_eval = [state.clone() for state in torch.cuda.get_rng_state_all()]
    with torch.no_grad():
        _, clean_hints, _ = reloaded(chunks[0], None)
    if clean_hints.shape != (2, 1000, 0):
        raise AssertionError("card08 evaluation must expose the canonical empty-hint output")
    if not torch.equal(counter_before_eval, reloaded._perturbation_rng_counter):
        raise AssertionError("card08 evaluation consumed its private perturbation stream")
    if not torch.equal(torch.get_rng_state(), cpu_rng_before_eval) or any(
        not torch.equal(actual, expected)
        for actual, expected in zip(
            torch.cuda.get_rng_state_all(), cuda_rng_before_eval, strict=True
        )
    ):
        raise AssertionError("card08 evaluation consumed a global random stream")

    exported = _build("state_perturbation", artifact, device)
    exported.load_state_dict(copy_state_dict(reloaded), strict=True)
    drop_gru_auxiliary_heads(exported)
    export_gap = assert_close(
        torch.from_numpy(predict_sequence(exported, sample, 1000, device)),
        torch.from_numpy(predict_sequence(plain, sample, 1000, device)),
        "card08 trained-state clean export",
    )
    if set(exported.state_dict()) != set(plain.state_dict()):
        raise AssertionError("card08 exported checkpoint retained its perturbation RNG buffer")
    return {
        "matched_split_clean_max_abs": max_clean_gap,
        "clean_state_max_abs": max_state_gap,
        "first_active_chunk_index": 3,
        "active_auxiliary_shape": list(active_discrepancy.shape),
        "active_loss": float(loss.detach().item()),
        "private_counter_after_restore": float(reloaded._perturbation_rng_counter.item()),
        "strict_reload_prediction_and_noise_max_abs": reload_gap,
        "evaluation_rng_nonconsumption": True,
        "clean_export_max_abs": export_gap,
    }


def _objective_checks(artifact: dict[str, Any], device: torch.device) -> dict[str, object]:
    """Check finite treatment-head gradients, masking and exact raw clipping classes."""
    from impact81.gru_objectives import (
        clip_event_objective_sums,
        gru_scaled_loss,
        nll_objective_sums,
    )

    rows = 64
    inputs = synthetic_inputs(2, rows, device, SEED + 23)
    target = synthetic_targets(2, rows, device, SEED + 29)
    mask = torch.arange(rows, device=device).unsqueeze(0).expand(2, rows) >= 8

    nll_model = _build("nll_transfer", artifact, device).train()
    nll_prediction, log_variance, _ = nll_model(inputs, None)
    if not torch.equal(log_variance, torch.zeros_like(log_variance)):
        raise AssertionError("card09 log variance head must start at exactly zero")
    nll_sums = nll_objective_sums(nll_prediction, log_variance, target, mask)
    nll_loss = gru_scaled_loss(nll_sums, nll_sums["count"], 1, "nll_transfer")
    nll_loss.backward()

    nll_gradients = assert_head_gradients(nll_model.log_variance_head, "card09 log variance")
    nll_mean_gradients = assert_head_gradients(nll_model.head, "card09 mean head")

    event_model = _build("clip_event", artifact, device).train()
    _, logits, _ = event_model(inputs, None)
    event_sums = clip_event_objective_sums(logits, target, mask)
    event_loss = gru_scaled_loss(event_sums, event_sums["count"], 1, "clip_event")
    event_loss.backward()
    event_gradients = assert_head_gradients(event_model.clip_event_heads, "card10 event")

    boundaries = torch.tensor(
        [
            [
                [-2.0001, 2.0001],
                [-2.0, 2.0],
                [-1.9999, 1.9999],
                [1.9999, -1.9999],
                [2.0, -2.0],
                [2.0001, -2.0001],
                [float("nan"), float("nan")],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    boundary_mask = torch.tensor([[True, True, True, True, True, True, False]], device=device)
    boundary_logits = torch.zeros((1, 7, 2, 3), dtype=torch.float32, device=device)
    expected_classes = torch.tensor([[0, 2], [1, 1], [1, 1], [1, 1], [1, 1], [2, 0]])
    for row, classes in enumerate(expected_classes.tolist()):
        for target_index, class_index in enumerate(classes):
            boundary_logits[0, row, target_index, class_index] = 4.0
    boundary_logits[0, 6] = float("nan")
    boundary_sums = clip_event_objective_sums(boundary_logits, boundaries, boundary_mask)
    expected_confusion = torch.tensor(
        [[[1, 0, 0], [0, 4, 0], [0, 0, 1]], [[1, 0, 0], [0, 4, 0], [0, 0, 1]]],
        dtype=torch.int64,
        device=device,
    )
    if not torch.equal(boundary_sums["confusion"], expected_confusion):
        raise AssertionError("card10 raw boundaries must map to the inclusive center class")
    if not bool(torch.isfinite(boundary_sums["loss_numerator"]).all()):
        raise AssertionError("card10 masked nonfinite rows contaminated CE")

    return {
        "nll_loss": float(nll_loss.detach().item()),
        "nll_head_gradient_norms": nll_gradients,
        "nll_mean_gradient_norms": nll_mean_gradients,
        "event_ce_loss": float(event_loss.detach().item()),
        "event_head_gradient_norms": event_gradients,
        "raw_boundary_confusion": boundary_sums["confusion"].cpu().tolist(),
        "masked_nonfinite_rows_excluded": True,
    }


def run_gru_checks(
    artifact: dict[str, Any], artifact_sha256: str, device: torch.device
) -> dict[str, object]:
    """Run bounded GRU state, objective, strict-reload and clean-export probes.

    This check runs only on a Linux CUDA receiver, uses synthetic data, and
    produces engineering evidence. It performs no fit, official-data score or
    competition-quality comparison.
    """
    if sys.platform != "linux" or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PR81 GRU receiver checks require Linux CUDA")
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") not in (None, "0"):
        raise RuntimeError("FP32 GRU receiver checks require CUBLAS emulation to be disabled")
    if len(artifact_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in artifact_sha256):
        raise ValueError("GRU preprocessing SHA-256 must be lowercase hexadecimal")
    device_name = torch.cuda.get_device_name(device)
    from wnn.training.sequence_precision import configure_precision_mode

    configure_precision_mode("fp32")
    rng_checks = _initial_rng_checks(MODEL_CONFIG, VARIANTS, SEED)
    common_checks, plain, sample, _ = _common_function_checks(
        MODEL_CONFIG, VARIANTS, SEED, artifact, device
    )
    state_checks = _state_perturbation_checks(artifact, device, plain, sample)
    objective_checks = _objective_checks(artifact, device)
    torch.cuda.synchronize(device)
    return {
        "status": "passed",
        "scope": "synthetic K16 GRU receiver contract only; no fit or quality score",
        "device": device_name,
        "precision_mode": "fp32",
        "preprocessing_sha256": artifact_sha256,
        "model_config": MODEL_CONFIG,
        "initialization_rng": rng_checks,
        "common_function_and_export": common_checks,
        "card08_state_perturbation": state_checks,
        "card09_card10_objectives": objective_checks,
        "tolerance_atol": ATOL,
        "quality_score": None,
    }
