"""FP32 GRU receiver admission for the B300-only cards08–10 handoff."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from impact81.bindings import SPLIT_SHA256, digest, source_identity, write_new
from impact81.preparation_later import GRU_RUNTIME_COMMIT
from impact81.preprocessing import load_preprocessing
from impact81.receiver_loop_checks import run_loop_checks


def run_checks(preprocessing_path: Path, output: Path, runtime_root: Path) -> None:
    """Write evidence only after actual GRU mechanism and control-loop checks pass."""
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("GRU receiver checks require an actual Linux B300")
    if output.exists() or output.is_symlink():
        raise FileExistsError("a receiver receipt cannot overwrite a previous attempt")
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") not in (None, "0"):
        raise RuntimeError("GRU recipe requires cuBLAS FP32 emulation disabled")
    device = torch.device("cuda", torch.cuda.current_device())
    name = torch.cuda.get_device_name(device)
    if "B300" not in name.upper():
        raise RuntimeError("GRU receiver must run on the declared B300 hardware")
    artifact = load_preprocessing(preprocessing_path)
    if (
        artifact["schema"] != "impact81.gru-preprocessing.v1"
        or artifact["fit_identity"]["split_sha256"] != SPLIT_SHA256
    ):
        raise ValueError("GRU check requires the pinned train-only GRU artifact")
    from impact81.receiver_gru_checks import run_gru_checks

    artifact_sha = digest(preprocessing_path)
    mechanisms = run_gru_checks(artifact, artifact_sha, device)
    control = run_loop_checks(artifact, artifact_sha, device, family="gru")
    signed_objective = _signed_profile_check()
    torch.cuda.synchronize(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(
        output,
        {
            "schema": "impact81.receiver-checks.v1",
            "status": "passed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "scope": "small synthetic engineering updates; no full profile or scientific fit",
            "runtime_commit": GRU_RUNTIME_COMMIT,
            "source_identity": source_identity(runtime_root),
            "preprocessing_sha256": artifact_sha,
            "precision_mode": "fp32",
            "extension_cards": ["08", "09", "10"],
            "device": {
                "name": name,
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "gru_checks": mechanisms,
            "control_engine_parity": control,
            "signed_profile_contract": signed_objective,
            "clip_event_feasibility": artifact["clip_event_feasibility"],
            "card10_feasible": artifact["clip_event_feasibility"]["feasible"],
            "scientific_fit_approved": False,
            "card10_profile_and_fit_require_feasible": True,
        },
    )


def _signed_profile_check() -> dict[str, bool]:
    """Exercise a negative synthetic NLL total inside the server-only receiver gate."""
    from impact81.profile_objective_admission import validate_profile_objective

    config = {"impact81": {"cell": {"card": "09", "arm": "treatment", "variant": "nll_transfer"}}}
    receipt = {
        "losses_total_scope": "primary_only_signed_objective_in_terms",
        "losses": {"total": 0.01},
        "objective_terms": {
            "primary_loss": 0.01,
            "weighted_auxiliary_loss": -0.10,
            "total_loss": -0.09,
        },
    }
    validate_profile_objective(receipt, config)
    inconsistent = receipt | {"objective_terms": dict(receipt["objective_terms"], total_loss=0.0)}
    try:
        validate_profile_objective(inconsistent, config)
    except ValueError:
        return {"synthetic_negative_total_accepted": True, "altered_total_rejected": True}
    raise AssertionError("signed NLL profile accepted an altered combined objective")
