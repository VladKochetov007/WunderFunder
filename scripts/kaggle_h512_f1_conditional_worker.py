from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kaggle_h512_f1_delivery import fit_and_export
from kaggle_h512_f1_preflight import run_profile_preflight
from kaggle_h512_f1_worker_support import (
    EXPECTED,
    FIT_NAME,
    PROFILE_NAME,
    RECEIPT_PATH,
    SWA_NAME,
    emit,
    write_receipt,
)


def main() -> None:
    """Run the sealed profile gate and stop unless conditional fit is admitted."""
    root = Path.cwd().resolve()
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "wnn.kaggle-h512-f1-conditional-worker-receipt.v1",
        "status": "RUNNING",
        "quality_fit_claim": None,
        "original_manifest_sha256": EXPECTED["original_manifest_sha256"],
        "derived_manifest_sha256": EXPECTED["derived_manifest_sha256"],
        "provider_billing_factor": 1.0,
        "dual_device_allocated_seconds_bound_factor": 2.0,
    }
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite worker evidence: {root / RECEIPT_PATH}")
    try:
        forbidden = [
            root / "artifacts/runs" / name for name in (PROFILE_NAME, FIT_NAME, SWA_NAME)
        ] + [root / "cache" / f"pred_{name}_valid.npy" for name in (FIT_NAME, SWA_NAME)]
        if any(path.exists() for path in forbidden):
            raise FileExistsError("Refusing to overwrite a previous profile, fit, or export")
        state = run_profile_preflight(root, started)
        admission = state["admission"]
        receipt.update(
            input_identity=state["identities"],
            provider_admission={
                key: admission[key]
                for key in (
                    "quota_observed_at_utc",
                    "remaining_unreserved_provider_gpu_seconds",
                    "gpu_reserved_seconds",
                    "quota_refresh_at_utc",
                    "provider_charge_factor",
                    "safety_reserve_seconds",
                    "outer_wall_cap_seconds",
                    "runner_finalizer_reserve_seconds",
                )
            },
            profile=state["records"],
            gate=state["gate"],
            device=state["device"],
            preflight_seconds=state["preflight_seconds"],
            precision_audit={
                "nvidia_tf32_override": "0",
                "flags_after_profile": state["tf32_flags_after_profile"],
            },
        )
        result = fit_and_export(root, started, state)
        receipt.update(result)
        if result["status"] == "CONDITIONAL_FIT_AND_EXPORT_COMPLETE":
            emit("worker_complete", started, status=result["status"], quality_fit_claim=None)
        write_receipt(root, receipt)
    except Exception as error:
        receipt.update(
            status="FAILED",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        if not RECEIPT_PATH.exists():
            write_receipt(root, receipt)
        emit("worker_failed", started, error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
