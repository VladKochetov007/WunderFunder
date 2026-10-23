"""Grade three frozen complete TL3 endpoints on the same exposed F0 groups."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# Score reports are versioned JSON records with endpoint-specific nested fields.
from endpoint_contract import SCRATCH_ROOT, physical_scratch, read_json, sha256_file
from endpoint_score_inputs import load_screen_inputs

PACKET = Path("research/campaign_2026-10-20/submission_delivery_r1/tl3_h512_endpoint_screen_r1")
PLAN_SHA256 = "852f64507fb9d96caa6ad7f8e82f7b7cc00a02eb7a046d7627d56eb42709bc94"


def _compare(
    targets: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """Compute one fixed paired component bootstrap with canonical arithmetic."""
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments

    candidate_moments, reference_moments = paired_component_moments(
        targets, candidate, reference, mask
    )
    return {
        "comparison": label,
        "paired_component_bootstrap": paired_component_bootstrap(
            candidate_moments,
            reference_moments,
            components,
            n_boot=2_000,
            seed=271_828,
            centered=True,
        ),
    }


def grade(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Score only the three fixed full endpoints on the exposed F0 panel."""
    from wnn.gmetric import global_wp

    values = load_screen_inputs(root.resolve(strict=True), plan)
    targets = values["targets"]
    mask = values["mask"]
    components = values["components"]
    baseline = values["baseline"]
    candidates = values["candidates"]
    scores = {
        "original_tl3": global_wp(targets, baseline, mask),
        "h512_cp5_replacement": global_wp(targets, candidates["cp5"], mask),
        "h512_swa345_replacement": global_wp(targets, candidates["swa345"], mask),
    }
    comparisons = [
        _compare(
            targets,
            mask,
            components,
            candidates["cp5"],
            baseline,
            "h512_cp5_replacement - original_tl3",
        ),
        _compare(
            targets,
            mask,
            components,
            candidates["swa345"],
            baseline,
            "h512_swa345_replacement - original_tl3",
        ),
        _compare(
            targets,
            mask,
            components,
            candidates["swa345"],
            candidates["cp5"],
            "h512_swa345_replacement - h512_cp5_replacement",
        ),
    ]
    return {
        "schema": "wnn.h512-tl3-endpoint-screen-grade.v1",
        "status": "GRADED_F0_FITTED_DIAGNOSTIC",
        "plan_sha256": PLAN_SHA256,
        "official_metric": "global_wp_clip2_v1",
        "target_identity": values["target_identity"],
        "endpoint_scores": scores,
        "fixed_comparisons": comparisons,
        "endpoint_sources": values["source_receipts"],
        "bootstrap": {"unit": "connected component", "n_boot": 2_000, "seed": 271_828},
        "selection": "Three predeclared endpoints only; no coefficient, epoch, or member search.",
        "evidence_limit": "F0 is exposed. The two retained TL3 members have unknown ancestry; all complete ensemble deltas are fitted diagnostics, not clean CV evidence.",
    }


def main() -> None:
    """Write the fixed endpoint grade as create-only JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    plan_path = root / PACKET / "ENDPOINT_SCREEN_PLAN_R1.json"
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("endpoint screen plan bytes differ from frozen plan")
    plan = read_json(plan_path)
    output = physical_scratch(args.output)
    if output.parent != SCRATCH_ROOT.resolve(strict=True) or output.exists():
        raise ValueError("grade output must be a new direct child of physical research scratch")
    started = time.monotonic()
    report = grade(root, plan)
    report["grade_script_sha256"] = sha256_file(Path(__file__).resolve())
    report["input_helper_sha256"] = sha256_file(
        Path(__file__).with_name("endpoint_score_inputs.py")
    )
    report["contract_sha256"] = sha256_file(Path(__file__).with_name("endpoint_contract.py"))
    report["wall_seconds"] = time.monotonic() - started
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "scores": report["endpoint_scores"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
