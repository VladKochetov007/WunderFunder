"""Run the bounded feature-only H512 native-precision diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import probe_config as config


def prelabel_checks(
    out: Path,
    selection: dict[str, Any],
    plan_sha: str,
    arrays: dict[str, np.ndarray],
    needs: list[np.ndarray],
    original: tuple[np.ndarray, dict[str, Any]],
    cp5: tuple[np.ndarray, dict[str, Any]],
    native_sha: str,
    backend: str,
    native_level: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Check native parity and Torch identity before mapping any targets."""
    from probe_native import cache_rows, digest_file, max_abs, require_hash

    selected = selection["members"]
    old_expected = cache_rows(original[0], original[1], selected)
    cp5_expected = cache_rows(cp5[0], cp5[1], selected)
    old_reconstructed = 0.34 * arrays["original_student"].astype(np.float64) + arrays[
        "retained_pair"
    ].astype(np.float64)
    cp5_reconstructed = 0.34 * arrays["h512_dynamic_int8"].astype(np.float64) + arrays[
        "retained_pair"
    ].astype(np.float64)
    parity = {
        "original_tl3_max_abs": max_abs(old_reconstructed, old_expected),
        "h512_cp5_max_abs": max_abs(cp5_reconstructed, cp5_expected),
        "tolerance": 1e-5,
    }
    parity["passed"] = max(parity["original_tl3_max_abs"], parity["h512_cp5_max_abs"]) <= 1e-5
    require_hash(config.TORCH_CACHE, config.TORCH_CACHE_SHA256)
    torch_sidecar_path = config.TORCH_CACHE.with_suffix(".json")
    torch_sidecar = json.loads(torch_sidecar_path.read_text(encoding="utf-8"))
    torch_values = np.load(config.TORCH_CACHE, mmap_mode="r", allow_pickle=False)
    groups = list(map(int, torch_sidecar.get("row_groups", [])))
    if (
        torch_values.shape != (1873, 20_000, 2)
        or torch_values.dtype != np.float32
        or groups != list(range(1873))
    ):
        raise ValueError("Torch H512 cache does not have full-valid float32 coverage")
    if (
        torch_sidecar.get("run") != "r01_h512_f0_s1003_local_cp5_r1"
        or torch_sidecar.get("split") != "valid"
        or torch_sidecar.get("epoch") != 5
        or torch_sidecar.get("split_manifest_sha256")
        != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
        or torch_sidecar.get("checkpoint_sha256") != config.TORCH_CHECKPOINT_SHA256
        or torch_sidecar.get("prediction_sha256") != config.TORCH_CACHE_SHA256
        or torch_sidecar.get("inference_precision") != "fp32"
        or torch_sidecar.get("feature_only") is not True
    ):
        raise ValueError("Torch H512 cache run/checkpoint/inference provenance changed")
    torch_rows = cache_rows(torch_values, {"row_groups": groups}, selected)
    active = np.broadcast_to(np.stack(needs)[:, :, None], torch_rows.shape)
    torch_rms = {
        "native_float32_fast": float(
            np.sqrt(
                np.mean(
                    (arrays["h512_float32_fast"][active] - torch_rows[active]).astype(np.float64)
                    ** 2
                )
            )
        ),
        "native_dynamic_int8": float(
            np.sqrt(
                np.mean(
                    (arrays["h512_dynamic_int8"][active] - torch_rows[active]).astype(np.float64)
                    ** 2
                )
            )
        ),
    }
    precision_rms = {
        "float32_fast_vs_dynamic_int8": float(
            np.sqrt(
                np.mean(
                    (
                        arrays["h512_float32_fast"][active] - arrays["h512_dynamic_int8"][active]
                    ).astype(np.float64)
                    ** 2
                )
            )
        ),
        "float32_exact_vs_dynamic_int8": float(
            np.sqrt(
                np.mean(
                    (
                        arrays["h512_float32_exact"][active] - arrays["h512_dynamic_int8"][active]
                    ).astype(np.float64)
                    ** 2
                )
            )
        ),
    }
    result = {
        "schema": "wnn.native-h512-precision-probe-prelabel.v1",
        "status": "PRELABEL_VALIDATION",
        "selection_plan_sha256": plan_sha,
        "membership": selected,
        "cache_sha256": {
            "original_tl3": config.ORIGINAL_CACHE_SHA256,
            "h512_cp5_native": config.CP5_CACHE_SHA256,
            "h512_torch": config.TORCH_CACHE_SHA256,
            "h512_torch_sidecar": digest_file(torch_sidecar_path),
        },
        "cache_parity": parity,
        "torch_h512_rms_difference_on_need_rows": torch_rms,
        "native_precision_rms_difference_on_need_rows": precision_rms,
        "native_level": native_level,
        "native_extension_sha256": native_sha,
        "quant_backend": backend,
        "target_accessed": False,
    }
    config.write_json(out / "prelabel_validation.json", result)
    return result, cp5_expected.astype(np.float32), old_reconstructed.astype(np.float32)


def score_after_parity(
    out: Path,
    repo: Path,
    selection: dict[str, Any],
    needs: list[np.ndarray],
    arrays: dict[str, np.ndarray],
    cp5_expected: np.ndarray,
    old_reconstructed: np.ndarray,
    prelabel: dict[str, Any],
    native_sha: str,
    backend: str,
    plan_sha: str,
) -> dict[str, Any]:
    """Map frozen targets after saved outputs and successful native parity."""
    from probe_grade import score_outputs
    from probe_native import digest_file, require_hash

    require_hash(config.TARGET_CACHE, config.TARGET_CACHE_SHA256)
    sys.path.insert(0, str(repo / "research/campaign_2026-09-30/audit"))
    from campaign_score import mmap_stored_npz

    targets = mmap_stored_npz(config.TARGET_CACHE)
    if (
        targets["y"].shape != (1873, 20_000, 2)
        or targets["mask"].shape != (1873, 20_000)
        or set(map(int, targets["row_group"])) != set(range(1873))
    ):
        raise ValueError("target cache lacks full-valid unique row coverage")
    derived = {
        "original_tl3_reconstructed": old_reconstructed,
        "h512_cp5_cached_native": cp5_expected,
        "h512_dynamic_int8": arrays["h512_dynamic_int8"],
        "h512_float32_fast": arrays["h512_float32_fast"],
        "h512_float32_exact": arrays["h512_float32_exact"],
        "original_student": arrays["original_student"],
        "retained_pair": arrays["retained_pair"],
        "h512_cp5_counterfactual_native": (
            cp5_expected.astype(np.float64)
            + 0.34
            * (
                arrays["h512_float32_fast"].astype(np.float64)
                - arrays["h512_dynamic_int8"].astype(np.float64)
            )
        ).astype(np.float32),
    }
    result = score_outputs(targets, selection["members"], needs, derived)
    result.update(
        {
            "schema": "wnn.native-h512-precision-probe-result.v1",
            "status": "SCORED_EIGHT_GROUP_F0_FITTED_DIAGNOSTIC",
            "evidence_limit": "Eight exposed F0 evaluation groups from distinct feature-only components; no uncertainty estimate or promotion claim.",
            "selection_plan_sha256": plan_sha,
            "selected_membership_sha256": selection["selected_membership_sha256"],
            "target_cache_sha256": config.TARGET_CACHE_SHA256,
            "cache_parity": prelabel["cache_parity"],
            "torch_h512_rms_difference_on_need_rows": prelabel[
                "torch_h512_rms_difference_on_need_rows"
            ],
            "native_precision_rms_difference_on_need_rows": prelabel[
                "native_precision_rms_difference_on_need_rows"
            ],
            "counterfactual_formula": "saved_native_CP5 + 0.34 * (H512_native_float32_fast - H512_native_dynamic_int8)",
            "native_level": "v3",
            "native_extension_sha256": native_sha,
            "quant_backend": backend,
            "output_npz_sha256": digest_file(out / "member_predictions.npz"),
        }
    )
    config.write_json(out / "precision_probe_result.json", result)
    return result


def run_probe(repo: Path, out: Path, selection: dict[str, Any], plan_sha: str) -> dict[str, Any]:
    """Run inference, save outputs, gate on parity, then score the fixed probe."""
    from probe_native import run_features
    from probe_prepare import (
        checked_sources,
        extract_packages,
        load_probe_native,
        make_models,
        save_outputs,
    )

    original_cache = checked_sources()
    sys.path.insert(0, str(config.REPLAY))
    sys.path.insert(0, str(repo / "src"))
    from f0_source import validate_f0_source

    source_all = validate_f0_source(
        repo, repo / "research/campaign_2026-09-29/artifacts/splits/fold0.json", None
    )
    selected = selection["members"]
    if any(int(item["row_group"]) not in source_all.groups for item in selected):
        raise ValueError("selection contains a non-evaluation F0 group")
    source = source_all
    with tempfile.TemporaryDirectory(
        prefix="native-h512-probe-", dir=config.SCRATCH
    ) as scratch_dir:
        old_package, cp5_package = extract_packages(Path(scratch_dir))
        module, native, native_sha, backend = load_probe_native(old_package, cp5_package)
        arrays, needs = run_features(
            source, selected, make_models(native, old_package, cp5_package)
        )
        save_outputs(out, arrays, plan_sha, selection, native_sha, backend)
        prelabel, cp5_expected, old_reconstructed = prelabel_checks(
            out,
            selection,
            plan_sha,
            arrays,
            needs,
            original_cache[:2],
            original_cache[2:],
            native_sha,
            backend,
            str(module.LEVEL),
        )
        if not prelabel["cache_parity"]["passed"]:
            raise RuntimeError("native construction parity failed; targets were not accessed")
    return score_after_parity(
        out,
        repo,
        selection,
        needs,
        arrays,
        cp5_expected,
        old_reconstructed,
        prelabel,
        native_sha,
        backend,
        plan_sha,
    )


def main() -> None:
    """Execute a reviewed fixed selection into its create-only scratch directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=config.ROOT)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo_root.resolve(), args.output_dir.resolve()
    plan_path = args.selection_plan.resolve()
    selection = json.loads(plan_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(__file__).parent))
    from prepare_selection import make_plan
    from probe_native import digest_file

    if selection != make_plan(repo) or selection.get("status") != "PREPARED_LABEL_FREE_SELECTION":
        raise ValueError("selection plan differs from the frozen deterministic recipe")
    if plan_path != out / "selection_plan.json":
        raise ValueError("selection plan must be the plan stored in the output directory")
    if {path.name for path in out.iterdir()} != {"selection_plan.json"}:
        raise FileExistsError("output directory must contain only the reviewed selection plan")
    if out.parent != config.SCRATCH.resolve():
        raise ValueError("probe outputs must be directly under approved physical scratch")
    result = run_probe(repo, out, selection, digest_file(plan_path))
    print(
        json.dumps({"status": result["status"], "result": str(out / "precision_probe_result.json")})
    )


if __name__ == "__main__":
    main()
