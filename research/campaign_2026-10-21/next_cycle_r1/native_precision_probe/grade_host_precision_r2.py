"""Gate saved eight-group tracks against the current-host TL3 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import probe_config as config
from grade_host_reference import (
    ADMISSION,
    ADMISSION_SHA,
    BINDING_SHA,
    MEMBERSHIP_SHA,
    NATIVE_GRADE_SHA,
    SAMPLE_DIR,
    SAMPLE_PINS,
    frozen_sources,
    host_reference,
    need,
    read_json,
    sampled_parity,
    sha,
)

OUTPUT_NAME = "native-h512-precision-host-reference-r2-20261022"
HOST_GRADE_SHA = "892cbf3a9092e9d73cfe751c75d5ba0e10cbbac5766d0f5bc70e159e9b3d0134"
HOST_PREDICTION_SHA = "9e4e220f517d7b5c5cd36fdac7c6876a9ce98c097d8ed1d7c23549ca211e41db"
HOST_SIDECAR_SHA = "d3f4850f24e0c67eef9be649a7ddafb9ec29c83617562ab9225a0439b386b161"
HOST_REPLAY_SHA = "20ed2f9d62400147e8b6960a128d7a3dc9dfac095085665100c4aab462aebcfe"
OLD_R1_PRELABEL_SHA = "530c89c6c4afbe5a15d78177b722b54a79310231b55f0308af07ce7df876a87e"
OLD_R2_PRELABEL_SHA = SAMPLE_PINS["original_endpoint_prelabel_validation.json"]
OLD_NPZ_SHA = "3c254fd21f0c41d8ee45aeb0fdbbd38a7d3c996554304053db733887fd80cd98"
OLD_MANIFEST_SHA = SAMPLE_PINS["outputs_manifest.json"]
TORCH_SIDECAR_SHA = "f9c41bb2264bbd519b0c1f3df42ed3826a388c177d0e159fa2f35ad13ae17a4c"
TOLERANCE = 1e-5


def check_historical_failures() -> tuple[dict[str, Any], dict[str, Any]]:
    """Pin the failed legacy-reference attempts as historical, target-free evidence."""
    r1_path = SAMPLE_DIR / "prelabel_validation.json"
    r2_path = SAMPLE_DIR / "original_endpoint_prelabel_validation.json"
    need(sha(r1_path) == OLD_R1_PRELABEL_SHA, "historical R1 receipt changed")
    need(sha(r2_path) == OLD_R2_PRELABEL_SHA, "historical R2 receipt changed")
    r1, r2 = read_json(r1_path), read_json(r2_path)
    need(r1.get("target_accessed") is False, "historical R1 opened targets")
    need(r1.get("cache_parity", {}).get("passed") is False, "historical R1 failure changed")
    need(r1["cache_parity"].get("h512_cp5_max_abs") <= TOLERANCE, "historical CP5 gate changed")
    need(r1["cache_parity"].get("original_tl3_max_abs") > TOLERANCE, "historical R1 gap changed")
    need(r2.get("target_accessed") is False, "historical R2 opened targets")
    need(r2.get("status") == "PRELABEL_VALIDATION_FAILED", "historical R2 status changed")
    need(r2.get("cache_parity", {}).get("passed") is False, "historical R2 failure changed")
    need(
        r2["cache_parity"].get("h512_cp5_saved_tracks_max_abs") <= TOLERANCE
        and r2["cache_parity"].get("original_exact_callback_max_abs") > TOLERANCE,
        "historical R2 parity values changed",
    )
    return r1, r2


def check_masks(
    source: Any, members: list[dict[str, Any]]
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Read only need_prediction and input features for the frozen eight groups."""
    from f0_replay import _read_features

    needs: list[np.ndarray] = []
    packed: list[bytes] = []
    warmup_rows = predicted_rows = 0
    selected_receipts = []
    expected = np.ones(source.sequence_length, dtype=np.bool_)
    expected[:99] = False
    for member in members:
        group, sequence_id = int(member["row_group"]), int(member["seq_ix"])
        need(group in source.groups, f"selected row group is outside F0: {group}")
        need(
            source.sequence_by_group.get(group) == sequence_id,
            f"selected sequence differs: {group}",
        )
        table = source.parquet.read_row_group(
            group, columns=list(source.columns), use_threads=False
        )
        features, mask = _read_features(table, source, sequence_id)
        del table, features
        need(
            mask.shape == (20_000,) and mask.dtype == np.bool_,
            f"need mask shape or dtype differs: {group}",
        )
        need(np.array_equal(mask, expected), f"warmup or prediction boundary differs: {group}")
        needs.append(mask.copy())
        packed.append(np.packbits(mask, bitorder="little").tobytes())
        warmup_rows += int((~mask).sum())
        predicted_rows += int(mask.sum())
        selected_receipts.append({"row_group": group, "sequence_id": sequence_id})
    need(warmup_rows == 792 and predicted_rows == 159_208, "eight-group need counts differ")
    return needs, {
        "groups_in_order": selected_receipts,
        "warmup_rows": warmup_rows,
        "need_prediction_rows": predicted_rows,
        "need_prediction_packed_little_sha256": hashlib.sha256(b"".join(packed)).hexdigest(),
        "columns_read": list(source.columns),
        "scoring_mask_or_targets_read": False,
    }


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    """Run target-free gates, then call the unchanged fixed scorer on success."""
    from prepare_selection import make_plan
    from probe_native import cache_rows, digest_file, max_abs, require_hash
    from probe_prepare import checked_sources
    from rerun_original_endpoint import load_and_check_saved_tracks
    from run_precision_probe import score_after_parity

    need(
        output_dir.parent.resolve() == config.SCRATCH.resolve(),
        "output must be under physical scratch",
    )
    need(output_dir.name == OUTPUT_NAME, "output path differs from the create-only R2 path")
    need(not output_dir.exists() and not output_dir.is_symlink(), "refusing to replace R2 output")

    frozen_sources(root)
    admission_path = root / ADMISSION
    need(sha(admission_path) == ADMISSION_SHA, "host replay admission changed")
    admission = read_json(admission_path)
    need(
        admission.get("status") == "ADMITTED_ONE_FULL_CURRENT_HOST_REFERENCE_REPLAY",
        "host replay is not admitted",
    )
    repo_grade_path = root / "research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json"
    scratch_grade_path = config.SCRATCH / "tl3_original_host_f0_grade_r1.json"
    need(sha(repo_grade_path) == HOST_GRADE_SHA, "repository host grade changed")
    need(sha(scratch_grade_path) == HOST_GRADE_SHA, "scratch host grade changed")
    host_grade = read_json(repo_grade_path)
    host_record = host_grade["host_reference"]
    need(
        host_grade.get("status") == "GRADED_F0_EXPOSED_FITTED_DIAGNOSTIC",
        "host grade status differs",
    )
    need(
        host_grade.get("source_binding_sha256") == BINDING_SHA, "host grade source binding differs"
    )
    need(host_grade.get("host_admission_sha256") == ADMISSION_SHA, "host grade admission differs")
    need(
        host_grade.get("endpoint_screen_plan_sha256")
        == "852f64507fb9d96caa6ad7f8e82f7b7cc00a02eb7a046d7627d56eb42709bc94",
        "host endpoint-screen plan differs",
    )
    need(host_grade.get("old_grade_sha256") == NATIVE_GRADE_SHA, "host grade old-grade pin differs")
    need(host_record.get("prediction_sha256") == HOST_PREDICTION_SHA, "host cache pin differs")
    need(host_record.get("sidecar_sha256") == HOST_SIDECAR_SHA, "host sidecar pin differs")
    need(host_record.get("receipt_sha256") == HOST_REPLAY_SHA, "host replay pin differs")
    need(
        host_grade.get("prelabel_sample_parity", {}).get("passed") is True,
        "host sample parity failed",
    )
    need(
        host_grade["prelabel_sample_parity"].get("membership_sha256") == MEMBERSHIP_SHA,
        "host sample membership differs",
    )

    selection_path = SAMPLE_DIR / "selection_plan.json"
    selection = read_json(selection_path)
    plan_sha = sha(selection_path)
    need(plan_sha == SAMPLE_PINS["selection_plan.json"], "fixed selection plan changed")
    need(selection == make_plan(root), "selection differs from deterministic label-free plan")
    members = selection["members"]
    need(selection.get("selected_membership_sha256") == MEMBERSHIP_SHA, "membership hash differs")
    need(
        len(members) == 8 and len({int(row["component"]) for row in members}) == 8,
        "selection is not eight components",
    )

    old_r1, old_r2 = check_historical_failures()
    tracks_manifest = SAMPLE_DIR / "outputs_manifest.json"
    tracks_path = SAMPLE_DIR / "member_predictions.npz"
    need(sha(tracks_manifest) == OLD_MANIFEST_SHA, "saved-output manifest changed")
    need(digest_file(tracks_path) == OLD_NPZ_SHA, "saved five-track NPZ changed")
    track_manifest = read_json(tracks_manifest)
    _, _, cp5_cache, cp5_sidecar = checked_sources()
    native_sha = str(old_r2.get("native_extension_sha256"))
    need(
        native_sha == "3cb4546229e80cca81cda9cdec1b9c6146b380271077e26f9324d8a4b6ecad9a",
        "native extension pin differs",
    )
    arrays = load_and_check_saved_tracks(SAMPLE_DIR, selection, plan_sha, native_sha)
    need(
        set(arrays)
        == {
            "h512_dynamic_int8",
            "h512_float32_fast",
            "h512_float32_exact",
            "original_student",
            "retained_pair",
        },
        "saved track roster differs",
    )

    sys.path.insert(0, str(config.REPLAY))
    from f0_source import validate_f0_source

    source = validate_f0_source(
        root,
        root / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        None,
    )
    groups = list(map(int, source.groups))
    sequence_ids = [int(source.sequence_by_group[group]) for group in groups]
    need(
        len(groups) == len(set(groups)) == 468 and len(set(sequence_ids)) == 468,
        "F0 source coverage differs",
    )

    host, host_info, host_paths = host_reference(root, admission, source, groups, sequence_ids)
    need(
        host_info["prediction_sha256"] == HOST_PREDICTION_SHA,
        "host cache differs from graded reference",
    )
    host_sidecar = read_json(host_paths["sidecar"])
    sample_gate = sampled_parity(host, groups, sequence_ids)
    needs, mask_info = check_masks(source, members)
    host_rows = cache_rows(host, host_sidecar, members)
    cp5_expected = cache_rows(cp5_cache, cp5_sidecar, members)
    original_reconstructed64 = 0.34 * arrays["original_student"].astype(np.float64) + arrays[
        "retained_pair"
    ].astype(np.float64)
    rebuilt_cp5 = 0.34 * arrays["h512_dynamic_int8"].astype(np.float64) + arrays[
        "retained_pair"
    ].astype(np.float64)
    original_max = max_abs(original_reconstructed64, host_rows)
    cp5_max = max_abs(rebuilt_cp5, cp5_expected)
    parity = {
        "passed": max(original_max, cp5_max) <= TOLERANCE,
        "original_tl3_max_abs": original_max,
        "h512_cp5_max_abs": cp5_max,
        "tolerance": TOLERANCE,
    }
    original_reconstructed = original_reconstructed64.astype(np.float32)
    # Keep the known failed cache-parity attempts intact; the new gate is independent.
    require_hash(config.TORCH_CACHE, config.TORCH_CACHE_SHA256)
    torch_sidecar_path = config.TORCH_CACHE.with_suffix(".json")
    require_hash(torch_sidecar_path, TORCH_SIDECAR_SHA)
    old_precision = {
        "torch_h512_rms_difference_on_need_rows": old_r1["torch_h512_rms_difference_on_need_rows"],
        "native_precision_rms_difference_on_need_rows": old_r1[
            "native_precision_rms_difference_on_need_rows"
        ],
        "source_receipt": "historical R1 feature-only prelabel receipt; metrics reused unchanged",
    }
    prelabel = {
        "schema": "wnn.native-h512-host-reference-precision-prelabel.v2",
        "status": "PRELABEL_VALIDATION_PASSED"
        if parity["passed"] and sample_gate["passed"]
        else "PRELABEL_VALIDATION_FAILED",
        "selection_plan_sha256": plan_sha,
        "selected_membership_sha256": MEMBERSHIP_SHA,
        "host_grade_sha256": HOST_GRADE_SHA,
        "host_prediction_sha256": HOST_PREDICTION_SHA,
        "host_sidecar_sha256": HOST_SIDECAR_SHA,
        "host_replay_receipt_sha256": HOST_REPLAY_SHA,
        "source_binding_sha256": BINDING_SHA,
        "host_admission_sha256": ADMISSION_SHA,
        "archive_sha256": config.ORIGINAL_ZIP_SHA256,
        "cp5_archive_sha256": config.CP5_ZIP_SHA256,
        "native_grade_sha256": config.NATIVE_GRADE_SHA256,
        "h512_cp5_cache_sha256": config.CP5_CACHE_SHA256,
        "h512_cp5_sidecar_sha256": config.CP5_SIDECAR_SHA256,
        "native_extension_sha256": native_sha,
        "quant_backend": old_r2["quant_backend"],
        "need_prediction": mask_info,
        "host_sample_parity": sample_gate,
        "cache_parity": {
            **parity,
        },
        **old_precision,
        "historical_legacy_reference_failures": {
            "r1_receipt_sha256": OLD_R1_PRELABEL_SHA,
            "r1_cache_parity": old_r1["cache_parity"],
            "r2_receipt_sha256": OLD_R2_PRELABEL_SHA,
            "r2_cache_parity": old_r2["cache_parity"],
        },
        "target_accessed": False,
        "scoring_performed": False,
    }
    output_dir.mkdir()
    config.write_json(output_dir / "host_reference_prelabel_r2.json", prelabel)
    if prelabel["status"] != "PRELABEL_VALIDATION_PASSED":
        raise RuntimeError("fresh-host or CP5 parity gate failed; target cache remains unopened")

    shutil.copyfile(selection_path, output_dir / "selection_plan.json")
    need(sha(output_dir / "selection_plan.json") == plan_sha, "copied selection plan hash differs")
    (output_dir / "member_predictions.npz").symlink_to(tracks_path)
    scorer_prelabel = {
        "cache_parity": prelabel["cache_parity"],
        "torch_h512_rms_difference_on_need_rows": old_precision[
            "torch_h512_rms_difference_on_need_rows"
        ],
        "native_precision_rms_difference_on_need_rows": old_precision[
            "native_precision_rms_difference_on_need_rows"
        ],
    }
    result = score_after_parity(
        output_dir,
        root,
        selection,
        needs,
        arrays,
        cp5_expected,
        original_reconstructed,
        scorer_prelabel,
        native_sha,
        str(old_r2["quant_backend"]),
        plan_sha,
    )
    require_hash(config.TARGET_CACHE, config.TARGET_CACHE_SHA256)
    report = {
        "schema": "wnn.native-h512-host-reference-precision-probe-r2.v1",
        "status": "SCORED_EIGHT_GROUP_F0_FITTED_DIAGNOSTIC",
        "evidence_limit": "Eight exposed F0 groups from distinct components; no CI, clean CV claim, coefficient search, or promotion claim.",
        "host_reference_prelabel": "host_reference_prelabel_r2.json",
        "host_reference_prelabel_sha256": sha(output_dir / "host_reference_prelabel_r2.json"),
        "precision_probe_result": "precision_probe_result.json",
        "precision_probe_result_sha256": sha(output_dir / "precision_probe_result.json"),
        "fresh_host_reference_sha256": HOST_PREDICTION_SHA,
        "host_grade_sha256": HOST_GRADE_SHA,
        "old_legacy_reference_failures_preserved": prelabel["historical_legacy_reference_failures"],
        "old_output_manifest_sha256": OLD_MANIFEST_SHA,
        "old_five_track_npz_sha256": OLD_NPZ_SHA,
        "old_array_sha256": track_manifest["array_sha256"],
        "target_cache_sha256": config.TARGET_CACHE_SHA256,
        "score": result,
        "target_accessed_after_prelabel_pass": True,
    }
    config.write_json(output_dir / "NATIVE_HOST_PRECISION_PROBE_R2.json", report)
    return report


def main() -> None:
    """Create the fixed R2 diagnostic only after fresh-host parity passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=config.ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.SCRATCH / OUTPUT_NAME,
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve() / "src"))
    report = run(args.root.resolve(strict=True), args.output_dir.resolve())
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
