"""Replay the exact pinned TL3 callback on the frozen eight-group probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import probe_config as config

ARRAY_NAME = "original_endpoint_exact_callback.npy"
RECEIPT_NAME = "original_endpoint_prelabel_validation.json"
OUTPUT_NAMES = {
    "h512_dynamic_int8",
    "h512_float32_fast",
    "h512_float32_exact",
    "original_student",
    "retained_pair",
}


def array_digest(values: np.ndarray) -> str:
    """Hash an array's contiguous bytes without changing the source array."""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_and_check_saved_tracks(
    output_dir: Path,
    selection: dict[str, Any],
    plan_sha: str,
    native_sha: str,
) -> dict[str, np.ndarray]:
    """Verify the original five-track artifact before reusing its CP5 control."""
    from probe_native import digest_file

    manifest_path = output_dir / "outputs_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_npz") != "member_predictions.npz":
        raise ValueError("saved member-track filename changed")
    predictions_path = output_dir / "member_predictions.npz"
    if manifest.get("status") != "FEATURE_ONLY_OUTPUTS_SAVED_BEFORE_TARGET_ACCESS":
        raise ValueError("saved member tracks are not marked feature-only")
    if manifest.get("target_accessed") is not False:
        raise ValueError("saved member tracks do not have a no-target receipt")
    if (
        manifest.get("selection_plan_sha256") != plan_sha
        or manifest.get("selected_membership_sha256") != selection["selected_membership_sha256"]
        or manifest.get("original_zip_sha256") != config.ORIGINAL_ZIP_SHA256
        or manifest.get("cp5_zip_sha256") != config.CP5_ZIP_SHA256
        or manifest.get("native_extension_sha256") != native_sha
    ):
        raise ValueError("saved member-track provenance differs from the frozen probe")
    if digest_file(predictions_path) != manifest.get("output_npz_sha256"):
        raise ValueError("saved member_predictions.npz hash changed")

    arrays: dict[str, np.ndarray] = {}
    with np.load(predictions_path, allow_pickle=False) as archive:
        if set(archive.files) != OUTPUT_NAMES:
            raise ValueError("saved member-track roster changed")
        for name in sorted(OUTPUT_NAMES):
            value = np.asarray(archive[name])
            expected_shape = (len(selection["members"]), 20_000, 2)
            if value.shape != expected_shape or value.dtype != np.float32:
                raise ValueError(f"saved track {name} has unexpected shape or dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"saved track {name} contains nonfinite values")
            if array_digest(value) != manifest["array_sha256"].get(name):
                raise ValueError(f"saved track {name} content hash changed")
            arrays[name] = value.copy()
    return arrays


def replay_exact_callback(
    repo: Path, package: Path, selection: dict[str, Any]
) -> tuple[np.ndarray, list[np.ndarray], str, str, str]:
    """Run the ZIP's PredictionModel callback over only selected feature rows."""
    from f0_replay import _read_features
    from f0_source import validate_f0_source
    from native_runtime import load_callback, sha256_file

    from utils import DataPoint

    sys.path.insert(0, str(repo / "src"))
    source = validate_f0_source(
        repo, repo / "research/campaign_2026-09-29/artifacts/splits/fold0.json", None
    )
    selected = selection["members"]
    if any(int(item["row_group"]) not in source.groups for item in selected):
        raise ValueError("selection contains a non-evaluation F0 group")
    sequence_ids = [int(item["seq_ix"]) for item in selected]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("selected groups must have distinct sequence IDs for reset checks")

    module, callback, native_path = load_callback(package)
    native_sha = sha256_file(native_path)
    if str(module.LEVEL) != "v3" or native_path.parent.name != "v3":
        raise RuntimeError("exact callback did not select the pinned v3 native library")
    result = np.zeros((len(selected), source.sequence_length, 2), dtype=np.float32)
    needs: list[np.ndarray] = []
    for position, member in enumerate(selected):
        row_group, seq_ix = int(member["row_group"]), int(member["seq_ix"])
        table = source.parquet.read_row_group(
            row_group, columns=list(source.columns), use_threads=False
        )
        features, need = _read_features(table, source, seq_ix)
        if features.shape != (source.sequence_length, len(source.feature_columns)):
            raise ValueError("selected feature group is not a complete sequence")
        needs.append(need.copy())
        for step, row in enumerate(features):
            prediction = callback.predict(DataPoint(seq_ix, step, bool(need[step]), row))
            if not need[step]:
                if prediction is not None:
                    raise ValueError("exact callback emitted a warmup prediction")
                continue
            value = np.asarray(prediction, dtype=np.float32)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("exact callback emitted malformed prediction")
            result[position, step] = value
        del table, features
    return (
        result,
        needs,
        native_sha,
        str(callback.quant_backend),
        sha256_file(package / "solution.py"),
    )


def selected_cache_rows(
    cache: np.ndarray, sidecar: dict[str, Any], selected: list[dict[str, Any]]
) -> np.ndarray:
    """Select aligned full-cache rows using frozen row-group identities."""
    from probe_native import cache_rows

    return cache_rows(cache, sidecar, selected)


def run(repo: Path, output_dir: Path, selection_path: Path) -> dict[str, Any]:
    """Replay and gate the exact old endpoint without opening any target artifact."""
    from prepare_selection import make_plan
    from probe_native import digest_file, max_abs
    from probe_prepare import checked_sources, extract_packages

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection != make_plan(repo) or selection.get("status") != "PREPARED_LABEL_FREE_SELECTION":
        raise ValueError("selection plan differs from the frozen deterministic recipe")
    if selection_path != output_dir / "selection_plan.json":
        raise ValueError("selection plan must be the one already stored in output_dir")
    required = {
        "selection_plan.json",
        "member_predictions.npz",
        "outputs_manifest.json",
        "prelabel_validation.json",
    }
    if not required.issubset({path.name for path in output_dir.iterdir()}):
        raise FileNotFoundError("original five-track feature-only outputs are incomplete")
    array_path, receipt_path = output_dir / ARRAY_NAME, output_dir / RECEIPT_NAME
    if array_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to replace a prior exact-endpoint replay")
    if output_dir.parent != config.SCRATCH.resolve():
        raise ValueError("probe outputs must be directly under approved physical scratch")

    original_cache, original_sidecar, cp5_cache, cp5_sidecar = checked_sources()
    plan_sha = digest_file(selection_path)
    with tempfile.TemporaryDirectory(
        prefix="native-h512-original-endpoint-", dir=config.SCRATCH
    ) as scratch_dir:
        original_package, _ = extract_packages(Path(scratch_dir))
        predictions, needs, native_sha, backend, solution_sha = replay_exact_callback(
            repo, original_package, selection
        )

    old_expected = selected_cache_rows(original_cache, original_sidecar, selection["members"])
    tracks = load_and_check_saved_tracks(output_dir, selection, plan_sha, native_sha)
    cp5_expected = selected_cache_rows(cp5_cache, cp5_sidecar, selection["members"])
    cp5_reconstructed = 0.34 * tracks["h512_dynamic_int8"].astype(np.float64) + tracks[
        "retained_pair"
    ].astype(np.float64)
    tolerance = 1e-5
    parity = {
        "original_exact_callback_max_abs": max_abs(predictions, old_expected),
        "h512_cp5_saved_tracks_max_abs": max_abs(cp5_reconstructed, cp5_expected),
        "tolerance": tolerance,
    }
    parity["passed"] = (
        max(
            parity["original_exact_callback_max_abs"],
            parity["h512_cp5_saved_tracks_max_abs"],
        )
        <= tolerance
    )

    with array_path.open("xb") as output:
        np.save(output, predictions, allow_pickle=False)
    receipt = {
        "schema": "wnn.native-h512-original-endpoint-rerun.v1",
        "status": "PRELABEL_VALIDATION_PASSED"
        if parity["passed"]
        else "PRELABEL_VALIDATION_FAILED",
        "selection_plan_sha256": plan_sha,
        "selected_membership_sha256": selection["selected_membership_sha256"],
        "saved_five_tracks_npz_sha256": digest_file(output_dir / "member_predictions.npz"),
        "original_zip_sha256": config.ORIGINAL_ZIP_SHA256,
        "original_solution_sha256": solution_sha,
        "cp5_zip_sha256": config.CP5_ZIP_SHA256,
        "native_extension_sha256": native_sha,
        "quant_backend": backend,
        "replay": "exact pinned original ZIP PredictionModel.predict; feature rows only; reset by source sequence ID",
        "needs_rows": int(sum(int(mask.sum()) for mask in needs)),
        "array_path": array_path.name,
        "array_shape": list(predictions.shape),
        "array_dtype": str(predictions.dtype),
        "array_sha256": digest_file(array_path),
        "cache_parity": parity,
        "target_accessed": False,
        "scoring_performed": False,
    }
    config.write_json(receipt_path, receipt)
    if not parity["passed"]:
        raise RuntimeError("corrected endpoint parity failed; targets remain unopened")
    return receipt


def main() -> None:
    """Run the create-only original-endpoint correction after root admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=config.ROOT)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, output_dir = args.repo_root.resolve(), args.output_dir.resolve()
    selection_path = args.selection_plan.resolve()
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(config.REPLAY))
    sys.path.insert(0, str(repo))
    result = run(repo, output_dir, selection_path)
    print(json.dumps({"status": result["status"], "receipt": str(output_dir / RECEIPT_NAME)}))


if __name__ == "__main__":
    main()
