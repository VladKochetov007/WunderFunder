from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
VAST_DIR = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast"
ATTEMPT = "e529-vast-h217-pair-v4-20261008-a3"
DOWNLOAD_DIR = VAST_DIR / "downloads" / ATTEMPT
PREDICTION_MEMBER = "cache/pred_h217_v3_tcn_s7_f0_ep3_valid.npy"
SIDECAR_MEMBER = "cache/pred_h217_v3_tcn_s7_f0_ep3_valid.json"
PROVENANCE_MEMBER = "artifacts/runs/bd26_h217_tcn_raw_s7_f0_v3_amp4096/provenance.json"
CHECKPOINT_MEMBER = "artifacts/runs/bd26_h217_tcn_raw_s7_f0_v3_amp4096/model_ep3.pt"
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGET_PATH = ROOT / "cache/valid_targets.npz"
RECIPE_PATH = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/H217_TCN_RAW_SCRATCH_STAGED_V1.json"
)
SCORER_PATH = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
GMETRIC_PATH = ROOT / "src/wnn/gmetric.py"
PREREG_PATH = VAST_DIR / "H217_A3_TCN_FOLD0_GRADE_PREREG_20261008_R1.json"
REPORT_PATH = VAST_DIR / "H217_A3_TCN_FOLD0_INDEPENDENT_GRADE_20261008_R1.json"
MARKDOWN_PATH = VAST_DIR / "H217_A3_TCN_FOLD0_INDEPENDENT_GRADE_20261008_R1.md"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large prediction and target files stay bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream: BinaryIO) -> str:
    """Hash one tar member incrementally."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def read_json_member(archive: tarfile.TarFile, name: str) -> dict:
    """Read a small JSON record from the downloaded uncompressed tar archive."""
    member = archive.extractfile(name)
    if member is None:
        raise FileNotFoundError(f"missing archive member: {name}")
    with member:
        return json.load(member)


def sha256_tar_member(archive: tarfile.TarFile, name: str) -> str:
    """Hash an artifact directly from the tar without extracting it."""
    member = archive.extractfile(name)
    if member is None:
        raise FileNotFoundError(f"missing archive member: {name}")
    with member:
        return sha256_stream(member)


def load_frozen_scorer():
    """Load the repository's frozen campaign scorer without importing a new metric."""
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("campaign_score", SCORER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen scorer at {SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.METRIC_ID != "global_wp_clip2_v1":
        raise ValueError(f"unexpected frozen metric: {module.METRIC_ID}")
    return module


def verify_fold_and_ancestry(
    split: dict, provenance: dict, sidecar: dict, expected_row_groups: list[int]
) -> dict[str, object]:
    """Verify exact source-qualified fit membership and exposed fold-0 exclusion."""
    fit = provenance["fit_row_groups"]
    expected_fit = split["fit"]
    actual_fit = {source: set(map(int, groups)) for source, groups in fit.items()}
    expected = {source: set(map(int, groups)) for source, groups in expected_fit.items()}
    if actual_fit != expected:
        raise ValueError("checkpoint provenance fit groups differ from frozen fold-0 fit")

    evaluation_groups = set(map(int, split["evaluation"]["valid"]))
    if len(evaluation_groups) != 468:
        raise ValueError("fold 0 must contain exactly 468 evaluation groups")
    if actual_fit.get("valid", set()) & evaluation_groups:
        raise ValueError("valid fit groups overlap the fold-0 evaluation groups")
    if set(map(int, provenance["evaluation_row_groups"])) != evaluation_groups:
        raise ValueError("checkpoint provenance evaluation membership differs from fold 0")

    component_of = {
        (source, int(node["row_group"])): int(node["component"])
        for source, nodes in split["sequences"].items()
        for node in nodes
    }
    fit_components = {
        component_of[(source, group)] for source, groups in actual_fit.items() for group in groups
    }
    evaluation_components = {component_of[("valid", group)] for group in evaluation_groups}
    if fit_components & evaluation_components:
        raise ValueError("frozen fold-0 fit and evaluation components overlap")

    args = provenance["args"]
    expected_args = {
        "seed": 7,
        "arch": "tcn",
        "hidden": 128,
        "layers": 12,
        "batch": 8,
        "epochs": 4,
        "checkpoint_epoch": 3,
        "train_source": "train+valid",
        "valid_fold": 0,
        "include_heldout_valid": False,
        "raw_only": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "precision": "fp16",
        "amp_initial_scale": 4096.0,
        "tcn_mix": 1,
        "init": "",
        "distill": "",
        "split_manifest_sha256": sidecar["split_manifest_sha256"],
    }
    mismatches = {
        key: {"expected": value, "actual": args.get(key)}
        for key, value in expected_args.items()
        if args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint training arguments differ: {mismatches}")

    row_group_digest = hashlib.sha256(
        np.asarray(expected_row_groups, dtype="<i8").tobytes()
    ).hexdigest()
    return {
        "fold": 0,
        "split_manifest_sha256": sidecar["split_manifest_sha256"],
        "evaluation_group_count": len(evaluation_groups),
        "evaluation_groups_sha256": hashlib.sha256(
            np.asarray(sorted(evaluation_groups), dtype="<i8").tobytes()
        ).hexdigest(),
        "fit_group_counts": {key: len(value) for key, value in actual_fit.items()},
        "fit_groups_match_frozen_manifest": True,
        "evaluation_groups_match_checkpoint_provenance": True,
        "fit_evaluation_component_overlap": 0,
        "prediction_row_group_count": len(expected_row_groups),
        "prediction_row_groups_sha256": row_group_digest,
        "prediction_row_groups_are_target_cache_order": True,
        "initialization": "fresh weights; no init checkpoint",
        "distillation_or_teacher_ancestry": "none",
        "exposure": "exposed discovery fold; not fresh confirmation",
    }


def main() -> None:
    """Verify the frozen artifacts, score one fixed TCN, and write its grade receipt."""
    if REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise FileExistsError("grade receipt already exists; refusing to overwrite")
    prereg = json.loads(PREREG_PATH.read_text())
    artifact_path = DOWNLOAD_DIR / "artifact_bundle.tar"
    result_path = DOWNLOAD_DIR / "result.json"
    manifest_path = DOWNLOAD_DIR / "artifact_manifest.json"
    split = json.loads(SPLIT_PATH.read_text())
    if hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest() != prereg["frozen_split"]["sha256"]:
        raise ValueError("frozen split manifest changed after preregistration")
    if (
        sha256_file(TARGET_PATH)
        != prereg["artifact_inputs"]["valid_targets_npz_stored_members_mmap_only"]["sha256"]
    ):
        raise ValueError("valid target cache changed after preregistration")
    if sha256_file(result_path) != prereg["artifact_inputs"]["result_json"]["sha256"]:
        raise ValueError("worker result changed after preregistration")
    if sha256_file(artifact_path) != prereg["artifact_inputs"]["artifact_bundle_tar"]["sha256"]:
        raise ValueError("delivered artifact bundle changed after preregistration")
    if sha256_file(RECIPE_PATH) != prereg["candidate"]["recipe_sha256"]:
        raise ValueError("frozen H217 recipe changed after preregistration")

    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if result.get("attempt_id") != ATTEMPT or result.get("exit_code") != 0:
        raise ValueError("worker terminal identity or exit status is invalid")
    if result.get("worker_success") is not True or result.get("grade_performed") is not False:
        raise ValueError("expected successful partial delivery awaiting its first grade")
    if manifest.get("attempt_id") != ATTEMPT or manifest.get("worker_terminal") is not True:
        raise ValueError("artifact manifest is not bound to this terminal attempt")
    manifest_files = {entry["path"]: entry for entry in manifest["files"]}
    if manifest_files["result.json"]["sha256"] != sha256_file(result_path):
        raise ValueError("artifact manifest result hash mismatch")
    if manifest_files["artifact_bundle.tar"]["sha256"] != sha256_file(artifact_path):
        raise ValueError("artifact manifest bundle hash mismatch")

    tcn = result["arms"]["tcn"]
    prediction_record = tcn["prediction"]
    if tcn.get("fit_status") != "passed" or tcn.get("export_status") != "passed":
        raise ValueError("the TCN fixed-epoch fit/export did not pass")
    if tcn["profile_gate"].get("status") != "passed":
        raise ValueError("the TCN numerical profile gate did not pass")
    if prediction_record.get("prediction_sha256") != prereg["candidate"]["prediction_sha256"]:
        raise ValueError("prediction identity differs from preregistration")
    if prediction_record.get("checkpoint_sha256") != prereg["candidate"]["checkpoint_sha256"]:
        raise ValueError("checkpoint identity differs from preregistration")

    with tarfile.open(artifact_path, "r:") as archive:
        provenance = read_json_member(archive, PROVENANCE_MEMBER)
        sidecar = read_json_member(archive, SIDECAR_MEMBER)
        sidecar_sha = sha256_tar_member(archive, SIDECAR_MEMBER)
        fit_log_sha = sha256_tar_member(archive, "diagnostics/fit_tcn.log")
        export_log_sha = sha256_tar_member(archive, "diagnostics/export_tcn_valid.log")
        if fit_log_sha != tcn["fit_phase"]["log_sha256"]:
            raise ValueError("fit log bytes do not match the worker phase receipt")
        if export_log_sha != tcn["export_phase"]["log_sha256"]:
            raise ValueError("export log bytes do not match the worker phase receipt")
        checkpoint_file = archive.extractfile(CHECKPOINT_MEMBER)
        if checkpoint_file is None:
            raise FileNotFoundError(f"missing archive member: {CHECKPOINT_MEMBER}")
        with checkpoint_file:
            checkpoint_sha = sha256_stream(checkpoint_file)
        if checkpoint_sha != prediction_record["checkpoint_sha256"]:
            raise ValueError("actual fixed epoch-3 checkpoint bytes do not match result.json")
        if provenance["checkpoint_sha256"].get("model_ep3.pt") != checkpoint_sha:
            raise ValueError("run provenance does not bind the fixed epoch-3 checkpoint")
        if sidecar.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("prediction sidecar does not bind the fixed epoch-3 checkpoint")
        if sidecar.get("prediction_sha256") != prediction_record["prediction_sha256"]:
            raise ValueError("prediction sidecar and result.json disagree")
        if sidecar.get("epoch") != 3 or sidecar.get("inference_precision") != "fp32":
            raise ValueError("prediction sidecar has the wrong epoch or inference precision")
        if sidecar.get("split_manifest_sha256") != prereg["frozen_split"]["sha256"]:
            raise ValueError("prediction sidecar does not bind the frozen fold-0 manifest")
        row_groups = list(map(int, sidecar["row_groups"]))
        if len(row_groups) != prediction_record["row_group_count"]:
            raise ValueError("prediction row-group count mismatch")

        scorer = load_frozen_scorer()
        targets = scorer.mmap_stored_npz(TARGET_PATH)
        target_groups = list(map(int, targets["row_group"]))
        if row_groups != target_groups:
            raise ValueError("prediction row order differs from target-cache row order")
        expected_row_digest = hashlib.sha256(
            np.asarray(row_groups, dtype="<i8").tobytes()
        ).hexdigest()
        if expected_row_digest != prediction_record["row_groups_sha256"]:
            raise ValueError("prediction row-group digest mismatch")
        membership = verify_fold_and_ancestry(split, provenance, sidecar, row_groups)
        if prediction_record["shape"] != [len(row_groups), 20_000, 2]:
            raise ValueError("result.json prediction shape is invalid")
        target_position = {group: index for index, group in enumerate(target_groups)}
        evaluation_groups = list(map(int, split["evaluation"]["valid"]))
        score_moments: list[np.ndarray] = []
        selected_rows = 0
        with tempfile.TemporaryDirectory(prefix="h217-a3-tcn-grade-") as temp_dir:
            prediction_path = Path(temp_dir) / "prediction.npy"
            prediction_file = archive.extractfile(PREDICTION_MEMBER)
            if prediction_file is None:
                raise FileNotFoundError(f"missing archive member: {PREDICTION_MEMBER}")
            prediction_digest = hashlib.sha256()
            with prediction_file, prediction_path.open("wb") as destination:
                for chunk in iter(lambda: prediction_file.read(1 << 20), b""):
                    prediction_digest.update(chunk)
                    destination.write(chunk)
            if prediction_digest.hexdigest() != prereg["candidate"]["prediction_sha256"]:
                raise ValueError("actual prediction bytes do not match the preregistered digest")
            prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
            if (
                prediction.shape != tuple(prediction_record["shape"])
                or prediction.dtype != np.float32
            ):
                raise ValueError("prediction array shape or dtype differs from the run receipt")

            need_prediction = np.arange(20_000) >= 99
            for group in evaluation_groups:
                index = target_position[group]
                y = np.asarray(targets["y"][index], dtype=np.float32)
                mask = np.asarray(targets["mask"][index], dtype=bool)
                p = np.asarray(prediction[index], dtype=np.float32)
                if not np.isfinite(y[need_prediction]).all():
                    raise ValueError(f"nonfinite required target values in group {group}")
                if not np.isfinite(p[need_prediction]).all():
                    raise ValueError(f"nonfinite required prediction values in group {group}")
                selected = need_prediction & mask
                selected_rows += int(selected.sum())
                score_moments.append(
                    np.stack(
                        [
                            scorer.moments(y[:, target], p[:, target], selected)
                            for target in range(2)
                        ]
                    )
                )

    pooled = scorer.pooled(np.stack(score_moments))
    report = {
        "schema_version": 1,
        "record_id": "H217_A3_TCN_FOLD0_INDEPENDENT_GRADE_20261008_R1",
        "created_utc": datetime.now(UTC).isoformat(),
        "attempt_id": ATTEMPT,
        "grade_status": "completed_descriptive_exposed_fold_grade",
        "worker_terminal_status": result["status"],
        "worker_success": result["worker_success"],
        "worker_exit_code": result["exit_code"],
        "matched_gru_arm": {
            "fit_status": result["arms"]["gru"].get("fit_status"),
            "reason": result["arms"]["gru"].get("error"),
            "used_as_comparator": False,
        },
        "claim_scope": prereg["claim_scope"],
        "metric_id": scorer.METRIC_ID,
        "metric": {
            **pooled,
            "evaluation_groups": len(evaluation_groups),
            "selected_rows": selected_rows,
            "selection": "need_prediction AND is_scored; need_prediction is row index >= 99",
            "weighting_and_clipping": "clip target and prediction to [-2, 2]; weight each row by abs(clipped target)",
            "aggregation": "pool selected rows separately for t0 and t1; report their arithmetic mean as wp",
        },
        "candidate": prereg["candidate"],
        "membership_and_ancestry": membership,
        "exposure_and_uncertainty": {
            "fold_exposure": "exposed discovery fold; not fresh confirmation",
            "training_seeds": 1,
            "training_seed_uncertainty_estimated": False,
            "public_transfer_estimated": False,
            "bootstrap": None,
            "all_data_comparison": None,
            "public_forecast": None,
            "blends_or_posthoc_slices": [],
        },
        "execution": {
            "profile_gate": tcn["profile_gate"],
            "fit_seconds": tcn["fit_phase"]["elapsed_seconds"],
            "fit_exit_code": tcn["fit_phase"]["exit_code"],
            "fit_peak_host_rss_bytes": tcn["fit_phase"]["peak_rss_bytes"],
            "fit_peak_vram_bytes": tcn["fit_phase"]["peak_vram_bytes"],
            "export_seconds": tcn["export_phase"]["elapsed_seconds"],
            "export_exit_code": tcn["export_phase"]["exit_code"],
            "export_peak_host_rss_bytes": tcn["export_phase"]["peak_rss_bytes"],
            "export_peak_vram_bytes": tcn["export_phase"]["peak_vram_bytes"],
        },
        "verified_artifacts": {
            "result_json_sha256": sha256_file(result_path),
            "artifact_bundle_sha256": sha256_file(artifact_path),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "split_manifest_sha256": sha256_file(SPLIT_PATH),
            "valid_targets_npz_sha256": sha256_file(TARGET_PATH),
            "recipe_sha256": sha256_file(RECIPE_PATH),
            "frozen_campaign_scorer_sha256": sha256_file(SCORER_PATH),
            "official_metric_source_sha256": sha256_file(GMETRIC_PATH),
            "checkpoint_sha256": checkpoint_sha,
            "prediction_sha256": prediction_record["prediction_sha256"],
            "prediction_sidecar_sha256": sidecar_sha,
            "fit_log_sha256": fit_log_sha,
            "export_log_sha256": export_log_sha,
            "pre_registration_sha256": sha256_file(PREREG_PATH),
        },
        "method": "campaign_score.mmap_stored_npz and campaign_score.moments/pooled; target members mapped without extraction, prediction memory-mapped from the delivered tar member; no Parquet footer, NPZ decompression, fit, GPU or provider operation",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    markdown = f"""# H217 A3 TCN fold-0 independent grade

The delivered TCN checkpoint passed its recorded fit, fixed-epoch export, and profile gate. This grade evaluates only the prespecified epoch-3 FP32 prediction on the 468 excluded fold-0 valid groups using `global_wp_clip2_v1`.

- Official pooled WP: **{pooled["wp"]:.9f}** (t0 {pooled["t0"]:.9f}; t1 {pooled["t1"]:.9f})
- Selected rows: {selected_rows:,} across {len(evaluation_groups)} evaluation groups
- Checkpoint SHA-256: `{prediction_record["checkpoint_sha256"]}`
- Prediction SHA-256: `{prediction_record["prediction_sha256"]}`
- Fold-0 split SHA-256: `{membership["split_manifest_sha256"]}`
- Actual fit membership: {membership["fit_group_counts"]["train"]:,} train-source plus {membership["fit_group_counts"]["valid"]:,} valid-source groups; exact match to the frozen manifest, with zero fit/evaluation component overlap

The fold is exposed and this is one training seed, so the result is discovery evidence rather than fresh confirmation. The matched GRU arm did not fit because its projected four-epoch fit/export exceeded its 7,200-second arm budget; it is not used as a comparator. No blend, all-data comparison, post-hoc slice, or public forecast is reported.

Targets were read from the campaign scorer's memory-mapped stored NPY members inside `cache/valid_targets.npz`. The prediction was memory-mapped from the delivered artifact tar. No Parquet footer, compressed NPZ member, fit, GPU, or provider operation was used.

Machine-readable receipt: `{REPORT_PATH.name}`. Re-run from the repository root with `scripts/limited` and the same one-CPU/1.5-GB/120-second limits:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \\
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \\
scripts/limited .venv/bin/python {Path(__file__).relative_to(ROOT)}
```
"""
    MARKDOWN_PATH.write_text(markdown)
    print(json.dumps({"report": str(REPORT_PATH), "metric": report["metric"]}, indent=2))


if __name__ == "__main__":
    main()
