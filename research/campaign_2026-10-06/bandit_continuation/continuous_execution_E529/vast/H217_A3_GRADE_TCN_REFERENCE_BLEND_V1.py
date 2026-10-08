from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

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
REFERENCE_RUN = "c20261003_combo_s3_f0"
REFERENCE_PREDICTION = ROOT / f"cache/pred_{REFERENCE_RUN}_valid.npy"
REFERENCE_SIDECAR = REFERENCE_PREDICTION.with_suffix(".json")
SOURCE_FAMILIES = ROOT / "research/campaign_2026-10-06/bandit_continuation/SOURCE_FAMILIES.json"
SCORER_PATH = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
GMETRIC_PATH = ROOT / "src/wnn/gmetric.py"
BASE_GRADE = VAST_DIR / "H217_A3_TCN_FOLD0_INDEPENDENT_GRADE_20261008_R1.json"
PREREG = VAST_DIR / "H217_A3_TCN_FOLD0_REFERENCE_BLEND_PREREG_20261008_R1.json"
REPORT = VAST_DIR / "H217_A3_TCN_FOLD0_REFERENCE_BLEND_20261008_R1.json"
MARKDOWN = VAST_DIR / "H217_A3_TCN_FOLD0_REFERENCE_BLEND_20261008_R1.md"
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)
N_BOOT = 2000
BOOTSTRAP_SEED = 271828


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_json(archive: tarfile.TarFile, member_name: str) -> dict:
    """Load a small JSON member from the delivered uncompressed tar archive."""
    member = archive.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"missing archive member: {member_name}")
    with member:
        return json.load(member)


def archive_member_sha256(archive: tarfile.TarFile, member_name: str) -> str:
    """Hash one archive member without holding the member in memory."""
    member = archive.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"missing archive member: {member_name}")
    digest = hashlib.sha256()
    with member:
        for chunk in iter(lambda: member.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_scorer():
    """Load the existing campaign scorer and require the official metric ID."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("campaign_score", SCORER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen scorer at {SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.METRIC_ID != "global_wp_clip2_v1":
        raise ValueError(f"unexpected frozen metric: {module.METRIC_ID}")
    return module


def weighted_pair_moments(
    candidate: np.ndarray, reference: np.ndarray, target: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Compute stable centered pair moments using the official target weights."""
    weight = np.abs(np.clip(target[selected], -2.0, 2.0)).astype(np.float64)
    first = np.clip(candidate[selected], -2.0, 2.0).astype(np.float64)
    second = np.clip(reference[selected], -2.0, 2.0).astype(np.float64)
    active = weight > 0
    if not active.any():
        return np.zeros(6, dtype=np.float64)
    weight, first, second = weight[active], first[active], second[active]
    total = float(weight.sum())
    first_delta, second_delta = first - first[0], second - second[0]
    mean_first = float(first[0] + np.dot(weight, first_delta) / total)
    mean_second = float(second[0] + np.dot(weight, second_delta) / total)
    centered_first = first - mean_first
    centered_second = second - mean_second
    return np.array(
        [
            total,
            mean_first,
            mean_second,
            np.dot(weight, centered_first * centered_first),
            np.dot(weight, centered_second * centered_second),
            np.dot(weight, centered_first * centered_second),
        ],
        dtype=np.float64,
    )


def summarize_pair_moments(scorer, moments: np.ndarray) -> dict[str, float]:
    """Return weighted population covariance and correlation from centered moments."""
    merged = np.zeros(6, dtype=np.float64)
    for value in moments:
        merged = scorer._merge_moments(merged, value)
    if merged[0] <= 0:
        raise ValueError("no positive official target weight in prediction-pair metric")
    denominator = np.sqrt(merged[3] * merged[4])
    correlation = 0.0 if denominator <= 1e-16 else float(merged[5] / denominator)
    return {
        "total_weight": float(merged[0]),
        "weighted_population_covariance": float(merged[5] / merged[0]),
        "weighted_pearson_correlation": correlation,
    }


def verify_reference_lineage(split: dict, reference_sidecar: dict) -> dict[str, object]:
    """Check the reference, its own teacher ancestors, and propensity ancestry."""
    expected_fit = {source: set(map(int, groups)) for source, groups in split["fit"].items()}
    evaluation_groups = set(map(int, split["evaluation"]["valid"]))
    component_of = {
        (source, int(node["row_group"])): int(node["component"])
        for source, nodes in split["sequences"].items()
        for node in nodes
    }
    evaluation_components = {component_of[("valid", group)] for group in evaluation_groups}
    source_families = json.loads(SOURCE_FAMILIES.read_text())
    family_records = {entry["run"]: entry for entry in source_families["families"]}
    ancestry = {}

    for run in (
        REFERENCE_RUN,
        "c20260929_since_s3_f0",
        "c20260929_since_s4_f0",
        "c20260929_prop_f0",
    ):
        run_dir = ROOT / "artifacts/runs" / run
        provenance_path = run_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        fit = {
            source: set(map(int, groups)) for source, groups in provenance["fit_row_groups"].items()
        }
        if run == "c20260929_prop_f0":
            expected_run_fit = {"valid": expected_fit["valid"]}
        else:
            expected_run_fit = expected_fit
        if fit != expected_run_fit:
            raise ValueError(f"{run}: fit groups differ from the frozen fold-0 membership")
        if fit.get("valid", set()) & evaluation_groups:
            raise ValueError(f"{run}: fitted on fold-0 evaluation groups")
        fit_components = {
            component_of[(source, group)] for source, groups in fit.items() for group in groups
        }
        overlap = fit_components & evaluation_components
        if overlap:
            raise ValueError(f"{run}: fit/evaluation component overlap ({len(overlap)})")
        if (
            provenance.get("split_manifest_sha256")
            != hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
        ):
            raise ValueError(f"{run}: provenance split hash differs from fold 0")
        if set(map(int, provenance.get("evaluation_row_groups", []))) != evaluation_groups:
            raise ValueError(f"{run}: evaluation membership differs from fold 0")

        checkpoint_name = "model.pt" if run.endswith("_prop_f0") else "model_ep3.pt"
        checkpoint_path = run_dir / checkpoint_name
        checkpoint_sha = sha256_file(checkpoint_path)
        if provenance["checkpoint_sha256"].get(checkpoint_name) != checkpoint_sha:
            raise ValueError(f"{run}: checkpoint bytes do not match provenance")
        family_record = family_records.get(run)
        if run != "c20260929_prop_f0":
            if family_record is None:
                raise ValueError(f"{run}: missing bound own-source family record")
            if family_record.get("checkpoint_sha256") != checkpoint_sha or family_record.get(
                "provenance_sha256"
            ) != sha256_file(provenance_path):
                raise ValueError(f"{run}: source family hashes do not match actual artifacts")

        ancestry[run] = {
            "checkpoint_sha256": checkpoint_sha,
            "provenance_sha256": sha256_file(provenance_path),
            "fit_groups": {source: len(groups) for source, groups in fit.items()},
            "fit_evaluation_component_overlap": 0,
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "selected_epoch": provenance.get("selected_epoch"),
            "init": provenance.get("args", {}).get("init"),
            "distill": provenance.get("args", {}).get("distill"),
            "propensity": provenance.get("args", {}).get("propensity"),
        }

    parent = json.loads((ROOT / "artifacts/runs" / REFERENCE_RUN / "provenance.json").read_text())
    prop_checkpoint = ancestry["c20260929_prop_f0"]["checkpoint_sha256"]
    if parent.get("propensity_checkpoint_sha256") != prop_checkpoint:
        raise ValueError(
            "parent GRU propensity checkpoint does not match audited own propensity run"
        )
    for teacher in ("c20260929_since_s3_f0", "c20260929_since_s4_f0"):
        if ancestry[teacher]["init"] or ancestry[teacher]["distill"]:
            raise ValueError(f"{teacher}: unexpected initializer or distillation ancestor")
        if ancestry[teacher]["propensity"] != "c20260929_prop_f0":
            raise ValueError(f"{teacher}: unexpected propensity ancestor")
    if parent["args"].get("init"):
        raise ValueError("parent GRU was not trained from fresh weights")
    if parent["args"].get("distill") != "c20260929_since_s3_f0,c20260929_since_s4_f0":
        raise ValueError("parent GRU teacher ancestry differs from the audited source family")
    if parent["args"].get("propensity") != "c20260929_prop_f0":
        raise ValueError("parent GRU propensity ancestry differs from the audited source family")
    if (
        reference_sidecar.get("split_manifest_sha256")
        != hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
    ):
        raise ValueError("parent prediction sidecar does not bind the frozen fold-0 split")
    if reference_sidecar.get("prediction_sha256") != sha256_file(REFERENCE_PREDICTION):
        raise ValueError("parent prediction sidecar does not bind the actual prediction bytes")

    return {
        "reference_run": REFERENCE_RUN,
        "reference_fit_groups_match_frozen_manifest": ancestry[REFERENCE_RUN]["fit_groups"]
        == {"train": 10535, "valid": 1405},
        "reference_evaluation_groups": len(evaluation_groups),
        "fit_evaluation_component_overlap": 0,
        "ancestry_runs": ancestry,
        "source_families_sha256": sha256_file(SOURCE_FAMILIES),
    }


def main() -> None:
    """Verify fixed ancestry, report every preregistered blend, and write a receipt."""
    if REPORT.exists() or MARKDOWN.exists():
        raise FileExistsError("reference comparison receipt already exists; refusing overwrite")
    prereg = json.loads(PREREG.read_text())
    base_grade = json.loads(BASE_GRADE.read_text())
    if base_grade["metric_id"] != "global_wp_clip2_v1":
        raise ValueError("standalone candidate grade has the wrong metric")
    if (
        base_grade["verified_artifacts"]["prediction_sha256"]
        != prereg["candidate"]["prediction_sha256"]
    ):
        raise ValueError("candidate grade prediction differs from comparison preregistration")
    if sha256_file(BASE_GRADE) != prereg["candidate"]["standalone_grade_sha256"]:
        raise ValueError("standalone candidate grade changed after comparison registration")
    split_hash = sha256_file(SPLIT_PATH)
    if split_hash != prereg["comparison_design"]["split_manifest"]["sha256"]:
        raise ValueError("fold-0 split changed after comparison registration")
    if sha256_file(TARGET_PATH) != prereg["comparison_design"]["target_cache"]["sha256"]:
        raise ValueError("valid target cache changed after comparison registration")
    if sha256_file(REFERENCE_PREDICTION) != prereg["reference"]["prediction_sha256"]:
        raise ValueError("reference prediction bytes differ from preregistration")
    if sha256_file(REFERENCE_SIDECAR) != prereg["reference"]["sidecar_sha256"]:
        raise ValueError("reference prediction sidecar differs from preregistration")
    if (
        sha256_file(SCORER_PATH)
        != "b6377a0e1d67030115c6d20a64ee00dc97ef15eaad7a806408a76ef70558ab39"
    ):
        raise ValueError("frozen scorer changed")
    if (
        sha256_file(GMETRIC_PATH)
        != "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
    ):
        raise ValueError("official metric implementation changed")

    split = json.loads(SPLIT_PATH.read_text())
    expected_groups = list(map(int, split["evaluation"]["valid"]))
    if len(expected_groups) != 468 or len(set(expected_groups)) != 468:
        raise ValueError("frozen fold-0 evaluation panel is not exactly 468 unique groups")
    reference_sidecar = json.loads(REFERENCE_SIDECAR.read_text())
    if (
        reference_sidecar.get("run") != REFERENCE_RUN
        or reference_sidecar.get("epoch") != 3
        or reference_sidecar.get("checkpoint_sha256") != prereg["reference"]["checkpoint_sha256"]
        or reference_sidecar.get("prediction_sha256") != prereg["reference"]["prediction_sha256"]
    ):
        raise ValueError("reference sidecar identity differs from preregistration")

    lineage = verify_reference_lineage(split, reference_sidecar)
    if lineage["source_families_sha256"] != prereg["reference"]["ancestry_manifest"]["sha256"]:
        raise ValueError("own-source family manifest differs from preregistration")
    scorer = load_frozen_scorer()
    targets = scorer.mmap_stored_npz(TARGET_PATH)
    target_groups = list(map(int, targets["row_group"]))
    expected_row_groups_sha = hashlib.sha256(
        np.asarray(target_groups, dtype="<i8").tobytes()
    ).hexdigest()
    if target_groups != list(map(int, reference_sidecar["row_groups"])):
        raise ValueError("parent prediction row order differs from target-cache order")
    if expected_row_groups_sha != prereg["reference"]["row_groups_sha256"]:
        raise ValueError("reference prediction row-group identity is incorrect")
    position = {group: index for index, group in enumerate(target_groups)}
    parent = np.load(REFERENCE_PREDICTION, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(target_groups), 20_000, 2)
    if parent.shape != expected_shape or parent.dtype != np.float32:
        raise ValueError("reference prediction shape or dtype is invalid")

    artifact_path = DOWNLOAD_DIR / "artifact_bundle.tar"
    if (
        sha256_file(artifact_path)
        != "e1b244d26dc3dee02089580b09aefaa3e57012a73f5a086c4b9d4de7863278e8"
    ):
        raise ValueError("candidate artifact bundle changed since its independent grade")
    result = json.loads((DOWNLOAD_DIR / "result.json").read_text())
    if (
        sha256_file(DOWNLOAD_DIR / "result.json")
        != base_grade["verified_artifacts"]["result_json_sha256"]
    ):
        raise ValueError("worker result changed since its independent grade")
    profile = result["arms"]["tcn"]["profile_gate"]
    if (
        result.get("attempt_id") != ATTEMPT
        or result.get("exit_code") != 0
        or result.get("worker_success") is not True
        or result["arms"]["tcn"].get("fit_status") != "passed"
        or result["arms"]["tcn"].get("export_status") != "passed"
        or profile.get("status") != "passed"
        or profile.get("update_evidence", {}).get("optimizer_completed_updates") != 230
        or profile.get("update_evidence", {}).get("amp_skipped_updates") != 0
    ):
        raise ValueError("candidate terminal fit/export/profile evidence is not valid")
    candidate_record = result["arms"]["tcn"]["prediction"]
    if candidate_record["prediction_sha256"] != prereg["candidate"]["prediction_sha256"]:
        raise ValueError("candidate prediction differs from preregistration")
    if candidate_record["checkpoint_sha256"] != prereg["candidate"]["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint differs from preregistration")

    evaluation_groups = expected_groups
    component_of = {
        (source, int(node["row_group"])): int(node["component"])
        for source, nodes in split["sequences"].items()
        for node in nodes
    }
    components = np.asarray(
        [component_of[("valid", group)] for group in evaluation_groups], dtype=np.int64
    )
    need_prediction = np.arange(20_000) >= 99
    base_moments: list[np.ndarray] = []
    blend_moments: dict[float, list[np.ndarray]] = {alpha: [] for alpha in ALPHAS}
    pair_moments = [[], []]
    selected_rows = 0
    scorer_module = importlib.util.spec_from_file_location(
        "h217_grade_helper", VAST_DIR / "H217_A3_GRADE_TCN_FOLD0_V1.py"
    )
    if scorer_module is None or scorer_module.loader is None:
        raise ImportError("cannot load standalone grade helper")
    helper = importlib.util.module_from_spec(scorer_module)
    scorer_module.loader.exec_module(helper)

    with (
        tarfile.open(artifact_path, "r:") as archive,
        tempfile.TemporaryDirectory(prefix="h217-a3-reference-grade-") as temp_dir,
    ):
        sidecar = helper.read_json_member(archive, SIDECAR_MEMBER)
        provenance = helper.read_json_member(archive, PROVENANCE_MEMBER)
        if sidecar.get("split_manifest_sha256") != split_hash:
            raise ValueError("candidate sidecar does not bind the frozen fold-0 split")
        if list(map(int, sidecar.get("row_groups", []))) != target_groups:
            raise ValueError("candidate prediction row order differs from target-cache order")
        candidate_checkpoint = archive.extractfile(CHECKPOINT_MEMBER)
        if candidate_checkpoint is None:
            raise FileNotFoundError("candidate fixed epoch-3 checkpoint is absent")
        with candidate_checkpoint:
            checkpoint_sha = helper.sha256_stream(candidate_checkpoint)
        if checkpoint_sha != candidate_record["checkpoint_sha256"]:
            raise ValueError("candidate checkpoint bytes differ from worker result")
        if provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") != checkpoint_sha:
            raise ValueError("candidate provenance does not bind its epoch-3 checkpoint")
        prediction_file = archive.extractfile(PREDICTION_MEMBER)
        if prediction_file is None:
            raise FileNotFoundError("candidate fixed epoch-3 prediction is absent")
        prediction_path = Path(temp_dir) / "tcn_prediction.npy"
        digest = hashlib.sha256()
        with prediction_file, prediction_path.open("wb") as destination:
            for chunk in iter(lambda: prediction_file.read(1 << 20), b""):
                digest.update(chunk)
                destination.write(chunk)
        if digest.hexdigest() != prereg["candidate"]["prediction_sha256"]:
            raise ValueError("candidate prediction bytes differ from preregistration")
        candidate = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if candidate.shape != expected_shape or candidate.dtype != np.float32:
            raise ValueError("candidate prediction shape or dtype is invalid")

        for group in evaluation_groups:
            index = position[group]
            y = np.asarray(targets["y"][index], dtype=np.float32)
            mask = np.asarray(targets["mask"][index], dtype=bool)
            candidate_rows = np.asarray(candidate[index], dtype=np.float32)
            parent_rows = np.asarray(parent[index], dtype=np.float32)
            if not np.isfinite(y[need_prediction]).all():
                raise ValueError(f"nonfinite target values in fold-0 evaluation group {group}")
            if (
                not np.isfinite(candidate_rows[need_prediction]).all()
                or not np.isfinite(parent_rows[need_prediction]).all()
            ):
                raise ValueError(f"nonfinite prediction values in fold-0 evaluation group {group}")
            selected = need_prediction & mask
            selected_rows += int(selected.sum())
            base_moments.append(
                np.stack(
                    [
                        scorer.moments(y[:, target], parent_rows[:, target], selected)
                        for target in range(2)
                    ]
                )
            )
            for alpha in ALPHAS:
                blend = (1.0 - alpha) * parent_rows + alpha * candidate_rows
                blend_moments[alpha].append(
                    np.stack(
                        [
                            scorer.moments(y[:, target], blend[:, target], selected)
                            for target in range(2)
                        ]
                    )
                )
            for target in range(2):
                pair_moments[target].append(
                    weighted_pair_moments(
                        candidate_rows[:, target], parent_rows[:, target], y[:, target], selected
                    )
                )

    baseline_array = np.stack(base_moments)
    baseline_score = scorer.pooled(baseline_array)
    blend_results = []
    for alpha in ALPHAS:
        candidate_array = np.stack(blend_moments[alpha])
        score = scorer.pooled(candidate_array)
        bootstrap = scorer.paired_component_bootstrap(
            candidate_array,
            baseline_array,
            components,
            n_boot=N_BOOT,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        blend_results.append(
            {
                "alpha_on_tcn": alpha,
                "score": score,
                "delta_vs_parent": {
                    "t0": score["t0"] - baseline_score["t0"],
                    "t1": score["t1"] - baseline_score["t1"],
                    "wp": score["wp"] - baseline_score["wp"],
                },
                "paired_component_bootstrap_vs_parent": bootstrap,
            }
        )
    pair_stats = {
        f"t{target}": summarize_pair_moments(scorer, np.stack(pair_moments[target]))
        for target in range(2)
    }
    report = {
        "schema_version": 1,
        "record_id": "H217_A3_TCN_FOLD0_REFERENCE_BLEND_20261008_R1",
        "created_utc": datetime.now(UTC).isoformat(),
        "attempt_id": ATTEMPT,
        "grade_status": "conditional_exposed_discovery_comparison",
        "claim_scope": prereg["claim_scope"],
        "metric_id": scorer.METRIC_ID,
        "evaluation": {
            "fold": 0,
            "evaluation_groups": len(evaluation_groups),
            "components": int(len(np.unique(components))),
            "selected_rows": selected_rows,
            "split_manifest_sha256": split_hash,
            "selection": "need_prediction AND is_scored; need_prediction is row index >= 99",
            "exposure": "exposed discovery fold; no fresh confirmation",
        },
        "candidate": prereg["candidate"],
        "reference": prereg["reference"],
        "reference_baseline_official_score": baseline_score,
        "predeclared_blends_all_reported": blend_results,
        "pair_prediction_covariance_correlation": {
            "definition": prereg["comparison_design"]["pair_covariance_and_correlation"],
            "per_target": pair_stats,
        },
        "lineage_validation": lineage,
        "uncertainty_and_limits": {
            "bootstrap": "paired component bootstrap over the same 468 frozen groups, fixed predictions, 2,000 replicates, seed 271828",
            "conditional_on_fixed_predictions": True,
            "training_seed_uncertainty_estimated": False,
            "public_transfer_estimated": False,
            "blend_weight_selection": "none; all five prespecified weights are reported",
            "all_data_comparison": None,
            "public_forecast": None,
            "comparability_limits": prereg["comparability_limits"],
        },
        "verified_artifacts": {
            "grader_script_sha256": sha256_file(Path(__file__)),
            "pre_registration_sha256": sha256_file(PREREG),
            "base_grade_sha256": sha256_file(BASE_GRADE),
            "candidate_bundle_sha256": sha256_file(artifact_path),
            "candidate_checkpoint_sha256": checkpoint_sha,
            "candidate_prediction_sha256": candidate_record["prediction_sha256"],
            "reference_prediction_sha256": sha256_file(REFERENCE_PREDICTION),
            "reference_sidecar_sha256": sha256_file(REFERENCE_SIDECAR),
            "valid_targets_sha256": sha256_file(TARGET_PATH),
            "split_manifest_sha256": split_hash,
            "source_families_sha256": sha256_file(SOURCE_FAMILIES),
            "frozen_campaign_scorer_sha256": sha256_file(SCORER_PATH),
            "official_metric_source_sha256": sha256_file(GMETRIC_PATH),
        },
        "method": "Frozen campaign scorer with stored target NPY members memory-mapped from NPZ and both FP32 prediction arrays memory-mapped; no Parquet footer, compressed NPZ member, fit, GPU or provider operation.",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    rows = [
        "| TCN weight | WP | t0 | t1 | delta vs parent | paired component 95% CI |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in blend_results:
        ci = entry["paired_component_bootstrap_vs_parent"]["ci95"]
        rows.append(
            f"| {entry['alpha_on_tcn']:.2f} | {entry['score']['wp']:.9f} | "
            f"{entry['score']['t0']:.9f} | {entry['score']['t1']:.9f} | "
            f"{entry['delta_vs_parent']['wp']:+.9f} | [{ci[0]:+.9f}, {ci[1]:+.9f}] |"
        )
    pair_lines = "\n".join(
        f"- {target}: weighted covariance {stats['weighted_population_covariance']:.9g}; "
        f"weighted correlation {stats['weighted_pearson_correlation']:.9f}"
        for target, stats in pair_stats.items()
    )
    markdown = f"""# H217 A3 TCN conditional fold-0 comparison

This continuation compares the fixed TCN against the own seed-3 `c20261003_combo_s3_f0` epoch-3 parent on the same 468 exposed fold-0 groups. The TCN standalone outcome had already been observed when this comparison was requested, so this is adaptive discovery analysis; the parent and blend outcomes were not inspected before the fixed weight set was registered. Every blend weight is reported and none is selected.

Parent score: **WP {baseline_score["wp"]:.9f}** (t0 {baseline_score["t0"]:.9f}; t1 {baseline_score["t1"]:.9f}).

{chr(10).join(rows)}

Pair prediction relationship on the same official mask and target-derived weights:

{pair_lines}

The parent, its two own since-teacher ancestors, and its own propensity source all bind to the fold-0 split and have zero fit/evaluation component overlap. The TCN is raw-only scratch; the parent GRU uses those teacher/propensity ancestors, so this is conditional prediction complementarity, not a matched family or compute-controlled comparison. The bootstrap is conditional on fixed predictions and exposed components; it does not estimate training-seed or public-transfer uncertainty. No all-data comparison or public forecast is made.

Targets use the frozen scorer's memory-mapped stored NPY members inside `valid_targets.npz`; the parent prediction is memory-mapped directly and the TCN prediction is memory-mapped from its delivered tar member. No Parquet footer, compressed NPZ member, fit, GPU, or provider operation was used.

Machine-readable receipt: `{REPORT.name}`. Re-run with `scripts/limited` under one CPU, 1.5 GB RAM, and 120 seconds:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \\
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \\
scripts/limited .venv/bin/python {Path(__file__).relative_to(ROOT)}
```
"""
    MARKDOWN.write_text(markdown)
    print(
        json.dumps(
            {"report": str(REPORT), "baseline": baseline_score, "blends": blend_results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
