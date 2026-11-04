from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.multistage_plot_ensemble import (  # noqa: E402
    export_stage4_ensemble_plots,
    export_stage4_trial_plots,
)
from wnn.multistage_search import (  # noqa: E402
    append_event,
    atomic_write_json,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now  # noqa: E402
from wnn.multistage_search_ensemble_finalized_inputs import (  # noqa: E402
    load_finalized_candidates,
    read_bound_campaign_reference,
    regular_campaign_file,
    verify_finalized_candidate_inputs_unchanged,
)
from wnn.multistage_search_ensemble_math import accumulate_quadratic, solve_simplex  # noqa: E402
from wnn.multistage_search_ensemble_optuna import search_wp_weights  # noqa: E402
from wnn.multistage_search_ensemble_result import build_stage4_ensemble_result  # noqa: E402
from wnn.splits import MULTISTAGE_AUXILIARY_SPLIT_SCHEMA, load_campaign_split  # noqa: E402


def run_stage4_from_finalized_results(
    run_dir: Path,
    data_root: Path,
    auxiliary_manifest: Path,
    portable_data_manifest: Path,
    candidate_refs: Sequence[str],
) -> dict[str, object]:
    """Run the frozen 24-trial CPU simplex search from finalizer-produced fit receipts."""
    run_dir = run_dir.resolve(strict=True)
    portable_manifest = regular_campaign_file(portable_data_manifest, "portable data registry")
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(portable_manifest)
    run_manifest = read_json(regular_campaign_file(run_dir / "run.json", "campaign run manifest"))
    _, admission = read_bound_campaign_reference(run_manifest.get("admission"), "admission")
    _, contract = read_bound_campaign_reference(run_manifest.get("contract"), "contract")
    data = admission.get("data")
    if not isinstance(data, Mapping) or data_root.resolve(strict=True) != Path(
        str(data.get("data_root"))
    ).resolve(strict=True):
        raise ValueError("stage 4 data root differs from the frozen campaign admission")
    primary_sha = data.get("primary_split_sha256")
    auxiliary_ref = data.get("auxiliary_manifest")
    if (
        not isinstance(primary_sha, str)
        or len(primary_sha) != 64
        or not isinstance(auxiliary_ref, Mapping)
    ):
        raise ValueError("stage 4 admission omits the pinned primary or auxiliary identity")
    frozen_aux, _ = read_bound_campaign_reference(
        run_manifest.get("auxiliary_manifest"), "auxiliary split"
    )
    auxiliary_manifest = regular_campaign_file(auxiliary_manifest, "auxiliary split")
    if auxiliary_manifest != frozen_aux:
        raise ValueError("stage 4 auxiliary split differs from the frozen campaign reference")
    auxiliary_sha = file_sha256(auxiliary_manifest)
    split = load_campaign_split(auxiliary_manifest, data_root)
    if (
        split.schema != MULTISTAGE_AUXILIARY_SPLIT_SCHEMA
        or split.evaluation_source != "train"
        or split.sha256 != auxiliary_sha
        or auxiliary_ref.get("sha256") != auxiliary_sha
        or split.primary_split_sha256 != primary_sha
    ):
        raise ValueError("stage 4 requires the exact frozen train-only auxiliary control split")
    expected_fit = [["train", group] for group in split.train_groups]
    expected_control = [["train", group] for group in split.evaluation_groups]
    candidates = load_finalized_candidates(
        run_dir,
        candidate_refs,
        auxiliary_sha,
        primary_sha,
        split,
        expected_fit,
        expected_control,
    )

    stage4 = admission.get("stage4", {})
    if not isinstance(stage4, Mapping):
        raise ValueError("stage 4 frozen admission omits its search policy")
    trial_cap = stage4.get("max_optuna_trials", stage4.get("optuna_trials", 24))
    candidate_cap = stage4.get("max_candidate_count", 32)
    if (
        type(trial_cap) is not int
        or trial_cap != 24
        or type(candidate_cap) is not int
        or not 2 <= candidate_cap <= 32
    ):
        raise ValueError("stage 4 frozen admission must request exactly24 Optuna trials")
    if len(candidates) > candidate_cap:
        raise ValueError(f"candidate count exceeds frozen Stage 4 cap of {candidate_cap}")
    if any(
        stage4.get(name, default) != default
        for name, default in (
            ("optuna_trials", 24),
            ("max_cpu_wall_seconds", 1800),
            ("max_search_wall_seconds", 1800),
            ("max_trial_wall_seconds", 600),
            ("max_optuna_trial_wall_seconds", 600),
        )
    ):
        raise ValueError("frozen Stage 4 time/trial caps differ from the bounded ensemble engine")
    if stage4.get("primary_valid_used", False) is not False:
        raise ValueError("Stage 4 admission must prohibit primary validation for weight search")
    output = run_dir / "stage4"
    if (
        output.is_symlink()
        or not output.is_dir()
        or any(
            (output / name).exists() or (output / name).is_symlink()
            for name in ("optuna.db", "ensemble_result.json", "runner_request.json")
        )
    ):
        raise FileExistsError("stage 4 must use its fresh prepared output and Optuna study")
    try:
        import matplotlib as mpl
        import optuna
        import pyarrow
        import scipy
    except ImportError as error:
        raise RuntimeError(
            "Stage 4 needs a CPU environment with Optuna, SciPy, PyArrow, Matplotlib, and NumPy; "
            "use the team's compatible research environment without GPU access"
        ) from error
    required_sampler_options = {"multivariate", "group", "constant_liar"}
    supported_sampler_options = set(inspect.signature(optuna.samplers.TPESampler).parameters)
    if not required_sampler_options <= supported_sampler_options:
        raise RuntimeError(
            "Stage 4 needs Optuna with TPESampler multivariate/group/constant_liar support"
        )
    source_paths = [
        ROOT / "scripts/multistage_stage4_runner.py",
        ROOT / "src/wnn/multistage_search_ensemble_descendants.py",
        ROOT / "src/wnn/multistage_search_ensemble_finalized_inputs.py",
        ROOT / "src/wnn/multistage_search_ensemble_inputs.py",
        ROOT / "src/wnn/multistage_search_ensemble_math.py",
        ROOT / "src/wnn/multistage_search_ensemble_optuna.py",
        ROOT / "src/wnn/multistage_search_ensemble_result.py",
        ROOT / "src/wnn/multistage_search_reviewed_successor_validation.py",
        ROOT / "src/wnn/multistage_search_successor_parent.py",
        ROOT / "src/wnn/multistage_search.py",
        ROOT / "src/wnn/splits.py",
        ROOT / "src/wnn/training/sequence_config.py",
        ROOT / "src/wnn/training/sequence_monitor_identity.py",
        ROOT / "src/wnn/training/sequence_cache.py",
        ROOT / "src/wnn/training/sequence_cache_ensemble.py",
        ROOT / "src/wnn/training/sequence_cache_ensemble_lineage.py",
        ROOT / "src/wnn/training/sequence_cache_ensemble_provenance.py",
        ROOT / "src/wnn/training/sequence_cache_weighted_pool_lineage.py",
        ROOT / "src/wnn/multistage_plot_ensemble.py",
    ]
    request = {
        "schema": "wnn.multistage-stage4-runner-request.v1",
        "created_at_utc": utc_now(),
        "command": sys.argv,
        "run_manifest_sha256": file_sha256(run_dir / "run.json"),
        "admission_sha256": run_manifest["admission"]["sha256"],
        "contract_sha256": run_manifest["contract"]["sha256"],
        "auxiliary_split_sha256": auxiliary_sha,
        "portable_data_manifest_sha256": file_sha256(portable_manifest),
        "primary_split_sha256": split.primary_split_sha256,
        "scope": "inner_train_proxy",
        "used_primary_valid": False,
        "cpu_only": True,
        "optuna_trials": trial_cap,
        "optuna_version": str(optuna.__version__),
        "scipy_version": str(scipy.__version__),
        "pyarrow_version": str(pyarrow.__version__),
        "runner_sources": {
            path.relative_to(ROOT).as_posix(): file_sha256(path) for path in source_paths
        },
        "matplotlib_version": str(mpl.__version__),
        "candidate_trial_results": [
            {
                "name": item["name"],
                "stage": item["stage"],
                "trial_number": item["trial_number"],
                "family": item["family"],
                "config_sha256": item["config_sha256"],
                "config_identity_sha256": item["config_identity_sha256"],
                "config_path": item["config_path"],
                "path": item["trial_result_path"],
                "sha256": item["trial_result_sha256"],
                "trial_request_path": item["trial_request_path"],
                "trial_request_sha256": item["trial_request_sha256"],
            }
            for item in candidates
        ],
    }
    atomic_write_json(output / "runner_request.json", request, replace=False)
    candidate_predictions = [item["predictions"] for item in candidates]
    quadratic, linear, constant, total_weight = accumulate_quadratic(
        data_root, split, candidate_predictions
    )
    warmstart_weights = solve_simplex(quadratic, linear)
    warmstart_mse = float(
        (
            warmstart_weights @ quadratic @ warmstart_weights
            - 2 * warmstart_weights @ linear
            + constant
        )
        / total_weight
    )
    weights, score, trial_count = search_wp_weights(
        run_dir,
        data_root,
        split,
        candidate_predictions,
        [str(item["name"]) for item in candidates],
        warmstart_weights,
    )
    if trial_count != trial_cap:
        raise RuntimeError(f"only {trial_count} of the admitted {trial_cap} trials were executed")
    study_name = f"{run_manifest['experiment_id']}-stage4-wp"
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{(output / 'optuna.db').as_posix()}",
    )
    if len(study.trials) != trial_cap or any(
        trial.state.name != "COMPLETE" for trial in study.trials
    ):
        raise RuntimeError("Stage 4 requires24 completed actual pooled-WP Optuna trials")
    if (weights < 0).any() or abs(float(weights.sum()) - 1.0) > 1e-10:
        raise ValueError("selected ensemble weights are not nonnegative and unit-sum")
    verify_finalized_candidate_inputs_unchanged(candidates)

    baseline = contract.get("baseline", {})
    result = build_stage4_ensemble_result(
        candidates,
        weights,
        score,
        trial_count,
        warmstart_weights,
        warmstart_mse,
        auxiliary_sha,
        split.primary_split_sha256,
        expected_control,
        baseline.get("canonical_fresh_TL3_wp"),
    )
    destination = output / "ensemble_result.json"
    atomic_write_json(destination, result, replace=False)
    trial_plots = export_stage4_trial_plots(run_dir)
    ensemble_plots = export_stage4_ensemble_plots(destination)
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "stage4_ensemble_complete",
            "at_utc": utc_now(),
            "stage": 4,
            "result_path": str(destination.resolve()),
            "result_sha256": file_sha256(destination),
            "objective_value": score["wp"],
            "status": "COMPLETE",
        },
    )
    return result | {
        "result_path": str(destination.resolve()),
        "result_sha256": file_sha256(destination),
        "trial_plots": trial_plots,
        "ensemble_plots": ensemble_plots,
    }


def parse_args() -> argparse.Namespace:
    """Parse campaign paths and exact finalizer-produced candidate references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--auxiliary-split", required=True, type=Path)
    parser.add_argument("--portable-data-manifest", required=True, type=Path)
    parser.add_argument(
        "--candidate", action="append", required=True, help="NAME=trial_result.json"
    )
    return parser.parse_args()


def main() -> None:
    """Run CPU-only Stage 4 without fitting models or inspecting primary validation rows."""
    args = parse_args()
    try:
        result = run_stage4_from_finalized_results(
            args.run_dir,
            args.data_root,
            args.auxiliary_split,
            args.portable_data_manifest,
            args.candidate,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"Stage 4 refused or incomplete: {error}") from error
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
