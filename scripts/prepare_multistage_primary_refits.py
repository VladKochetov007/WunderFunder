from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.multistage_primary_refit_inputs import validate_and_rank_stage1  # noqa: E402
from wnn.multistage_primary_report import (  # noqa: E402
    AUXILIARY_CONTROL_COUNT,
    AUXILIARY_FIT_COUNT,
    AUXILIARY_SPLIT_SHA256,
    PRIMARY_SPLIT_SHA256,
    PRIMARY_TRAIN_COUNT,
    PRIMARY_VALID_COUNT,
)
from wnn.multistage_search import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
)
from wnn.multistage_search_ensemble_finalized_inputs import regular_campaign_file  # noqa: E402
from wnn.splits import CampaignSplit, load_campaign_split  # noqa: E402
from wnn.training.sequence_config import load_config  # noqa: E402
from wnn.training.sequence_provenance import source_identity  # noqa: E402

SOURCE4_SHA256 = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"
PROFILE_UPDATES = 50
PROFILE_WALL_SECONDS = 180
FIT_WALL_SECONDS = 7200


def _portable_config(raw_config: dict[str, object]) -> dict[str, object]:
    """Remove only redundant nested precision without changing normalized values."""
    result = dict(raw_config)
    model = result.get("model")
    if not isinstance(model, dict):
        raise ValueError("candidate config has no model mapping")
    model = dict(model)
    if "precision_mode" in model:
        if model["precision_mode"] != result.get("precision_mode"):
            raise ValueError("nested precision differs from the top-level setting")
        model.pop("precision_mode")
    result["model"] = model
    return result


def _primary_split(
    path: Path, data_root: Path, auxiliary_path: Path, portable: Path
) -> tuple[CampaignSplit, CampaignSplit]:
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(portable)
    primary = load_campaign_split(path, data_root)
    auxiliary = load_campaign_split(auxiliary_path, data_root)
    if (
        primary.sha256 != PRIMARY_SPLIT_SHA256
        or len(primary.train_groups) != PRIMARY_TRAIN_COUNT
        or primary.valid_groups
        or len(primary.evaluation_groups) != PRIMARY_VALID_COUNT
        or auxiliary.sha256 != AUXILIARY_SPLIT_SHA256
        or len(auxiliary.train_groups) != AUXILIARY_FIT_COUNT
        or len(auxiliary.evaluation_groups) != AUXILIARY_CONTROL_COUNT
    ):
        raise ValueError("primary or auxiliary membership differs from the frozen protocol")
    return primary, auxiliary


def prepare(args: argparse.Namespace) -> dict[str, object]:
    """Write reviewed full-train refit plans without profiling or fitting models."""
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("preparation output must be fresh")
    run_dir = args.run_dir.resolve(strict=True)
    stage4 = regular_campaign_file(args.stage4_result, "Stage4 result")
    if not stage4.is_relative_to(run_dir):
        raise ValueError("Stage4 result must be inside the campaign root")
    data_root = args.data_root.resolve(strict=True)
    runtime_root = args.runtime_root.resolve(strict=True)
    source_root = args.training_source_root.resolve(strict=True)
    primary_path = regular_campaign_file(args.primary_split_manifest, "primary manifest")
    auxiliary_path = regular_campaign_file(args.auxiliary_split_manifest, "auxiliary manifest")
    portable = regular_campaign_file(args.portable_data_manifest, "portable data registry")
    dependency_root = args.dependency_root.resolve(strict=True)
    primary, auxiliary = _primary_split(primary_path, data_root, auxiliary_path, portable)
    source = source_identity(source_root, source_root / "scripts/train_sequence.py")
    if source.get("sha256") != SOURCE4_SHA256:
        raise ValueError("training source differs from the reviewed Source4 snapshot")
    ranked, stage4_request_sha = validate_and_rank_stage1(
        run_dir, stage4, primary.sha256, auxiliary.sha256, auxiliary
    )
    if len(ranked) < 2:
        raise ValueError("at least two direct Stage1 candidates are needed")
    selected = ranked[: min(args.count, len(ranked))]
    output.mkdir(parents=True, exist_ok=False)
    config_dir, request_dir = output / "configs", output / "requests"
    config_dir.mkdir()
    request_dir.mkdir()
    prepared: list[dict[str, object]] = []
    for rank, item in enumerate(selected, 1):
        candidate = item["candidate"]
        config = _portable_config(item["raw_config"])
        fit_id = f"primary-refit-{rank:02d}-s1t{int(candidate['trial_number']):06d}-{candidate['family']}"
        config_path = config_dir / f"{fit_id}.json"
        atomic_write_json(config_path, config, replace=False)
        normalized = load_config(config_path)
        if normalized != item["normalized_config"]:
            raise ValueError("portable config changed the original normalized recipe")
        config_sha = file_sha256(config_path)
        profile_output = runtime_root / "profiles" / f"{fit_id}-profile"
        approved_profile = profile_output / "approved_profile_receipt.json"
        fit_output = runtime_root / "runs" / fit_id
        trainer = source_root / "scripts/train_sequence.py"
        profile_argv = [
            args.python,
            str(trainer),
            "--config",
            str(config_path.resolve()),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(primary_path),
            "--output",
            str(profile_output),
            "--profile-updates",
            str(PROFILE_UPDATES),
        ]
        fit_argv = [
            args.python,
            str(trainer),
            "--config",
            str(config_path.resolve()),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(primary_path),
            "--output",
            str(fit_output),
            "--profile-receipt",
            str(approved_profile),
        ]
        request = {
            "schema": "wnn.multistage-primary-refit-request.v1",
            "status": "pending_profile_and_independent_review",
            "fit_id": fit_id,
            "selected_by": {
                "stage": 1,
                "trial_number": candidate["trial_number"],
                "family": candidate["family"],
                "fixed_final_inner_wp": item["inner_wp"],
            },
            "config": {
                "path": str(config_path.resolve()),
                "sha256": config_sha,
                "normalized_identity_sha256": canonical_json_sha256(normalized),
                "source_config_sha256": candidate["config_sha256"],
                "epochs_preserved": normalized["epochs"],
                "seed_preserved": normalized["seed"],
                "precision_mode": "fp32",
                "no_teacher": True,
                "no_initializer": True,
            },
            "training_source": {"sha256": source["sha256"], "files": source["files"]},
            "primary_split": {
                "path": str(primary_path),
                "sha256": primary.sha256,
                "fit_train_sequences": len(primary.train_groups),
                "fit_valid_sequences": 0,
                "evaluation_sequences": len(primary.evaluation_groups),
            },
            "profile": {
                "updates": PROFILE_UPDATES,
                "max_wall_seconds": PROFILE_WALL_SECONDS,
                "output": str(profile_output),
                "argv": profile_argv,
                "environment_template": {
                    "CUDA_VISIBLE_DEVICES": "<independently reviewed GPU UUID>",
                    "CUBLAS_EMULATE_SINGLE_PRECISION": "0",
                    "WNN_PORTABLE_DATA_MANIFEST": str(portable),
                    "WNN_PROJECT_ROOT": str(data_root),
                    "PYTHONPATH": f"{source_root / 'src'}:{dependency_root}",
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                },
                "primary_valid_score_used_for_admission": False,
            },
            "fit": {
                "max_wall_seconds": FIT_WALL_SECONDS,
                "output": str(fit_output),
                "argv_after_approval_and_reservation": fit_argv,
                "resume": False,
                "initializer": None,
                "teacher_or_cache": None,
                "transferred_auxiliary_weights": False,
                "primary_valid_role": "post-fit evaluation only; development-exposed",
            },
            "review_gate": "fresh Source4 profile; independent exact-config/source/data review; Stage4 reserve_fit; native guardian",
            "auxiliary_checkpoint_sha256_selection_evidence_only": candidate["checkpoint_sha256"],
        }
        request_path = request_dir / f"{fit_id}.json"
        atomic_write_json(request_path, request, replace=False)
        prepared.append(
            {
                "fit_id": fit_id,
                "request_path": str(request_path.resolve()),
                "request_sha256": file_sha256(request_path),
                "config_path": str(config_path.resolve()),
                "config_sha256": config_sha,
                "fixed_final_inner_wp": item["inner_wp"],
                "source_result_path": candidate["trial_result_path"],
                "source_result_sha256": candidate["trial_result_sha256"],
            }
        )
    receipt = {
        "schema": "wnn.multistage-primary-refit-selection.v1",
        "status": "prepared_pending_review_and_profiles",
        "selection_metric": "fixed-final inner_train_proxy pooled WP",
        "selection_used_primary_valid": False,
        "stage4_result": {
            "path": str(stage4),
            "sha256": file_sha256(stage4),
            "runner_request_sha256": stage4_request_sha,
        },
        "primary_split_sha256": primary.sha256,
        "auxiliary_split_sha256": auxiliary.sha256,
        "source4_sha256": source["sha256"],
        "ranked_direct_stage1": [
            {
                "trial_number": row["candidate"]["trial_number"],
                "family": row["candidate"]["family"],
                "fixed_final_inner_wp": row["inner_wp"],
                "result_path": row["candidate"]["trial_result_path"],
                "result_sha256": row["candidate"]["trial_result_sha256"],
            }
            for row in ranked
        ],
        "selected": prepared,
        "interpretation": "Auxiliary checkpoints rank candidates only. Each refit starts fresh on all 10,325 official training sequences; any refit quality claim requires its new primary score.",
    }
    atomic_write_json(output / "selection_receipt.json", receipt, replace=False)
    atomic_write_json(
        output / "independent_review_inputs.json",
        {
            "schema": "wnn.multistage-primary-refit-review-inputs.v1",
            "status": "pending",
            "selection_receipt_sha256": file_sha256(output / "selection_receipt.json"),
            "selected": prepared,
            "required_checks": [
                "Source4 strict-FP32 profile and actual-device/resource identity",
                "exact raw/normalized config identity with original epochs and seeds",
                "10,325 fit / 0 fit-valid / 1,873 evaluation membership",
                "no teacher, initializer, resumed optimizer state, or auxiliary weight transfer",
                "primary valid used only for fixed-epoch evaluation; label as development-exposed",
            ],
            "approval": None,
        },
        replace=False,
    )
    return receipt


def parse_args() -> argparse.Namespace:
    """Read frozen campaign paths and output count for the local preparer."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-dir",
        "stage4-result",
        "data-root",
        "primary-split-manifest",
        "auxiliary-split-manifest",
        "portable-data-manifest",
        "training-source-root",
        "dependency-root",
        "runtime-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--count", type=int, choices=(2, 3, 4), default=4)
    return parser.parse_args()


def main() -> None:
    """Create review inputs only; this command never starts a profile or fit."""
    try:
        receipt = prepare(parse_args())
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"Primary refit preparation refused: {error}") from error
    sys.stdout.write(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
