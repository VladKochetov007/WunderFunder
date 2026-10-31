from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from wnn.multistage_search import file_sha256, read_json
from wnn.multistage_search_successor_proposal_cap_evidence import (
    validate_frozen_stage2_limits,
    validate_stage2_proposal_history,
)

OVERRIDE_SHA256 = "e59d3a312099263636b4b63e89eb65d00a7c44e5d37aa479b35fbb585521af10"
REVIEW_SHA256 = "54500c257a8f844d2cb4bdd9da91767746b0953558d58be36c81c6b838fa1af1"
CONTINUATION_SHA256 = "03796b7cd5f1d57c6aa44590acf51edaab28f9b036cdaaeeb8ae5ef88e165562"
RECIPE_REVIEW_SHA256 = "2d0dd946af20a7c476d54a15a35d898e6402d52074510810afa504a2c913b8bf"
CAMPAIGN_RUN_SHA256 = "d1f32df19d671bab3bd648bfc004eec65dced7b8d6815076c29c3e5dd4baae8e"
PRIMARY_SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SPLIT_SHA256 = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
RECIPE_DIR = "research/experiment_drafts/multistage-optuna-b300-20261031"
CAPS = {"1": 16, "2": 8, "3": 12, "4": 12}


@dataclass(frozen=True)
class Stage2ProposalLimit:
    """Approved proposal ceiling and optional single permitted new trial."""

    max_trials: int
    permitted_next_trial_number: int | None = None
    amendment_path: Path | None = None
    amendment_sha256: str | None = None
    review_path: Path | None = None
    review_sha256: str | None = None


def resolve_stage2_proposal_limit(
    run_dir: Path,
    run_manifest: Mapping[str, object],
    admission: Mapping[str, object],
    study: object,
) -> Stage2ProposalLimit:
    """Use the exact reviewed sidecar only after checking trial-0 failure evidence."""
    stage2 = admission.get("stage2")
    base_cap = stage2.get("max_trials") if isinstance(stage2, Mapping) else None
    if type(base_cap) is not int or base_cap < 1:
        raise ValueError("Stage2 admission has no valid proposal cap")
    if base_cap != 6:
        return Stage2ProposalLimit(base_cap)
    trials = list(study.get_trials(deepcopy=False))
    numbers = sorted(getattr(trial, "number", -1) for trial in trials)
    if numbers != list(range(len(numbers))) or len(numbers) > 7:
        raise ValueError("Stage2 Optuna proposals are not a contiguous 0-through-6 history")
    if len(numbers) < base_cap:
        return Stage2ProposalLimit(base_cap)
    receipt_dir = run_dir.resolve(strict=True).parent / "receipts"
    override_path = receipt_dir / "STAGE2_PROPOSAL_CAP_OVERRIDE_R1.json"
    review_path = receipt_dir / "INDEPENDENT_STAGE2_PROPOSAL_CAP_AMENDMENT_R1.json"
    if not override_path.is_file() or not review_path.is_file():
        if len(numbers) == base_cap:
            return Stage2ProposalLimit(base_cap)
        raise ValueError("Stage2 proposal 6 exists without the approved cap sidecar")
    _validate_sidecar_pair(
        run_dir.resolve(strict=True), run_manifest, admission, trials, override_path, review_path
    )
    return Stage2ProposalLimit(
        max_trials=7,
        permitted_next_trial_number=6 if len(numbers) == base_cap else None,
        amendment_path=override_path.resolve(strict=True),
        amendment_sha256=OVERRIDE_SHA256,
        review_path=review_path.resolve(strict=True),
        review_sha256=REVIEW_SHA256,
    )


def _validate_sidecar_pair(
    run_dir: Path,
    run_manifest: Mapping[str, object],
    admission: Mapping[str, object],
    trials: Sequence[object],
    override_path: Path,
    review_path: Path,
) -> None:
    if (
        override_path.is_symlink()
        or review_path.is_symlink()
        or file_sha256(override_path) != OVERRIDE_SHA256
        or file_sha256(review_path) != REVIEW_SHA256
        or file_sha256(run_dir / "run.json") != CAMPAIGN_RUN_SHA256
    ):
        raise ValueError("Stage2 proposal amendment, review, or campaign bytes changed")
    recipe_root = Path(__file__).resolve().parents[2] / RECIPE_DIR
    continuation = recipe_root / "recipes" / "R2_CONTINUATION.json"
    recipe_review = recipe_root / "INDEPENDENT_R2_RECIPE_REVIEW.json"
    if (
        file_sha256(continuation) != CONTINUATION_SHA256
        or file_sha256(recipe_review) != RECIPE_REVIEW_SHA256
    ):
        raise ValueError("Stage2 cap sidecar no longer matches the reviewed R2 recipe")
    override, review = read_json(override_path), read_json(review_path)
    recipe, recipe_receipt = read_json(continuation), read_json(recipe_review)
    _validate_artifact_bindings(override, review, recipe, recipe_receipt)
    validate_frozen_stage2_limits(run_dir, run_manifest, admission)
    validate_stage2_proposal_history(run_dir, admission, trials)


def _validate_artifact_bindings(
    override: Mapping[str, object],
    review: Mapping[str, object],
    recipe: Mapping[str, object],
    recipe_review: Mapping[str, object],
) -> None:
    bindings = override.get("bindings")
    review_bindings = review.get("bindings")
    target, decision = override.get("target"), review.get("decision")
    known_failure = override.get("known_failed_proposal")
    basis = review.get("review_basis")
    reviewed_failure = basis.get("known_trial0_failure") if isinstance(basis, Mapping) else None
    if not all(isinstance(item, Mapping) for item in (bindings, review_bindings, target, decision)):
        raise ValueError("Stage2 cap amendment has malformed bindings")
    expected_bindings = {
        "continuation_path": f"{RECIPE_DIR}/recipes/R2_CONTINUATION.json",
        "continuation_sha256": CONTINUATION_SHA256,
        "recipe_review_path": f"{RECIPE_DIR}/INDEPENDENT_R2_RECIPE_REVIEW.json",
        "recipe_review_sha256": RECIPE_REVIEW_SHA256,
        "campaign_run_json_sha256": CAMPAIGN_RUN_SHA256,
        "primary_split_sha256": PRIMARY_SPLIT_SHA256,
        "auxiliary_split_sha256": AUXILIARY_SPLIT_SHA256,
    }
    limits = recipe.get("limits")
    data = recipe.get("data")
    if (
        dict(bindings) != expected_bindings
        or dict(review_bindings) != expected_bindings
        or override.get("schema") != "wnn.multistage-stage-proposal-cap-override.v1"
        or override.get("status") != "proposed_for_independent_approval"
        or override.get("experiment_id") != "E-MULTISTAGE-OPTUNA-B300-20261031-R1"
        or review.get("schema") != "wnn.multistage-independent-runtime-projection-amendment.v1"
        or review.get("review_id") != "INDEPENDENT_STAGE2_PROPOSAL_CAP_AMENDMENT_R1"
        or review.get("reviewer") != "multistage_independent_critic"
        or review.get("status") != "approved"
        or review.get("engineering_admission_only") is not True
        or review.get("scientific_candidate_promotion") is not False
        or review.get("override_path") != f"{RECIPE_DIR}/STAGE2_PROPOSAL_CAP_OVERRIDE_R1.json"
        or review.get("override_sha256") != OVERRIDE_SHA256
        or target.get("stage") != 2
        or target.get("field") != "max_trials"
        or target.get("expected_current_value") != 6
        or target.get("proposed_value") != 7
        or target.get("permitted_new_trial_number") != 6
        or decision.get("stage") != 2
        or decision.get("field") != "max_trials"
        or decision.get("expected_existing_value") != 6
        or decision.get("approved_derived_value") != 7
        or recipe_review.get("status") != "approved"
        or recipe_review.get("continuation_sha256") != CONTINUATION_SHA256
        or not isinstance(limits, Mapping)
        or limits.get("stage_fit_caps") != CAPS
        or limits.get("max_total_fits") != 48
        or limits.get("max_total_profile_attempts") != 56
        or limits.get("campaign_wall_seconds") != 21600
        or limits.get("collection_reserve_seconds") != 1200
        or limits.get("maximum_profile_wall_seconds") != 180
        or recipe.get("stage2", {}).get("additional_fit_cap") != 6
        or not isinstance(data, Mapping)
        or data.get("primary_split_sha256") != PRIMARY_SPLIT_SHA256
        or data.get("auxiliary_split_sha256") != AUXILIARY_SPLIT_SHA256
        or not isinstance(known_failure, Mapping)
        or known_failure.get("trial_number") != 0
        or known_failure.get("process_exit_code") != 1
        or known_failure.get("rank0_log") != "absent"
        or known_failure.get("native_profile_receipt") != "absent"
        or not isinstance(reviewed_failure, Mapping)
        or reviewed_failure.get("trial_number") != 0
        or reviewed_failure.get("reported_exit_code") != 1
    ):
        raise ValueError(
            "Stage2 proposal cap is not the exact independently approved R2 projection"
        )
