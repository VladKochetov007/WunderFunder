from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json

REQUEST_POLICY_R2_SHA = "153655303a89dfd9dd2372a20238327aba36337a36141373c51c5cc1e30c2f0e"
REQUEST_CHECKER_R2_SHA = "af50cd5f744f996971fd64330d9e89f67415beeea72d02a25edadd0b87d9a19c"
REVIEW_POLICY_R4_SHA = "35defc15dc51ea98bf33ed262b11b896b7c17cf73a6c2075e2c57c88ad12873f"
REVIEW_CHECKER_R4_SHA = "3408e3a1aa0e86ec19f61758582ea1a8b965f4649ddc5a55e7e03aae8913b198"
REVIEW_POLICY_R5_SHA = "409aaf50fd31cbb527aac924c302e4e97586245224e4085c04edb71dee1a1e65"
REVIEW_CHECKER_R5_SHA = "0e00c69de327b0c28a014b4345ef3cfc021051c17e86726d9f53fac0db58ceb1"
REVIEW_POLICY_R6_SHA = "2e5ae7c2edc8a41cf09f7ee8d2f17078b9b80f8bf23eaa6c667ce04254bc49e6"
REVIEW_CHECKER_R6_SHA = "084e971373e535223516115836b42a643b57c65f2d1577454c94140e50d1c94c"
CAMPAIGN_RUN_R2_SHA = "d1f32df19d671bab3bd648bfc004eec65dced7b8d6815076c29c3e5dd4baae8e"


def validate_successor_review(
    run_dir: Path,
    trial_dir: Path,
    stage: int,
    trial_number: int,
    request: Mapping[str, object],
    review_path: Path,
) -> dict[str, object]:
    """Validate an exact frozen R4 receipt or the narrow R5 R2-readonly re-audit."""
    run_dir = run_dir.resolve(strict=True)
    request_path = trial_dir / "trial_request.json"
    profile_dir = trial_dir / "profile"
    native_path = Path(str(request["profile_receipt_path"])).resolve(strict=True)
    attempt_path = profile_dir / "profile_attempt_receipt.json"
    native = read_json(native_path)
    attempt_sha = file_sha256(attempt_path)
    review_path = review_path.resolve(strict=True)
    review = read_json(review_path)
    policy_sha = review.get("review_policy_sha256")
    binding = review.get("request_subject_review_binding")
    if (
        review.get("schema") != "wnn.multistage-successor-profile-review.v1"
        or review.get("status") != "approved"
        or review.get("reviewer") != "independent"
        or review.get("stage") != stage
        or review.get("trial_number") != trial_number
        or review.get("native_profile_sha256") != file_sha256(native_path)
        or review.get("identity_sha256") != canonical_json_sha256(native.get("identity"))
        or review.get("trial_request_sha256") != file_sha256(request_path)
        or review.get("profile_attempt_receipt_sha256") != attempt_sha
        or review.get("config_sha256") != request.get("config_sha256")
        or review.get("auxiliary_split_sha256") != request.get("auxiliary_split_sha256")
        or review.get("primary_split_sha256") != request.get("primary_split_sha256")
        or review.get("source_sha256") != request.get("source_sha256")
        or review.get("initializer_path") != request.get("initializer_path")
        or review.get("initializer_sha256") != request.get("initializer_sha256")
        or review.get("teacher_cache_manifest_sha256")
        != request.get("teacher_cache_manifest_sha256")
        or not isinstance(binding, Mapping)
    ):
        raise ValueError("successor review does not bind the exact trial/profile/lineage request")
    _validate_request_compatibility(request, binding)
    if policy_sha == REVIEW_POLICY_R4_SHA:
        _validate_review_artifact_pair(review, policy_sha, REVIEW_CHECKER_R4_SHA, "R4")
        policy_path = review.get("review_policy_path")
        if (
            not isinstance(policy_path, str)
            or not policy_path.endswith("PROFILE_REVIEW_POLICY_AUTOMATED_R4.json")
            or binding.get("reviewer_script_sha256") != REQUEST_CHECKER_R2_SHA
            or "reviewer_script_sha256" not in request
        ):
            raise ValueError("R4 receipt lacks its original request/reviewer identity binding")
        return _review_summary(review_path, review, "R4", policy_sha, REVIEW_CHECKER_R4_SHA)
    if policy_sha == REVIEW_POLICY_R5_SHA:
        _validate_review_artifact_pair(review, policy_sha, REVIEW_CHECKER_R5_SHA, "R5")
        _validate_readonly_r2_binding(run_dir, request, binding)
        return _review_summary(review_path, review, "R5", policy_sha, REVIEW_CHECKER_R5_SHA)
    if policy_sha == REVIEW_POLICY_R6_SHA:
        _validate_review_artifact_pair(review, policy_sha, REVIEW_CHECKER_R6_SHA, "R6")
        _validate_readonly_r2_binding(run_dir, request, binding)
        return _review_summary(review_path, review, "R6", policy_sha, REVIEW_CHECKER_R6_SHA)
    raise ValueError("successor review policy is not one of the frozen R4/R5/R6 versions")


def _validate_request_compatibility(
    request: Mapping[str, object], binding: Mapping[str, object]
) -> None:
    if (
        request.get("schema") != "wnn.multistage-trial-plan.v1"
        or request.get("primary_valid_used") is not False
        or request.get("objective_scope") != "inner_train_proxy"
        or request.get("profile_checker_policy_sha256") != REQUEST_POLICY_R2_SHA
        or request.get("profile_policy_sha256") != REQUEST_POLICY_R2_SHA
        or binding.get("profile_checker_policy_sha256") != REQUEST_POLICY_R2_SHA
        or binding.get("profile_policy_sha256") != REQUEST_POLICY_R2_SHA
        or binding.get("reviewer_script_sha256") != REQUEST_CHECKER_R2_SHA
        or binding.get("read_only_compatibility_match") is not True
    ):
        raise ValueError("successor review is not the exact read-only R2 request compatibility")


def _validate_review_artifact_pair(
    review: Mapping[str, object], policy_sha: str, checker_sha: str, revision: str
) -> None:
    policy_path = Path(str(review.get("review_policy_path"))).resolve(strict=True)
    expected_policy_name = f"PROFILE_REVIEW_POLICY_AUTOMATED_{revision}.json"
    expected_checker_name = f"review_multistage_profile_{revision.lower()}.py"
    checker_path = policy_path.parent / expected_checker_name
    if (
        policy_path.name != expected_policy_name
        or file_sha256(policy_path) != policy_sha
        or not checker_path.is_file()
        or file_sha256(checker_path) != checker_sha
        or review.get("reviewer_policy_sha256") != policy_sha
        or review.get("reviewer_policy_script_sha256") != checker_sha
        or review.get("reviewer_script_sha256") != checker_sha
    ):
        raise ValueError(f"{revision} reviewer files or receipt bindings changed")


def _validate_readonly_r2_binding(
    run_dir: Path, request: Mapping[str, object], binding: Mapping[str, object]
) -> None:
    run_path = run_dir / "run.json"
    run_sha = file_sha256(run_path)
    run = read_json(run_path)
    checker_ref = run.get("profile_checker_policy")
    reviewer_ref = run.get("profile_reviewer_script")
    if (
        "reviewer_script_sha256" in request
        or binding.get("request_reviewer_script_key_absent") is not True
        or binding.get("campaign_run_manifest_path") != str(run_path.resolve())
        or binding.get("campaign_run_manifest_sha256") != CAMPAIGN_RUN_R2_SHA
        or run_sha != CAMPAIGN_RUN_R2_SHA
        or not isinstance(checker_ref, Mapping)
        or not isinstance(reviewer_ref, Mapping)
        or binding.get("campaign_profile_checker_policy_path") != checker_ref.get("path")
        or binding.get("campaign_profile_checker_policy_sha256") != REQUEST_POLICY_R2_SHA
        or binding.get("campaign_profile_reviewer_script_path") != reviewer_ref.get("path")
        or binding.get("campaign_profile_reviewer_script_sha256") != REQUEST_CHECKER_R2_SHA
        or checker_ref.get("sha256") != REQUEST_POLICY_R2_SHA
        or reviewer_ref.get("sha256") != REQUEST_CHECKER_R2_SHA
    ):
        raise ValueError("successor re-audit is not backed by the exact campaign R2 run")
    if (
        file_sha256(Path(str(checker_ref["path"])).resolve(strict=True)) != REQUEST_POLICY_R2_SHA
        or file_sha256(Path(str(reviewer_ref["path"])).resolve(strict=True))
        != REQUEST_CHECKER_R2_SHA
    ):
        raise ValueError("read-only compatibility proof no longer matches the original R2 files")


def _review_summary(
    path: Path, review: Mapping[str, object], revision: str, policy_sha: str, checker_sha: str
) -> dict[str, object]:
    return {
        "revision": revision,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "policy_sha256": policy_sha,
        "checker_sha256": checker_sha,
        "request_subject_review_binding": review["request_subject_review_binding"],
    }
