from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.e540_kaggle_worker import (
    EXPECTED_PROFILE_ATTEMPTS,
    expected_fit_updates,
    parse_trainer_output,
    trainer_argv,
    validate_frozen_membership,
    validate_profile,
)
from scripts.kaggle_bundle import local_bundle_descriptors
from scripts.kaggle_common import verify_files


def _profile_output(*, step_wall: str = "12.5", skipped: int = 0) -> str:
    evidence = {
        "optimizer_attempts": EXPECTED_PROFILE_ATTEMPTS,
        "optimizer_completed_updates": EXPECTED_PROFILE_ATTEMPTS - skipped,
        "amp_skipped_updates": skipped,
        "loss_checks": EXPECTED_PROFILE_ATTEMPTS,
        "loss_finite": True,
        "gradient_finite": skipped == 0,
        "valid_for_throughput": skipped == 0,
    }
    return (
        f"PROFILE {{\"step_wall\": {step_wall}, \"gpu_total\": 8.0}}\n"
        f"PROFILE_UPDATE_EVIDENCE {json.dumps(evidence)}\n"
    )


def test_profile_gate_requires_three_timed_batches_finite_wall_and_zero_skips() -> None:
    parsed = parse_trainer_output(_profile_output())
    step_wall, evidence = validate_profile(parsed)
    assert step_wall == 12.5
    assert evidence["optimizer_attempts"] == 316
    assert EXPECTED_PROFILE_ATTEMPTS == 316

    with pytest.raises(ValueError, match="finite and positive"):
        validate_profile(parse_trainer_output(_profile_output(step_wall="NaN")))
    with pytest.raises(ValueError, match="316 finite optimizer updates"):
        validate_profile(parse_trainer_output(_profile_output(skipped=1)))


def test_v3_profile_and_fit_pin_the_same_amp_scale_and_no_auxiliary_paths() -> None:
    profile = trainer_argv("ple", profile=True)
    fit = trainer_argv("ple", profile=False)
    assert profile[profile.index("--amp-init-scale") + 1] == "4096"
    assert fit[fit.index("--amp-init-scale") + 1] == "4096"
    for argv in (profile, fit):
        assert argv[argv.index("--loss") + 1] == "wmse"
        assert argv[argv.index("--loss-norm") + 1] == "chunk"
        assert argv[argv.index("--aux-struct") + 1] == "0"
        assert argv[argv.index("--propensity") + 1] == ""
        assert argv[argv.index("--distill") + 1] == ""
        assert "--no-eval-during-fit" in argv
    assert "--profile-steps" not in fit


def test_fit_update_count_uses_drop_last_outer_batches_and_79_chunk_updates() -> None:
    assert expected_fit_updates(340, 172) == (340 + 172) // 2 * 79
    assert expected_fit_updates(341, 172) == (341 + 172) // 2 * 79
    # Frozen V3 membership selects 1,597 train + 252 valid sequences. The
    # batch-2 drop_last loader consumes 1,848 of those 1,849 sequences.
    assert expected_fit_updates(1597, 252) == 72996


def test_external_bundle_inputs_are_not_hashed_in_the_local_controller_checkout() -> None:
    files = {
        "scripts/train.py": {"path": "scripts/train.py", "size": 4, "sha256": "a" * 64},
        "datasets/train.parquet": {
            "path": "datasets/train.parquet",
            "size": 29_000_000_000,
            "sha256": "b" * 64,
        },
    }
    assert local_bundle_descriptors(
        files, {"datasets/train.parquet": "train.parquet"}
    ) == [files["scripts/train.py"]]


def test_input_identity_receipt_persists_expected_and_observed_hashes(tmp_path: Path) -> None:
    value = b"sealed-small-input"
    path = tmp_path / "input.json"
    path.write_bytes(value)
    descriptor = {
        "path": "input.json",
        "size": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }
    receipt = tmp_path / "identity.json"
    observations = verify_files(tmp_path, [descriptor], identity_receipt=receipt)
    assert observations[0]["status"] == "match"
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "verified"
    assert payload["observations"][0]["observed"] == {
        "size": len(value),
        "sha256": descriptor["sha256"],
    }

    altered = {**descriptor, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_files(tmp_path, [altered], identity_receipt=receipt)
    failed = json.loads(receipt.read_text())
    assert failed["observations"][0]["status"] == "mismatch"


def _synthetic_membership() -> tuple[dict[str, object], SimpleNamespace]:
    import numpy as np

    train = list(range(512))
    valid = list(range(2_000, 2_512)) + [3_000]
    evaluation = list(range(4_000, 4_468))
    components = [
        ("train", group, group) for group in train
    ] + [
        ("valid", group, group - 2_000) for group in valid[:-1]
    ] + [
        ("valid", 3_000, 0)
    ] + [
        ("valid", group, 512 + index) for index, group in enumerate(evaluation)
    ]
    selected_components = list(range(512))
    groups = {"train": train, "valid": valid}
    members = sorted((split, group) for split, values in groups.items() for group in values)

    def digest_groups(values: list[int]) -> str:
        return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()

    subset: dict[str, object] = {
        "split_manifest_sha256": "s" * 64,
        "selection_seed": 540,
        "eligible_component_count": 3_148,
        "selected_component_count": 512,
        "selected_components": selected_components,
        "selected_component_sha256": digest_groups(selected_components),
        "selected_groups": groups,
        "selected_group_counts": {name: len(values) for name, values in groups.items()},
        "selected_group_sha256": {
            name: digest_groups(values) for name, values in groups.items()
        },
        "selected_members_sha256": hashlib.sha256(
            json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest(),
        "evaluation_groups": evaluation,
        "evaluation_group_count": 468,
        "evaluation_membership_sha256": digest_groups(evaluation),
    }
    evaluation_components = set(range(512, 980))
    component_by_member = {
        (split, group): component for split, group, component in components
    }

    def assert_fit_groups(fit_train: list[int], fit_valid: list[int]) -> None:
        assert set(fit_train).issubset(train)
        assert set(fit_valid).issubset(valid)
        fitted_components = {component_by_member[("train", group)] for group in fit_train}
        fitted_components.update(
            component_by_member[("valid", group)] for group in fit_valid
        )
        assert not fitted_components.intersection(evaluation_components)

    campaign = SimpleNamespace(
        sha256="s" * 64,
        train_groups=tuple(train),
        valid_groups=tuple(valid),
        evaluation_groups=tuple(evaluation),
        components=tuple(components),
        assert_fit_groups=assert_fit_groups,
    )
    return subset, campaign


def test_exact_membership_receipt_binds_component_and_group_sets() -> None:
    subset, campaign = _synthetic_membership()
    receipt = validate_frozen_membership(subset, subset, campaign)
    assert receipt["selected_component_count"] == 512
    assert receipt["evaluation_group_count"] == 468
    assert receipt["fit_group_counts"] == {"train": 512, "valid": 513}
    assert receipt["component_complete"] is True


def test_membership_gate_rejects_partial_selected_component() -> None:
    import numpy as np

    subset, campaign = _synthetic_membership()
    subset["selected_groups"]["valid"].remove(3_000)
    subset["selected_group_counts"]["valid"] -= 1
    subset["selected_group_sha256"]["valid"] = hashlib.sha256(
        np.asarray(subset["selected_groups"]["valid"], dtype="<i8").tobytes()
    ).hexdigest()
    members = sorted(
        (split, group)
        for split, values in subset["selected_groups"].items()
        for group in values
    )
    subset["selected_members_sha256"] = hashlib.sha256(
        json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="not component-complete"):
        validate_frozen_membership(subset, subset, campaign)


def test_ple_checkpoint_receipt_binds_exact_fit_members_and_stored_knots() -> None:
    import numpy as np

    from scripts.e540_kaggle_worker import validate_numeric_checkpoint

    subset, _ = _synthetic_membership()
    knots = np.zeros((112, 9), dtype=np.float32)
    counts = np.ones(112, dtype=np.int64)
    knot_digest = hashlib.sha256(
        knots.astype("<f4", copy=False).tobytes()
        + counts.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    numeric = {
        "mode": "ple",
        "n_features": 112,
        "n_intervals": 8,
        "mapped_input_width": 1107,
        "split_manifest_sha256": subset["split_manifest_sha256"],
        "fit_members": subset["selected_groups"],
        "fit_members_sha256": subset["selected_members_sha256"],
        "fit_member_count": 1_025,
        "samples_per_sequence": 128,
        "candidate_sample_rows": 131_200,
        "retained_sample_rows": 65_536,
        "knot_sha256": knot_digest,
    }
    model_state = {
        "numeric_feature_map.knots": knots,
        "numeric_feature_map.knot_counts": counts,
        "numeric_feature_map.knots_fitted": np.asarray(True),
    }
    identity = validate_numeric_checkpoint("ple", numeric, subset, model_state)
    assert identity["fit_members_sha256"] == subset["selected_members_sha256"]
    assert identity["knot_sha256"] == knot_digest
    changed_state = {**model_state, "numeric_feature_map.knots": knots + 1}
    with pytest.raises(ValueError, match="knot-buffer digest"):
        validate_numeric_checkpoint("ple", numeric, subset, changed_state)
