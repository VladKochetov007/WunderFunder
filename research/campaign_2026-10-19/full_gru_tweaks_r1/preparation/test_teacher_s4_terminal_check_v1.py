"""Synthetic identity checks for the S4 teacher terminal verifier."""

from __future__ import annotations

import numpy as np
import pytest
import teacher_s4_terminal_check_v1 as check


def test_terminal_exit_identity_is_bound_to_s4() -> None:
    """Accept only the captured S4 native unit and invocation."""
    launch = {"unit": check.UNIT, "invocation_id": check.INVOCATION}
    evidence = {
        **launch,
        "run_name": check.RUN,
        "active_state": "active",
        "sub_state": "exited",
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
    }
    check._check_exit(evidence, launch)
    with pytest.raises(ValueError, match="invocation_id identity differs"):
        check._check_exit({**evidence, "invocation_id": "02a8c00b433b49a185c2700624397769"}, launch)


def test_seed_binding_and_membership_order_are_s4() -> None:
    """Pin seed four and preserve the fit/evaluation permutation rules."""
    assert dict(check.CRITICAL_ARGS)["seed"] == 4
    args = {
        "train_source": "train+valid",
        "valid_fold": 0,
        "valid_train_fraction": 1.0,
        "train_seqs": 30,
        "valid_seqs": 4,
        "full_valid": False,
        "eval_source": "vfold",
        "seed": 4,
    }
    split = {
        "fit": {"train": list(range(50)), "valid": list(range(100, 108))},
        "evaluation": {"valid": list(range(200, 208))},
    }

    train, fit_valid, evaluation = check._expected_membership(args, split)

    assert train == np.random.default_rng(4).permutation(split["fit"]["train"])[:30].tolist()
    assert train != np.random.default_rng(3).permutation(split["fit"]["train"])[:30].tolist()
    assert fit_valid == split["fit"]["valid"]
    assert (
        evaluation
        == np.random.default_rng(12345).permutation(split["evaluation"]["valid"])[:4].tolist()
    )


def test_checkpoint_arguments_reject_seed_three() -> None:
    """Reject an otherwise self-consistent S3 argument identity."""
    args = dict(check.CRITICAL_ARGS)
    check._check_args(args, args)
    args["seed"] = 3
    with pytest.raises(ValueError, match="argument seed differs"):
        check._check_args(args, args)
