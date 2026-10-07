from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

ProfileEvidence = Callable[[int, int, int, int, bool, bool], dict[str, object]]


def load_profile_evidence_helper() -> ProfileEvidence:
    """Compile only the pure helper so this test does not load PyTorch/CUDA."""
    trainer = Path(__file__).resolve().parents[1] / "scripts" / "train_rnn.py"
    tree = ast.parse(trainer.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "profile_update_evidence"
    )
    isolated = ast.Module(body=[helper], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(isolated), str(trainer), "exec"), namespace)
    return cast(ProfileEvidence, namespace["profile_update_evidence"])


profile_update_evidence = load_profile_evidence_helper()


def test_finite_amp_profile_evidence_serializes_as_usable() -> None:
    evidence = profile_update_evidence(2, 2, 0, 2, True, True)

    restored = json.loads(json.dumps(evidence))
    assert restored["optimizer_attempts"] == 2
    assert restored["optimizer_completed_updates"] == 2
    assert restored["amp_skipped_updates"] == 0
    assert restored["loss_finite"] is True
    assert restored["gradient_finite"] is True
    assert restored["valid_for_throughput"] is True


def test_amp_skip_is_reported_and_rejects_throughput_profile() -> None:
    evidence = profile_update_evidence(3, 2, 1, 3, True, True)

    restored = json.loads(json.dumps(evidence))
    assert restored["optimizer_attempts"] == 3
    assert restored["optimizer_completed_updates"] == 2
    assert restored["amp_skipped_updates"] == 1
    assert restored["gradient_finite"] is False
    assert restored["valid_for_throughput"] is False


def test_inconsistent_counts_and_disabled_amp_do_not_claim_finite_gradients() -> None:
    inconsistent = profile_update_evidence(3, 2, 0, 3, True, True)
    disabled_amp = profile_update_evidence(2, 2, 0, 2, True, False)

    assert inconsistent["counts_consistent"] is False
    assert inconsistent["gradient_finite"] is None
    assert inconsistent["valid_for_throughput"] is False
    assert disabled_amp["gradient_finite"] is None
    assert disabled_amp["valid_for_throughput"] is False


@pytest.mark.parametrize(
    ("loss_checks", "loss_finite", "expected_loss_finite"),
    [(2, False, False), (1, True, None)],
)
def test_nonfinite_or_incomplete_loss_evidence_is_rejected(
    loss_checks: int,
    loss_finite: bool,
    expected_loss_finite: bool | None,
) -> None:
    evidence = profile_update_evidence(2, 2, 0, loss_checks, loss_finite, True)

    restored = json.loads(json.dumps(evidence))
    assert restored["loss_finite"] is expected_loss_finite
    assert restored["valid_for_throughput"] is False
