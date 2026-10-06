from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "cadence_materializer", Path(__file__).with_name("materialize_specs.py")
)
assert SPEC is not None and SPEC.loader is not None
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)

from wnn.training_controls import (  # noqa: E402
    DistilledLoss,
    distilled_macro_losses,
    ema_decay_per_update,
)


def _value(argv: list[str], flag: str) -> str:
    """Read one option from a fixed argv list and reject accidental duplicates."""
    positions = [index for index, value in enumerate(argv) if value == flag]
    assert len(positions) == 1
    return argv[positions[0] + 1]


def _minimal_config() -> DistilledLoss:
    return DistilledLoss(
        alpha=0.0,
        bootstrap_horizon=0,
        ema=0.0,
        ema_metric_weight=False,
        structural=0.0,
        structural_start=0,
        spread_bins=0,
        next_event=0.0,
        next_start=0,
    )


def _toy_credit_gradient(detach_rows: int) -> float:
    """Measure whether the second microchunk's loss credits the first transition."""
    scale = torch.tensor(0.3, requires_grad=True)
    labels = torch.ones(1, 4, 2)
    weights = torch.ones(1, 4)

    def forward(rows: slice, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        previous = scale.new_zeros(()) if state is None else state
        next_state = previous + scale
        output = next_state.expand(1, rows.stop - rows.start, 2)
        return output, next_state

    pieces = distilled_macro_losses(
        forward=forward,
        ema_forward=None,
        state=None,
        ema_state=None,
        labels=labels,
        teacher=labels,
        row_weights=weights,
        next_change=None,
        structural_labels=None,
        next_labels=None,
        rows=slice(0, 4),
        micro_rows=2,
        detach_rows=detach_rows,
        config=_minimal_config(),
    )
    for contribution, _, _ in pieces:
        contribution.backward()
    assert scale.grad is not None
    return float(scale.grad)


def test_registered_arms_share_strong_recipe_and_differ_only_on_update_credit() -> None:
    args = {arm: materializer.cadence_args("screen", 3, 0, arm) for arm in materializer.ARM_ROWS}
    expected = {
        "update1000": (1000, 1000),
        "pooled2000": (2000, 1000),
        "connected2000": (2000, 2000),
    }
    for arm, argv in args.items():
        assert _value(argv, "--chunk") == "1000"
        assert _value(argv, "--update-rows") == str(expected[arm][0])
        assert _value(argv, "--detach-rows") == str(expected[arm][1])
        assert _value(argv, "--ema-reference-rows") == "1000"
        assert _value(argv, "--ema-decay") == "0.999"
        assert _value(argv, "--batch-order") == "seeded"
        assert _value(argv, "--checkpoint-epoch") == "3"
        assert _value(argv, "--seed") == "3"
        assert "--no-eval-during-fit" in argv
    assert args["update1000"] != args["pooled2000"]
    assert args["pooled2000"] != args["connected2000"]


def test_row_clock_ema_retains_equal_nominal_mass_per_processed_rows() -> None:
    one_k = ema_decay_per_update(0.999, 1000, 1000)
    two_k = ema_decay_per_update(0.999, 2000, 1000)
    assert one_k == pytest.approx(0.999)
    assert two_k == pytest.approx(0.998001)
    assert two_k == pytest.approx(one_k**2)


@pytest.mark.parametrize("update_rows, windows", [(1000, 20), (2000, 10)])
def test_update_windows_cover_the_entire_sequence_without_a_tail(
    update_rows: int, windows: int
) -> None:
    sequence_rows = 20_000
    boundaries = list(range(0, sequence_rows, update_rows))
    assert len(boundaries) == windows
    assert boundaries[0] == 0
    assert boundaries[-1] + update_rows == sequence_rows
    assert all(
        right - left == update_rows
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def test_pooled_macro_objective_differs_from_equal_mean_of_chunk_losses() -> None:
    labels = torch.ones(1, 4, 2)
    row_weights = torch.tensor([[1.0, 1.0, 3.0, 3.0]])

    def forward(rows: slice, state: None) -> tuple[torch.Tensor, None]:
        prediction = 0.0 if rows.start == 0 else 3.0
        return torch.full((1, rows.stop - rows.start, 2), prediction), None

    pieces = list(
        distilled_macro_losses(
            forward=forward,
            ema_forward=None,
            state=None,
            ema_state=None,
            labels=labels,
            teacher=labels,
            row_weights=row_weights,
            next_change=None,
            structural_labels=None,
            next_labels=None,
            rows=slice(0, 4),
            micro_rows=2,
            detach_rows=2,
            config=_minimal_config(),
        )
    )
    pooled_loss = sum(float(loss) for loss, _, _ in pieces)
    assert pooled_loss == pytest.approx(3.25)
    assert pooled_loss != pytest.approx((1.0 + 4.0) / 2)


def test_credit_arm_changes_gradient_path_while_values_remain_pooled() -> None:
    truncated_gradient = _toy_credit_gradient(detach_rows=2)
    connected_gradient = _toy_credit_gradient(detach_rows=4)
    assert truncated_gradient == pytest.approx(-1.1)
    assert connected_gradient == pytest.approx(-1.5)
    assert connected_gradient != pytest.approx(truncated_gradient)


def test_semantics_guard_fails_closed_if_row_clock_source_drifts(tmp_path: Path) -> None:
    trainer = tmp_path / "scripts/train_rnn.py"
    controls = tmp_path / "src/wnn/training_controls.py"
    dynamics = tmp_path / "src/wnn/training_dynamics.py"
    trainer.parent.mkdir(parents=True)
    controls.parent.mkdir(parents=True)
    shutil.copy(ROOT / "src/wnn/training_dynamics.py", dynamics)
    shutil.copy(ROOT / "scripts/train_rnn.py", trainer)
    source = (ROOT / "src/wnn/training_controls.py").read_text()
    controls.write_text(
        source.replace("main_mass = main_weights.sum()", "main_mass = changed.sum()")
    )
    with pytest.raises(RuntimeError, match="main_mass = main_weights.sum"):
        materializer.verify_supported_semantics(tmp_path)


def test_screen_contrast_matrix_and_fold_blocks_are_frozen() -> None:
    assert materializer.BLOCKS == ((3, 0), (4, 0), (3, 1))
    assert materializer.COMPARISONS == [
        ["pooled2000", "update1000"],
        ["connected2000", "pooled2000"],
        ["connected2000", "update1000"],
    ]
