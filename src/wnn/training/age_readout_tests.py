"""Small synthetic checks for age-readout resume, ordering and evaluation safety."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import Tensor

from .age_readout_artifacts import PredictionSink
from .age_readout_checkpoint import restore_panel_checkpoint, save_panel_checkpoint
from .age_readout_cohort import ROWS_PER_SEQUENCE, WIDTH, FitBatch, iter_fit_batches
from .age_readout_fit import ARMS, _new_totals, _predict_eval, _train_batch
from .age_readout_model import AgeResidualHead
from .age_readout_portable_tests import check_portable_rebuild_and_page_release
from .age_readout_runtime import CUBLAS_WORKSPACE_CONFIG


def run_delivery_invariants(tmp_root: Path) -> dict[str, object]:
    """Run bounded synthetic checks using only the supplied physical-disk directory."""
    resolved = tmp_root.resolve()
    if any(resolved.is_relative_to(path) for path in (Path("/tmp"), Path("/dev/shm"))):
        raise ValueError("synthetic test storage must be outside /tmp and /dev/shm")
    resolved.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG
        or not torch.are_deterministic_algorithms_enabled()
    ):
        raise AssertionError("runtime did not configure deterministic PyTorch execution")
    _check_checkpoint_resume(resolved)
    _check_batch_tail_order()
    _check_prediction_coverage(resolved)
    _check_eval_column_invariance()
    check_portable_rebuild_and_page_release(resolved)
    return {
        "checkpoint_next_update_exact": True,
        "batch_tail_order_exact": True,
        "prediction_duplicates_and_incomplete_coverage_rejected": True,
        "evaluation_target_weight_invariance_exact": True,
        "portable_cache_rebuild_sha_exact": True,
        "sequence_page_release_preserves_data": True,
        "storage_root": str(resolved),
    }


def _check_checkpoint_resume(tmp_root: Path) -> None:
    torch.manual_seed(77)
    first, second = _synthetic_batch(13), _synthetic_batch(29)
    uninterrupted_models, uninterrupted_optimizers = _new_panel()
    initial_state = {
        key: value.detach().clone()
        for key, value in uninterrupted_models[ARMS[0]].state_dict().items()
    }
    _train_batch(
        first, uninterrupted_models, uninterrupted_optimizers, _new_totals(torch.device("cpu"))
    )
    _train_batch(
        second, uninterrupted_models, uninterrupted_optimizers, _new_totals(torch.device("cpu"))
    )
    identity: dict[str, object] = {"synthetic": "checkpoint-next-update.v1"}
    with TemporaryDirectory(dir=tmp_root) as directory:
        path = Path(directory) / "epoch0.pt"
        saved_models, saved_optimizers = _new_panel(initial_state)
        _train_batch(first, saved_models, saved_optimizers, _new_totals(torch.device("cpu")))
        save_panel_checkpoint(path, identity, 0, saved_models, saved_optimizers, [{"epoch": 0}])
        restored_models, restored_optimizers = _new_panel(initial_state)
        next_epoch, records = restore_panel_checkpoint(
            path,
            identity,
            restored_models,
            restored_optimizers,
            torch.device("cpu"),
            ARMS,
            3,
        )
        if next_epoch != 1 or records != [{"epoch": 0}]:
            raise AssertionError("checkpoint did not restore its next epoch and records")
        _train_batch(second, restored_models, restored_optimizers, _new_totals(torch.device("cpu")))
    for arm in ARMS:
        _assert_tree_equal(
            uninterrupted_models[arm].state_dict(), restored_models[arm].state_dict()
        )
        _assert_tree_equal(
            uninterrupted_optimizers[arm].state_dict(), restored_optimizers[arm].state_dict()
        )


def _check_batch_tail_order() -> None:
    cache = np.zeros((2, ROWS_PER_SEQUENCE, WIDTH), dtype="<f4")
    markers = np.arange(2 * ROWS_PER_SEQUENCE, dtype=np.float32)
    cache[:, :, 112] = markers.reshape(2, ROWS_PER_SEQUENCE)
    cache[:, :, 114] = 1.0
    released: list[int] = []
    batches = iter_fit_batches(
        cache,
        2,
        torch.zeros(211),
        torch.ones(211),
        torch.device("cpu"),
        release_sequence=released.append,
    )
    sizes: list[int] = []
    ordered_markers: list[Tensor] = []
    for batch in batches:
        sizes.append(len(batch.target))
        ordered_markers.append(batch.target[:, 0].clone())
    if sizes != [8192, 8192, 8192, 8192, 7232]:
        raise AssertionError(f"tail batch sizes changed: {sizes}")
    if not torch.equal(torch.cat(ordered_markers), torch.from_numpy(markers)):
        raise AssertionError("fit batches changed source row order or omitted the tail")
    if released != [0, 1]:
        raise AssertionError("fit cache page release did not follow complete sequence boundaries")


def _check_prediction_coverage(tmp_root: Path) -> None:
    with TemporaryDirectory(dir=tmp_root) as directory:
        root = Path(directory)
        output_dirs = {arm: root / arm for arm in ARMS}
        sink = PredictionSink(output_dirs, sequence_count=2)
        values = np.zeros((ROWS_PER_SEQUENCE, 2), dtype=np.float32)
        sink.write("A", 0, values)
        _expect_error(
            ValueError,
            lambda: sink.write("A", 0, values),
            "duplicate prediction callback was accepted",
        )
        _expect_error(RuntimeError, sink.publish, "incomplete prediction coverage was published")
        if any((path / "prediction.npy").exists() for path in output_dirs.values()):
            raise AssertionError("coverage failure created a published prediction")
        for arm_index, arm in enumerate(ARMS):
            for sequence_index in range(2):
                if arm == "A" and sequence_index == 0:
                    continue
                value = np.full_like(values, arm_index * 2 + sequence_index)
                sink.write(arm, sequence_index, value)
        published = sink.publish()
        for arm_index, arm in enumerate(ARMS):
            prediction = np.load(published[arm]["path"], mmap_mode="r", allow_pickle=False)
            for sequence_index in range(2):
                expected = np.float32(arm_index * 2 + sequence_index)
                if not np.all(prediction[sequence_index] == expected):
                    raise AssertionError("published prediction sequence order changed")


def _check_eval_column_invariance() -> None:
    cache = np.zeros((1, ROWS_PER_SEQUENCE, WIDTH), dtype="<f4")
    cache[:, :, 115:117] = np.float32(0.25)
    changed = cache.copy()
    changed[:, :, 112:115] = np.nan
    torch.manual_seed(101)
    models = {arm: AgeResidualHead() for arm in ARMS}
    for model in models.values():
        torch.nn.init.uniform_(model.output.weight, -0.1, 0.1)
        torch.nn.init.uniform_(model.output.bias, -0.1, 0.1)
    mean, std = torch.zeros(211), torch.ones(211)
    before = _capture_eval(cache, mean, std, models)
    after = _capture_eval(changed, mean, std, models)
    for arm in ARMS:
        if not np.array_equal(before[arm], after[arm]):
            raise AssertionError(f"arm {arm} read evaluation target or weight columns")


def _capture_eval(
    cache: np.ndarray, mean: Tensor, std: Tensor, models: dict[str, AgeResidualHead]
) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}

    def capture(arm: str, index: int, values: np.ndarray) -> None:
        """Collect one synthetic sequence prediction by its arm identity."""
        if index != 0:
            raise AssertionError("synthetic evaluation yielded an unexpected sequence")
        outputs[arm] = values

    _predict_eval(cache, 1, mean, std, models, torch.device("cpu"), capture)
    return outputs


def _new_panel(
    initial_state: dict[str, Tensor] | None = None,
) -> tuple[dict[str, AgeResidualHead], dict[str, torch.optim.Optimizer]]:
    template = AgeResidualHead()
    if initial_state is not None:
        template.load_state_dict(initial_state)
    models = {arm: AgeResidualHead() for arm in ARMS}
    for model in models.values():
        model.load_state_dict(template.state_dict())
    optimizers = {arm: torch.optim.AdamW(models[arm].parameters()) for arm in ARMS}
    return models, optimizers


def _synthetic_batch(seed: int) -> FitBatch:
    generator = torch.Generator().manual_seed(seed)
    return FitBatch(
        base=torch.randn((7, 211), generator=generator),
        group_age=torch.randn((7, 112), generator=generator),
        field_age=torch.randn((7, 112), generator=generator),
        target=torch.randn((7, 2), generator=generator),
        row_weight=torch.ones(7),
        parent=torch.randn((7, 2), generator=generator),
    )


def _assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if not torch.equal(left, right):
            raise AssertionError("checkpoint restore changed a tensor value")
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError("checkpoint restore changed a mapping key")
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise AssertionError("checkpoint restore changed a sequence length")
        for left_item, right_item in zip(left, right, strict=True):
            _assert_tree_equal(left_item, right_item)
    elif left != right:
        raise AssertionError("checkpoint restore changed scalar state")


def _expect_error(error_type: type[Exception], action: Callable[[], object], message: str) -> None:
    try:
        action()
    except error_type:
        return
    raise AssertionError(message)
