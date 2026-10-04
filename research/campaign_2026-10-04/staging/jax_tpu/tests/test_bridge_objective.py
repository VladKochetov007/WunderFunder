from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.outcomes import sequence_class, structure_labels

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from wnn_jax.bridge import (  # noqa: E402
    SEQUENCE_LENGTH,
    PredictionCache,
    SequenceSupervision,
    _class_and_structure,
    _next_change_aux,
    absolute_propensity_index,
    collate_sequences,
    combo_sbd_loss,
    fixed_sbd_target,
)


def test_fold_teacher_sidecar_requires_exact_provenance_but_no_normalization_fields(
    tmp_path,
) -> None:
    run = "c20260929_since_s3_f0"
    split = "train"
    epoch = 3
    manifest_hash = "f" * 64
    prediction = np.zeros((1, SEQUENCE_LENGTH, 2), dtype=np.float32)
    array_path = tmp_path / "cache" / f"pred_{run}_{split}.npy"
    array_path.parent.mkdir()
    np.save(array_path, prediction)
    checkpoint_path = tmp_path / "artifacts" / "runs" / run / f"model_ep{epoch}.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fixed checkpoint fixture")
    target_index_path = tmp_path / "cache" / f"{split}_targets.npz"
    np.savez(target_index_path, row_group=np.asarray([17], dtype=np.int64))
    sidecar = {
        "run": run,
        "split": split,
        "epoch": epoch,
        "split_manifest_sha256": manifest_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        "row_groups": [17],
    }
    array_path.with_suffix(".json").write_text(json.dumps(sidecar))

    cache = PredictionCache.open(tmp_path, run, split, epoch, manifest_hash)
    np.testing.assert_array_equal(cache.get(17), prediction[0])


def test_absolute_propensity_lookup_preserves_hole_528_and_group_529() -> None:
    encoded_absolute_ids = np.arange(530, dtype=np.int64)
    assert encoded_absolute_ids[absolute_propensity_index(529, 530, "train")] == 529
    with pytest.raises(ValueError, match="corrupt TRAIN row group 528"):
        absolute_propensity_index(528, 530, "train")
    with pytest.raises(IndexError):
        absolute_propensity_index(530, 530, "train")


def test_structure_classes_and_next_change_edges_match_reference_fixtures() -> None:
    quanta = np.asarray([0.12543, 0.24133, 0.46854, 0.65687], dtype=np.float32)
    observed_classes = []
    for expected_class, quantum in enumerate(quanta):
        labels = np.tile(np.asarray([[0.0, -quantum]], dtype=np.float32), (2, 1))
        class_id, structure = _class_and_structure(labels)
        reference_labels = torch.from_numpy(labels[None])
        reference_class = sequence_class(reference_labels)
        reference_structure = structure_labels(reference_labels, reference_class)
        observed_classes.append(class_id)
        assert class_id == expected_class
        np.testing.assert_array_equal(np.asarray([class_id]), reference_class.numpy())
        for index, reference in enumerate(reference_structure):
            np.testing.assert_array_equal(structure[:, index], reference[0].numpy())
        np.testing.assert_array_equal(structure[:, 0], np.zeros(2, dtype=np.int64))
        np.testing.assert_array_equal(structure[:, 1], np.ones(2, dtype=np.int64))
        np.testing.assert_array_equal(structure[:, 2], np.full(2, expected_class))
    np.testing.assert_array_equal(observed_classes, np.arange(4, dtype=np.int64))

    # A D change at row 2 is strictly after rows 0 and 1; the final row has no
    # later change and must be masked. Every observed jump is upward.
    corner_labels = np.asarray([[0, 0], [0, 0], [1, 0], [1, -1]], dtype=np.float32)
    rows, direction, valid = _next_change_aux(corner_labels, np.ones(4, dtype=bool))
    np.testing.assert_allclose(rows[valid], [np.log1p(2), np.log1p(1), np.log1p(1)])
    np.testing.assert_array_equal(direction[valid], [1, 1, 1])
    np.testing.assert_array_equal(valid, [True, True, True, False])

    # Compare even the masked placeholders to the frozen production function.
    # Import only that function's AST, avoiding trainer imports or execution.
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "scripts/train_rnn.py").is_file()
    )
    source = root / "scripts/train_rnn.py"
    parsed = ast.parse(source.read_text(), filename=str(source))
    node = next(
        item
        for item in parsed.body
        if isinstance(item, ast.FunctionDef) and item.name == "next_change_labels"
    )
    namespace = {"torch": torch}
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, str(source), "exec"), namespace)
    reference = namespace["next_change_labels"](
        torch.from_numpy(corner_labels[None]), torch.ones(1, 4, dtype=torch.bool)
    )
    for actual, expected in zip((rows, direction, valid), reference, strict=True):
        np.testing.assert_allclose(actual, expected[0].numpy(), rtol=1e-7, atol=1e-7)


def _sample(*, scored: np.ndarray, base_weight: float) -> SequenceSupervision:
    rows = len(scored)
    valid = np.asarray([False, True, True, True][:rows])
    return SequenceSupervision(
        inputs=np.zeros((rows, 211), dtype=np.float32),
        labels=np.ones((rows, 2), dtype=np.float32),
        fixed_target=np.zeros((rows, 2), dtype=np.float32),
        main_weights=valid.astype(np.float32) * base_weight,
        since_scale=np.linspace(0.5, 1.0, rows, dtype=np.float32),
        structural_labels=np.zeros((rows, 3), dtype=np.int32),
        next_rows=np.ones(rows, dtype=np.float32),
        next_up=np.zeros(rows, dtype=np.float32),
        next_valid=np.asarray([False, True, True, False][:rows]),
        valid=valid,
        scored=scored,
    )


def test_fixed_sbd_target_bootstraps_only_when_change_is_after_horizon() -> None:
    labels = np.zeros((12, 2), dtype=np.float32)
    labels[11, 0] = 1.0
    teachers = np.stack((np.arange(12), -np.arange(12)), axis=-1).astype(np.float32)
    target = fixed_sbd_target(labels, teachers)

    # Rows 0..2 survive through t+8 and use teacher(t+8) for the bootstrap.
    clipped = np.clip(teachers, -2.0, 2.0)
    np.testing.assert_array_equal(target[:3], clipped[8:11] * 0.5 + clipped[:3] * 0.5)
    # At row 3 the target changes exactly at t+8, so the original y is retained.
    np.testing.assert_array_equal(target[3], clipped[3] * 0.5)
    # Near the sequence end the t+8 prediction is unavailable, also retaining y.
    np.testing.assert_array_equal(target[4], clipped[4] * 0.5)


def test_batch_true_mask_renormalization_precedes_since_decay() -> None:
    scored = np.asarray([False, True, False, True], dtype=bool)
    samples = [
        _sample(scored=np.zeros(4, dtype=bool), base_weight=2.0),
        _sample(scored=scored, base_weight=3.0),
    ]
    batch = collate_sequences(samples)

    valid = batch["valid"].astype(np.float32)
    scored_rows = batch["scored"].astype(np.float32)
    has_mask = scored_rows.any(axis=1)[:, None]
    base = np.stack([sample.main_weights for sample in samples])
    scale = (base * has_mask).sum(dtype=np.float64) / (valid * (scored_rows + 0.03) * has_mask).sum(
        dtype=np.float64
    )
    corrected = np.where(
        has_mask,
        valid * (scored_rows + 0.03) * scale,
        base,
    )
    expected = corrected * batch["since_scale"]
    np.testing.assert_allclose(batch["main_weights"], expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        batch["next_weights"],
        expected * np.stack([s.next_valid for s in samples]),
        rtol=0,
        atol=1e-7,
    )


def test_23_output_combo_objective_matches_torch_loss_formula() -> None:
    prediction = jnp.asarray(np.linspace(-0.4, 0.5, 2 * 4 * 23).reshape(2, 4, 23))
    ema_prediction = jnp.asarray(np.linspace(0.3, -0.2, 2 * 4 * 23).reshape(2, 4, 23))
    labels = jnp.asarray([[[0.5, -1.0], [0.25, 0.75], [-0.5, 1.5], [0.0, 0.0]]] * 2)
    fixed_target = jnp.asarray(np.linspace(-0.7, 0.8, 2 * 4 * 2).reshape(2, 4, 2))
    main_weights = jnp.asarray([[0.0, 0.5, 1.0, 0.2], [0.3, 0.0, 0.7, 1.0]])
    next_weights = jnp.asarray([[0.0, 0.5, 0.0, 0.2], [0.3, 0.0, 0.7, 0.0]])
    structural_labels = jnp.asarray(
        np.tile(np.asarray([[[1, 2, 3]] * 4], dtype=np.int32), (2, 1, 1))
    )
    next_rows = jnp.asarray([[0.0, 4.0, 1.0, 5.0], [2.0, 0.0, 6.0, 1.0]])
    next_up = jnp.asarray([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    batch = {
        "labels": labels,
        "fixed_target": fixed_target,
        "main_weights": main_weights,
        "structural_labels": structural_labels,
        "next_rows": next_rows,
        "next_up": next_up,
        "next_weights": next_weights,
    }
    actual = combo_sbd_loss(prediction, batch, ema_prediction)

    pred = torch.as_tensor(np.asarray(prediction)).float()
    ema_pred = torch.as_tensor(np.asarray(ema_prediction)).float()
    target = torch.as_tensor(np.asarray(labels)).float().clamp(-2, 2)
    fixed = torch.as_tensor(np.asarray(fixed_target)).float().clamp(-2, 2)
    weights = torch.as_tensor(np.asarray(main_weights)).float()
    primary_weights = target.abs() * weights[..., None]
    primary = (primary_weights * (pred[..., :2] - fixed).square()).sum()
    primary = primary / primary_weights.sum().clamp_min(1e-8)
    consistency = (
        weights[..., None] * (pred[..., :2] - ema_pred[..., :2]).square()
    ).sum() / weights.sum().clamp_min(1e-8)
    struct = torch.zeros(())
    structures = torch.as_tensor(np.asarray(structural_labels)).long()
    for start, width, target_index in ((2, 12, 0), (14, 3, 1), (17, 4, 2)):
        ce = torch.nn.functional.cross_entropy(
            pred[..., start : start + width].reshape(-1, width),
            structures[..., target_index].reshape(-1),
            reduction="none",
        ).reshape(2, 4)
        struct = struct + (ce * weights).sum() / weights.sum().clamp_min(1e-8)
    next_w = torch.as_tensor(np.asarray(next_weights)).float()
    next_dist = (torch.as_tensor(np.asarray(next_rows)).float() - 3.0) / 2.0
    next_mse = (pred[..., 21] - next_dist).square()
    next_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred[..., 22], torch.as_tensor(np.asarray(next_up)).float(), reduction="none"
    )
    next_aux = ((next_mse + next_bce) * next_w).sum() / next_w.sum().clamp_min(1e-8)
    expected = primary + consistency + 0.3 * struct + 0.3 * next_aux
    np.testing.assert_allclose(float(actual), float(expected), rtol=2e-6, atol=2e-7)
