"""Real-data bridge for the fixed fold-0 SBD + auxiliary JAX objective.

This module deliberately does not import Torch. It reads raw row groups and
cached predictions with PyArrow/NumPy, then returns fixed tensors suitable for
the JAX core. It is a staged bridge, not an executable experiment harness.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from jax import lax

from wnn.data import (
    CORRUPT_TRAIN_ROW_GROUPS,
    FEATURE_COLUMNS,
    N_FEATURES,
    SEQUENCE_LENGTH,
    WARMUP,
)
from wnn.features import FEATURE_NAMES, event_features
from wnn_jax.rnn import AdamWState, ComputePrecision, adamw_update, forward_chunk

SBD_TEACHERS = (
    "c20260929_since_s3_f0",
    "c20260929_since_s4_f0",
)
SBD_TEACHER_EPOCHS = (3, 3)
TARGET_QUANTA = np.asarray([0.12543, 0.24133, 0.46854, 0.65687], dtype=np.float32)


@dataclass(frozen=True)
class FoldGroups:
    """Frozen fit/evaluation row groups from campaign fold-0 JSON."""

    train: tuple[int, ...]
    valid_fit: tuple[int, ...]
    evaluation: tuple[int, ...]
    manifest_sha256: str

    @property
    def fit_union(self) -> tuple[tuple[str, int], ...]:
        return tuple(("train", group) for group in self.train) + tuple(
            ("valid", group) for group in self.valid_fit
        )


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fold_groups(path: Path) -> FoldGroups:
    """Read and validate the fit/evaluation membership in a fold manifest."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    train = tuple(map(int, manifest["fit"]["train"]))
    valid_fit = tuple(map(int, manifest["fit"]["valid"]))
    evaluation = tuple(map(int, manifest["evaluation"]["valid"]))
    sets = (set(train), set(valid_fit), set(evaluation))
    members = (train, valid_fit, evaluation)
    if any(
        len(values) != len(group_list)
        for values, group_list in zip(sets, members, strict=True)
    ):
        raise ValueError("fold manifest contains duplicate row groups")
    if sets[1] & sets[2] or not train or not valid_fit or not evaluation:
        raise ValueError("fold fit/evaluation VALID membership is empty or overlaps")
    if int(manifest.get("fold", 0)) != 0:
        raise ValueError("this staged bridge is restricted to fold 0")
    return FoldGroups(train, valid_fit, evaluation, hashlib.sha256(raw).hexdigest())


def seeded_group_batches(
    groups: tuple[tuple[str, int], ...], *, seed: int, epoch: int, batch_size: int
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Create one deterministic order that precision arms can share unchanged."""
    if not groups or batch_size < 1 or epoch < 0:
        raise ValueError("group batches require non-empty membership and positive size/epoch")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(len(groups))
    usable = (len(order) // batch_size) * batch_size
    return tuple(
        tuple(groups[int(index)] for index in order[start : start + batch_size])
        for start in range(0, usable, batch_size)
    )


def group_batch_sha256(
    batches: tuple[tuple[tuple[str, int], ...], ...]
) -> str:
    """Stable identity for proving both precision arms consume the same groups."""
    payload = json.dumps(batches, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class PredictionCache:
    """A sidecar-verified prediction array addressed by original row-group id."""

    run: str
    split: str
    epoch: int
    manifest_sha256: str
    row_groups: tuple[int, ...]
    values: np.ndarray
    array_sha256: str
    checkpoint_sha256: str

    @classmethod
    def open(
        cls, root: Path, run: str, split: str, expected_epoch: int, manifest_sha256: str
    ) -> PredictionCache:
        """Open one fixed SBD teacher cache and verify its declared identity/hash."""
        array_path = root / "cache" / f"pred_{run}_{split}.npy"
        sidecar_path = array_path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("run") != run or sidecar.get("split") != split:
            raise ValueError(f"teacher cache identity differs for {run}/{split}")
        if int(sidecar.get("epoch", -1)) != expected_epoch:
            raise ValueError(f"teacher checkpoint epoch differs for {run}/{split}")
        if sidecar.get("split_manifest_sha256") != manifest_sha256:
            raise ValueError(f"teacher cache split manifest differs for {run}/{split}")
        checkpoint_path = root / "artifacts" / "runs" / run / f"model_ep{expected_epoch}.pt"
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != sidecar.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"teacher checkpoint hash differs for {run}/epoch{expected_epoch}")
        if sha256_file(array_path) != sidecar.get("prediction_sha256"):
            raise ValueError(f"teacher cache content hash differs for {run}/{split}")
        groups = tuple(map(int, sidecar["row_groups"]))
        if len(set(groups)) != len(groups):
            raise ValueError(f"teacher cache repeats row groups for {run}/{split}")
        target_index_path = root / "cache" / f"{split}_targets.npz"
        with np.load(target_index_path, allow_pickle=False) as index:
            target_row_groups = tuple(map(int, index["row_group"]))
        if groups != target_row_groups:
            raise ValueError(
                f"teacher {run}/{split} row groups differ from the trainer target-cache order"
            )
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if (
            values.ndim != 3
            or values.shape[:2] != (len(groups), SEQUENCE_LENGTH)
            or values.shape[2] < 2
        ):
            raise ValueError(f"teacher cache shape differs for {run}/{split}: {values.shape}")
        return cls(
            run,
            split,
            expected_epoch,
            manifest_sha256,
            groups,
            values,
            sidecar["prediction_sha256"],
            sidecar["checkpoint_sha256"],
        )

    def get(self, row_group: int) -> np.ndarray:
        """Return the fixed two-channel teacher sequence for a row group."""
        try:
            position = self._positions[int(row_group)]
        except AttributeError:
            self._positions = {group: index for index, group in enumerate(self.row_groups)}
            position = self._positions.get(int(row_group))
        if position is None:
            raise KeyError(f"teacher {self.run}/{self.split} has no row group {row_group}")
        return np.asarray(self.values[position, :, :2], dtype=np.float32)


class ComboTeacherMean:
    """Equal mean of the frozen since-s3 and since-s4 fold-0 teachers."""

    def __init__(self, root: Path, split: str, manifest_sha256: str) -> None:
        self.split = split
        self.caches = tuple(
            PredictionCache.open(root, run, split, epoch, manifest_sha256)
            for run, epoch in zip(SBD_TEACHERS, SBD_TEACHER_EPOCHS, strict=True)
        )
        reference = self.caches[0].row_groups
        if any(cache.row_groups != reference for cache in self.caches[1:]):
            raise ValueError("the two SBD teacher caches have different group order")

    def get(self, row_group: int) -> np.ndarray:
        """Match the trainer's `sum(scale * pred) / number_of_teachers` rule."""
        return (self.caches[0].get(row_group) + self.caches[1].get(row_group)) * 0.5


def absolute_propensity_index(row_group: int, n_row_groups: int, split: str) -> int:
    """Validate and return the absolute Parquet row-group propensity index."""
    group = int(row_group)
    if split not in ("train", "valid"):
        raise ValueError(f"unsupported propensity split: {split}")
    if group < 0 or group >= n_row_groups:
        raise IndexError(f"propensity row group {group} outside [0, {n_row_groups})")
    if split == "train" and group in CORRUPT_TRAIN_ROW_GROUPS:
        raise ValueError(f"corrupt TRAIN row group {group} cannot be used for propensity lookup")
    return group


class PropensityCache:
    """Read fixed propensity probabilities by absolute Parquet row-group id."""

    def __init__(self, root: Path, split: str, name: str = "c20260929_prop_f0") -> None:
        self.split = split
        array_path = root / "cache" / f"propcache_{name}_{split}.npy"
        metadata_path = array_path.with_suffix(".json")
        model_path = root / "artifacts" / "runs" / name / "model.pt"
        if not array_path.is_file() or not metadata_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"missing propensity cache or source checkpoint for {name}/{split}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("model") != name or metadata.get("split") != split:
            raise ValueError(f"propensity cache identity differs for {name}/{split}")
        if metadata.get("model_sha256") != sha256_file(model_path):
            raise ValueError(f"propensity checkpoint hash differs for {name}/{split}")
        parquet_path = root / "datasets" / f"{split}.parquet"
        self.n_row_groups = pq.ParquetFile(parquet_path).num_row_groups
        self.values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if self.values.shape != (self.n_row_groups, SEQUENCE_LENGTH):
            raise ValueError(f"propensity cache shape differs for {split}: {self.values.shape}")
        if int(metadata.get("rows", self.n_row_groups)) != self.n_row_groups:
            raise ValueError(f"propensity cache row-group count differs for {split}")
        self.array_sha256 = sha256_file(array_path)
        self.checkpoint_sha256 = metadata["model_sha256"]

    def get(self, row_group: int) -> np.ndarray:
        """Return one cached sequence probability vector."""
        index = absolute_propensity_index(row_group, self.n_row_groups, self.split)
        return np.asarray(self.values[index], dtype=np.float32)


class RawSequenceStore:
    """Reusable PyArrow row-group readers for the TRAIN and VALID parquet files."""

    def __init__(self, root: Path) -> None:
        self.files = {
            split: pq.ParquetFile(root / "datasets" / f"{split}.parquet")
            for split in ("train", "valid")
        }

    def read(
        self, split: str, row_group: int, *, require_official_mask: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read raw feature/label/scored columns for one complete sequence."""
        if split not in self.files:
            raise ValueError(f"unsupported dataset split: {split}")
        parquet = self.files[split]
        names = parquet.schema_arrow.names
        columns = [*FEATURE_COLUMNS, "t0", "t1"]
        has_scored = "is_scored" in names
        has_need_prediction = "need_prediction" in names
        if require_official_mask and not (has_scored and has_need_prediction):
            raise ValueError(
                "official evaluation requires both is_scored and need_prediction columns"
            )
        if has_scored:
            columns.append("is_scored")
        if has_need_prediction:
            columns.append("need_prediction")
        table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {row_group} has {table.num_rows} rows")
        raw = np.ascontiguousarray(
            np.stack([table.column(name).to_numpy() for name in FEATURE_COLUMNS], axis=0)
            .astype(np.float32, copy=False)
            .T
        )
        labels = np.column_stack(
            [table.column("t0").to_numpy(), table.column("t1").to_numpy()]
        ).astype(np.float32, copy=False)
        scored = (
            table.column("is_scored").to_numpy().astype(bool, copy=False)
            if has_scored
            else np.zeros(SEQUENCE_LENGTH, dtype=bool)
        )
        need_prediction = (
            table.column("need_prediction").to_numpy().astype(bool, copy=False)
            if has_need_prediction
            else np.ones(SEQUENCE_LENGTH, dtype=bool)
        )
        return raw, labels, scored, need_prediction


@dataclass(frozen=True)
class SequenceSupervision:
    """One sequence's model inputs and all fixed labels/weights for combo training."""

    inputs: np.ndarray
    labels: np.ndarray
    fixed_target: np.ndarray
    main_weights: np.ndarray
    since_scale: np.ndarray
    structural_labels: np.ndarray
    next_rows: np.ndarray
    next_up: np.ndarray
    next_valid: np.ndarray
    valid: np.ndarray
    scored: np.ndarray


@dataclass(frozen=True)
class EvaluationSequence:
    """Sequence data used for inference and the official scored-row mask."""

    inputs: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    scored: np.ndarray
    need_prediction: np.ndarray


def _class_and_structure(labels: np.ndarray) -> tuple[int, np.ndarray]:
    """Match `sequence_class` and `structure_labels` from the Torch trainer."""
    errors = [
        np.max(
            np.abs(
                np.rint(labels / quantum).astype(np.float32) * quantum - labels
            )
        )
        for quantum in TARGET_QUANTA
    ]
    class_id = int(np.argmin(errors))
    ticks = np.rint(labels / TARGET_QUANTA[class_id]).astype(np.int64)
    t0, t1 = ticks[:, 0], ticks[:, 1]
    up, down = t0 >= 1, t1 >= 1
    spread = np.where(up, -t1 - 2, np.where(down, -t0 - 2, -(t0 + t1)))
    spread = np.maximum(spread, 1)
    u = np.where(up, t0 + spread, -t1)
    branch = np.where(u < 0, 0, np.where(u > spread, 2, 1))
    labels_out = np.stack(
        [np.clip(spread, 1, 12) - 1, branch, np.full_like(branch, class_id)], axis=-1
    )
    return class_id, labels_out.astype(np.int32)


def _next_label_change(labels: np.ndarray) -> np.ndarray:
    """First pair-label change strictly after each row, or sequence length."""
    rows = np.arange(len(labels), dtype=np.int32)
    changed = np.zeros(len(labels), dtype=bool)
    changed[1:] = np.any(labels[1:] != labels[:-1], axis=-1)
    marker = np.where(changed, rows, len(labels))
    next_at_or_after = np.minimum.accumulate(marker[::-1])[::-1]
    return np.concatenate((next_at_or_after[1:], np.asarray([len(labels)], dtype=np.int32)))


def fixed_sbd_target(
    labels: np.ndarray,
    teacher_rows: np.ndarray,
    *,
    alpha: float = 0.5,
    horizon: int = 8,
) -> np.ndarray:
    """Build the fixed `(1-alpha)*T_h + alpha*teacher(t)` SBD target."""
    if labels.ndim != 2 or labels.shape[1] != 2 or teacher_rows.shape != labels.shape:
        raise ValueError("labels and teacher_rows must have matching [time,2] shapes")
    if alpha != 0.5 or horizon != 8:
        raise ValueError("staged production bridge supports only SBD alpha=.5 and horizon=8")
    length = len(labels)
    rows = np.arange(length)
    ahead = rows + horizon
    alive = (_next_label_change(labels) > ahead) & (ahead < length)
    bootstrap = np.clip(labels, -2.0, 2.0)
    bootstrap = bootstrap.copy()
    bootstrap[alive] = np.clip(teacher_rows[ahead[alive]], -2.0, 2.0)
    teacher = np.clip(teacher_rows, -2.0, 2.0)
    return (np.float32(1.0 - alpha) * bootstrap + np.float32(alpha) * teacher).astype(
        np.float32, copy=False
    )


def _next_change_aux(
    labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match `next_change_labels`: log1p rows to next D change and its direction."""
    difference = labels[:, 0] - labels[:, 1]
    rows = np.arange(len(labels), dtype=np.int32)
    changed = np.zeros(len(labels), dtype=bool)
    changed[1:] = difference[1:] != difference[:-1]
    marker = np.where(changed, rows, len(labels))
    next_at_or_after = np.minimum.accumulate(marker[::-1])[::-1]
    next_row = np.concatenate((next_at_or_after[1:], np.asarray([len(labels)], dtype=np.int32)))
    has = next_row < len(labels)
    index = np.minimum(next_row, len(labels) - 1)
    previous = np.maximum(index - 1, 0)
    direction = (difference[index] - difference[previous] > 0).astype(np.float32)
    distance = np.log1p((next_row - rows).astype(np.float32))
    return distance, direction, valid & has


def _main_weights(
    labels: np.ndarray,
    propensity: np.ndarray,
) -> np.ndarray:
    """Return the pre-batch valid * (prop_v1**2 + epsilon) fit weights."""
    valid = np.arange(len(labels)) >= WARMUP
    return valid.astype(np.float32) * (propensity**2 + np.float32(0.03))


def _since_scale(
    inputs: np.ndarray, *, since_weight: float = 32.0, since_floor: float = 0.25
) -> np.ndarray:
    """Match the causal since_i0px decay applied after valid-mask correction."""
    since_column = N_FEATURES + FEATURE_NAMES.index("since_i0px")
    since = np.expm1(inputs[:, since_column]).astype(np.float32)
    return np.maximum(1.0 / (1.0 + since / np.float32(since_weight)), since_floor).astype(
        np.float32
    )


def load_sequence_supervision(
    root: Path,
    split: str,
    row_group: int,
    teacher: ComboTeacherMean,
    propensity: PropensityCache,
    store: RawSequenceStore | None = None,
    *,
    groups: FoldGroups,
    alpha: float = 0.5,
    horizon: int = 8,
) -> SequenceSupervision:
    """Build exact fixed SBD targets and original 23-output combo supervision."""
    if split not in ("train", "valid") or teacher.split != split or propensity.split != split:
        raise ValueError("staged production bridge supports only SBD alpha=.5 and horizon=8")
    allowed = groups.train if split == "train" else groups.valid_fit
    if int(row_group) not in allowed:
        raise ValueError(f"{split} row group {row_group} is outside fold-0 fit membership")
    sequence_store = store or RawSequenceStore(root)
    raw, labels, scored, _ = sequence_store.read(split, row_group)
    event = event_features(raw)
    inputs = np.concatenate((raw, event), axis=-1).astype(np.float32, copy=False)
    if inputs.shape != (SEQUENCE_LENGTH, 211):
        raise ValueError(f"linear GRU expects 211 inputs, got {inputs.shape}")
    valid = np.arange(SEQUENCE_LENGTH) >= WARMUP
    main_weights = _main_weights(labels, propensity.get(row_group))
    since_scale = _since_scale(inputs)

    teacher_rows = teacher.get(row_group)
    fixed_target = fixed_sbd_target(labels, teacher_rows, alpha=alpha, horizon=horizon)

    _, structure = _class_and_structure(labels)
    next_rows, next_up, next_valid = _next_change_aux(labels, valid)
    return SequenceSupervision(
        inputs=inputs,
        labels=labels,
        fixed_target=fixed_target.astype(np.float32, copy=False),
        main_weights=main_weights,
        since_scale=since_scale,
        structural_labels=structure,
        next_rows=next_rows.astype(np.float32, copy=False),
        next_up=next_up,
        next_valid=next_valid,
        valid=valid,
        scored=scored,
    )


def load_evaluation_sequence(
    root: Path, row_group: int, store: RawSequenceStore | None = None
) -> EvaluationSequence:
    """Load one fold-held-out VALID sequence without consulting teacher caches."""
    sequence_store = store or RawSequenceStore(root)
    raw, labels, scored, need_prediction = sequence_store.read(
        "valid", row_group, require_official_mask=True
    )
    inputs = np.concatenate((raw, event_features(raw)), axis=-1).astype(np.float32, copy=False)
    if inputs.shape != (SEQUENCE_LENGTH, 211):
        raise ValueError(f"linear GRU expects 211 inputs, got {inputs.shape}")
    valid = np.arange(SEQUENCE_LENGTH) >= WARMUP
    return EvaluationSequence(inputs, labels, valid, scored, need_prediction)


def collate_evaluation_sequences(samples: list[EvaluationSequence]) -> dict[str, np.ndarray]:
    """Stack evaluation values while preserving the official row masks."""
    if not samples:
        raise ValueError("cannot collate an empty evaluation batch")
    return {
        name: np.stack([getattr(sample, name) for sample in samples])
        for name in EvaluationSequence.__dataclass_fields__
    }


def collate_sequences(samples: list[SequenceSupervision]) -> dict[str, np.ndarray]:
    """Stack sequences into fixed-shape CPU arrays without changing row order."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch = {
        name: np.stack([getattr(sample, name) for sample in samples])
        for name in SequenceSupervision.__dataclass_fields__
    }
    valid = batch["valid"].astype(np.float32)
    scored = batch["scored"].astype(np.float32)
    has_mask = scored.any(axis=1)[:, None]
    true_weights = scored + np.float32(0.03)
    numerator = np.sum(batch["main_weights"] * has_mask, dtype=np.float64)
    denominator = np.sum(valid * true_weights * has_mask, dtype=np.float64)
    scale = np.float32(numerator / max(denominator, 1e-8))
    corrected = np.where(
        has_mask,
        valid * true_weights * scale,
        batch["main_weights"],
    ).astype(np.float32)
    batch["main_weights"] = corrected * batch["since_scale"]
    batch["next_weights"] = batch["main_weights"] * batch.pop("next_valid").astype(np.float32)
    return batch


def slice_time_batch(
    batch: Mapping[str, np.ndarray | jax.Array], start: int, stop: int
) -> dict[str, jax.Array]:
    """Slice every time-indexed batch value for one fixed TBPTT chunk."""
    if start < 0 or stop <= start or stop > SEQUENCE_LENGTH:
        raise ValueError("chunk bounds must be a non-empty interval within the sequence")
    return {
        name: jnp.asarray(value[:, start:stop])
        for name, value in batch.items()
    }


def combo_sbd_loss(
    prediction: jax.Array,
    batch: Mapping[str, jax.Array],
    ema_prediction: jax.Array | None = None,
) -> jax.Array:
    """Exact combo wmse + EMA consistency + .3 structural + .3 next-change loss."""
    if prediction.shape[-1] != 23:
        raise ValueError(f"SBD combo objective requires 23 outputs, got {prediction.shape[-1]}")
    prediction = prediction.astype(jnp.float32)
    labels = jnp.clip(batch["labels"].astype(jnp.float32), -2.0, 2.0)
    row_weights = batch["main_weights"].astype(jnp.float32)
    primary_weights = jnp.abs(labels) * row_weights[..., None]
    primary = jnp.sum(
        primary_weights
        * (
            prediction[..., :2]
            - jnp.clip(batch["fixed_target"].astype(jnp.float32), -2.0, 2.0)
        )
        ** 2
    ) / jnp.maximum(jnp.sum(primary_weights), 1e-8)

    if ema_prediction is None:
        raise ValueError("the registered combo objective requires its fixed incoming EMA teacher")
    consistency_weights = row_weights[..., None]
    consistency = jnp.sum(
        consistency_weights
        * (
            prediction[..., :2]
            - lax.stop_gradient(ema_prediction[..., :2].astype(jnp.float32))
        )
        ** 2
    ) / jnp.maximum(jnp.sum(row_weights), 1e-8)

    struct_labels = batch["structural_labels"].astype(jnp.int32)
    structural = jnp.asarray(0.0, jnp.float32)
    for start, width, target_index in ((2, 12, 0), (14, 3, 1), (17, 4, 2)):
        log_probs = jax.nn.log_softmax(prediction[..., start : start + width], axis=-1)
        selected = jnp.take_along_axis(
            log_probs, struct_labels[..., target_index, None], axis=-1
        )[..., 0]
        structural = structural - jnp.sum(selected * row_weights) / jnp.maximum(
            jnp.sum(row_weights), 1e-8
        )

    next_weights = batch["next_weights"].astype(jnp.float32)
    next_distance = (batch["next_rows"].astype(jnp.float32) - 3.0) / 2.0
    next_mse = (prediction[..., 21] - next_distance) ** 2
    next_logit = prediction[..., 22]
    next_bce = jax.nn.softplus(next_logit) - batch["next_up"].astype(jnp.float32) * next_logit
    next_aux = jnp.sum((next_mse + next_bce) * next_weights) / jnp.maximum(
        jnp.sum(next_weights), 1e-8
    )
    return (
        primary
        + consistency
        + jnp.float32(0.3) * structural
        + jnp.float32(0.3) * next_aux
    )


@partial(jax.jit, static_argnames=("precision",))
def train_combo_chunk(
    params: Mapping[str, jax.Array],
    optimizer_state: AdamWState,
    batch: Mapping[str, jax.Array],
    recurrent_state: jax.Array | None,
    *,
    ema_params: Mapping[str, jax.Array],
    ema_state: jax.Array | None = None,
    learning_rate: float,
    precision: ComputePrecision,
    weight_decay: float = 0.01,
) -> tuple[dict[str, jax.Array], AdamWState, jax.Array, jax.Array, jax.Array]:
    """One combo update returning detached student and EMA recurrent carries."""
    inputs = batch["inputs"]

    def objective(
        current: Mapping[str, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        prediction, next_state = forward_chunk(
            current, inputs, recurrent_state, precision=precision
        )
        ema_prediction, next_ema_state = forward_chunk(
            ema_params, inputs, ema_state, precision=precision
        )
        return combo_sbd_loss(prediction, batch, ema_prediction), (next_state, next_ema_state)

    (loss, (next_state, next_ema_state)), gradients = jax.value_and_grad(
        objective, has_aux=True
    )(params)
    next_params, next_optimizer = adamw_update(
        params,
        gradients,
        optimizer_state,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return (
        next_params,
        next_optimizer,
        lax.stop_gradient(next_state),
        lax.stop_gradient(next_ema_state),
        loss,
    )


def evaluation_row_mask(batch: Mapping[str, jax.Array]) -> jax.Array:
    """Official score rows: `need_prediction AND is_scored AND row >= 99`."""
    valid = batch["valid"].astype(jnp.bool_)
    scored = batch["scored"].astype(jnp.bool_)
    need_prediction = batch["need_prediction"].astype(jnp.bool_)
    return valid & scored & need_prediction
