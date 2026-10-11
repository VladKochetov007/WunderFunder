from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler


def input_strata(classes: np.ndarray, activity: np.ndarray) -> np.ndarray:
    """Cross market class with training-input activity terciles, never targets.

    Caller supplies one finite input-only activity summary per training sequence.
    Ties may collapse terciles; fitting cuts on evaluation inputs is forbidden.
    """
    if classes.ndim != 1 or activity.shape != classes.shape or not len(classes):
        raise ValueError("classes/activity must be equally sized nonempty vectors")
    if not np.isfinite(activity).all() or not np.isfinite(classes).all():
        raise ValueError("strata metadata must be finite")
    if not np.equal(classes, classes.astype(int)).all() or np.any(classes < 0):
        raise ValueError("classes must be nonnegative integers")
    cuts = np.quantile(activity, [1 / 3, 2 / 3])
    return classes.astype(np.int64) * 3 + np.searchsorted(cuts, activity, side="right")


def stratified_order(strata: Sequence[int], batch_size: int, seed: int) -> list[int]:
    """Proportional per-batch quotas with scrambled Sobol tie ordering.

    Every index occurs once. Cumulative deficits choose strata; integer quota
    error and exhausted small strata prevent exact proportions in every batch.
    Sobol affects tie priority only: this is stratification, not a QMC theorem
    for SGD. Storage O(N) indices, no sequence tensors and no target reads.
    """
    from scipy.stats import qmc

    labels = np.asarray(strata, dtype=np.int64)
    if labels.ndim != 1 or not len(labels) or batch_size < 1:
        raise ValueError("nonempty strata and positive batch size required")
    values, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(labels == v)).tolist() for v in values]
    used = np.zeros(len(values), dtype=np.int64)
    shares = counts / len(labels)
    batches = math.ceil(len(labels) / batch_size)
    sobol = qmc.Sobol(d=1, scramble=True, seed=seed)
    shifts = sobol.random_base2(math.ceil(math.log2(batches))).ravel()[:batches]
    order: list[int] = []
    for batch in range(batches):
        indices: list[int] = []
        priority = np.roll(np.arange(len(values)), int(shifts[batch] * len(values)))
        for _ in range(min(batch_size, len(labels) - len(order))):
            deficit = (len(order) + len(indices) + 1) * shares - used
            deficit[used == counts] = -np.inf
            chosen = int(priority[np.argmax(deficit[priority])])
            indices.append(pools[chosen][used[chosen]])
            used[chosen] += 1
        rng.shuffle(indices)
        order.extend(indices)
    return order


class EpochOrderSampler(Sampler[int]):
    """Explicit seed/epoch permutation with optional complete reverse order.

    This is a separate seeded control, not parity with RandomSampler's global
    RNG. When drop_last is set, reverse only the retained complete-batch prefix;
    the same unused tail stays last in both arms so fitting membership is equal.
    Pairing requires the SAME initialization seed and epoch; not seed+1.
    Epoch -1 is the shared normalization pass and never reverses the order.
    """

    def __init__(
        self,
        size: int,
        seed: int,
        is_antithetic: bool = False,
        strata: Sequence[int] | None = None,
        batch_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        if size < 1 or batch_size < 1 or strata is not None and len(strata) != size:
            raise ValueError("positive size and matching strata required")
        self.size = size
        self.seed = seed
        self.is_antithetic = is_antithetic
        self.strata = strata
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select an explicit deterministic epoch, including normalization epoch -1."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        seed = (self.seed + 1009 * (self.epoch + 1)) % (2**32)
        if self.strata is not None:
            order = stratified_order(self.strata, self.batch_size, seed)
        else:
            order = torch.randperm(
                self.size, generator=torch.Generator().manual_seed(seed)
            ).tolist()
        if self.is_antithetic and self.epoch >= 0:
            kept = self.size - self.size % self.batch_size if self.drop_last else self.size
            order = order[:kept][::-1] + order[kept:]
        return iter(order)


class AntiCurriculumSampler(EpochOrderSampler):
    """Hard-first resampling that anneals to the ordinary uniform permutation.

    Difficulty is a per-sequence mean baseline loss: either supplied (precomputed, fit sequences only) or
    learned online by ``update``. In epoch e of E the harder half (difficulty at or above the median of the
    known values) has sampling weight 1 + strength * (1 - e / (E - 1)): 2 at epoch 0 for strength 1, uniform
    in the last epoch. Epoch size is unchanged; draws are with replacement while the weights are unequal.
    Epoch -1 (normalization) and a uniform epoch use the parent's permutation. Unknown difficulty counts as
    easy, so the first running-loss epoch is uniform.
    """

    def __init__(self, size: int, seed: int, epochs: int, batch_size: int,
                 difficulty: np.ndarray | None = None, strength: float = 1.0) -> None:
        super().__init__(size, seed, False, None, batch_size, True)
        if difficulty is not None and (difficulty.shape != (size,) or not np.isfinite(difficulty).all()):
            raise ValueError("difficulty must be a finite vector with one entry per training sequence")
        self.epochs = epochs
        self.strength = strength
        self.difficulty = np.full(size, np.nan) if difficulty is None else difficulty.astype(np.float64).copy()
        self.last_order: list[int] = []

    def update(self, indices, losses) -> None:
        """Running per-sequence loss (mean of old and new once seen)."""
        for i, value in zip(torch.as_tensor(indices).tolist(), torch.as_tensor(losses).tolist()):
            old = self.difficulty[i]
            self.difficulty[i] = value if np.isnan(old) else 0.5 * (old + value)

    def anneal(self) -> float:
        return self.strength * max(0.0, 1.0 - self.epoch / max(1, self.epochs - 1))

    def weights(self) -> np.ndarray:
        known = ~np.isnan(self.difficulty)
        weights = np.ones(self.size)
        if known.any():
            weights[known & (self.difficulty >= np.median(self.difficulty[known]))] += self.anneal()
        return weights

    def __iter__(self) -> Iterator[int]:
        if self.epoch < 0 or self.anneal() <= 0.0 or np.isnan(self.difficulty).all():
            order = list(super().__iter__())
        else:
            seed = (self.seed + 1009 * (self.epoch + 1)) % (2**32)
            draw = torch.multinomial(torch.from_numpy(self.weights()), self.size, replacement=True,
                                     generator=torch.Generator().manual_seed(seed))
            order = draw.tolist()
        self.last_order = order
        return iter(order)


@torch.no_grad()
def init_qmc_(model: nn.Module, seed: int) -> None:
    """Replace Linear and GRU/LSTM default uniform draws using scipy Sobol.

    Preserves each scalar marginal, not iid joint distribution. A separately
    scrambled net per tensor uses input columns as coordinates and output rows
    as points. Biases use one dimension. This deliberately couples entries.
    Only a plain RNNModel with Linear inp/head and GRU/LSTM is supported. Other
    parameters must be rejected before mutation, rather than guessed.
    """
    from scipy.stats import qmc

    bounds: dict[int, float] = {}
    for module in model.modules():
        if isinstance(module, nn.Linear):
            for parameter in module.parameters(recurse=False):
                bounds[id(parameter)] = 1 / math.sqrt(module.in_features)
        elif isinstance(module, (nn.GRU, nn.LSTM)):
            for parameter in module.parameters(recurse=False):
                bounds[id(parameter)] = 1 / math.sqrt(module.hidden_size)
    parameters = list(model.named_parameters())
    unsupported = [name for name, p in parameters if id(p) not in bounds]
    if unsupported:
        raise ValueError(f"QMC init unsupported parameters: {unsupported}")
    for i, (_, parameter) in enumerate(parameters):
        count = parameter.shape[0]
        dimension = parameter.shape[1] if parameter.ndim == 2 else 1
        net = qmc.Sobol(d=dimension, scramble=True, seed=(seed + 7919 * (i + 1)) % (2**32))
        uniform = net.random_base2(math.ceil(math.log2(count)))[:count].ravel()
        values = (2 * uniform - 1) * bounds[id(parameter)]
        parameter.copy_(
            torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device).reshape(
                parameter.shape
            )
        )
