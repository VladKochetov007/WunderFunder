"""Out-of-sample path distillation for students (research flags --ood-*, hypothesis H-350).

After a student is warm-started from a parent, a fraction of its training batches is replaced by *synthetic input
paths* built only from fit-split sequences. A synthetic path has no label; its target is the mean of frozen teacher
GRUs run online on the very same path (own carried state), so the student sees how the teachers behave away from the
observed data distribution (regime jumps, interpolated books, jittered values, stretched time).

Causality: every generator below maps source rows to output rows so that output row t depends only on source rows
at or before t (splice copies contiguous segments, mixup is row-wise, noise is row-wise with forward fill on the
source change mask, warp repeats fully static rows). Event features are recomputed from the generated raw rows by the
trainer's ordinary ``build_inputs``, hence causal as well.

Not implemented: ``sign_flip_bidask``. The 112 columns are anonymised; swapping bid and ask roles needs a verified
price reflection (and the teachers were never fitted on reflected books), so a flipped path would not be a possible
market path and the teacher output on it carries no transferable behaviour. Requesting it raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from wnn.augment import warp_index
from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP

OOD_AUGS = ("splice", "warp", "noise", "mixup_paths", "sign_flip_bidask")
OOD_MODES = ("mix", "only", "alternate")
OOD_WEIGHTS = ("teacher", "uniform")
SPLICE_MIN, SPLICE_MAX = 200, 1500
# columns blended by mixup_paths: book prices and volumes of both instruments; trades and extras come from path A
CONTINUOUS_COLUMNS = np.r_[0:44, 52:96]
# raw column groups that share one noise draw per row (keeps bid/ask ladders of a group moving together)
NOISE_GROUPS = [(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104), (104, 112)]
DEFAULT_PARAMS = {"splice": (), "warp": (0.5, 0.7, 1.4), "noise": (0.05,), "mixup_paths": (0.5,)}
# input-affecting checkpoint arguments the teacher must share with the student (both go through one build_inputs)
INPUT_ARGS = ("raw_only", "online_norm", "heuristics", "input_lead", "snap_codebook", "research_features",
              "privileged_mode", "input_act", "input_act_table", "frozen_extractor", "sentinel_values",
              "sentinel_indicators", "sentinel_base_width", "prop_input")
INPUT_DEFAULTS = {"raw_only": False, "online_norm": False, "heuristics": False, "input_lead": 0,
                  "snap_codebook": "", "research_features": "none", "privileged_mode": "none", "input_act": "none",
                  "input_act_table": "", "frozen_extractor": "", "sentinel_values": None, "sentinel_indicators": "none",
                  "sentinel_base_width": 0, "prop_input": ""}


def parse_ood_aug(spec: str) -> dict[str, tuple[float, ...]]:
    """``splice,warp:P/LO/HI,noise:SIGMA,mixup_paths:P`` -> {name: parameters}; omitted parameters use defaults."""
    result: dict[str, tuple[float, ...]] = {}
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, _, raw = item.partition(":")
        if name not in OOD_AUGS:
            raise ValueError(f"unknown --ood-aug {name!r}; choose from {OOD_AUGS}")
        if name == "sign_flip_bidask":
            raise ValueError("sign_flip_bidask is not implemented: anonymised columns have no verified bid/ask "
                             "reflection, so a flipped book is not a possible market path (see module docstring)")
        if name in result:
            raise ValueError(f"--ood-aug {name} given twice")
        params = tuple(float(v) for v in raw.split("/")) if raw else DEFAULT_PARAMS[name]
        if len(params) != len(DEFAULT_PARAMS[name]):
            raise ValueError(f"--ood-aug {name} takes {len(DEFAULT_PARAMS[name])} parameter(s)")
        if name == "warp" and not (0 < params[0] <= 1 and 0 < params[1] <= params[2]):
            raise ValueError("warp needs P/LO/HI with 0 < P <= 1 and 0 < LO <= HI")
        if name == "noise" and not 0 <= params[0] <= 0.5:
            raise ValueError("noise sigma is in quantile units and must lie in [0, 0.5]")
        if name == "mixup_paths" and not 0 < params[0] <= 1:
            raise ValueError("mixup_paths probability must lie in (0, 1]")
        result[name] = params
    if not result:
        raise ValueError("--ood-aug needs at least one augmentation")
    return result


def parse_teacher_specs(spec: str) -> list[tuple[str, int]]:
    """``RUN@EP[,RUN@EP...]`` -> [(run, epoch)]. The epoch is mandatory so the teacher is an explicit snapshot."""
    result = []
    for item in filter(None, (part.strip() for part in spec.split(","))):
        run, sep, epoch = item.partition("@")
        if not run or not sep or not epoch.lstrip("ep").isdigit():
            raise ValueError(f"--ood-teachers entries look like RUN@EP or RUN@epEP, got {item!r}")
        result.append((run, int(epoch.lstrip("ep"))))
    if not result:
        raise ValueError("--ood-teachers is empty")
    return result


def is_synthetic_batch(mode: str, frac: float, seed: int, epoch: int, batch: int) -> bool:
    """Deterministic in (seed, epoch, batch) so a resumed fit repeats the same real/synthetic pattern."""
    if mode == "only":
        return True
    if not 0 < frac <= 1:
        raise ValueError("--ood-frac must lie in (0, 1]")
    if mode == "alternate":
        period = max(1, round(1.0 / frac))
        return batch % period == period - 1
    return bool(np.random.default_rng([seed, epoch, batch, 1]).random() < frac)


@dataclass
class PathSampler:
    """Draw synthetic raw paths (B, rows, 112) from fit-split sequences only.

    ``fit_ids`` are (split, row_group) pairs; every read goes through ``read`` and is recorded in ``used`` so tests
    and provenance can assert that no evaluation sequence was touched. ``forbidden`` (the evaluation groups of the
    same splits) is checked disjoint from the fit set at construction.
    """

    fit_ids: Sequence[tuple[str, int]]
    read: Callable[[str, int], np.ndarray]
    augs: dict[str, tuple[float, ...]]
    seed: int
    rows: int = SEQUENCE_LENGTH
    sources: int = 24
    quantile_table: np.ndarray | None = None
    forbidden: frozenset[tuple[str, int]] = frozenset()
    used: set[tuple[str, int]] = field(default_factory=set)
    last_trace: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fit_ids = [(str(s), int(g)) for s, g in self.fit_ids]
        if not self.fit_ids:
            raise ValueError("no fit sequences to sample paths from")
        clash = set(self.fit_ids) & set(self.forbidden)
        if clash:
            raise ValueError(f"{len(clash)} sampler sequences are evaluation sequences")
        if self.rows <= WARMUP:
            raise ValueError("synthetic paths must be longer than the warm-up")
        self._table = self.quantile_table
        self._fit_set = frozenset(self.fit_ids)
        self._pool: list = []

    # -- sources -----------------------------------------------------------------------------------------------
    def _load(self, ident: tuple[str, int]) -> np.ndarray:
        if ident not in self._fit_set:
            raise AssertionError(f"{ident} is outside the fit split")
        self.used.add(ident)
        x = np.asarray(self.read(*ident), dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != N_FEATURES:
            raise ValueError("source sequence must be (rows, 112)")
        return x

    def _table_for_noise(self) -> np.ndarray:
        if self._table is None:  # deterministic and independent of the batch: first fit sequences by a seeded draw
            rng = np.random.default_rng([self.seed, 99])
            picks = rng.choice(len(self.fit_ids), size=min(16, len(self.fit_ids)), replace=False)
            rows = np.concatenate([self._load(self.fit_ids[i])[::50] for i in sorted(picks)])
            knots = min(257, len(rows))
            self._table = np.quantile(rows, np.linspace(0, 1, knots), axis=0).T.astype(np.float32)
        table = np.asarray(self._table, dtype=np.float32)
        if table.ndim != 2 or table.shape[0] != N_FEATURES:
            raise ValueError("quantile table must be (112, knots)")
        return table

    # -- generators --------------------------------------------------------------------------------------------
    def _base(self, pool: list[tuple[tuple[str, int], np.ndarray]], rng, segments: list[dict]) -> np.ndarray:
        rows = self.rows
        if "splice" not in self.augs:
            ident, source = pool[int(rng.integers(len(pool)))]
            start = int(rng.integers(0, max(1, len(source) - rows + 1)))
            out = source[start:start + rows]
            if len(out) < rows:  # short source: repeat the final row (static) rather than invent data
                out = np.concatenate([out, np.repeat(out[-1:], rows - len(out), axis=0)])
            segments.append({"id": ident, "start": start, "length": rows, "at": 0})
            return out.copy()
        parts, filled, last = [], 0, -1
        while filled < rows:
            length = min(int(rng.integers(SPLICE_MIN, SPLICE_MAX + 1)), rows - filled)
            j = int(rng.integers(len(pool)))
            if len(pool) > 1 and j == last:
                j = (j + 1 + int(rng.integers(len(pool) - 1))) % len(pool)
            last = j
            ident, source = pool[j]
            length = min(length, len(source))
            start = int(rng.integers(0, len(source) - length + 1))
            parts.append(source[start:start + length])
            segments.append({"id": ident, "start": start, "length": length, "at": filled})
            filled += length
        return np.concatenate(parts)

    def _mixup(self, a: np.ndarray, b: np.ndarray, rng, trace: dict) -> np.ndarray:
        lam = float(rng.uniform(0.25, 0.75))
        out = a.copy()
        out[:, CONTINUOUS_COLUMNS] = lam * a[:, CONTINUOUS_COLUMNS] + (1.0 - lam) * b[:, CONTINUOUS_COLUMNS]
        trace["mixup_lambda"] = lam
        return out

    def _noise(self, x: np.ndarray, sigma: float, rng, trace: dict) -> np.ndarray:
        """Jitter in quantile space on rows where the source column changed; static rows stay static (forward fill)."""
        table = self._table_for_noise()
        knots = table.shape[1]
        levels = np.linspace(0.0, 1.0, knots)
        changed = np.ones_like(x, dtype=bool)
        changed[1:] = x[1:] != x[:-1]
        eps = np.empty_like(x)
        draws = rng.standard_normal((len(x), len(NOISE_GROUPS))).astype(np.float32)
        for g, (lo, hi) in enumerate(NOISE_GROUPS):
            eps[:, lo:hi] = draws[:, g:g + 1]
        jittered = np.empty_like(x)
        for c in range(N_FEATURES):
            u = np.interp(x[:, c], table[c], levels)
            jittered[:, c] = np.interp(np.clip(u + sigma * eps[:, c], 0.0, 1.0), levels, table[c])
        index = np.maximum.accumulate(np.where(changed, np.arange(len(x))[:, None], 0), axis=0)
        trace["noise_sigma"] = sigma
        return np.take_along_axis(jittered, index, axis=0)

    def one(self, rng, torch_seed: int) -> tuple[np.ndarray, int, dict]:
        pool = self._pool
        trace: dict = {"segments": []}
        path = self._base(pool, rng, trace["segments"])
        if "mixup_paths" in self.augs and rng.random() < self.augs["mixup_paths"][0]:
            other: list[dict] = []
            path = self._mixup(path, self._base(pool, rng, other), rng, trace)
            trace["mixup_segments"] = other
        if "noise" in self.augs and rng.random() < 0.5:
            path = self._noise(path, self.augs["noise"][0], rng, trace)
        kept = self.rows
        if "warp" in self.augs and rng.random() < self.augs["warp"][0]:
            _, low, high = self.augs["warp"]
            factor = low + (high - low) * float(rng.random())
            torch.manual_seed(torch_seed)  # warp_index draws from the global torch RNG; restored by the caller
            index, kept = warp_index(torch.from_numpy(path), factor)
            path = path[index.numpy()]
            trace["warp_factor"] = factor
            trace["kept"] = kept
        return path, kept, trace

    def batch(self, batch_size: int, epoch: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, rows, 112) float32 and kept lengths, deterministic in (seed, epoch, step)."""
        rng = np.random.default_rng([self.seed, epoch, step, 2])
        picks = rng.choice(len(self.fit_ids), size=min(self.sources, len(self.fit_ids)), replace=False)
        self._pool = [(self.fit_ids[i], self._load(self.fit_ids[i])) for i in sorted(picks)]
        paths, kept, self.last_trace = [], [], []
        saved = torch.random.get_rng_state()
        try:
            for k in range(batch_size):
                path, n, trace = self.one(rng, int(rng.integers(1 << 31)))
                paths.append(path)
                kept.append(n)
                self.last_trace.append(trace)
        finally:
            torch.random.set_rng_state(saved)
            self._pool = []
        return torch.from_numpy(np.stack(paths)), torch.tensor(kept, dtype=torch.long)


# -- teachers ---------------------------------------------------------------------------------------------------
def check_teacher_compat(student_args: dict, teacher_args: dict, label: str) -> None:
    """Teacher and student share one input pipeline (build_inputs globals), so the input-affecting args must agree."""
    for key in INPUT_ARGS:
        default = INPUT_DEFAULTS[key]
        if teacher_args.get(key, default) != student_args.get(key, default):
            raise ValueError(f"teacher {label} differs from the student in input option {key}")


def check_teacher_ancestry(teacher_provenance: dict, student_eval_groups: Sequence[int], label: str,
                           split_sha: str | None) -> None:
    """The teacher must come from the student's clean fold and may not have fitted any evaluation sequence."""
    if split_sha is None or teacher_provenance.get("split_manifest_sha256") != split_sha:
        raise ValueError(f"teacher {label} was not fitted on the student's clean fold manifest")
    fit = teacher_provenance.get("fit_row_groups", {})
    clash = set(map(int, fit.get("valid", []))) & set(map(int, student_eval_groups))
    if clash:
        raise ValueError(f"teacher {label} fitted {len(clash)} evaluation sequences")


class OnlineTeachers:
    """Frozen teacher GRUs: mean of their first two outputs on a batch, each run chunk by chunk with carried state."""

    def __init__(self, models: Sequence[torch.nn.Module], forward: Callable, chunk: int, autocast: Callable):
        self.models = list(models)
        self.forward, self.chunk, self.autocast = forward, chunk, autocast
        for model in self.models:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        total = None
        for model in self.models:
            state, outs = None, []
            for start in range(0, inputs.shape[1], self.chunk):
                with self.autocast():
                    out, state = self.forward(model, inputs, slice(start, start + self.chunk), state)
                outs.append(out[..., :2].float())
            pred = torch.cat(outs, dim=1)
            total = pred if total is None else total + pred
        return total / len(self.models)


def ood_loss(out: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor, weight: str = "teacher") -> torch.Tensor:
    """Weighted MSE of the student toward the (clipped) teacher target on a synthetic chunk.

    ``teacher``: weights |clip(teacher)| * valid, the unlabelled analogue of the real distillation weights
    |clip(y)| * valid (the label is unknown on a synthetic path). ``uniform``: valid rows only.
    """
    target = teacher.float().clamp(-2, 2)
    row = valid[..., None].float()
    w = target.abs() * row if weight == "teacher" else row.expand_as(target)
    return (w * (out[..., :2].float() - target) ** 2).sum() / w.sum().clamp_min(1e-8)


def run_synthetic_batch(x: torch.Tensor, kept: torch.Tensor, *, teachers: OnlineTeachers, build_inputs: Callable,
                        forward: Callable, model: torch.nn.Module, windows: Sequence[slice], autocast: Callable,
                        detach: Callable, step: Callable[[torch.Tensor], None], weight: str = "teacher",
                        scale: float = 1.0) -> torch.Tensor:
    """One synthetic batch: teacher targets online, then one optimizer ``step(loss)`` per training window."""
    inputs = build_inputs(x)
    target = teachers(inputs) * scale
    rows = torch.arange(inputs.shape[1], device=inputs.device)[None]
    valid = (rows >= WARMUP) & (rows < kept.to(inputs.device)[:, None])
    state, losses = None, []
    for window in windows:
        with autocast():
            out, state = forward(model, inputs, window, state)
        state = detach(state)
        loss = ood_loss(out, target[:, window], valid[:, window], weight)
        step(loss)
        losses.append(loss.detach())
    return torch.stack(losses).mean()
