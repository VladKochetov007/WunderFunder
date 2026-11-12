"""K16 GRU model variants for PR81 cards08–10, composed over the PR79 adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from wnn.training.sequence_gru import GRUSequenceAdapter
from wnn.training.sequence_input_stem import validate_input_stem_kernel

from impact81.perturbation import perturb_hidden_state
from wnn.training.sequence_model_helpers import GRUSequenceState, check_sequence_input
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import configure_precision_mode

PERTURBATION_CHUNK_ROWS = 1000
PERTURBATION_PREFIX_ROWS = 128
PERTURBATION_COMPARE_ROWS = 32
PERTURBATION_EVERY_CHUNKS = 4
PERTURBATION_COEFFICIENT = 0.01
PERTURBATION_SCALE_SQUARED = 0.01**2 + 1e-8


def _number(config: Mapping[str, object], name: str, default: float) -> float:
    """Read a finite numeric model setting without accepting Boolean values."""
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"GRU model setting {name!r} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"GRU model setting {name!r} must be finite")
    return parsed


class Impact81GRUAdapter(GRUSequenceAdapter):
    """Preserve canonical K16 parameter paths while exposing card-specific outputs."""

    def __init__(self, model_config: Mapping[str, object], variant: str, seed: int) -> None:
        hidden, layers = model_config.get("hidden"), model_config.get("layers")
        hint_dim = model_config.get("hint_dim", 0)
        if any(type(value) is not int or value < 1 for value in (hidden, layers)):
            raise ValueError("K16 GRU requires positive integer hidden width and layer count")
        if type(hint_dim) is not int or hint_dim != 0:
            raise ValueError("cards08–10 use the frozen zero-width hint contract")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("GRU initialization seed must be an unsigned 32-bit integer")
        precision_mode = str(model_config.get("precision_mode", "fp32"))
        if precision_mode != "fp32":
            raise ValueError("cards08–10 preserve the canonical strict-FP32 GRU recipe")
        if validate_input_stem_kernel(model_config.get("input_stem_kernel", 0)) != 16:
            raise ValueError("cards08–10 require the canonical K16 input stem")
        dropout = _number(model_config, "dropout", 0.0)
        if dropout != 0.0:
            raise ValueError("cards08–10 freeze zero GRU dropout for matched clean trajectories")
        if variant not in (
            "perturbation_control",
            "state_perturbation",
            "nll_transfer",
            "clip_event",
        ):
            raise ValueError(f"unsupported PR81 GRU treatment variant: {variant!r}")
        configure_precision_mode(precision_mode)
        super().__init__(
            hidden,
            layers,
            dropout,
            hint_dim,
            str(model_config.get("hint_projection", "learned")),
            precision_mode,
            16,
        )
        self.variant = variant
        self.seed = seed
        self._auxiliary_heads_dropped = False
        if variant == "state_perturbation":
            self.register_buffer("_perturbation_rng_counter", torch.zeros((), dtype=torch.float32))
        elif variant == "nll_transfer":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 8109)
                self.log_variance_head = nn.Linear(hidden, 2)
                nn.init.zeros_(self.log_variance_head.weight)
                nn.init.zeros_(self.log_variance_head.bias)
        elif variant == "clip_event":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 8110)
                self.clip_event_heads = nn.ModuleList(nn.Linear(hidden, 3) for _ in range(2))

    def _run_recurrent(
        self, encoded: Tensor, hidden: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the canonical GRU and mean head to one encoded row segment."""
        recurrent, next_hidden = self.gru(encoded, hidden)
        prediction = self.head(torch.cat((recurrent, encoded), dim=-1))
        return prediction, recurrent, next_hidden.float()

    def _forward_core(
        self, values: Tensor, state: GRUSequenceState | None
    ) -> tuple[Tensor, Tensor, GRUSequenceState]:
        """Run the canonical normalized K16 stem, GRU, mean head and next state."""
        check_sequence_input(self, values, self.precision_mode)
        hidden, history, position = self._hidden_and_history(values, state)
        standardized = (values - self.mean) / self.std
        if self.input_stem is None:
            raise RuntimeError("K16 GRU input stem is missing")
        standardized, next_history = self.input_stem(standardized, history)
        if next_history is None:
            raise RuntimeError("K16 GRU did not return causal input history")
        encoded = self.encode(standardized)
        prediction, recurrent, next_hidden = self._run_recurrent(encoded, hidden)
        next_state = GRUSequenceState(next_hidden, next_history, position + values.shape[1])
        return prediction, recurrent, next_state

    def _forward_perturbation(
        self, values: Tensor, state: GRUSequenceState | None
    ) -> tuple[Tensor, Tensor, GRUSequenceState]:
        """Split every training chunk identically and fork only the declared state."""
        start_position = 0 if state is None else state.next_position
        if start_position % PERTURBATION_CHUNK_ROWS or values.shape[1] != PERTURBATION_CHUNK_ROWS:
            raise ValueError("card08 training requires aligned full 1,000-row chunks")
        prefix_end = PERTURBATION_PREFIX_ROWS
        compare_end = prefix_end + PERTURBATION_COMPARE_ROWS
        check_sequence_input(self, values, self.precision_mode)
        hidden, history, position = self._hidden_and_history(values, state)
        standardized = (values - self.mean) / self.std
        if self.input_stem is None:
            raise RuntimeError("K16 GRU input stem is missing")
        standardized, next_history = self.input_stem(standardized, history)
        if next_history is None:
            raise RuntimeError("K16 GRU did not return causal input history")
        encoded = self.encode(standardized)
        prefix, _, prefix_hidden = self._run_recurrent(encoded[:, :prefix_end], hidden)
        active = (start_position // PERTURBATION_CHUNK_ROWS + 1) % PERTURBATION_EVERY_CHUNKS == 0
        noisy_prediction: Tensor | None = None
        if active and self.variant == "state_perturbation":
            counter = getattr(self, "_perturbation_rng_counter", None)
            if not isinstance(counter, Tensor):
                raise RuntimeError("card08 perturbation RNG buffer was removed before training")
            noisy_hidden = perturb_hidden_state(prefix_hidden, counter, self.seed)
            noisy_prediction, _, _ = self._run_recurrent(
                encoded[:, prefix_end:compare_end], noisy_hidden
            )
        clean_middle, _, middle_hidden = self._run_recurrent(
            encoded[:, prefix_end:compare_end], prefix_hidden
        )
        suffix, _, final_hidden = self._run_recurrent(encoded[:, compare_end:], middle_hidden)
        final_state = GRUSequenceState(final_hidden, next_history, position + values.shape[1])
        auxiliary = (
            noisy_prediction - clean_middle.detach()
            if noisy_prediction is not None
            else values.new_empty((values.shape[0], 0, 2))
        )
        return torch.cat((prefix, clean_middle, suffix), dim=1), auxiliary, final_state

    def forward(
        self, values: Tensor, state: Tensor | GRUSequenceState | None = None
    ) -> tuple[Tensor, Tensor, Tensor | GRUSequenceState]:
        """Return canonical predictions plus a training-only card08/09/10 auxiliary."""
        if self._auxiliary_heads_dropped and self.training:
            raise RuntimeError("an exported GRU with removed auxiliary heads cannot resume fitting")
        if not self.training:
            return super().forward(values, state)
        if self.variant in ("perturbation_control", "state_perturbation"):
            if state is not None and not isinstance(state, GRUSequenceState):
                raise TypeError("card08 K16 state must include hidden state and input history")
            return self._forward_perturbation(values, state)
        if not isinstance(state, (GRUSequenceState, type(None))):
            raise TypeError("K16 auxiliary GRU state must include hidden state and input history")
        prediction, recurrent, next_state = self._forward_core(values, state)
        if self.variant == "nll_transfer":
            return prediction, self.log_variance_head(recurrent).clamp(-5.0, 3.0), next_state
        if self.variant == "clip_event":
            logits = torch.stack([head(recurrent) for head in self.clip_event_heads], dim=-2)
            return prediction, logits, next_state
        raise RuntimeError(f"no training forward is defined for {self.variant!r}")


def build_gru_model(model_config: dict[str, Any], variant: str, seed: int) -> nn.Module:
    """Build a canonical K16 control or its isolated PR81 auxiliary treatment.

    Args:
        model_config: Canonical validated GRU model recipe from the pinned PR79 runtime.
        variant: One of ``gru_plain``, ``perturbation_control``, ``state_perturbation``,
            ``nll_transfer`` or ``clip_event``.
        seed: Fresh paired-cell seed used only for deterministic module initialization.

    Returns:
        A canonical GRU adapter or a subclass with unchanged shared parameter paths.
    """
    if model_config.get("architecture") != "gru":
        raise ValueError("PR81 GRU models require the pinned canonical GRU architecture")
    if variant == "gru_plain":
        return build_sequence_model(dict(model_config))
    return Impact81GRUAdapter(model_config, variant, seed)


def drop_gru_auxiliary_heads(model: nn.Module) -> nn.Module:
    """Discard training-only heads/RNG after strict checkpoint load and select clean eval."""
    model.eval()
    if not isinstance(model, Impact81GRUAdapter) or model._auxiliary_heads_dropped:
        return model
    if model.variant == "nll_transfer":
        delattr(model, "log_variance_head")
    elif model.variant == "clip_event":
        delattr(model, "clip_event_heads")
    elif model.variant == "state_perturbation":
        delattr(model, "_perturbation_rng_counter")
    model._auxiliary_heads_dropped = True
    return model
