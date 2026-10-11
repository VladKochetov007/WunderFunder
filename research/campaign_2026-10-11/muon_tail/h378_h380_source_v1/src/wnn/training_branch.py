from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, MutableMapping
from typing import Protocol


class OptimizerLike(Protocol):
    """Minimal optimizer surface needed by the branch-local LR schedule."""

    param_groups: list[MutableMapping[str, object]]


State = Mapping[str, object]
Group = MutableMapping[str, object]
EXPECTED_PARENT_SOURCE_REF = (
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v6_lead"
)
EXPECTED_PARENT_SOURCE_DIGEST = "5ea4c9260dbffa939c47d991afb0a20eed50da5ba9d6685b924c397fe27bb262"


def contract_source_digest(sources: Mapping[str, str]) -> str:
    """Hash a producer's trainer/package map in its canonical serialized form."""
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_parent_source_binding(parent_sources: Mapping[str, str], binding: State) -> str:
    """Check a parent state against the independently recorded producer source map."""
    if binding.get("format") != "wnn.branch-parent-source-binding":
        raise ValueError("unsupported parent source binding format")
    expected = binding.get("contract_sources")
    digest = binding.get("contract_sources_sha256")
    if not isinstance(expected, dict) or not isinstance(digest, str):
        raise ValueError("parent source binding lacks its contract source map")
    actual_digest = contract_source_digest(expected)
    if actual_digest != digest:
        raise ValueError("parent source binding digest does not match its source map")
    if digest != EXPECTED_PARENT_SOURCE_DIGEST:
        raise ValueError("parent source binding does not name the reviewed frozen producer tree")
    if binding.get("producer_source_ref") != EXPECTED_PARENT_SOURCE_REF:
        raise ValueError("parent source binding references an unexpected producer tree")
    if dict(parent_sources) != expected:
        changed = sorted(
            key
            for key in set(parent_sources) | set(expected)
            if parent_sources.get(key) != expected.get(key)
        )
        raise ValueError(f"parent training-state source map differs from producer: {changed}")
    return actual_digest


def validate_parent_state_schema(state: State) -> tuple[Mapping[str, object], int]:
    """Require a complete, terminal epoch-boundary state before branching."""
    required = {
        "format",
        "version",
        "boundary",
        "contract",
        "next_epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "swa",
        "loader",
        "auxiliary",
        "progress",
        "rng",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"parent training state is incomplete: {missing}")
    if (
        state.get("format") != "wnn.epoch-training-state"
        or state.get("version") != 1
        or state.get("boundary") != "epoch"
    ):
        raise ValueError("parent is not a supported epoch training state")
    contract = state.get("contract")
    next_epoch = state.get("next_epoch")
    if (
        not isinstance(contract, Mapping)
        or isinstance(next_epoch, bool)
        or not isinstance(next_epoch, int)
    ):
        raise ValueError("parent training state has invalid contract or epoch")
    if next_epoch < 1 or contract.get("epochs") != next_epoch:
        raise ValueError("branch parent must be saved at its declared final epoch")
    return contract, next_epoch


def validate_branch_contracts(
    parent: Mapping[str, object],
    child: Mapping[str, object],
    *,
    tail_epochs: int,
    tail_steps: int,
    parent_next_epoch: int,
) -> None:
    """Allow only declared epoch/schedule changes across a full-state branch."""
    allowed_contract_fields = {"epochs", "total_steps", "sources", "options"}
    if set(parent) - allowed_contract_fields != set(child) - allowed_contract_fields:
        raise ValueError("branch contract fields differ outside epochs, schedule, and source")
    for key in set(parent) | set(child):
        if key in allowed_contract_fields:
            continue
        if parent.get(key) != child.get(key):
            raise ValueError(f"branch contract field differs: {key}")

    parent_epochs = parent.get("epochs")
    child_epochs = child.get("epochs")
    if not isinstance(parent_epochs, int) or not isinstance(child_epochs, int):
        raise ValueError("branch contract epoch counts are invalid")
    if parent_epochs != parent_next_epoch or child_epochs != parent_epochs + tail_epochs:
        raise ValueError("branch must append its declared tail after the complete parent schedule")

    parent_options = parent.get("options")
    child_options = child.get("options")
    if not isinstance(parent_options, Mapping) or not isinstance(child_options, Mapping):
        raise ValueError("branch contract is missing training options")
    allowed_option_changes = {"epochs", "checkpoint_epoch"}
    if set(parent_options) != set(child_options):
        raise ValueError("branch option fields differ")
    for key in parent_options:
        if key in allowed_option_changes:
            continue
        if parent_options.get(key) != child_options.get(key):
            raise ValueError(f"branch training option differs: {key}")

    parent_steps = parent.get("total_steps")
    child_steps = child.get("total_steps")
    if not isinstance(parent_steps, int) or not isinstance(child_steps, int):
        raise ValueError("branch schedule step counts are invalid")
    if parent_steps <= 0 or tail_steps <= 0 or child_steps != parent_steps + tail_steps:
        raise ValueError("child schedule must append exactly its declared tail optimizer attempts")


def inherit_parent_options(
    args: object,
    parent_options: Mapping[str, object],
    *,
    explicit_fields: set[str],
    allowed_overrides: set[str],
) -> None:
    """Load a verified parent's recipe, rejecting explicitly supplied drift."""
    for key, value in parent_options.items():
        if not hasattr(args, key):
            raise ValueError(f"parent training option is unavailable in child source: {key}")
        if key in explicit_fields and key not in allowed_overrides and getattr(args, key) != value:
            raise ValueError(f"explicit child training option differs from parent: {key}")
        if key not in allowed_overrides:
            setattr(args, key, value)


def restore_branch_resume_options(
    args: object, contract: Mapping[str, object], *, explicit_fields: set[str]
) -> None:
    """Recreate a branch's complete strict child recipe before rebuilding its objects."""
    options = contract.get("options")
    if not isinstance(options, Mapping):
        raise ValueError("saved branch contract lacks its complete child options")
    # These values are recomputed from the declared manifest paths later in setup;
    # the final strict contract comparison verifies their exact saved identities.
    derived_options = {"fit_groups_manifest_sha256", "split_manifest_sha256"}
    inherit_parent_options(
        args,
        {key: value for key, value in options.items() if key not in derived_options},
        explicit_fields=explicit_fields,
        allowed_overrides=set(),
    )
    # This persistence control is intentionally excluded from contract options, but
    # every continuation must keep writing atomic branch checkpoints for recovery.
    args.save_training_state = True


def _group_scale(group: Group) -> float:
    value = group.get("lr_scale", 1.0)
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("optimizer lr_scale must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("optimizer lr_scale must be positive and finite")
    return result


def _group_beta1(group: Group) -> float:
    betas = group.get("betas")
    if not isinstance(betas, (tuple, list)) or not betas:
        raise ValueError("tail branch requires optimizer beta1 in every group")
    value = betas[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("optimizer beta1 must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result < 1:
        raise ValueError("optimizer beta1 must be finite and in [0, 1)")
    return result


class TailCosineScheduler:
    """Cosine LR tail that changes group LR while holding restored betas fixed."""

    FORMAT = "wnn.cosine-tail-scheduler"
    VERSION = 1

    def __init__(
        self,
        optimizer: OptimizerLike,
        *,
        total_steps: int,
        peak_lr: float,
        final_lr: float,
    ) -> None:
        if total_steps < 1:
            raise ValueError("tail schedule requires at least one optimizer attempt")
        if not math.isfinite(peak_lr) or not math.isfinite(final_lr):
            raise ValueError("tail learning rates must be finite")
        if peak_lr <= 0 or final_lr < 0 or final_lr > peak_lr:
            raise ValueError("tail learning rates must satisfy peak > 0 and 0 <= final <= peak")
        if not optimizer.param_groups:
            raise ValueError("tail schedule requires optimizer parameter groups")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.peak_lr = float(peak_lr)
        self.final_lr = float(final_lr)
        self.group_scales = [_group_scale(group) for group in optimizer.param_groups]
        self.frozen_beta1 = [_group_beta1(group) for group in optimizer.param_groups]
        self.position = 0
        self._apply_lr()

    def _assert_beta1(self) -> None:
        current = [_group_beta1(group) for group in self.optimizer.param_groups]
        if current != self.frozen_beta1:
            raise ValueError("optimizer beta1 changed during the declared frozen-momentum tail")

    def _lr_at(self, position: int) -> float:
        if self.total_steps == 1:
            return self.peak_lr
        phase = min(max(position, 0), self.total_steps - 1) / (self.total_steps - 1)
        return self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (
            1.0 + math.cos(math.pi * phase)
        )

    def _apply_lr(self) -> None:
        self._assert_beta1()
        lr = self._lr_at(self.position)
        for group, scale in zip(self.optimizer.param_groups, self.group_scales, strict=True):
            group["lr"] = lr * scale

    def step(self) -> None:
        """Advance by one attempted optimizer update, including an AMP skip."""
        self._assert_beta1()
        if self.position < self.total_steps - 1:
            self.position += 1
        self._apply_lr()

    def get_last_lr(self) -> list[float]:
        """Return the current LR for each optimizer group."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, object]:
        """Serialize tail clock, rates, group scaling, and frozen beta1 values."""
        self._assert_beta1()
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "final_lr": self.final_lr,
            "group_scales": list(self.group_scales),
            "frozen_beta1": list(self.frozen_beta1),
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a matching tail clock after the optimizer state is restored."""
        expected = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "final_lr": self.final_lr,
            "group_scales": list(self.group_scales),
            "frozen_beta1": list(self.frozen_beta1),
        }
        mismatches = [key for key, value in expected.items() if state.get(key) != value]
        if mismatches:
            raise ValueError(f"saved tail schedule differs: {mismatches}")
        position = state.get("position")
        if not isinstance(position, int) or not 0 <= position < self.total_steps:
            raise ValueError("saved tail scheduler position is invalid")
        self.position = position
        self._apply_lr()


def tail_auxiliary_coefficients(
    *,
    aux_struct: float,
    aux_next: float,
    mode: str,
    position: int,
    total_steps: int,
) -> tuple[float, float]:
    """Get effective branch-local auxiliary weights without changing model args."""
    if mode == "none":
        return aux_struct, aux_next
    if mode != "linear-to-zero":
        raise ValueError(f"unsupported tail auxiliary schedule: {mode}")
    if total_steps < 1 or not 0 <= position < total_steps:
        raise ValueError("tail auxiliary schedule position is invalid")
    fraction = 1.0 if total_steps == 1 else position / (total_steps - 1)
    multiplier = 1.0 - fraction
    return aux_struct * multiplier, aux_next * multiplier
