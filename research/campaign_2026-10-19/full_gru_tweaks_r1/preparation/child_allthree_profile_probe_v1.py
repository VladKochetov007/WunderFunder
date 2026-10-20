#!/usr/bin/env python3
"""Run the canonical bounded all-three child profile after its parent is accepted."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from child_allthree_profile_contract_v1 import (
    DICTIONARY_ACCEPTANCE_SHA256,
    DICTIONARY_BYTES,
    DICTIONARY_SHA256,
    PARENT_EP3_SHA256,
    PARENT_INVOCATION,
    PARENT_RUN,
    PARENT_SPLIT_SHA256,
    SOURCE_COMMIT,
    require_sha256,
    sha256_file,
    validate_exact_stop,
    validate_parent_acceptance,
    validate_profile_args,
    verify_parent_extension,
)

# `Any` is confined to the trusted PyTorch checkpoint state mapping.


def _load_parent_state(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load only the pinned checkpoint through the existing restricted CPU reader."""
    import numpy as np
    import torch
    from propensity_terminal_support import _load_torch_file, _numpy_safe_globals

    allowlist = [*_numpy_safe_globals(np), torch.torch_version.TorchVersion]
    payload, actual_sha256, _ = _load_torch_file(path, torch, allowlist=allowlist)
    if actual_sha256 != expected_sha256:
        raise ValueError("parent checkpoint changed during its restricted CPU load")
    if not isinstance(payload, dict):
        raise ValueError("parent epoch-3 checkpoint is not a state mapping")
    args, state = payload.get("args", {}), payload.get("model")
    if not isinstance(state, dict) or not isinstance(args, dict):
        raise ValueError("parent epoch-3 checkpoint lacks canonical args/model maps")
    if (
        args.get("seed") != 1001
        or args.get("hidden") != 256
        or args.get("layers") != 2
        or args.get("representation_mode") not in (None, "none")
        or args.get("mean_variance_head", 0) != 0
    ):
        raise ValueError("accepted parent checkpoint is not the 211-input/23-output baseline")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _verify_dictionary_cache(path: Path, expected_sha256: str) -> None:
    """Require the exact existing regular cache; never permit fitting a replacement."""
    from propensity_terminal_support import _hash_file

    actual_sha256, identity = _hash_file(path, DICTIONARY_BYTES)
    if identity["bytes"] != DICTIONARY_BYTES or actual_sha256 != expected_sha256:
        raise ValueError("shared fold-0 dictionary cache differs from its pinned artifact")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--parent-acceptance", type=Path, required=True)
    parser.add_argument("--expected-parent-acceptance-sha256", required=True)
    parser.add_argument("--parent-exact-stop", type=Path, required=True)
    parser.add_argument("--expected-parent-exact-stop-sha256", required=True)
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--dictionary-acceptance", type=Path, required=True)
    parser.add_argument("--expected-distill-cache-root", type=Path, required=True)
    return parser


def _training_argv(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split probe identity arguments from canonical trainer argv after `--`."""
    values = list(argv)
    if "--" not in values:
        raise ValueError("put the unchanged trainer argv after a literal `--` separator")
    separator = values.index("--")
    if separator == len(values) - 1:
        raise ValueError("canonical trainer argv is empty")
    return _parser().parse_args(values[:separator]), values[separator + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    """Validate accepted artifacts, then run the canonical bounded profile via its hook."""
    values = list(sys.argv[1:] if argv is None else argv)
    probe, trainer_argv = _training_argv(values)
    checkpoint_sha = require_sha256(
        probe.expected_parent_checkpoint_sha256, "parent epoch-3 checkpoint SHA-256"
    )
    if checkpoint_sha != PARENT_EP3_SHA256:
        raise ValueError("parent checkpoint differs from the captured native producer SHA")
    source_root = probe.source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise FileNotFoundError("pinned source root is missing")
    sys.path.insert(0, str(source_root / "src"))
    import torch

    from wnn.data import SEQUENCE_LENGTH
    from wnn.training.cli import build_parser
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.runner import run_training
    from wnn.training_dynamics import build_optimizer as canonical_build_optimizer

    if SEQUENCE_LENGTH != 20_000:
        raise ValueError(
            "canonical source sequence length differs from the full 20,000-row profile"
        )
    args = build_parser().parse_args(trainer_argv)
    expected_cache = Path(args.representation_cache).resolve(strict=True)
    validate_profile_args(args, expected_cache, probe.expected_distill_cache_root)
    split_path = Path(args.split_manifest).resolve(strict=True)
    if sha256_file(split_path) != PARENT_SPLIT_SHA256:
        raise ValueError("profile split manifest differs from the accepted parent fold")
    if sha256_file(probe.dictionary_acceptance) != DICTIONARY_ACCEPTANCE_SHA256:
        raise ValueError("shared dictionary acceptance JSON differs from its accepted digest")
    dictionary_receipt = json.loads(probe.dictionary_acceptance.read_text(encoding="utf-8"))
    cache_record = dictionary_receipt.get("cache", {})
    if (
        dictionary_receipt.get("status") != "ARTIFACT_AND_PROVENANCE_ACCEPTED"
        or cache_record.get("bytes") != DICTIONARY_BYTES
        or cache_record.get("sha256") != DICTIONARY_SHA256
        or not isinstance(cache_record.get("path"), str)
        or Path(cache_record["path"]).resolve() != expected_cache
    ):
        raise ValueError(
            "shared fold-0 dictionary cache receipt differs from its accepted identity"
        )
    _verify_dictionary_cache(expected_cache, DICTIONARY_SHA256)
    parent_receipt = validate_parent_acceptance(
        probe.parent_acceptance, probe.expected_parent_acceptance_sha256, checkpoint_sha
    )
    validate_exact_stop(probe.parent_exact_stop, probe.expected_parent_exact_stop_sha256)
    project_root = Path(os.environ["WNN_PROJECT_ROOT"]).resolve(strict=True)
    checkpoint = project_root / "artifacts" / "runs" / PARENT_RUN / "model_ep3.pt"
    parent_state = _load_parent_state(checkpoint, checkpoint_sha)
    output_dir = project_root / "artifacts" / "runs" / args.name
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("profile run output must be absent before create-only execution")
    for name, value in (("WNN_THREADS", "1"), ("WNN_LOADER_WORKERS", "0")):
        if os.environ.get(name) != value:
            raise ValueError(f"unit environment must set {name}={value}")
    environment = TrainingEnvironment()

    def checked_optimizer(
        model: torch.nn.Module, train_args: argparse.Namespace
    ) -> torch.optim.Optimizer:
        """Check the loaded extension and call the unchanged canonical AdamW builder."""
        _verify_dictionary_cache(expected_cache, DICTIONARY_SHA256)
        evidence = verify_parent_extension(parent_state, model.state_dict())
        record = {
            "schema": "fg19-child-allthree-profile-initial-state.v1",
            "parent_run": PARENT_RUN,
            "parent_invocation_id": PARENT_INVOCATION,
            "parent_checkpoint_sha256": checkpoint_sha,
            "parent_acceptance_sha256": probe.expected_parent_acceptance_sha256,
            "parent_exact_stop_sha256": probe.expected_parent_exact_stop_sha256,
            "dictionary_acceptance_sha256": DICTIONARY_ACCEPTANCE_SHA256,
            "dictionary_cache_sha256": DICTIONARY_SHA256,
            "split_manifest_sha256": PARENT_SPLIT_SHA256,
            "source_commit": SOURCE_COMMIT,
            "profile": {
                "sequence_length": SEQUENCE_LENGTH,
                "batch": train_args.batch,
                "chunk": train_args.chunk,
                "warmup_batches": train_args.profile_warmup,
                "measured_batches": train_args.profile_steps,
                "batch_count": train_args.max_batches,
                "updates_per_batch": SEQUENCE_LENGTH // train_args.chunk,
                "optimizer_attempts": train_args.max_batches
                * (SEQUENCE_LENGTH // train_args.chunk),
                "input_representation": train_args.representation_mode,
                "objective": train_args.mean_variance_objective,
            },
            "initialization": evidence,
            "parent_acceptance_status": parent_receipt["status"],
        }
        print("CHILD_PROFILE_INITIAL_STATE " + json.dumps(record, sort_keys=True), flush=True)
        if train_args.optim != "adamw" or train_args.precision != "fp32":
            raise ValueError("optimizer hook received non-FP32/non-AdamW configuration")
        return canonical_build_optimizer(model, train_args)

    environment.build_optimizer = checked_optimizer
    run_training(trainer_argv, environment=environment)
    if any(
        (output_dir / item).exists() for item in ("model.pt", "model_ep5.pt", "training_state.pt")
    ):
        raise RuntimeError("profile mode unexpectedly wrote a persistent model checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
