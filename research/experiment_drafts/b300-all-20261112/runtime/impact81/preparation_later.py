"""Train-only GRU moments and raw-label prevalence for cards08–10."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from impact81.bindings import write_new
from wnn.data import FEATURE_COLUMNS
from wnn.training.sequence_data import SequenceData

# Artifact dictionaries contain heterogeneous, explicitly validated provenance values.
GRU_RUNTIME_COMMIT = "67c3bd818ea243177eeb1dc789cd3d145eba0e81"


def prepare(data: SequenceData, output: Path, cell: dict[str, Any]) -> None:
    """Persist canonical moments and count tail events from fitting labels only."""
    from impact81.preprocessing import fit_identity

    if cell["card"] not in ("08", "09", "10"):
        raise ValueError("GRU preparation is restricted to cards08–10")
    identity = fit_identity(data)
    mean, std = data.normalization()
    output.mkdir(parents=True, exist_ok=False)
    rows = np.zeros((2, 3), dtype=np.int64)
    sequences = np.zeros((2, 3), dtype=np.int64)
    for source_name, group in data.fit.members:
        if source_name != "train":
            raise ValueError("tail feasibility cannot consume validation labels")
        source = data.fit.sources[source_name]
        if source._file is None:
            source._file = source.environment.sequence_file(source.path)
        raw = source._file.read(group, targets=True, mask=False).targets
        if raw is None or raw.shape != (20_000, 2):
            raise ValueError("tail feasibility requires raw full-sequence t0/t1 labels")
        eligible = raw[99:]
        finite = np.isfinite(eligible)
        classes = (eligible < -2, (eligible >= -2) & (eligible <= 2), eligible > 2)
        counts = np.stack([(finite & selected).sum(axis=0) for selected in classes], axis=-1)
        rows += counts
        sequences += counts > 0
    feasible = bool((rows[:, (0, 2)] >= 100).all() and (sequences[:, (0, 2)] >= 32).all())
    feasibility = {
        "schema": "impact81.clip-event-feasibility.v1",
        "target_view": "raw_preclip_t0_t1",
        "class_order": ["less_than_minus2", "closed_minus2_plus2", "greater_than_plus2"],
        "training_mask": "all_after_warmup_and_finite_target",
        "warmup_rows": 99,
        "fit_sequences": len(data.fit.members),
        "row_counts_by_target_class": rows.tolist(),
        "sequence_counts_by_target_class": sequences.tolist(),
        "minimum_tail_rows": 100,
        "minimum_tail_sequences": 32,
        "feasible": feasible,
        "decision": "eligible_for_receiver_checks" if feasible else "fixed_recipe_infeasible",
    }
    write_new(
        output / "preprocessing.json",
        {
            "schema": "impact81.gru-preprocessing.v1",
            "runtime_commit": GRU_RUNTIME_COMMIT,
            "fit_identity": identity,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "clip_event_feasibility": feasibility,
        },
    )


def load_gru_preprocessing(value: dict[str, Any], data: SequenceData | None) -> dict[str, Any]:
    """Validate raw-label feasibility and source-qualified canonical moments."""
    from impact81.preprocessing import fit_identity

    if value.get("runtime_commit") != GRU_RUNTIME_COMMIT:
        raise ValueError("GRU preprocessing runtime pin differs")
    mean, std = np.asarray(value["mean"]), np.asarray(value["std"])
    if mean.shape != (112,) or std.shape != (112,):
        raise ValueError("GRU preprocessing must contain112 moments")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("GRU preprocessing moments must be finite with positive scales")
    if value["fit_identity"]["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("GRU preprocessing feature order mismatch")
    validate_feasibility(value["clip_event_feasibility"])
    if data is not None:
        if value["fit_identity"] != fit_identity(data):
            raise ValueError("GRU preprocessing differs from actual train membership")
        actual_mean, actual_std = data.normalization()
        if not torch.equal(
            actual_mean, torch.tensor(value["mean"], dtype=torch.float32)
        ) or not torch.equal(actual_std, torch.tensor(value["std"], dtype=torch.float32)):
            raise ValueError("GRU moments differ from the canonical normalizer")
    return value


def validate_feasibility(value: dict[str, Any], *, require_feasible: bool = False) -> None:
    """Reject inconsistent counts; card10 requires the frozen tail prevalence minimum."""
    if (
        value.get("schema") != "impact81.clip-event-feasibility.v1"
        or value.get("target_view") != "raw_preclip_t0_t1"
        or value.get("training_mask") != "all_after_warmup_and_finite_target"
        or value.get("warmup_rows") != 99
        or value.get("fit_sequences") != 10_325
        or value.get("minimum_tail_rows") != 100
        or value.get("minimum_tail_sequences") != 32
        or value.get("class_order")
        != ["less_than_minus2", "closed_minus2_plus2", "greater_than_plus2"]
    ):
        raise ValueError("card10 feasibility contract differs from the frozen recipe")
    rows = np.asarray(value.get("row_counts_by_target_class"))
    sequences = np.asarray(value.get("sequence_counts_by_target_class"))
    if rows.shape != (2, 3) or sequences.shape != (2, 3):
        raise ValueError("feasibility counts must cover two targets and three classes")
    if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(sequences.dtype, np.integer):
        raise ValueError("feasibility counts must be integers")
    if (
        (rows < 0).any()
        or (sequences < 0).any()
        or (sequences > 10_325).any()
        or (sequences > rows).any()
        or (rows.sum(axis=1) > 10_325 * 19_901).any()
    ):
        raise ValueError("feasibility counts exceed the fitting population")
    feasible = bool((rows[:, (0, 2)] >= 100).all() and (sequences[:, (0, 2)] >= 32).all())
    if value.get("feasible") is not feasible:
        raise ValueError("feasibility decision differs from the recorded tail counts")
    expected_decision = "eligible_for_receiver_checks" if feasible else "fixed_recipe_infeasible"
    if value.get("decision") != expected_decision:
        raise ValueError("feasibility decision label differs from its recorded counts")
    if require_feasible and not feasible:
        raise ValueError(
            "card10 fixed recipe is infeasible: preserve counts; do not retune thresholds"
        )
