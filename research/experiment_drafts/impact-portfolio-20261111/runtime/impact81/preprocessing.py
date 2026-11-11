from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from impact81.bindings import RUNTIME_COMMIT, SPLIT_SHA256, canonical, read_object, write_new
from wnn.data import FEATURE_COLUMNS
from wnn.training.sequence_data import SequenceData

# The canonical data provenance is a validated heterogeneous JSON mapping.
ROWS_PER_SEQUENCE = 64


def fit_identity(data: SequenceData) -> dict[str, Any]:
    """Bind preprocessing to complete train-only membership and canonical population moments."""
    if (
        data.split.sha256 != SPLIT_SHA256
        or data.split.valid_groups
        or len(data.fit.members) != 10325
        or len(data.evaluation.members) != 1873
        or any(source != "train" for source, _ in data.fit.members)
    ):
        raise ValueError("PR81 requires the complete pinned train-only official split")
    data.normalization()
    return {
        "split_sha256": data.split.sha256,
        "fit_members_sha256": canonical([list(item) for item in data.fit.members]),
        "normalization": data.provenance["normalization"],
        "feature_columns": list(FEATURE_COLUMNS),
    }


def prepare(data: SequenceData, output: Path) -> None:
    """Prepare train-only knot estimates on the receiving server; never read valid labels."""
    identity = fit_identity(data)
    mean, std = data.normalization()
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "train_quantile_samples.f32"
    shape = (len(data.fit.members) * ROWS_PER_SEQUENCE, 112)
    samples = np.memmap(scratch, dtype="float32", mode="w+", shape=shape)
    positions = np.linspace(0, 19_999, ROWS_PER_SEQUENCE, dtype=np.int64)
    for index, (source_name, group) in enumerate(data.fit.members):
        source = data.fit.sources[source_name]
        if source._file is None:
            source._file = source.environment.sequence_file(source.path)
        raw = source._file.read(group, targets=False, mask=False).features
        selected = np.asarray(raw[positions], dtype=np.float32)
        if selected.shape != (ROWS_PER_SEQUENCE, 112) or not np.isfinite(selected).all():
            raise ValueError("invalid train-only quantile sample")
        samples[index * ROWS_PER_SEQUENCE : (index + 1) * ROWS_PER_SEQUENCE] = (
            selected - mean.numpy()
        ) / std.numpy()
    samples.flush()
    knots, counts = [], []
    for feature in range(112):
        values = np.quantile(samples[:, feature], np.linspace(0, 1, 16), method="linear")
        unique = np.unique(values.astype(np.float32))
        if not np.isfinite(unique).all():
            raise ValueError("nonfinite quantile knots")
        counts.append(len(unique))
        knots.append(np.pad(unique, (0, 16 - len(unique)), mode="edge").tolist())
    artifact = {
        "schema": "impact81.preprocessing.v1",
        "runtime_commit": RUNTIME_COMMIT,
        "fit_identity": identity,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "knots": knots,
        "counts": counts,
        "estimator": {
            "method": "linear_empirical_quantiles",
            "rows_per_sequence": ROWS_PER_SEQUENCE,
            "positions": positions.tolist(),
            "sample_rows": shape[0],
            "population": "all train sequences; fixed label-independent row sample",
            "exact_all_row_quantiles": False,
        },
    }
    write_new(output / "preprocessing.json", artifact)
    del samples
    scratch.unlink()  # Only this attempt's successfully consumed temporary sample file.


def load_preprocessing(path: Path, data: SequenceData | None = None) -> dict[str, Any]:
    """Validate frozen knot/moment metadata; compare to actual fitting data when available."""
    value = read_object(path)
    if value.get("schema") == "impact81.gru-preprocessing.v1":
        from impact81.preparation_later import load_gru_preprocessing

        return load_gru_preprocessing(value, data)
    if (
        value.get("schema") != "impact81.preprocessing.v1"
        or value.get("runtime_commit") != RUNTIME_COMMIT
    ):
        raise ValueError("preprocessing schema/runtime mismatch")
    knots = np.asarray(value["knots"], dtype=np.float32)
    mean, std = np.asarray(value["mean"]), np.asarray(value["std"])
    counts = value["counts"]
    if knots.shape != (112, 16) or mean.shape != (112,) or std.shape != (112,):
        raise ValueError("preprocessing dimensions differ from raw112/16 knots")
    if not all(np.isfinite(item).all() for item in (knots, mean, std)) or not (std > 0).all():
        raise ValueError("nonfinite or nonpositive preprocessing values")
    if len(counts) != 112 or any(type(n) is not int or not 1 <= n <= 16 for n in counts):
        raise ValueError("invalid distinct-knot counts")
    for feature, count in enumerate(counts):
        if not np.all(np.diff(knots[feature, :count]) > 0):
            raise ValueError("distinct knots must be strictly increasing")
    if value["fit_identity"]["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("preprocessing feature order mismatch")
    if data is not None:
        if value["fit_identity"] != fit_identity(data):
            raise ValueError("preprocessing ancestry differs from actual train membership/moments")
        actual_mean, actual_std = data.normalization()
        if not torch.equal(
            actual_mean, torch.tensor(value["mean"], dtype=torch.float32)
        ) or not torch.equal(actual_std, torch.tensor(value["std"], dtype=torch.float32)):
            raise ValueError("preprocessing moment bytes differ from the canonical normalizer")
    return value
