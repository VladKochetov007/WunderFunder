"""Small metadata builders shared by the fixed R15 execution coordinator."""

from __future__ import annotations

import numpy as np
import torch

from .representation import DictionaryAutoencoder, R15Predictor


def validate_loaded_panel(sequences: list[dict], manifest: dict) -> dict[str, object]:
    """Check panel identities, NPY headers, component coverage and disjointness."""
    if (
        manifest.get("split_sha256")
        != "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
        or manifest.get("membership_file") != "MEMBERSHIP.json"
    ):
        raise ValueError("R15 canonical F0 split identity differs")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 96 or len(sequences) != 96:
        raise ValueError("R15 requires the accepted 96-sequence panel")
    roles: dict[str, set[int]] = {"fit": set(), "eval": set()}
    keys: set[tuple[str, str]] = set()
    mapped = 0
    fit_stride_rows, eval_scored_rows = 0, 0
    for record, sequence in zip(records, sequences, strict=True):
        for name in ("role", "source", "group", "component", "seq_ix"):
            if record.get(name) != sequence.get(name):
                raise ValueError("R15 loaded sequence identity differs from manifest")
        role, component = str(sequence["role"]), int(sequence["component"])
        if role not in roles or component in roles[role]:
            raise ValueError("R15 role metadata is invalid or duplicated")
        roles[role].add(component)
        source_group = (str(sequence["source"]), str(sequence["group"]))
        if source_group in keys:
            raise ValueError("R15 source-qualified group is duplicated")
        keys.add(source_group)
        if set(record.get("files", {})) != {"x", "y", "scored"}:
            raise ValueError("R15 requires source-bound x, y and scored NPYs")
        mapped += 3
        for key, shape, dtype in (
            ("x", (20_000, 112), np.dtype("float32")),
            ("y", (20_000, 2), np.dtype("float32")),
            ("scored", (20_000,), np.dtype("bool")),
        ):
            array = sequence[key]
            if tuple(array.shape) != shape or np.dtype(array.dtype) != dtype:
                raise ValueError(f"R15 {key} array header differs from the fixed panel")
        mask = np.asarray(sequence["scored"], dtype=bool)
        if mask[:99].any():
            raise ValueError("R15 sample scored mask violates the 99-row warmup")
        if role == "fit":
            fit_stride_rows += int(mask[np.arange(0, 20_000, 40)].sum())
        else:
            eval_scored_rows += int(mask.sum())
    if len(roles["fit"]) != 64 or len(roles["eval"]) != 32 or roles["fit"] & roles["eval"]:
        raise ValueError("R15 fit/eval component pools are incomplete or overlapping")
    if mapped != 288:
        raise ValueError("R15 expected all 288 source-bound NPY mappings")
    if fit_stride_rows != 31_808 or eval_scored_rows != 59_436:
        raise ValueError("R15 original-mask fit/evaluation row counts differ")
    return {
        "fit_components": 64,
        "eval_components": 32,
        "fit_stride_rows": fit_stride_rows,
        "eval_scored_rows": eval_scored_rows,
        "mapped_npy_count": mapped,
    }


def fit_dictionary_diagnostics(
    model: DictionaryAutoencoder, fit_inputs: np.ndarray
) -> dict[str, object]:
    """Measure reconstruction and active-code counts on supervised fit rows only."""
    counts = np.zeros(16, dtype=np.int64)
    reconstruction_sq = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(fit_inputs), 2048):
            batch = torch.from_numpy(np.ascontiguousarray(fit_inputs[start : start + 2048]))
            code, rebuilt, _ = model(batch)
            counts += (code != 0).sum(dim=0).cpu().numpy()
            reconstruction_sq += float((rebuilt - batch).double().square().sum().item())
    return {
        "fit_stride_reconstruction_mse": reconstruction_sq / (len(fit_inputs) * 112),
        "fit_stride_active_code_rows_by_factor": counts.tolist(),
        "fit_stride_code_active_factors": int(np.count_nonzero(counts)),
        "fit_stride_row_count": int(len(fit_inputs)),
    }


def model_arrays(
    dictionaries: dict[str, DictionaryAutoencoder],
    predictors: dict[str, R15Predictor],
    raw_mean: np.ndarray,
    raw_scale: np.ndarray,
    feature_means: dict[str, np.ndarray],
    feature_scales: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Collect final fitted tensors and normalizers for the private NPZ only."""
    arrays = {"raw_input_mean": raw_mean, "raw_input_scale": raw_scale}
    for name, model in dictionaries.items():
        arrays[f"{name}_encoder_weight"] = model.encoder.weight.detach().cpu().numpy()
        arrays[f"{name}_decoder_weight"] = model.decoder.weight.detach().cpu().numpy()
        arrays[f"{name}_downstream_mean"] = feature_means[name]
        arrays[f"{name}_downstream_scale"] = feature_scales[name]
    for name, model in predictors.items():
        for key, value in model.state_dict().items():
            arrays[f"{name}_predictor_{key.replace('.', '_')}"] = value.detach().cpu().numpy()
    return arrays
