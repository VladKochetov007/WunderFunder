"""Freeze component-disjoint membership and stage bounded disk arrays for a pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS, TARGET_COLUMNS
from wnn.features import event_features
from wnn.moment_geometry.features import EXTRA_WIDTH, geometry_features, infer_geometry
from wnn.moment_geometry.provenance import bind_sources, validate_storage
from wnn.splits import connected_components

LOGGER = logging.getLogger(__name__)


def freeze_membership(root: Path, output: Path) -> dict[str, object]:
    """Select whole fitting components from folds2/3 and exposed evaluation from folds0/1."""
    with np.load(root / "cache/chain_folds.npz", allow_pickle=False) as source:
        rows = {key: source[key] for key in source.files}
    with np.load(root / "cache/chains.npz", allow_pickle=False) as graph:
        assert np.array_equal(graph["split"], rows["split"])
        assert np.array_equal(graph["row_group"], rows["row_group"])
        recovered = connected_components(graph["next"])
    for chain in np.unique(rows["chain"]):
        members = rows["chain"] == chain
        if np.unique(rows["fold"][members]).size != 1 or np.unique(recovered[members]).size != 1:
            raise ValueError("component/fold cache disagreement")
    if len(np.unique(recovered)) != len(np.unique(rows["chain"])):
        raise ValueError("cached chain partition differs from successor graph")
    rng = np.random.default_rng(20261017)
    fitting = rng.choice(np.unique(rows["chain"][rows["fold"] >= 2]), 12, replace=False)
    evaluation = {
        str(fold): rng.choice(
            np.unique(rows["chain"][(rows["fold"] == fold) & (rows["split"] == "valid")]),
            8,
            replace=False,
        )
        for fold in (0, 1)
    }
    fit_mask = np.isin(rows["chain"], fitting)
    eval_mask = (rows["split"] == "valid") & np.isin(
        rows["chain"], np.concatenate(list(evaluation.values()))
    )
    fit = [
        (str(s), int(g), int(c), int(f))
        for s, g, c, f in zip(
            rows["split"][fit_mask],
            rows["row_group"][fit_mask],
            rows["chain"][fit_mask],
            rows["fold"][fit_mask],
            strict=True,
        )
    ]
    valid = [
        (str(s), int(g), int(c), int(f))
        for s, g, c, f in zip(
            rows["split"][eval_mask],
            rows["row_group"][eval_mask],
            rows["chain"][eval_mask],
            rows["fold"][eval_mask],
            strict=True,
        )
    ]
    if {r[2] for r in fit} & {r[2] for r in valid}:
        raise ValueError("fitting/evaluation component overlap")
    identities = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "cache/chain_folds.npz", root / "cache/chains.npz")
    }
    result = {
        "fit": fit,
        "eval": valid,
        "cache_hashes": identities,
        "evaluation_exposure": "exposed",
        "fit_folds": [2, 3],
        "eval_folds": [0, 1],
        "seed": 20261017,
        "note": "One fitting pool; folds0/1 are two evaluation slices of each seed, not independent fitted repetitions. Complete fitting components include permitted train and valid rows. Evaluation retains every valid sequence of selected components, excluding all their train counterparts from fitting.",
    }
    (output / "membership.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def stage_raw(root: Path, output: Path, membership: dict[str, object]) -> None:
    """Keep one Parquet reader and one complete sequence in RAM, stage arrays on disk."""
    validate_storage(output)
    binding = bind_sources(root, membership)
    (output / "source_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    arrays = {}
    for role in ("fit", "eval"):
        count = len(membership[role])
        for name, width in (("raw", 112), ("y", 2)):
            arrays[role, name] = np.lib.format.open_memmap(
                output / f"{role}_{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, 20000, width),
            )
        arrays[role, "mask"] = np.lib.format.open_memmap(
            output / f"{role}_mask.npy", mode="w+", dtype=bool, shape=(count, 20000)
        )
        np.save(output / f"{role}_component.npy", [row[2] for row in membership[role]])
        np.save(output / f"{role}_fold.npy", [row[3] for row in membership[role]])
    for split in ("train", "valid"):
        reader = pq.ParquetFile(root / "datasets" / f"{split}.parquet")
        columns = ["seq_ix", *FEATURE_COLUMNS, *TARGET_COLUMNS, "need_prediction"]
        if split == "valid":
            columns.append("is_scored")
        for role in ("fit", "eval"):
            for index, (source, group, _, _) in enumerate(membership[role]):
                if source != split:
                    continue
                table = reader.read_row_group(group, columns=columns, use_threads=False)
                expected_seq_ix = binding["selected_sequences"][role][index]["seq_ix"]
                actual_ids = table["seq_ix"].to_numpy()
                if not np.all(actual_ids == expected_seq_ix):
                    raise ValueError("source-qualified sequence identity changed")
                if table.num_rows != 20000:
                    raise ValueError("incomplete source sequence")
                arrays[role, "raw"][index] = np.column_stack(
                    [table[c].to_numpy() for c in FEATURE_COLUMNS]
                )
                arrays[role, "y"][index] = np.column_stack(
                    [table[c].to_numpy() for c in TARGET_COLUMNS]
                )
                mask = table["need_prediction"].to_numpy().copy()
                if role == "eval":
                    mask &= table["is_scored"].to_numpy()
                arrays[role, "mask"][index] = mask
        reader.close()
        del reader
    for array in arrays.values():
        array.flush()


def stage_features(output: Path) -> dict[str, object]:
    """Fit constants on fitting sequences, then independently transform each causal history."""
    fit = np.load(output / "fit_raw.npy", mmap_mode="r")
    beta, intercept, atoms = infer_geometry(fit)
    total, sums, squares = 0, np.zeros(231), np.zeros(231)
    for raw in fit:
        values = np.concatenate(
            [raw, event_features(raw), geometry_features(raw, beta, intercept, atoms)], axis=-1
        )
        total += len(values)
        sums += values.sum(axis=0, dtype=np.float64)
        squares += np.square(values.astype(np.float64)).sum(axis=0)
    mean = sums / total
    scale = np.maximum(np.sqrt(np.maximum(squares / total - mean**2, 0)), 0.05)
    for role in ("fit", "eval"):
        raw_data = np.load(output / f"{role}_raw.npy", mmap_mode="r")
        features = np.lib.format.open_memmap(
            output / f"{role}_x.npy", mode="w+", dtype=np.float32, shape=(len(raw_data), 20000, 231)
        )
        for index, raw in enumerate(raw_data):
            values = np.concatenate(
                [raw, event_features(raw), geometry_features(raw, beta, intercept, atoms)], axis=-1
            )
            features[index] = (values - mean) / scale
        features.flush()
    prefix = np.array(fit[0, :512], copy=True)
    changed = np.array(fit[0], copy=True)
    changed[512:] += np.float32(0.731)
    reference = geometry_features(fit[0], beta, intercept, atoms)[:512]
    future_changed = geometry_features(changed, beta, intercept, atoms)[:512]
    sliced = geometry_features(prefix, beta, intercept, atoms)
    if not np.array_equal(reference, future_changed) or not np.array_equal(reference, sliced):
        raise ValueError("causal suffix invariance failed")
    result = {
        "selected": ["i0_p0", "i0_p3", "i1_p0", "i1_p3"],
        "beta": beta.tolist(),
        "intercept": intercept.tolist(),
        "atoms": atoms.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "extra_width": EXTRA_WIDTH,
        "causal_suffix_invariance": "exact",
        "fit_sequences": len(fit),
    }
    (output / "preprocessing.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run the declared preparation phase; membership freezes before data are opened."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("membership", "raw", "features"), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    validate_storage(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "membership":
        membership = freeze_membership(args.root, args.output)
        LOGGER.info("Fit %d sequences; eval %d", len(membership["fit"]), len(membership["eval"]))
    elif args.phase == "raw":
        stage_raw(args.root, args.output, json.loads((args.output / "membership.json").read_text()))
    else:
        LOGGER.info("%s", stage_features(args.output))


if __name__ == "__main__":
    main()
