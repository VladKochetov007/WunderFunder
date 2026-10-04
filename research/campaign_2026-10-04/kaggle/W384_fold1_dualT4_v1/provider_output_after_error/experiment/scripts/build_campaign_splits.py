"""Freeze cycle-safe, component-excluded outer-fold manifests without reading labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS
from wnn.splits import connected_components, file_sha256, load_campaign_split

ROOT = Path(__file__).resolve().parents[1]


def source_identity(path: Path, *, is_parquet: bool = False, hash_file: bool = False) -> dict:
    """Record a bounded-memory identity; large parquet files use their footer."""
    stat = path.stat()
    identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_file:
        identity["sha256"] = file_sha256(path)
    if is_parquet:
        with path.open("rb") as stream:
            stream.seek(-8, 2)
            trailer = stream.read(8)
            footer_bytes = int.from_bytes(trailer[:4], "little")
            if trailer[4:] != b"PAR1" or footer_bytes > stat.st_size - 8:
                raise ValueError(f"invalid parquet footer: {path}")
            stream.seek(-8 - footer_bytes, 2)
            digest = hashlib.sha256()
            remaining = footer_bytes + 8
            while remaining:
                chunk = stream.read(min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError(f"truncated parquet footer: {path}")
                digest.update(chunk)
                remaining -= len(chunk)
            identity.update(footer_bytes=footer_bytes, footer_sha256=digest.hexdigest())
    return identity


def build_manifests(root: Path, output: Path, n_folds: int = 4) -> list[Path]:
    """Preserve cached validation folds and exclude their graph components from all fits."""
    with np.load(root / "cache/chains.npz", allow_pickle=False) as graph:
        splits, groups = graph["split"], graph["row_group"]
        components = connected_components(graph["next"])
    with np.load(root / "cache/chain_folds.npz", allow_pickle=False) as cached:
        fold_keys = list(zip(cached["split"].tolist(), cached["row_group"].tolist(), strict=True))
        if len(set(fold_keys)) != len(fold_keys):
            raise ValueError("duplicate sequence in cached folds")
        fold_of = dict(zip(fold_keys, cached["fold"].tolist(), strict=True))
    keys = list(zip(splits.tolist(), groups.tolist(), strict=True))
    if len(set(keys)) != len(keys) or set(keys) != set(fold_of):
        raise ValueError("chain and fold caches have different or duplicate sequences")
    folds = np.array([fold_of[key] for key in keys])
    if np.any(folds < 0) or np.any(folds >= n_folds):
        raise ValueError("cached folds do not match the requested fold count")
    for component in np.unique(components[splits == "valid"]):
        if len(np.unique(folds[(components == component) & (splits == "valid")])) != 1:
            raise ValueError("validation component crosses cached folds")
    sources = {
        f"datasets/{split}.parquet": source_identity(
            root / f"datasets/{split}.parquet", is_parquet=True
        )
        for split in ("train", "valid")
    }
    for filename in ("chains.npz", "chain_folds.npz"):
        sources[f"cache/{filename}"] = source_identity(root / "cache" / filename, hash_file=True)
    sequences = {}
    for split in ("train", "valid"):
        index_path = root / f"cache/{split}_targets.npz"
        with np.load(index_path, allow_pickle=False) as index:
            row_groups, sequence_ids = index["row_group"], index["seq_ix"]
        if len(set(row_groups.tolist())) != len(row_groups):
            raise ValueError(f"duplicate row group in {index_path}")
        sequence_of = dict(zip(row_groups.tolist(), sequence_ids.tolist(), strict=True))
        selected = np.flatnonzero(splits == split)
        if set(sequence_of) != set(groups[selected].tolist()):
            raise ValueError(f"target index and graph disagree for {split}")
        identity = source_identity(index_path)
        digest = hashlib.sha256()
        for values in (row_groups, sequence_ids):
            digest.update(np.asarray(values, dtype="<i8").tobytes())
        identity["index_sha256"] = digest.hexdigest()
        sources[f"cache/{split}_targets.npz"] = identity
        sequences[split] = [
            {
                "row_group": int(groups[i]),
                "seq_ix": int(sequence_of[int(groups[i])]),
                "component": int(components[i]),
            }
            for i in selected
        ]
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for fold in range(n_folds):
        evaluation = (splits == "valid") & (folds == fold)
        evaluated_components = np.unique(components[evaluation])
        is_excluded = np.isin(components, evaluated_components)
        fitting = {
            split: sorted(map(int, groups[(splits == split) & ~is_excluded]))
            for split in ("train", "valid")
        }
        manifest = {
            "version": 1,
            "fold": fold,
            "n_folds": n_folds,
            "component_method": "undirected connected components of feature-only successor graph",
            "fold_assignment": "unchanged cache/chain_folds.npz validation assignment",
            "sources": sources,
            "sequences": sequences,
            "fit": fitting,
            "evaluation": {"valid": sorted(map(int, groups[evaluation]))},
            "excluded": {
                "train": sorted(map(int, groups[(splits == "train") & is_excluded])),
                "valid": sorted(map(int, groups[(splits == "valid") & is_excluded])),
                "corrupt_train": sorted(CORRUPT_TRAIN_ROW_GROUPS),
            },
            "counts": {
                "fit_train": len(fitting["train"]),
                "fit_valid": len(fitting["valid"]),
                "evaluation_valid": int(evaluation.sum()),
                "excluded_train": int(((splits == "train") & is_excluded).sum()),
                "evaluation_components": len(evaluated_components),
                "total_components": len(np.unique(components)),
            },
        }
        path = output / f"fold{fold}.json"
        encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        if path.exists():
            if path.read_bytes() != encoded:
                raise FileExistsError(f"refusing to replace frozen manifest: {path}")
        else:
            with path.open("xb") as stream:
                stream.write(encoded)
        frozen = load_campaign_split(path, root)
        print(
            json.dumps({"manifest": str(path), "sha256": frozen.sha256, **manifest["counts"]}),
            flush=True,
        )
        written.append(path)
    return written


def main() -> None:
    """Build all outer-fold manifests in an explicitly chosen artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=4)
    args = parser.parse_args()
    if args.n_folds < 2:
        parser.error("--n-folds must be at least two")
    build_manifests(args.root.resolve(), args.output_dir.resolve(), args.n_folds)


if __name__ == "__main__":
    main()
