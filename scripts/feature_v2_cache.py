"""Additional causal histories bound to the existing frozen-backbone sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from feature_cache import bound_path
from feature_screen import digest_file, load_cache
from feature_v2_filters import fit_router, histories, route, warmup_summary

from wnn.data import SequenceFile
from wnn.splits import load_campaign_split


def training_sample(groups: tuple[int, ...], limit: int) -> list[int]:
    """Prespecify the router's TRAIN sample without inspecting labels or outcomes."""
    if not 4 <= limit <= 2048:
        raise ValueError("Router training limit must be between 4 and 2048 sequences")
    return sorted(
        sorted(
            map(int, groups),
            key=lambda group: hashlib.sha256(f"feature-router-v2:{group}:271828".encode()).digest(),
        )[:limit]
    )


def train_router(
    root: Path, manifest_path: Path, labels_path: Path, output: Path, limit: int = 1024
) -> None:
    """Fit market classes using TRAIN labels and only each sequence's warmup inputs."""
    if output.exists() or output.with_suffix(".json").exists() or output.suffix != ".npz":
        raise ValueError("Router output must be a new .npz path")
    manifest = json.loads(manifest_path.read_text())
    split_path = bound_path(root, manifest["campaign_split"])
    split = load_campaign_split(split_path, root)
    # Restrict the semantic label source, in addition to controller content sealing.
    if labels_path.resolve() != (root / "cache/train_targets.npz").resolve():
        raise ValueError("Router labels must come from the explicit TRAIN target cache")
    with np.load(labels_path, allow_pickle=False) as stored:
        labels, identifiers, groups = stored["market_class"], stored["seq_ix"], stored["row_group"]
    sequences = SequenceFile(root / "datasets/train.parquet")
    if (
        identifiers.shape != labels.shape
        or groups.shape != labels.shape
        or len(np.unique(groups)) != len(groups)
        or np.any(groups < 0)
        or np.any(groups >= len(sequences))
    ):
        raise ValueError("TRAIN label cache sequence count mismatch")
    label_index = {int(group): index for index, group in enumerate(groups)}
    selected = training_sample(split.train_groups, limit)
    split.assert_fit_groups(selected, [])
    rows, targets, used, unknown = [], [], [], []
    for group in selected:
        index = label_index[group]
        if labels[index] not in (0, 1, 2, 3):
            unknown.append(group)
            continue
        sequence = sequences.read(group, targets=False, mask=False)
        if identifiers[index] != sequence.seq_ix:
            raise ValueError("TRAIN label cache sequence identity mismatch")
        rows.append(warmup_summary(sequence.features))
        targets.append(labels[index])
        used.append(group)
    if not rows:
        raise ValueError("No known TRAIN classes in the prespecified router sample")
    parameters = fit_router(np.stack(rows), np.asarray(targets, dtype=np.int64))
    component_of = {(source, group): component for source, group, component in split.components}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **parameters)
    metadata = {
        "version": 2,
        "sha256": digest_file(output),
        "label_source": "TRAIN market_class only",
        "feature_rows": [0, 98],
        "fitted_row_groups": used,
        "selected_row_groups": selected,
        "unknown_label_groups": unknown,
        "fit_components": sorted({component_of[("train", group)] for group in used}),
        "component_graph_sha256": digest_file(split_path),
        "labels_sha256": digest_file(labels_path),
        "class_counts": np.bincount(targets, minlength=4).tolist(),
        "ridge": 0.01,
        "selection": "none; fixed balanced ridge; no VALID labels read",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def extract(
    root: Path, manifest_path: Path, base_path: Path, router_path: Path, output: Path
) -> None:
    """Reuse exactly the sampled backbone rows; no resampling or backbone fitting."""
    if output.exists() or output.with_suffix(".json").exists() or output.suffix != ".npz":
        raise ValueError("History output must be a new .npz path")
    base, base_meta = load_cache(base_path)
    manifest = json.loads(manifest_path.read_text())
    if base_meta["selection_manifest_sha256"] != digest_file(manifest_path):
        raise ValueError("Base cache does not match the declared row manifest")
    split_path = bound_path(root, manifest["campaign_split"])
    split = load_campaign_split(split_path, root)
    router_meta = json.loads(router_path.with_suffix(".json").read_text())
    if (
        router_meta["sha256"] != digest_file(router_path)
        or router_meta["component_graph_sha256"] != digest_file(split_path)
        or base_meta["component_graph_sha256"] != digest_file(split_path)
        or router_meta["label_source"] != "TRAIN market_class only"
        or router_meta["feature_rows"] != [0, 98]
        or set(router_meta["fit_components"]) & set(map(int, base["component"]))
    ):
        raise ValueError("Router provenance, component exclusion or content mismatch")
    split.assert_fit_groups(router_meta["fitted_row_groups"], [])
    with np.load(router_path, allow_pickle=False) as stored:
        parameters = {key: stored[key] for key in stored.files}
    source = manifest["source"]
    sequences = SequenceFile(root / f"datasets/{source}.parquet")
    component_of = {(source, group): component for source, group, component in split.components}
    results: dict[str, np.ndarray] = {}
    seen = np.zeros(len(base["row"]), dtype=bool)
    for group in manifest["row_groups"]:
        sequence = sequences.read(group, targets=False, mask=False)
        positions = np.flatnonzero(base["sequence"] == sequence.seq_ix)
        if not len(positions):
            continue
        rows = base["row"][positions]
        if (
            np.any(seen[positions])
            or np.any(rows >= len(sequence.features))
            or np.any(base["component"][positions] != component_of[(source, group)])
        ):
            raise ValueError("History sequence, component or row alignment failed")
        seen[positions] = True
        values = histories(sequence.features)
        routing = int(route(warmup_summary(sequence.features)[None], parameters)[0])
        values["regime"] = np.full((len(sequence.features), 1), routing, dtype=np.int64)
        for key, value in values.items():
            if key not in results:
                results[key] = np.empty((len(base["row"]), value.shape[1]), dtype=value.dtype)
            results[key][positions] = value[rows]
    if not seen.all():
        raise ValueError("The manifest did not reproduce all base cache rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output, **results, row=base["row"], sequence=base["sequence"], component=base["component"]
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": 2,
                "sha256": digest_file(output),
                "rows": len(base["row"]),
                "base_cache_sha256": base_meta["sha256"],
                "router_sha256": router_meta["sha256"],
                "router_fit_components": router_meta["fit_components"],
                "component_graph_sha256": base_meta["component_graph_sha256"],
                "feature_semantics": "feature_v2_filters.histories; exact base-cache row identity",
                "array_widths": {key: value.shape[1] for key, value in results.items()},
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Generate a separately versioned TRAIN router or exact-history cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("router")
    training.add_argument("--labels", type=Path, required=True)
    training.add_argument("--limit", type=int, default=1024)
    extraction = sub.add_parser("histories")
    extraction.add_argument("--base", type=Path, required=True)
    extraction.add_argument("--router", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "router":
        train_router(args.root, args.manifest, args.labels, args.output, args.limit)
    else:
        extract(args.root, args.manifest, args.base, args.router, args.output)


if __name__ == "__main__":
    main()
