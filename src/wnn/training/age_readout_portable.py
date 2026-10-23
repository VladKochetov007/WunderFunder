"""Rebuild and validate one attested cache fold on a portable CPU worker."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from wnn.training.online_relationship_cache import (
    SequenceCache,
    load_sequence_cache,
    sequence_cache_paths,
)

from .age_readout_portable_builder import (
    copy_pinned_file,
    decode_header,
    read_npy_header,
    reader_for_sources,
    rebuild_cache_file,
)
from .age_readout_portable_identity import (
    cache_receipt,
    payload_pin,
    verify_loaded_cache_metadata,
)
from .age_readout_portable_paths import PortablePaths
from .age_readout_portable_sources import verify_raw_sources
from .age_readout_portable_validate import (
    attest_bundle,
    fold_metadata,
    transport_path,
)

JsonObject = dict[str, Any]


def load_portable_caches(
    repo_root: Path,
    recipe: JsonObject,
    recipe_sha: str,
    fold: JsonObject,
    fold_id: int,
    paths: PortablePaths,
) -> tuple[SequenceCache, SequenceCache, dict[str, object]]:
    """Rebuild all 117 columns and require exact original bytes before fitting."""
    transport, source_recipe, source_sha, source_fold, receipt, milestone = attest_bundle(
        repo_root,
        recipe,
        recipe_sha,
        fold_id,
        paths.bundle_root,
        paths.transport_sha256,
    )
    bundle_fold = transport["folds"][str(fold_id)]
    cache_dir = paths.cache_dir(fold_id)
    if cache_dir.exists():
        raise FileExistsError(f"portable cache output already exists: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=False)
    fit_cache: SequenceCache | None = None
    eval_cache: SequenceCache | None = None
    close_reader = None
    try:
        fit_refs, eval_refs, parent_sidecar, original_sources = fold_metadata(
            repo_root,
            source_sha,
            source_fold,
            fold_id,
            bundle_fold,
            receipt,
            milestone,
            paths.bundle_root,
        )
        source_identity = verify_raw_sources(
            repo_root, source_fold, paths.raw_sources, original_sources
        )
        reader, close_reader = reader_for_sources(paths.raw_sources)
        fit_pin = payload_pin(receipt, fold_id, "fit_sequences")
        eval_pin = payload_pin(receipt, fold_id, "evaluation_sequences")
        fit_header = decode_header(
            bundle_fold["original_cache_headers"]["fit_sequences"],
            tuple(fit_pin["header"]["shape"]),
        )
        eval_header = decode_header(
            bundle_fold["original_cache_headers"]["evaluation_sequences"],
            tuple(eval_pin["header"]["shape"]),
        )
        fit_path, fit_meta = sequence_cache_paths(cache_dir, "in_sample_fit")
        eval_path, eval_meta = sequence_cache_paths(cache_dir, "feature_only_eval")
        _build_one(
            fit_path,
            fit_header,
            fit_refs,
            reader,
            paths.bundle_root,
            bundle_fold["fit_suffix5"],
            fit_pin["sha256"],
            evaluation=False,
        )
        _build_one(
            eval_path,
            eval_header,
            eval_refs,
            reader,
            paths.bundle_root,
            bundle_fold["evaluation_predictions"],
            eval_pin["sha256"],
            evaluation=True,
        )
        _copy_sidecars(paths.bundle_root, bundle_fold, cache_dir)
        fit_cache = load_sequence_cache(
            fit_path,
            fit_meta,
            expected_scope="in_sample_fit",
            expected_recipe_sha256=source_sha,
            expected_fold=fold_id,
            verify_payload=False,
        )
        eval_cache = load_sequence_cache(
            eval_path,
            eval_meta,
            expected_scope="feature_only_eval",
            expected_recipe_sha256=source_sha,
            expected_fold=fold_id,
            verify_payload=False,
        )
        verify_loaded_cache_metadata(
            fit_cache, eval_cache, fit_refs, eval_refs, source_fold, source_recipe, repo_root
        )
        cache_receipt_data = cache_receipt(
            recipe,
            source_sha,
            fold,
            source_fold,
            fit_cache,
            eval_cache,
            parent_sidecar,
            milestone["parent_caches"][str(fold_id)],
            cache_dir,
            bundle_fold,
            paths,
            source_identity,
            receipt,
            transport["actual_parent_metadata_check_sha256"],
        )
        return fit_cache, eval_cache, cache_receipt_data
    except BaseException:
        _close_optional(fit_cache)
        _close_optional(eval_cache)
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise
    finally:
        if close_reader is not None:
            close_reader()


def close_portable_caches(fit: SequenceCache, evaluation: SequenceCache) -> None:
    """Close large read mappings before deleting a fold's derived cache."""
    _close_optional(fit)
    _close_optional(evaluation)


def remove_portable_cache(cache_dir: Path) -> None:
    """Remove only this fold's derived cache after fit or profile finishes."""
    shutil.rmtree(cache_dir)


def _build_one(
    destination: Path,
    header: bytes,
    refs: list[dict[str, Any]],
    reader: Any,
    bundle_root: Path,
    suffix: JsonObject,
    full_sha: str,
    *,
    evaluation: bool,
) -> None:
    path = transport_path(bundle_root, suffix)
    observed_header, shape, dtype = read_npy_header(path)
    if (
        shape != tuple(suffix["shape"])
        or dtype != np.dtype("<f4")
        or len(observed_header) != int(suffix["header_bytes"])
        or path.stat().st_size != int(suffix["bytes"])
    ):
        raise ValueError(f"transport suffix header or byte count changed: {path.name}")
    rebuild_cache_file(
        destination,
        header,
        refs,
        reader,
        path,
        suffix["sha256"],
        tuple(suffix["shape"]),
        int(suffix["header_bytes"]),
        full_sha,
        evaluation=evaluation,
    )


def _copy_sidecars(bundle_root: Path, fold: JsonObject, cache_dir: Path) -> None:
    names = {
        "fit_sequences": "fit_sequences.json",
        "evaluation_sequences": "evaluation_sequences.json",
        "evaluation_predictions": "evaluation_predictions.json",
    }
    for name, destination_name in names.items():
        item = fold["sidecars"][name]
        copy_pinned_file(
            transport_path(bundle_root, item),
            cache_dir / destination_name,
            item["sha256"],
        )
    parent = fold["evaluation_predictions"]
    copy_pinned_file(
        transport_path(bundle_root, parent),
        cache_dir / "evaluation_predictions.npy",
        parent["sha256"],
    )


def _close_optional(cache: SequenceCache | None) -> None:
    if cache is None:
        return
    mapping = getattr(cache.array, "_mmap", None)
    if mapping is not None:
        mapping.close()
