"""Freeze component-disjoint membership from bounded cached sequence metadata."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import TypeAlias

import numpy as np

from wnn.moment_geometry.membership_all import select_all_eligible
from wnn.moment_geometry.membership_serialization import write_membership_json
from wnn.moment_geometry.provenance import validate_chain_cache_sources
from wnn.moment_geometry.staging_resources import estimate_staging_bytes
from wnn.splits import connected_components

MembershipRow: TypeAlias = tuple[str, int, int, int]
MAX_CACHE_FILE_BYTES = 2_000_000
MAX_CACHE_ARRAY_BYTES = 4_000_000
MAX_BASE_MEMBERSHIP_BYTES = 100_000
KNOWN_BAD_TRAIN_ROW_GROUP = 528


def _read_rows(root: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load the two tiny chain caches only after compressed and expanded size checks."""
    rows: dict[str, np.ndarray] = {}
    cache_sizes: dict[str, int] = {}
    for name in ("chain_folds.npz", "chains.npz"):
        path = root / "cache" / name
        compressed_bytes = path.stat().st_size
        if compressed_bytes <= 0 or compressed_bytes > MAX_CACHE_FILE_BYTES:
            raise ValueError(f"chain cache exceeds its {MAX_CACHE_FILE_BYTES}-byte bound: {name}")
        with zipfile.ZipFile(path) as archive:
            expanded_bytes = sum(member.file_size for member in archive.infolist())
            if expanded_bytes > MAX_CACHE_ARRAY_BYTES:
                raise ValueError(f"expanded chain cache exceeds its bound: {name}")
            cache_sizes[name] = expanded_bytes
        with np.load(path, allow_pickle=False) as archive:
            loaded = {field: archive[field] for field in archive.files}
        actual_bytes = sum(array.nbytes for array in loaded.values())
        if actual_bytes > MAX_CACHE_ARRAY_BYTES:
            raise ValueError(f"loaded chain cache exceeds its array bound: {name}")
        if name == "chain_folds.npz":
            rows = loaded
        else:
            if any(
                not np.array_equal(rows[field], loaded[field])
                for field in ("split", "row_group", "chain")
            ):
                raise ValueError("chain caches disagree on source-qualified sequence identities")
            rows["graph_split"] = loaded["split"]
            rows["graph_row_group"] = loaded["row_group"]
            rows["next"] = loaded["next"]
    return rows, cache_sizes


def _membership_rows(rows: dict[str, np.ndarray], components: set[int]) -> list[MembershipRow]:
    """Return every cache row belonging to the requested whole components."""
    selected = np.isin(rows["chain"], np.fromiter(components, dtype=np.int64))
    return [
        (str(split), int(group), int(component), int(fold))
        for split, group, component, fold in zip(
            rows["split"][selected],
            rows["row_group"][selected],
            rows["chain"][selected],
            rows["fold"][selected],
            strict=True,
        )
    ]


def _read_baseline(
    path: Path,
) -> tuple[list[MembershipRow], list[MembershipRow], dict[str, str]]:
    """Read a small frozen membership and preserve its ordered fit/evaluation rows."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_BASE_MEMBERSHIP_BYTES:
        raise ValueError("baseline membership exceeds its bounded JSON read size")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("fit"), list):
        raise ValueError("baseline membership is missing its fitting rows")
    if not isinstance(value.get("eval"), list):
        raise ValueError("baseline membership is missing its evaluation rows")

    def rows_from(value: list[object]) -> list[MembershipRow]:
        result = []
        for row in value:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError(
                    "baseline membership row must contain source, group, component, fold"
                )
            source, group, component, fold = row
            if source not in ("train", "valid") or not all(
                isinstance(item, int) for item in (group, component, fold)
            ):
                raise ValueError("baseline membership row has invalid source-qualified values")
            result.append((source, group, component, fold))
        return result

    cache_hashes = value.get("cache_hashes")
    if not isinstance(cache_hashes, dict) or any(
        not isinstance(key, str) or not isinstance(digest, str)
        for key, digest in cache_hashes.items()
    ):
        raise ValueError("baseline membership is missing its source-cache hashes")
    return rows_from(value["fit"]), rows_from(value["eval"]), cache_hashes


def _validate_component_cache(rows: dict[str, np.ndarray]) -> None:
    """Require cached component labels to match the cached successor graph."""
    if not np.array_equal(rows["split"], rows["graph_split"]):
        raise ValueError("chain cache split order differs")
    if not np.array_equal(rows["row_group"], rows["graph_row_group"]):
        raise ValueError("chain cache row-group order differs")
    recovered = connected_components(rows["next"])
    for component in np.unique(rows["chain"]):
        members = rows["chain"] == component
        if np.unique(rows["fold"][members]).size != 1 or np.unique(recovered[members]).size != 1:
            raise ValueError("component/fold cache disagreement")
    if len(np.unique(recovered)) != len(np.unique(rows["chain"])):
        raise ValueError("cached chain partition differs from successor graph")


def _baseline_membership(
    root: Path,
    rows: dict[str, np.ndarray],
    path: Path,
    fit_components: int,
    seed: int,
    cache_hashes: dict[str, str],
) -> tuple[list[MembershipRow], list[MembershipRow], dict[str, object]]:
    """Extend existing fitting components while retaining every evaluation row verbatim."""
    baseline_fit, evaluation, baseline_hashes = _read_baseline(path)
    if baseline_hashes != cache_hashes:
        raise ValueError("baseline membership was generated from different chain-cache sources")
    base_components = {row[2] for row in baseline_fit}
    eval_components = {row[2] for row in evaluation}
    if len(evaluation) != 46 or len(base_components) != 12:
        raise ValueError("expected the frozen R2 membership of 12 fit components and 46 eval rows")
    if fit_components < len(base_components):
        raise ValueError("expanded membership cannot drop an existing fitting component")
    if _membership_rows(rows, base_components) != baseline_fit:
        raise ValueError("baseline fitting rows are not complete cache components")
    expected_eval = [row for row in _membership_rows(rows, eval_components) if row[0] == "valid"]
    if expected_eval != evaluation:
        raise ValueError(
            "baseline evaluation rows differ from complete source-qualified components"
        )
    bad_rows = (rows["split"] == "train") & (rows["row_group"] == KNOWN_BAD_TRAIN_ROW_GROUP)
    bad_components = set(map(int, np.unique(rows["chain"][bad_rows])))
    if int(np.sum(bad_rows)) > 1 or len(bad_components) > 1:
        raise ValueError("known bad train row group 528 is not uniquely indexed")
    bad_component = next(iter(bad_components)) if bad_components else None
    if bad_component is not None and bad_component in base_components:
        raise ValueError("known bad train row group 528 is already in the frozen fit membership")
    candidates = np.unique(rows["chain"][np.isin(rows["fold"], (2, 3))])
    excluded = base_components | eval_components | bad_components
    eligible = np.asarray(sorted(set(map(int, candidates)) - excluded), dtype=np.int64)
    add_count = fit_components - len(base_components)
    if add_count > len(eligible):
        raise ValueError("not enough eligible complete fitting components")
    selected_extra = np.random.default_rng(seed).choice(eligible, add_count, replace=False)
    selected_components = base_components | set(map(int, selected_extra))
    fitting = _membership_rows(rows, selected_components)
    if {row[2] for row in fitting} & eval_components:
        raise ValueError("fitting/evaluation component overlap")
    if any(row[3] not in (2, 3) for row in fitting):
        raise ValueError("expanded fitting components must be from folds 2 and 3")
    try:
        baseline_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        baseline_path = str(path)
    return (
        fitting,
        evaluation,
        {
            "fit_components_total": fit_components,
            "existing_fit_components": len(base_components),
            "additional_fit_components": add_count,
            "membership_seed": seed,
            "sampling": "sorted eligible component IDs sampled without replacement by NumPy default_rng",
            "eligible_component_count": len(eligible),
            "excluded_train_row": {"source": "train", "row_group": KNOWN_BAD_TRAIN_ROW_GROUP},
            "excluded_train_component": bad_component,
            "known_bad_row_status": (
                "component_excluded" if bad_component is not None else "absent_from_chain_cache"
            ),
            "reused_evaluation_membership": baseline_path,
        },
    )


def _legacy_membership(
    rows: dict[str, np.ndarray], fit_components: int, seed: int
) -> tuple[list[MembershipRow], list[MembershipRow]]:
    """Preserve the original seeded fit/evaluation sampling defaults."""
    rng = np.random.default_rng(seed)
    fitting_components = rng.choice(
        np.unique(rows["chain"][rows["fold"] >= 2]), fit_components, replace=False
    )
    evaluation_components = {
        str(fold): rng.choice(
            np.unique(rows["chain"][(rows["fold"] == fold) & (rows["split"] == "valid")]),
            8,
            replace=False,
        )
        for fold in (0, 1)
    }
    fitting = _membership_rows(rows, set(map(int, fitting_components)))
    eval_ids = set(map(int, np.concatenate(list(evaluation_components.values()))))
    evaluation = [row for row in _membership_rows(rows, eval_ids) if row[0] == "valid"]
    return fitting, evaluation


def freeze_membership(
    root: Path,
    output: Path,
    fit_components: int | None = 12,
    membership_seed: int = 20261017,
    reuse_evaluation_membership: Path | None = None,
) -> dict[str, object]:
    """Write preregistered source-qualified component membership from tiny caches only."""
    rows, cache_member_bytes = _read_rows(root)
    _validate_component_cache(rows)
    cache_hashes = {
        f"cache/{name}": hashlib.sha256((root / "cache" / name).read_bytes()).hexdigest()
        for name in ("chain_folds.npz", "chains.npz")
    }
    cache_binding = validate_chain_cache_sources(root, cache_hashes)
    all_eligible: dict[str, object] = {}
    if fit_components is None:
        if reuse_evaluation_membership is not None:
            raise ValueError("all-eligible mode derives its evaluation membership directly")
        fitting, evaluation, all_eligible = select_all_eligible(rows, KNOWN_BAD_TRAIN_ROW_GROUP)
        sampling: dict[str, object] = all_eligible
        recorded_seed: int | None = None
    elif reuse_evaluation_membership is None:
        if (fit_components, membership_seed) != (12, 20261017):
            raise ValueError(
                "nondefault membership requires an explicit frozen evaluation membership"
            )
        fitting, evaluation = _legacy_membership(rows, fit_components, membership_seed)
        sampling: dict[str, object] = {
            "fit_components_total": fit_components,
            "membership_seed": membership_seed,
            "sampling": "legacy seed samples fitting components and exposed evaluation rows",
        }
        recorded_seed = membership_seed
    else:
        if fit_components is None:
            raise ValueError("expanded component count must be explicit when reusing evaluation")
        fitting, evaluation, sampling = _baseline_membership(
            root,
            rows,
            reuse_evaluation_membership,
            fit_components,
            membership_seed,
            cache_hashes,
        )
        recorded_seed = membership_seed
    fit_components_set = {row[2] for row in fitting}
    eval_components_set = {row[2] for row in evaluation}
    if fit_components_set & eval_components_set:
        raise ValueError("fitting/evaluation component overlap")
    estimates = estimate_staging_bytes(len(fitting), len(evaluation))
    result = {
        "fit": fitting,
        "eval": evaluation,
        "cache_hashes": cache_hashes,
        "cache_source_binding": cache_binding,
        "evaluation_exposure": "exposed",
        "fit_folds": [2, 3],
        "eval_folds": [0, 1],
        "seed": recorded_seed,
        "sampling": sampling,
        "note": "One fitting pool; folds 0/1 are exposed evaluation. Fitting contains complete eligible fold-2/3 components, including permitted train and valid sequences. Evaluation scoring rows are valid-source rows; train counterparts in the same component are recorded and held out.",
        "resource_estimate": {
            "cache_compressed_bytes": sum(
                (root / "cache" / name).stat().st_size for name in cache_member_bytes
            ),
            "cache_npy_member_bytes": sum(cache_member_bytes.values()),
            "fit_sequences": len(fitting),
            "eval_sequences": len(evaluation),
            "eval_component_member_sequences": int(
                all_eligible.get("eval_component_member_sequence_count", len(evaluation))
            ),
            **estimates,
            "metadata_phase_parquet_reads": 0,
            "metadata_phase_cache_payload_bound_bytes": 2 * MAX_CACHE_ARRAY_BYTES,
            "stage_process_ram_reservation_bytes": 2_800_000_000,
            "aggregate_task_ram_cap_bytes": 5_000_000_000,
            "stage_ram_basis": "One ParquetFile reader, one sequence at a time, and per-sequence memmap cache advice; RAM reservation is the prior successful R1 staging bound, not a new measurement.",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_membership_json(output / "membership.json", result)
    return result
