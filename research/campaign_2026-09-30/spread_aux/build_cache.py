from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import logging
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from map_support import (
    PRICE_COLUMNS,
    QUANTA,
    ROWS,
    UNKNOWN,
    load_table,
    lookup,
    packed_codes,
    read_prices,
    sha256,
)

from wnn.splits import load_campaign_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
LOG = logging.getLogger(__name__)


def write_json(path: Path, value: object) -> None:
    """Publish actor-owned metadata atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_sources(selection: dict) -> None:
    """Reject changed fitting selection, source code, or parquet identities."""
    for relative, digest in selection["source_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"frozen source changed: {relative}")
    for relative, identity in selection["parquet_identities"].items():
        stat = (ROOT / relative).stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"frozen parquet changed: {relative}")


def freeze() -> dict:
    """Freeze component-hashed512group selection before any target values."""
    if (HERE / "selection.json").exists():
        raise FileExistsError("selection already frozen")
    contract = json.loads((HERE / "preregistration.json").read_text())
    split = load_campaign_split(SPLIT, ROOT)
    raw_split = json.loads(SPLIT.read_text())
    components = {(source, group): component for source, group, component in split.components}
    evaluated = {components["valid", group] for group in split.evaluation_groups}
    choices: dict[int, list[int]] = defaultdict(list)
    for group in split.train_groups:
        component = components["train", group]
        if component in evaluated:
            raise AssertionError("evaluation-linked map-fitting component")
        choices[component].append(group)
    salt = contract["selection_salt"]
    ordered = sorted(
        choices, key=lambda c: hashlib.sha256(f"{salt}:component:{c}".encode()).digest()
    )
    chosen = [
        (min(choices[c], key=lambda g: hashlib.sha256(f"{salt}:group:{g}".encode()).digest()), c)
        for c in ordered[: contract["map_train_groups"]]
    ]
    if len(chosen) != 512:
        raise ValueError("not enough distinct TRAIN fitting components")
    n_spread = next(
        ast.literal_eval(node.value)
        for node in ast.parse((ROOT / "src/wnn/outcomes.py").read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "N_SP_AUX" for target in node.targets)
    )
    if n_spread != contract["spread_head_bins"]:
        raise ValueError("spread head does not match actual outcome constants")
    files = [
        HERE / "preregistration.json",
        Path(__file__),
        HERE / "map_support.py",
        SPLIT,
        ROOT / "src/wnn/outcomes.py",
        ROOT / "src/wnn/features.py",
        ROOT / "scripts/train_rnn.py",
        ROOT / contract["shared_initial_checkpoint"],
    ]
    selection = {
        "map_fit_train_groups": [g for g, _ in chosen],
        "map_fit_components": [c for _, c in chosen],
        "fit_train_groups": list(split.train_groups),
        "fit_valid_groups": list(split.valid_groups),
        "evaluation_valid_groups": list(split.evaluation_groups),
        "evaluation_components": sorted(evaluated),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "split_manifest_sha256": split.sha256,
        "split_manifest_path": str(SPLIT.relative_to(ROOT)),
        "parquet_identities": {
            key: value for key, value in raw_split["sources"].items() if key.startswith("datasets/")
        },
        "selection_before_any_target_read": True,
        "selection_reads": "feature-only component metadata and target index IDs only, never target values",
        "map_vote_columns": PRICE_COLUMNS + ["t0", "t1"],
        "cache_columns": PRICE_COLUMNS,
        "forbidden_columns": ["is_scored", "need_prediction", "seq_ix"],
        "N_SP_AUX": n_spread,
    }
    write_json(HERE / "selection.json", selection)
    return selection


def fit_map(selection: dict) -> dict:
    """Count only TRAIN interior votes and preserve all rejected/conflict codes."""
    if (HERE / "interior_code_table.json").exists():
        raise FileExistsError("map already fitted/frozen")
    verify_sources(selection)
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    counts: dict[bytes, Counter[tuple[int, int]]] = defaultdict(Counter)
    groups = []
    for group in selection["map_fit_train_groups"]:
        data = source.read_row_group(group, columns=PRICE_COLUMNS + ["t0", "t1"], use_threads=False)
        prices = np.column_stack([data.column(name).to_numpy() for name in PRICE_COLUMNS]).astype(
            np.float32
        )
        targets = np.column_stack([data.column(name).to_numpy() for name in ["t0", "t1"]]).astype(
            np.float64
        )
        errors = np.max(
            np.abs(
                np.rint(targets[None] / QUANTA[:, None, None]) * QUANTA[:, None, None] - targets
            ),
            axis=(1, 2),
        )
        cls = int(errors.argmin())
        if errors[cls] > 0.002 or len(prices) != ROWS:
            raise ValueError(f"invalid lattice/length: TRAIN group{group}")
        ticks = np.rint(targets / QUANTA[cls]).astype(np.int32)
        spreads = -ticks.sum(axis=1)
        interior = (ticks <= 0).all(axis=1) & (spreads >= 1)
        starts = np.r_[0, np.flatnonzero(np.any(prices[1:] != prices[:-1], axis=1)) + 1]
        codes = packed_codes(prices)
        total_votes = 0
        for first, last in zip(starts, np.r_[starts[1:], ROWS], strict=True):
            rows = np.arange(first, last, 7)
            rows = rows[interior[rows]]
            counts[bytes(codes[first])].update((cls, int(value)) for value in spreads[rows])
            total_votes += len(rows)
        groups.append(
            {
                "group": group,
                "diagnostic_class": cls,
                "interior_votes": total_votes,
                "interior_rows": int(interior.sum()),
                "total_rows": ROWS,
            }
        )
    accepted, all_votes = [], []
    for code, votes in sorted(counts.items()):
        if not votes:
            continue
        (cls, spread), support = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0]
        is_accepted = support >= 3 and support / votes.total() >= 0.95
        record = {
            "code_float32_round4_hex": code.hex(),
            "class": cls,
            "spread_ticks": spread,
            "fit_vote_count": support,
            "fit_total_votes": votes.total(),
        }
        all_votes.append(
            {
                **record,
                "accepted": is_accepted,
                "all_votes": [
                    {"class": c, "spread_ticks": s, "votes": n}
                    for (c, s), n in sorted(votes.items())
                ],
            }
        )
        if is_accepted:
            accepted.append(record)
    write_json(HERE / "interior_code_table.json", accepted)
    write_json(HERE / "all_code_votes.json", all_votes)
    result = {
        "accepted_codes": len(accepted),
        "voted_codes": len(all_votes),
        "class_conflict_codes": sum(len({c for c, _ in v}) > 1 for v in counts.values()),
        "spread_conflict_codes": sum(len({s for _, s in v}) > 1 for v in counts.values()),
        "selection_sha256": sha256(HERE / "selection.json"),
        "table_sha256": sha256(HERE / "interior_code_table.json"),
        "all_votes_sha256": sha256(HERE / "all_code_votes.json"),
        "fit_groups": groups,
        "target_reads": "only frozen512TRAIN groups; VALID targets/masks/evaluation data never read",
    }
    write_json(HERE / "map_results.json", result)
    return result


def cache(selection: dict) -> dict:
    """Apply frozen current-code map to fitting rows only, with bounded resume."""
    verify_sources(selection)
    table = load_table(HERE / "interior_code_table.json")
    map_result = json.loads((HERE / "map_results.json").read_text())
    if sha256(HERE / "interior_code_table.json") != map_result["table_sha256"]:
        raise ValueError("frozen mapping changed")
    if (HERE / "metadata.json").exists():
        raise FileExistsError("complete cache already exists")
    started = time.monotonic()
    progress_path = HERE / "cache_progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"counts": {}, "completed": {}}
    )
    for split in ["train", "valid"]:
        groups = selection[f"fit_{split}_groups"]
        path = HERE / f"spread_{split}.partial.npy"
        if not path.exists():
            values = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.uint8, shape=(len(groups), ROWS)
            )
            values[:] = UNKNOWN
            values.flush()
        else:
            values = np.load(path, mmap_mode="r+")
        if values.shape != (len(groups), ROWS) or values.dtype != np.uint8:
            raise ValueError("partial cache shape/type changed")
        source = pq.ParquetFile(ROOT / f"datasets/{split}.parquet")
        done = progress["completed"].get(split, 0)
        for position in range(done, len(groups)):
            prices = read_prices(source, groups[position])
            encoded = lookup(prices, table)
            values[position] = encoded
            progress["counts"][f"{split}/{groups[position]}"] = {
                "known": int((encoded != UNKNOWN).sum()),
                "rows": ROWS,
            }
            progress["completed"][split] = position + 1
            if (position + 1) % 100 == 0:
                values.flush()
                write_json(progress_path, progress)
                LOG.info("%s %s/%s", split, position + 1, len(groups))
            if time.monotonic() - started > 265:
                values.flush()
                write_json(progress_path, progress)
                return {
                    "status": "partial",
                    "resume_same_selection": True,
                    "completed": progress["completed"],
                }
        values.flush()
        write_json(progress_path, progress)
        del source, values
        gc.collect()
    files = {}
    for split in ["train", "valid"]:
        path = HERE / f"spread_{split}.partial.npy"
        final = HERE / f"spread_{split}.npy"
        path.rename(final)
        files[split] = {
            "file": final.name,
            "sha256": sha256(final),
            "bytes": final.stat().st_size,
            "row_groups": selection[f"fit_{split}_groups"],
            "shape": [len(selection[f"fit_{split}_groups"]), ROWS],
        }
    metadata = {
        "status": "complete",
        "split_manifest_sha256": selection["split_manifest_sha256"],
        "selection_sha256": sha256(HERE / "selection.json"),
        "map_sha256": map_result["table_sha256"],
        "fit_train_groups": selection["fit_train_groups"],
        "fit_valid_groups": selection["fit_valid_groups"],
        "evaluation_valid_groups": selection["evaluation_valid_groups"],
        "files": files,
        "dtype": "uint8",
        "unknown": UNKNOWN,
        "N_SP_AUX": selection["N_SP_AUX"],
        "lookup_inputs": "current22float32prices only, no targets/classes/masks/IDs",
        "supervision_only": True,
        "counts": progress["counts"],
    }
    write_json(HERE / "metadata.json", metadata)
    return {
        "status": "complete",
        "metadata_sha256": sha256(HERE / "metadata.json"),
        "artifact_array_bytes": sum(entry["bytes"] for entry in files.values()),
    }


def main() -> None:
    """Execute one bounded actor phase; preserve reports and exact sources."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "map", "cache"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.monotonic()
    report = {
        "phase": args.phase,
        "run_id": args.run_id,
        "competition_scoring": False,
        "source_sha256": sha256(Path(__file__)),
        "started_unix": time.time(),
    }
    try:
        if args.phase == "freeze":
            result = freeze()
            report["result"] = {
                "map_groups": len(result["map_fit_train_groups"]),
                "selection_sha256": sha256(HERE / "selection.json"),
            }
        else:
            selection = json.loads((HERE / "selection.json").read_text())
            report["result"] = fit_map(selection) if args.phase == "map" else cache(selection)
        report["exit_code"] = 0
    except (AssertionError, RuntimeError, ValueError, OSError) as error:
        report.update({"exit_code": 1, "failure": str(error)})
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with (HERE / f"job_{args.run_id}.json").open("x") as output:
            output.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
