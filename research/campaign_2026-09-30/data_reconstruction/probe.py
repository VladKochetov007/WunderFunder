from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
PRICE = [f"i0_p{i}" for i in range(22)]


def digest(path: Path) -> str:
    """Hash compact provenance artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict[str, object]:
    """Freeze chain-disjoint TRAIN membership without accessing targets."""
    excluded: set[int] = set()
    for path in (
        ROOT / "research/campaign_2026-09-29/provided_decode/sample_manifest.json",
        ROOT / "research/campaign_2026-09-29/provided_decode/preregistration_v2.json",
        ROOT / "research/campaign_2026-09-29/clock_reproduction/preregistration.json",
    ):
        excluded.update(json.loads(path.read_text())["evaluation_components"])
    metadata = ROOT / "cache/chain_folds.npz"
    with np.load(metadata) as data:
        split, group, chain = (data[k] for k in ("split", "row_group", "chain"))
        candidates = [
            (int(g), int(c))
            for g, c in zip(group[split == "train"], chain[split == "train"], strict=True)
            if int(c) not in excluded
        ]
    seed = json.loads((HERE / "preregistration.json").read_text())["seed"]
    candidates.sort(
        key=lambda pair: hashlib.sha256(f"{seed}:{pair[0]}:{pair[1]}".encode()).digest()
    )
    test: list[tuple[int, int]] = []
    seen: set[int] = set()
    for pair in candidates:
        if pair[1] not in seen:
            test.append(pair)
            seen.add(pair[1])
        if len(test) == 32:
            break
    fit = [pair for pair in candidates if pair[1] not in seen][:64]
    if {c for _, c in fit} & seen:
        raise ValueError("Component overlap")
    result = {
        "fit_train_groups": [g for g, _ in fit],
        "fit_components": [c for _, c in fit],
        "test_train_groups": [g for g, _ in test],
        "test_components": [c for _, c in test],
        "prior_evaluation_components_excluded": sorted(excluded),
        "chain_metadata_sha256": digest(metadata),
        "selected_without_targets_or_masks": True,
    }
    destination = HERE / "selection.json"
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise ValueError("Frozen selection changed")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def read(source: pq.ParquetFile, group: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Read only current price codes and two diagnostic TRAIN targets."""
    table = source.read_row_group(group, columns=PRICE + ["t0", "t1"], use_threads=False)
    x = np.column_stack([table.column(k).to_numpy() for k in PRICE]).astype(np.float32)
    y = np.column_stack([table.column(k).to_numpy() for k in ("t0", "t1")]).astype(np.float64)
    errors = np.max(
        np.abs(np.rint(y[None] / QUANTA[:, None, None]) * QUANTA[:, None, None] - y), axis=(1, 2)
    )
    cls = int(errors.argmin())
    if errors[cls] > 0.002:
        raise ValueError(f"Lattice mismatch in group {group}")
    return x, np.rint(y / QUANTA[cls]).astype(np.int32), cls, float(errors[cls])


def events(x: np.ndarray) -> np.ndarray:
    """Return complete representation-code changes including the origin."""
    return np.r_[0, np.flatnonzero(np.any(x[1:] != x[:-1], axis=1)) + 1]


def invert(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert the declared offset-two pair law for diagnostics only."""
    a, b = y.T
    branch = np.where(a > 0, 2, np.where(b > 0, 0, 1))
    spread = np.where(a > 0, -b - 2, np.where(b > 0, -a - 2, -a - b))
    return spread, branch


def key(row: np.ndarray) -> bytes:
    """Preserve the predeclared code-key grammar."""
    return np.round(row, 4).tobytes()


def accept(counts: dict[bytes, Counter[int]]) -> dict[bytes, int]:
    """Accept supported interior-spread identities."""
    output = {}
    for code, values in counts.items():
        if not values:
            continue
        value, support = values.most_common(1)[0]
        if support >= 3 and support / values.total() >= 0.95:
            output[code] = value
    return output


def cycles(moves: dict[tuple[bytes, bytes], tuple[int, int]]) -> dict[str, object]:
    """Check the absolute-potential hypothesis on accepted supervised moves."""
    graph: dict[bytes, list[tuple[bytes, np.ndarray]]] = defaultdict(list)
    reverse = Counter()
    examples = []
    for (a, b), delta in moves.items():
        d = np.array(delta)
        graph[a].append((b, d))
        graph[b].append((a, -d))
        if (b, a) in moves:
            closure = tuple((d + moves[b, a]).tolist())
            reverse[str(closure)] += 1
    potential: dict[bytes, np.ndarray] = {}
    contradictions = 0
    for start in graph:
        if start in potential:
            continue
        potential[start] = np.zeros(2, dtype=np.int64)
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for b, delta in graph[a]:
                wanted = potential[a] + delta
                if b not in potential:
                    potential[b] = wanted
                    queue.append(b)
                elif np.any(wanted != potential[b]):
                    contradictions += 1
                    if len(examples) < 5:
                        examples.append(
                            {
                                "from_sha256": hashlib.sha256(a).hexdigest(),
                                "to_sha256": hashlib.sha256(b).hexdigest(),
                                "closure_ticks": (wanted - potential[b]).tolist(),
                            }
                        )
    return {
        "nodes": len(graph),
        "directed_accepted_edges": len(moves),
        "reverse_pair_closure_directed_counts": dict(reverse),
        "potential_constraint_checks": sum(map(len, graph.values())),
        "nonzero_potential_constraint_checks": contradictions,
        "closure_examples": examples,
        "interpretation": "Nonzero closure rules out absolute-price potential on this supervised table; does not rule out relative-shape states or common-reference contamination.",
    }


def main() -> None:
    """Run a bounded supervised-coordinate falsifier on fixed TRAIN chains."""
    started = time.monotonic()
    selection = freeze()
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    interior = [defaultdict(Counter) for _ in range(4)]
    all_spread = [defaultdict(Counter) for _ in range(4)]
    votes = [defaultdict(Counter) for _ in range(4)]
    samples_x: list[list[np.ndarray]] = [[] for _ in range(4)]
    samples_y: list[list[np.ndarray]] = [[] for _ in range(4)]
    fit_count: Counter[int] = Counter()
    for group in selection["fit_train_groups"]:
        x, y, cls, _ = read(source, group)
        fit_count[cls] += 1
        ev = events(x)
        keys = [key(row) for row in x[ev]]
        spread, branch = invert(y)
        for code, start, stop in zip(keys, ev, np.r_[ev[1:], len(x)], strict=True):
            all_spread[cls][code].update(spread[start:stop:7].tolist())
            rows = np.arange(start, stop, 7)
            rows = rows[(branch[rows] == 1) & (spread[rows] >= 1)]
            interior[cls][code].update(spread[rows].tolist())
        for i, row in enumerate(ev[1:], 1):
            if branch[row - 1] == branch[row] == 1:
                dy = y[row] - y[row - 1]
                votes[cls][keys[i - 1], keys[i]][(-int(dy[0]), int(dy[1]))] += 1
        loc = ev[(branch[ev] == 1) & (spread[ev] >= 1)]
        if len(loc) > 256:
            loc = loc[np.linspace(0, len(loc) - 1, 256, dtype=int)]
        samples_x[cls].append(x[loc].astype(np.float64))
        samples_y[cls].append(spread[loc].astype(np.float64))
    maps = [accept(counts) for counts in interior]
    coefficients = []
    accepted_moves = []
    for cls in range(4):
        xx = np.concatenate(samples_x[cls])
        yy = np.concatenate(samples_y[cls])
        design = np.column_stack((np.ones(len(xx)), xx))
        coefficients.append(
            np.linalg.solve(design.T @ design + np.diag([0] + [0.001] * 22), design.T @ yy)
        )
        accepted_moves.append(
            {
                pair: counter.most_common(1)[0][0]
                for pair, counter in votes[cls].items()
                if counter.most_common(1)[0][1] >= 3
                and counter.most_common(1)[0][1] / counter.total() >= 0.8
            }
        )
    summaries = []
    totals = [Counter() for _ in range(4)]
    offsets = [Counter() for _ in range(4)]
    errors_linear = [defaultdict(list) for _ in range(4)]
    for group, component in zip(
        selection["test_train_groups"], selection["test_components"], strict=True
    ):
        x, y, cls, lattice_error = read(source, group)
        ev = events(x)
        keys = [key(row) for row in x[ev]]
        sp, branch = invert(y)
        target_change = np.any(y[1:] != y[:-1], axis=1)
        price_change = np.any(x[1:] != x[:-1], axis=1)
        local = Counter(
            {
                "groups": 1,
                "rows": len(x),
                "events": len(ev) - 1,
                "target_change": int(target_change.sum()),
                "target_change_without_price": int((target_change & ~price_change).sum()),
                "spread_change_without_price": int(((sp[1:] != sp[:-1]) & ~price_change).sum()),
                "invalid_spread_rows": int((sp < 1).sum()),
                "both_positive_rows": int(np.all(y > 0, axis=1).sum()),
            }
        )
        local_offset: Counter[int] = Counter()
        for code, start, stop in zip(keys, ev, np.r_[ev[1:], len(x)], strict=True):
            rows = np.arange(start, stop)
            mapped = maps[cls].get(code)
            if mapped is None:
                continue
            local["mapped_interior_code_rows"] += len(rows)
            interior_rows = rows[branch[rows] == 1]
            local["interior_test_mapped_rows"] += len(interior_rows)
            local["interior_test_spread_exact_rows"] += int((sp[interior_rows] == mapped).sum())
            outside = rows[branch[rows] != 1]
            local["outside_test_mapped_rows"] += len(outside)
            losing = np.minimum(y[outside, 0], y[outside, 1])
            local_offset.update((-losing - mapped).tolist())
        for i, row in enumerate(ev[1:], 1):
            pair = keys[i - 1], keys[i]
            if branch[row - 1] == branch[row] == 1:
                local["interior_transition_events"] += 1
                move = accepted_moves[cls].get(pair)
                if move is not None:
                    local["mapped_interior_transition_events"] += 1
                    dy = y[row] - y[row - 1]
                    local["exact_mapped_transition_events"] += move == (-int(dy[0]), int(dy[1]))
        for b in range(3):
            loc = ev[branch[ev] == b]
            if len(loc) > 256:
                loc = loc[np.linspace(0, len(loc) - 1, 256, dtype=int)]
            pred = np.column_stack((np.ones(len(loc)), x[loc])) @ coefficients[cls]
            errors_linear[cls][str(b)].extend((pred - sp[loc]).tolist())
        totals[cls].update(local)
        offsets[cls].update(local_offset)
        summaries.append(
            {
                "group": group,
                "component": component,
                "diagnostic_target_class": cls,
                "lattice_max_error": lattice_error,
                "counts": dict(local),
                "outside_offsets": dict(local_offset),
            }
        )
    result = {
        "preregistration_sha256": digest(HERE / "preregistration.json"),
        "selection_sha256": digest(HERE / "selection.json"),
        "source_sha256": digest(Path(__file__)),
        "scope": "TRAIN-only held-out components, all rows without prediction/scoring masks; target-supervised diagnostic, no deployable gain claim",
        "by_class": {},
        "by_group": summaries,
    }
    for cls in range(4):
        result["by_class"][str(cls)] = {
            "fit_groups": fit_count[cls],
            "fit_interior_code_count": len(interior[cls]),
            "accepted_interior_spread_codes": len(maps[cls]),
            "fit_code_spread_ambiguous": sum(len(v) > 1 for v in all_spread[cls].values()),
            "fit_interior_code_spread_ambiguous": sum(len(v) > 1 for v in interior[cls].values()),
            "test_counts": dict(totals[cls]),
            "test_outside_loss_offsets_from_interior_only_map": dict(offsets[cls]),
            "fixed_raw_price_linear_spread_control": {
                branch: {
                    "event_samples": len(err),
                    "mae_ticks": float(np.mean(np.abs(err))),
                    "rmse_ticks": float(np.sqrt(np.mean(np.square(err)))),
                    "round_exact_fraction": float(np.mean(np.rint(err) == 0)),
                }
                for branch, err in errors_linear[cls].items()
                if err
            },
            "supervised_transition_cycles": cycles(accepted_moves[cls]),
        }
    result["resources"] = {
        "wall_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(HERE / "results.json"), "resources": result["resources"]}))


if __name__ == "__main__":
    main()
