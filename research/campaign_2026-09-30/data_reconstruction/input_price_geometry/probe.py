"""Input-only price geometry; no target-based code or movement tables."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rankspectrum(x: np.ndarray) -> dict[str, object]:
    r = rankdata(x, axis=0, method="average")
    r -= r.mean(0)
    norms = np.sqrt((r * r).sum(0))
    r /= np.maximum(norms, 1e-30)
    corr = r.T @ r
    eig, vec = np.linalg.eigh(corr)
    loading = vec[:, -1]
    if loading[0] < 0:
        loading = -loading
    trace = float(eig.sum())
    pairs = corr[np.triu_indices(x.shape[1], 1)]
    signs = np.where(loading < 0, -1, 1)
    corrected = corr * signs[:, None] * signs[None, :]
    return {
        "rows": len(x),
        "active_columns": int((norms > 0).sum()),
        "signed_rho_median": float(np.median(pairs)),
        "absolute_rho_median": float(np.median(np.abs(pairs))),
        "negative_pair_fraction": float((pairs < 0).mean()),
        "top_loading": loading.tolist(),
        "top_loading_negative_count": int((loading < 0).sum()),
        "top_loading_nearzero_count": int((np.abs(loading) <= 1e-8).sum()),
        "sign_corrected_rho_median": float(np.median(corrected[np.triu_indices(x.shape[1], 1)])),
        "spectrum": (eig[::-1] / max(trace, 1e-30)).tolist(),
        "top_share": float(eig[-1] / max(trace, 1e-30)),
        "top_two_share": float(eig[-2:].sum() / max(trace, 1e-30)),
        "effective_rank": trace * trace / max(float((eig * eig).sum()), 1e-30),
        "correlation": corr.tolist(),
    }


def monotonicity(values: np.ndarray, orientations: list[int] | None = None) -> dict[str, object]:
    order = np.argsort(values[:, 0], kind="stable")
    v = values[order]
    out = []
    for j in range(1, v.shape[1]):
        dif = np.diff(v[:, j])
        sx = rankdata(v[:, 0])
        sy = rankdata(v[:, j])
        sx -= sx.mean()
        sy -= sy.mean()
        den = np.sqrt((sx * sx).sum() * (sy * sy).sum())
        rho = float(sx @ sy / max(den, 1e-30))
        orientation = orientations[j - 1] if orientations is not None else (1 if rho >= 0 else -1)
        signs = np.sign(dif)
        nonzero = signs[signs != 0]
        bad = float((orientation * dif < 0).mean()) if len(dif) else 0
        out.append(
            {
                "sibling": j,
                "rho": rho,
                "orientation": orientation,
                "increasing": int((dif > 0).sum()),
                "decreasing": int((dif < 0).sum()),
                "ties": int((dif == 0).sum()),
                "off_orientation_fraction": bad,
                "turning_points": int((nonzero[1:] != nonzero[:-1]).sum()),
                "monotone_support": len(v) >= 64 and orientation * rho >= 0.99 and bad <= 0.001,
            }
        )
    return {
        "accepted_leaders": len(v),
        "all_siblings_support": all(o["monotone_support"] for o in out),
        "siblings": out,
    }


def block_keys(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype="<f4").view(np.dtype((np.void, 44))).ravel()


def graph_summary(nodes: set[bytes], edges: set[tuple[bytes, bytes]]) -> dict[str, object]:
    adjacency: dict[bytes, set[bytes]] = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    degree = Counter(len(adjacency[n]) for n in nodes)
    unseen = set(nodes)
    sizes = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return {
        "nodes": len(nodes),
        "directed_edges": len(edges),
        "undirected_edges": len({tuple(sorted(e)) for e in edges}),
        "degree_histogram": dict(sorted(degree.items())),
        "nodes_degree_above2": sum(n for k, n in degree.items() if k > 2),
        "component_sizes": sorted(sizes, reverse=True),
        "strict_unit_step_line_necessary_gate": max(degree, default=0) <= 2,
        "interpretation": "degree>2 excludes only strictunitneighbor scalarstate graph; multijumps/shape/relative contexts not excluded",
    }


def main() -> None:
    started = time.perf_counter()
    selection = json.loads((OWN / "selection.json").read_text())
    frozen = json.loads((OWN / "source_hashes.json").read_text())
    for name, value in frozen.items():
        if digest(OWN / name) != value:
            raise ValueError("sourcefreezechanged " + name)
    data = ROOT / "datasets/train.parquet"
    identity = selection["source_train_identity"]
    if data.stat().st_size != identity["bytes"] or data.stat().st_mtime_ns != identity["mtime_ns"]:
        raise ValueError("TRAIN changed")
    source = pq.ParquetFile(data)
    fit_votes = {key: defaultdict(Counter) for key in ("i0_A", "i0_B", "i1_A", "i1_B")}
    fit_nodes = {key: set() for key in fit_votes}
    fit_edges = {key: set() for key in fit_votes}
    fit_arrays = {key: [] for key in fit_votes}
    spectra = []
    test_arrays = []
    for role in ("fit", "test"):
        for node in selection[role]:
            names = [f"i{i}_p{j}" for i in range(2) for j in range(22)]
            raw = source.read_row_group(node["row_group"], columns=names, use_threads=False)
            x = np.column_stack([raw.column(n).to_numpy() for n in names]).astype(np.float32)
            for inst in range(2):
                p = x[:, inst * 22 : (inst + 1) * 22]
                for weight, z in (("rows", p), ("unique22", np.unique(p, axis=0))):
                    spectra.append(
                        {
                            "role": role,
                            **node,
                            "instrument": inst,
                            "weighting": weight,
                            "full22": rankspectrum(z),
                            "first11": rankspectrum(z[:, :11]),
                            "last11": rankspectrum(z[:, 11:]),
                        }
                    )
                for side, start in (("A", 0), ("B", 11)):
                    key = f"i{inst}_{side}"
                    block = p[:, start : start + 11]
                    if role == "fit":
                        unique, counts = np.unique(block, axis=0, return_counts=True)
                        fit_arrays[key].append(unique)
                        for row, count in zip(unique, counts, strict=True):
                            fit_votes[key][float(row[0])][
                                np.ascontiguousarray(row[1:], dtype="<f4").tobytes()
                            ] += int(count)
                        keys = block_keys(block)
                        fit_nodes[key].update(k.tobytes() for k in np.unique(keys))
                        changed = keys[1:] != keys[:-1]
                        fit_edges[key].update(
                            (a.tobytes(), b.tobytes())
                            for a, b in zip(keys[:-1][changed], keys[1:][changed], strict=True)
                        )
            if role == "test":
                test_arrays.append((node, x))
            print(json.dumps({"read": role, **node}), flush=True)
    maps = {}
    mapping_records = {}
    fit_monotone = {}
    graphs = {}
    for key, votes in fit_votes.items():
        accepted = {}
        records = []
        for leader, counter in votes.items():
            output, support = counter.most_common(1)[0]
            total = counter.total()
            if support >= 3 and support / total >= 0.999:
                accepted[leader] = output
                records.append(
                    {
                        "leader": leader,
                        "output_float32_hex": output.hex(),
                        "support": support,
                        "total": total,
                    }
                )
        maps[key] = accepted
        mapping_records[key] = records
        values = np.array(
            [[leader, *np.frombuffer(out, dtype="<f4")] for leader, out in accepted.items()],
            dtype=np.float64,
        )
        fit_monotone[key] = monotonicity(values)
        fit_unique = np.unique(np.concatenate(fit_arrays[key]), axis=0)
        graphs[key] = graph_summary(fit_nodes[key], fit_edges[key])
        graphs[key]["distinct_fit_leaders"] = len(votes)
        graphs[key]["unique_block_codes"] = len(fit_unique)
        graphs[key]["raw_scalar_functional_conflicts"] = sum(len(c) > 1 for c in votes.values())
    test_records = []
    for node, x in test_arrays:
        for inst in range(2):
            for side, start in (("A", 0), ("B", 11)):
                key = f"i{inst}_{side}"
                block = x[:, inst * 22 + start : inst * 22 + start + 11]
                keys = block_keys(block)
                prediction = [maps[key].get(float(v)) for v in block[:, 0]]
                known = np.array([p is not None for p in prediction])
                exact = np.array(
                    [
                        p is not None and p == np.ascontiguousarray(row[1:], dtype="<f4").tobytes()
                        for p, row in zip(prediction, block, strict=True)
                    ]
                )
                unique = np.unique(block, axis=0)
                u_known = np.array([float(row[0]) in maps[key] for row in unique])
                u_exact = np.array(
                    [
                        maps[key].get(float(row[0]))
                        == np.ascontiguousarray(row[1:], dtype="<f4").tobytes()
                        for row in unique
                    ]
                )
                changes = np.flatnonzero(keys[1:] != keys[:-1]) + 1
                known_node = np.array([k.tobytes() in fit_nodes[key] for k in keys])
                edge_known = np.array(
                    [(keys[t - 1].tobytes(), keys[t].tobytes()) in fit_edges[key] for t in changes]
                )
                leaders = np.unique(block[known, 0])
                values = []
                for lead in leaders:
                    rows = block[(block[:, 0] == lead) & known]
                    values.append(rows[0])
                mono = (
                    monotonicity(
                        np.asarray(values, dtype=np.float64),
                        [v["orientation"] for v in fit_monotone[key]["siblings"]],
                    )
                    if len(values) >= 2
                    else None
                )
                support = bool(
                    fit_monotone[key]["all_siblings_support"]
                    and mono is not None
                    and mono["all_siblings_support"]
                    and known.mean() >= 0.95
                    and exact.sum() / max(1, known.sum()) >= 0.999
                    and len(leaders) >= 64
                )
                test_records.append(
                    {
                        **node,
                        "key": key,
                        "rows": len(block),
                        "known_rows": int(known.sum()),
                        "exact_known_rows": int(exact.sum()),
                        "unknown_rows": int((~known).sum()),
                        "unique_codes": len(unique),
                        "known_unique_codes": int(u_known.sum()),
                        "exact_known_unique_codes": int(u_exact.sum()),
                        "known_distinct_leaders": len(leaders),
                        "monotonicity_on_known_test_leaders": mono,
                        "orientation_frozen_on_fit": [
                            v["orientation"] for v in fit_monotone[key]["siblings"]
                        ],
                        "fit_node_coverage_rows": int(known_node.sum()),
                        "transition_rows": len(changes),
                        "known_fit_transition_rows": int(edge_known.sum()),
                        "unknown_fit_transition_rows": int((~edge_known).sum()),
                        "input_ordinal_support_gate": support,
                    }
                )
    with (OWN / "fit_feature_maps.json").open("x") as f:
        json.dump(mapping_records, f, indent=2)
        f.write("\n")
    result = {
        "spectra": spectra,
        "fit_monotonicity": fit_monotone,
        "fit_graphs": graphs,
        "test": test_records,
        "all_input_ordinal_support_gates": all(
            r["input_ordinal_support_gate"] for r in test_records
        ),
        "rows_fit": 8 * 20000,
        "rows_test": 8 * 20000,
        "selection_sha256": digest(OWN / "selection.json"),
        "source_sha256": digest(OWN / "probe.py"),
        "preregistration_sha256": digest(OWN / "preregistration.json"),
        "maps_sha256": digest(OWN / "fit_feature_maps.json"),
        "duration_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
        "cpu_limit_seconds": resource.getrlimit(resource.RLIMIT_CPU)[0],
        "targets_masks_VALID_external_read": False,
        "physical_tick_displacement_identified": False,
        "durable_or_cgroup_execution_claim": False,
    }
    with (OWN / "results.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("spectra", "fit_monotonicity", "fit_graphs", "test")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
