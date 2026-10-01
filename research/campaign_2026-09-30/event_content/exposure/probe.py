"""Feature-only exposure falsifier; preregistered data access gate enforced."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[3]
PRICE = [f"i0_p{i}" for i in range(22)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basis() -> np.ndarray:
    return np.array(
        [[1 - 2 * ((n & j).bit_count() % 2) for j in range(96)] for n in range(128)],
        dtype=np.float64,
    )


def exposure(
    ordinal: np.ndarray, name: str, grams: dict[str, np.ndarray], table: np.ndarray
) -> dict[str, object]:
    hist = np.bincount(ordinal % 128, minlength=128).astype(np.int64)
    gram = table.T @ (hist[:, None] * table)
    grams[name] = gram
    eig = np.linalg.eigvalsh(gram)
    threshold = max(1e-8, float(eig[-1]) * 1e-8)
    rank = int((eig > threshold).sum())
    unique = np.unique(ordinal)
    distinct_mod = int((hist > 0).sum())
    rawdistincthist = np.bincount(unique % 128, minlength=128)
    mean = (hist[:, None] * table).sum(0) / max(1, int(hist.sum()))
    variance = (hist[:, None] * table**2).sum(0) / max(1, int(hist.sum())) - mean**2
    return {
        "rows": len(ordinal),
        "ordinal_min": int(ordinal.min()) if len(ordinal) else None,
        "ordinal_max": int(ordinal.max()) if len(ordinal) else None,
        "hist128": hist.tolist(),
        "distinct_mod128": distinct_mod,
        "distinct_ordinals": len(unique),
        "aliased_distinct_ordinals": len(unique) - distinct_mod,
        "alias_distinctordinal_multiplicity128": rawdistincthist.tolist(),
        "maximum_alias_distinctordinal_multiplicity": int(rawdistincthist.max()),
        "maximum_repeatedquery_or_write_count_permod": int(hist.max()),
        "repeated_exposures_beyond_modclasses": len(ordinal) - distinct_mod,
        "rank": rank,
        "eigenvalue_min": float(eig[0]),
        "eigenvalue_max": float(eig[-1]),
        "rank_threshold": threshold,
        "gram_key": name,
        "column0_uncentered_norm": float(gram[0, 0]),
        "nonintercept_variance_min": float(variance[1:].min()),
        "nonintercept_variance_max": float(variance[1:].max()),
    }


def main() -> None:
    started = time.perf_counter()
    # Root creates authorization only on two affirmative completion messages.
    auth = json.loads((OWN / "launch_authorization.json").read_text())
    if auth.get("approved") is not True:
        raise ValueError("independent implementation gates not authorized")
    for key in ("critic_D1_grade", "builder_integration_CPU_proof"):
        evidence = auth[key]
        if (
            evidence.get("passed") is not True
            or digest(ROOT / evidence["path"]) != evidence["sha256"]
        ):
            raise ValueError(f"gate evidence absent/changed: {key}")
    frozen = json.loads((OWN / "frozen_sources.json").read_text())
    for name, expected in frozen.items():
        if digest(OWN / name) != expected:
            raise ValueError(f"frozen local source changed: {name}")
    selection = json.loads((OWN / "selection.json").read_text())
    table_path = ROOT / "research/campaign_2026-09-30/data_reconstruction/interior_code_table.json"
    if digest(table_path) != selection["class_table_sha256"]:
        raise ValueError("TRAIN class diagnostic table changed")
    fold_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    if digest(fold_path) != selection["fold_manifest_sha256"]:
        raise ValueError("frozen fold changed")
    data = ROOT / "datasets/train.parquet"
    identity = selection["source_train_identity"]
    if data.stat().st_size != identity["bytes"] or data.stat().st_mtime_ns != identity["mtime_ns"]:
        raise ValueError("TRAIN identity changed")
    class_map = {
        bytes.fromhex(v["code_float32_round4_hex"]): int(v["class"])
        for v in json.loads(table_path.read_text())
    }
    table = basis()
    source = pq.ParquetFile(data)
    records = []
    grams = {}
    aggregate_hist = {}
    class_totals = {
        str(c): {"rows": 0, "endpoint_writes": 0, "code_writes": 0} for c in (-1, 0, 1, 2, 3)
    }
    for node in selection["selected"]:
        group, component = node["row_group"], node["component"]
        raw = source.read_row_group(group, columns=PRICE + ["a6"], use_threads=False)
        x = np.column_stack([raw.column(n).to_numpy() for n in PRICE]).astype(np.float32)
        a6 = raw.column("a6").to_numpy().astype(np.float32)
        endpoint = a6 == np.float32(-5.199337482452393)
        code = np.r_[False, np.any(x[1:] != x[:-1], axis=1)]
        packed = (
            np.ascontiguousarray(np.round(x, 4), dtype="<f4").view(np.dtype((np.void, 88))).ravel()
        )
        unique, inverse = np.unique(packed, return_inverse=True)
        mapped = np.array([class_map.get(k.tobytes(), -1) for k in unique], dtype=np.int8)[inverse]
        rec = {
            "row_group": group,
            "component": component,
            "rows": len(x),
            "endpoint_writes": int(endpoint.sum()),
            "code_writes": int(code.sum()),
            "endpoint_origin": bool(endpoint[0]),
            "code_origin_suppressed": True,
            "nonorigin_endpoint_without_rawcode": int((endpoint[1:] & ~code[1:]).sum()),
            "nonorigin_rawcode_without_endpoint": int((code[1:] & ~endpoint[1:]).sum()),
            "mapped_class_counts": {str(c): int((mapped == c).sum()) for c in (-1, 0, 1, 2, 3)},
            "clocks": {},
            "class_diagnostics": {},
        }
        for label, gate in (("row", np.ones(len(x), bool)), ("endpoint", endpoint), ("code", code)):
            count = np.cumsum(gate, dtype=np.int64)
            ordinal = (
                np.arange(len(x), dtype=np.int64) if label == "row" else np.maximum(count - 1, 0)
            )
            nonempty = count > 0
            written = ordinal[gate]
            clock = {
                "writes": int(gate.sum()),
                "emptyhistory_rows": int((~nonempty).sum()),
                "empty_entire_sequence": not bool(gate.any()),
                "complete128write_cycles": int(gate.sum()) // 128,
                "queries": exposure(ordinal, f"g{group}_{label}_query", grams, table),
                "nonempty_queries": exposure(
                    ordinal[nonempty], f"g{group}_{label}_nonempty_query", grams, table
                ),
                "writes_exposure": exposure(written, f"g{group}_{label}_write", grams, table),
            }
            clock["support_pass"] = (
                clock["nonempty_queries"]["rank"] == 96
                and clock["writes_exposure"]["rank"] == 96
                and clock["writes"] >= 128
            )
            rec["clocks"][label] = clock
            for typ, ord_ in (
                ("query", ordinal),
                ("nonempty_query", ordinal[nonempty]),
                ("write", written),
            ):
                key = f"{label}_{typ}"
                aggregate_hist.setdefault(key, np.zeros(128, dtype=np.int64))
                aggregate_hist[key] += np.bincount(ord_ % 128, minlength=128)
            for c in (-1, 0, 1, 2, 3):
                key = str(c)
                mask = mapped == c
                rec["class_diagnostics"].setdefault(key, {})[label] = {
                    "rows": int(mask.sum()),
                    "writes": int((mask & gate).sum()),
                    "emptyhistory_rows": int((mask & ~nonempty).sum()),
                    "query_mod128_counts": np.bincount(ordinal[mask] % 128, minlength=128).tolist(),
                    "write_mod128_counts": np.bincount(
                        ordinal[mask & gate] % 128, minlength=128
                    ).tolist(),
                }
        for c in (-1, 0, 1, 2, 3):
            mask = mapped == c
            tot = class_totals[str(c)]
            tot["rows"] += int(mask.sum())
            tot["endpoint_writes"] += int((mask & endpoint).sum())
            tot["code_writes"] += int((mask & code).sum())
        rec["support_pass"] = all(clock["support_pass"] for clock in rec["clocks"].values())
        records.append(rec)
        print(
            json.dumps(
                {
                    k: rec[k]
                    for k in (
                        "row_group",
                        "component",
                        "endpoint_writes",
                        "code_writes",
                        "support_pass",
                        "mapped_class_counts",
                    )
                }
            ),
            flush=True,
        )
    aggregate = {}
    for name, hist in aggregate_hist.items():
        gram = table.T @ (hist[:, None] * table)
        grams["aggregate_" + name] = gram
        eig = np.linalg.eigvalsh(gram)
        aggregate[name] = {
            "hist128": hist.tolist(),
            "rank": int((eig > max(1e-8, eig[-1] * 1e-8)).sum()),
            "eigenvalue_min": float(eig[0]),
            "eigenvalue_max": float(eig[-1]),
            "gram_key": "aggregate_" + name,
        }
    np.savez_compressed(OWN / "grams.npz", **grams)
    result = {
        "selection_sha256": digest(OWN / "selection.json"),
        "preregistration_sha256": digest(OWN / "preregistration.json"),
        "authorization_sha256": digest(OWN / "launch_authorization.json"),
        "frozen_sources_sha256": digest(OWN / "frozen_sources.json"),
        "grams_sha256": digest(OWN / "grams.npz"),
        "components": records,
        "aggregate": aggregate,
        "class_totals": class_totals,
        "unknown_class_label": -1,
        "unknown_rows_retained": True,
        "support_pass": all(r["support_pass"] for r in records),
        "duration_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "columns_read": PRICE + ["a6"],
        "targets_masks_or_VALID_read": False,
    }
    (OWN / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("components", "aggregate")}),
        flush=True,
    )
    # A failed scientific supportgate is a valid completed measurement, not runtime failure.


if __name__ == "__main__":
    main()
