"""Independent feature-only D1 exposure replay using Walsh XOR algebra."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[2]
AUTHOR = AUDIT.parent / "event_content/exposure"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.monotonic()
    properties = subprocess.check_output(
        ["systemctl", "--user", "show", os.environ["WNN_SCOPE_UNIT"] + ".scope",
         "-p", "MemoryMax", "-p", "MemorySwapMax", "-p", "CPUQuotaPerSecUSec", "-p", "Slice"], text=True
    )
    limits = dict(line.split("=", 1) for line in properties.splitlines())
    assert limits == {"MemoryMax": "2147483648", "MemorySwapMax": "0",
                      "CPUQuotaPerSecUSec": "1s", "Slice": "wnn-campaign.slice"}
    selection = json.loads((AUTHOR / "selection.json").read_text())
    reported = json.loads((AUTHOR / "results.json").read_text())
    fold_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    prior_path = AUDIT.parent / "data_reconstruction/selection.json"
    table_path = AUDIT.parent / "data_reconstruction/interior_code_table.json"
    assert sha(fold_path) == selection["fold_manifest_sha256"]
    assert sha(prior_path) == selection["prior_selection_sha256"]
    assert sha(table_path) == selection["class_table_sha256"]
    for key, name in [("selection_sha256", "selection.json"),
                      ("preregistration_sha256", "preregistration.json"),
                      ("authorization_sha256", "launch_authorization.json"),
                      ("frozen_sources_sha256", "frozen_sources.json"),
                      ("grams_sha256", "grams.npz")]:
        assert sha(AUTHOR / name) == reported[key]
    authorization = json.loads((AUTHOR / "launch_authorization.json").read_text())
    assert authorization["approved"] is True
    for key in ["critic_D1_grade", "builder_integration_CPU_proof"]:
        gate = authorization[key]
        assert gate["passed"] is True and sha(ROOT / gate["path"]) == gate["sha256"]
    for name, expected in json.loads((AUTHOR / "frozen_sources.json").read_text()).items():
        assert sha(AUTHOR / name) == expected
    fold = json.loads(fold_path.read_text())
    prior = json.loads(prior_path.read_text())
    components = {r["row_group"]: r["component"] for r in fold["sequences"]["train"]}
    excluded = {components[g] for g in prior["fit_train_groups"] + prior["test_train_groups"]}
    assert sorted(excluded) == selection["excluded_prior_canonical_components"]
    eligible = [g for g in fold["fit"]["train"] if g != 528 and components[g] not in excluded]
    assert len(eligible) == selection["eligible_group_count"]
    ordered = sorted((hashlib.sha256(f"d1-exposure-v1:{g}:{components[g]}".encode()).hexdigest(), g)
                     for g in eligible)
    chosen, seen = [], set()
    for digest, group in ordered:
        if components[group] not in seen:
            chosen.append({"row_group": group, "component": components[group], "selection_hash": digest})
            seen.add(components[group])
        if len(chosen) == 16:
            break
    assert chosen == selection["selected"]
    data = ROOT / "datasets/train.parquet"
    identity = selection["source_train_identity"]
    assert data.stat().st_size == identity["bytes"] and data.stat().st_mtime_ns == identity["mtime_ns"]
    with data.open("rb") as stream:
        stream.seek(-8, 2)
        tail = stream.read(8)
        assert tail[4:] == b"PAR1"
        footer_size = int.from_bytes(tail[:4], "little")
        assert footer_size == identity["footer_bytes"]
        stream.seek(-8 - footer_size, 2)
        # Frozen split identity includes the eight-byte Parquet trailer.
        assert hashlib.sha256(stream.read(footer_size + 8)).hexdigest() == identity["footer_sha256"]
    class_map = {bytes.fromhex(v["code_float32_round4_hex"]): v["class"]
                 for v in json.loads(table_path.read_text())}
    # H_i H_j = H_(i XOR j): derive every Gram from a 128-point transform.
    walsh = np.array([[(-1) ** ((n & j).bit_count()) for j in range(128)] for n in range(128)], dtype=np.int64)
    xor = np.bitwise_xor(np.arange(96)[:, None], np.arange(96)[None, :])
    stored_grams = np.load(AUTHOR / "grams.npz", allow_pickle=False)
    aggregate: dict[str, np.ndarray] = {}
    totals = {str(c): {"rows": 0, "endpoint_writes": 0, "code_writes": 0} for c in (-1, 0, 1, 2, 3)}
    checked_grams = 0

    def verify_exposure(ordinal: np.ndarray, expected: dict) -> None:
        nonlocal checked_grams
        hist = np.bincount(ordinal % 128, minlength=128)
        assert hist.tolist() == expected["hist128"]
        transformed = walsh.T @ hist
        gram = transformed[xor].astype(np.float64)
        assert np.array_equal(gram, stored_grams[expected["gram_key"]])
        eig = np.linalg.eigvalsh(gram)
        threshold = max(1e-8, eig[-1] * 1e-8)
        assert int(np.count_nonzero(eig > threshold)) == expected["rank"] == 96
        np.testing.assert_allclose([eig[0], eig[-1]], [expected["eigenvalue_min"], expected["eigenvalue_max"]], atol=1e-7, rtol=1e-12)
        if "rows" in expected:
            unique = np.unique(ordinal)
            distinct = np.count_nonzero(hist)
            uniquehist = np.bincount(unique % 128, minlength=128)
            assert expected["rows"] == len(ordinal)
            assert [expected["ordinal_min"], expected["ordinal_max"]] == [int(ordinal.min()), int(ordinal.max())]
            assert expected["distinct_mod128"] == distinct
            assert expected["distinct_ordinals"] == len(unique)
            assert expected["aliased_distinct_ordinals"] == len(unique) - distinct
            assert expected["alias_distinctordinal_multiplicity128"] == uniquehist.tolist()
            assert expected["maximum_alias_distinctordinal_multiplicity"] == int(uniquehist.max())
            assert expected["maximum_repeatedquery_or_write_count_permod"] == int(hist.max())
            assert expected["repeated_exposures_beyond_modclasses"] == len(ordinal) - distinct
            variance = 1 - (transformed[:96] / len(ordinal)) ** 2
            np.testing.assert_allclose([variance[1:].min(), variance[1:].max()], [expected["nonintercept_variance_min"], expected["nonintercept_variance_max"]], atol=1e-14, rtol=1e-14)
            assert expected["column0_uncentered_norm"] == len(ordinal)
        checked_grams += 1

    source = pq.ParquetFile(data)
    prices = [f"i0_p{i}" for i in range(22)]
    for node, rec in zip(chosen, reported["components"], strict=True):
        group = node["row_group"]
        assert rec["row_group"] == group and rec["component"] == node["component"]
        raw = source.read_row_group(group, columns=prices + ["a6"], use_threads=False)
        x = np.asarray(np.column_stack([raw.column(p).to_numpy() for p in prices]), dtype=np.float32)
        endpoint = raw.column("a6").to_numpy().astype(np.float32) == np.float32(-5.199337482452393)
        code = np.zeros(len(x), dtype=bool)
        code[1:] = np.count_nonzero(x[1:] != x[:-1], axis=1) > 0
        packed = np.ascontiguousarray(np.round(x, 4), dtype="<f4").view("V88").reshape(-1)
        classes = np.fromiter((class_map.get(row.tobytes(), -1) for row in packed), dtype=np.int8, count=len(x))
        assert rec["rows"] == len(x) == 20000
        assert rec["endpoint_writes"] == int(endpoint.sum()) and rec["code_writes"] == int(code.sum())
        assert rec["endpoint_origin"] == bool(endpoint[0]) and rec["code_origin_suppressed"] is True
        assert rec["nonorigin_endpoint_without_rawcode"] == int((endpoint[1:] & ~code[1:]).sum())
        assert rec["nonorigin_rawcode_without_endpoint"] == int((code[1:] & ~endpoint[1:]).sum())
        for label, gate in [("row", np.ones(len(x), dtype=bool)), ("endpoint", endpoint), ("code", code)]:
            count = np.add.accumulate(gate, dtype=np.int64)
            ordinal = np.arange(len(x)) if label == "row" else np.clip(count - 1, 0, None)
            nonempty = count != 0
            clock = rec["clocks"][label]
            assert clock["writes"] == int(gate.sum()) >= 128
            assert clock["emptyhistory_rows"] == int((~nonempty).sum())
            assert clock["empty_entire_sequence"] is False
            assert clock["complete128write_cycles"] == int(gate.sum()) // 128
            assert clock["support_pass"] is True
            for name, selected, result_key in [("query", ordinal, "queries"), ("nonempty_query", ordinal[nonempty], "nonempty_queries"), ("write", ordinal[gate], "writes_exposure")]:
                verify_exposure(selected, clock[result_key])
                key = f"{label}_{name}"
                aggregate[key] = aggregate.get(key, np.zeros(128, dtype=np.int64)) + np.bincount(selected % 128, minlength=128)
            for c in (-1, 0, 1, 2, 3):
                mask = classes == c
                expected = rec["class_diagnostics"][str(c)][label]
                assert expected == {"rows": int(mask.sum()), "writes": int((mask & gate).sum()),
                                    "emptyhistory_rows": int((mask & ~nonempty).sum()),
                                    "query_mod128_counts": np.bincount(ordinal[mask] % 128, minlength=128).tolist(),
                                    "write_mod128_counts": np.bincount(ordinal[mask & gate] % 128, minlength=128).tolist()}
        for c in (-1, 0, 1, 2, 3):
            mask = classes == c
            assert rec["mapped_class_counts"][str(c)] == int(mask.sum())
            totals[str(c)]["rows"] += int(mask.sum())
            totals[str(c)]["endpoint_writes"] += int((mask & endpoint).sum())
            totals[str(c)]["code_writes"] += int((mask & code).sum())
        assert rec["support_pass"] is True
    for name, hist in aggregate.items():
        # Aggregate ranks and Grams use the exact histogram without materializing rows.
        verify_exposure(np.repeat(np.arange(128), hist), reported["aggregate"][name])
    assert totals == reported["class_totals"]
    assert reported["support_pass"] is True and reported["unknown_rows_retained"] is True
    result = {"passed": True, "script_sha256": sha(Path(__file__)),
              "author_results_sha256": sha(AUTHOR / "results.json"),
              "selection_sha256": sha(AUTHOR / "selection.json"),
              "raw_replayed_components": 16, "raw_replayed_rows": 320000,
              "verified_exact_grams": checked_grams, "class_totals": totals,
              "endpoint_origins": sum(r["endpoint_origin"] for r in reported["components"]),
              "nonorigin_endpoint_without_code": sum(r["nonorigin_endpoint_without_rawcode"] for r in reported["components"]),
              "maximum_empty_history_rows": max(r["clocks"][clock]["emptyhistory_rows"] for r in reported["components"] for clock in ("endpoint", "code")),
              "method": "independent raw projection/counters; Gram from Walsh XOR product identity",
              "columns_read": prices + ["a6"], "targets_masks_or_VALID_read": False,
              "scope_properties": limits, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "duration_seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "limits": "Rank/exposure only; period128 aliases remain; code class table diagnostic only; C/R changes write and decay."}
    with (AUDIT / "event_content_exposure_independent_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
