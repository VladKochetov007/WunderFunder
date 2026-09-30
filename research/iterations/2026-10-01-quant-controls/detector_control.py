"""DETECTOR02: one frozen raw-run-packet control; no price targets or inference."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PLAN_SHA = "59814066828b301ee7098a90903b36aa69dad4dfb21ad239400c1f98336294bb"
SEEDS = (20261011, 20261012, 20261013)
CONDITIONS = ["original"] + [f"run128_{s}" for s in SEEDS]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def load_spec():
    assert sha(HERE / "detector-plan.json") == PLAN_SHA
    plan = json.loads((HERE / "detector-plan.json").read_text())
    names = (("authoritative_selection", "selection_sha256"),
             ("original_result", "original_result_sha256"),
             ("legacy_detector_module", "legacy_module_sha256"),
             ("original_plan", "original_plan_sha256"))
    for path, digest in names:
        assert sha(HERE / plan["inputs"][path]) == plan["inputs"][digest], path
    path = HERE / plan["inputs"]["legacy_detector_module"]
    spec = importlib.util.spec_from_file_location("detector02_legacy", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    return plan, legacy


def packetize(x):
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 4
    bits = np.ascontiguousarray(x).view(np.uint32)
    if not len(bits):
        return bits.copy(), np.empty(0, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.any(bits[1:] != bits[:-1], axis=1)) + 1]
    lengths = np.diff(np.r_[starts, len(bits)])
    vectors = bits[starts].copy()
    np.testing.assert_array_equal(np.repeat(vectors, lengths, axis=0), bits)
    return vectors, lengths


def bit_multiset(vectors, lengths):
    result = Counter()
    for vector, length in checked_zip(vectors, lengths):
        result[vector.tobytes()] += int(length)
    return result


def transition_multiset(vectors):
    return Counter(a.tobytes() + b.tobytes() for a, b in zip(vectors[:-1], vectors[1:]))


def quantiles(values):
    return np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist() if len(values) else None


def checked_zip(*arrays):
    """Strict zip also usable for local synthetic checks on Python 3.9."""
    assert len({len(a) for a in arrays}) <= 1
    return zip(*arrays)


def occupancy(x):
    valid = np.isfinite(x).all(1) & (np.abs(x) < 5.19).all(1)
    with np.errstate(invalid="ignore"):
        nonneutral = np.ptp(x, axis=1) > 1e-6
    return {"invalid": int((~valid).sum()), "valid_neutral": int((valid & ~nonneutral).sum()),
            "valid_nonneutral": int((valid & nonneutral).sum())}


def permute_raw(x, seed, chain):
    vectors, lengths = packetize(x)
    key = f"DETECTOR02:run128:{seed}:{chain}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    pieces, block_lengths = [], []
    for start in range(0, len(vectors), 128):
        stop = min(start + 128, len(vectors))
        order = rng.permutation(stop - start) + start
        np.testing.assert_array_equal(np.sort(order), np.arange(start, stop))
        piece = np.repeat(vectors[order], lengths[order], axis=0)
        # Check the actual reconstructed block, including any adjacent mergers.
        reconstructed_v, reconstructed_l = packetize(piece.view(np.float32))
        assert bit_multiset(vectors[start:stop], lengths[start:stop]) == bit_multiset(reconstructed_v, reconstructed_l)
        assert Counter((v.tobytes(), int(n)) for v, n in zip(vectors[order], lengths[order])) == Counter(
            (v.tobytes(), int(n)) for v, n in zip(vectors[start:stop], lengths[start:stop]))
        assert len(piece) == int(lengths[start:stop].sum())
        pieces.append(piece)
        block_lengths.append(len(piece))
    outbits = np.concatenate(pieces) if pieces else np.empty((0, 4), np.uint32)
    y = outbits.view(np.float32)
    assert y.shape == x.shape and occupancy(y) == occupancy(x)
    newvectors, _ = packetize(y)
    oldtr, newtr = transition_multiset(vectors), transition_multiset(newvectors)
    common = sum((oldtr & newtr).values())
    stats = {"original_packets": len(vectors), "reconstructed_packets": len(newvectors),
             "merged_packet_boundaries": len(vectors) - len(newvectors), "packet_blocks": len(block_lengths),
             "block_raw_row_lengths_min_q25_median_q75_q95_max": quantiles(block_lengths),
             "raw_rows_bitwise_changed": int(np.any(outbits != x.view(np.uint32), axis=1).sum()),
             "raw_rows": len(x), "boundary_transition_multiset_intersection": common,
             "original_boundary_transition_count": max(len(vectors) - 1, 0),
             "new_boundary_transition_count": max(len(newvectors) - 1, 0),
             "occupancy": occupancy(y), "all_block_multisets_and_attached_lengths_verified": True}
    return y, stats


def detect_record(legacy, x):
    with np.errstate(invalid="ignore"):
        ids, counts = legacy.detect(x)
    for cut in (1, 99, 5000, 10000, 19999):
        cut = min(cut, len(x))
        with np.errstate(invalid="ignore"):
            np.testing.assert_array_equal(legacy.detect(x[:cut])[0], ids[ids < cut])
    kept = ids[:4096]
    counts = dict(counts, retained_first_raw_row=int(kept[0]) if len(kept) else None,
                  retained_last_raw_row=int(kept[-1]) if len(kept) else None,
                  retained_covered_raw_prefix_rows=int(kept[-1]) + 1 if len(kept) else 0,
                  cap_retention_fraction=len(kept) / len(ids) if len(ids) else None,
                  capped=len(ids) > 4096)
    return x[kept].copy(), counts


def literal_detect(x):
    found = []
    for i in range(1, len(x)):
        a, b = x[i - 1], x[i]
        if (np.isfinite(a).all() and np.isfinite(b).all() and
                (np.abs(a) < 5.19).all() and (np.abs(b) < 5.19).all() and
                np.any(a != b) and np.ptp(b) > 1e-6):
            found.append(i)
    return np.array(found, dtype=np.int64)


def bootstrap(chains):
    complete = [c for c in chains if all(c["conditions"][k]["n"] > 0 for k in CONDITIONS)]
    excluded = [{"chain": c["chain"], "empty_conditions": [k for k in CONDITIONS if not c["conditions"][k]["n"]]}
                for c in chains if not all(c["conditions"][k]["n"] > 0 for k in CONDITIONS)]
    strata = [[i for i, c in enumerate(complete) if c["half"] == h and c["class"] == cl]
              for h in (0, 1) for cl in range(4)]
    result = {"paired_complete_chains": len(complete), "excluded": excluded,
              "strata_sizes_half_then_class": list(map(len, strata)), "draws": 1000, "seed": 20261031,
              "conditioning": "Fixed original centers, condition-matched priors, sample, and three surrogate draws; no nuisance-fit uncertainty"}
    if not complete:
        result.update(equal_chain=None, event_weighted=None)
        return result
    ns = np.array([[c["conditions"][k]["n"] for k in CONDITIONS] for c in complete])
    gs = np.array([[c["conditions"][k]["loss_sum_G_F_T"][1] - c["conditions"][k]["loss_sum_G_F_T"][2]
                    for k in CONDITIONS] for c in complete])
    def calc(ids):
        equal = (gs[ids] / ns[ids]).mean(0)
        weighted = gs[ids].sum(0) / ns[ids].sum(0)
        return np.array([np.r_[a, a[1:].mean(), a[0] - a[1:].mean()] for a in (equal, weighted)])
    rng = np.random.default_rng(20261031)
    draws = np.array([calc(np.concatenate([rng.choice(s, len(s), replace=True) for s in strata if len(s)]))
                      for _ in range(1000)])
    point = calc(np.arange(len(complete)))
    names = CONDITIONS + ["mean_three_surrogates", "original_minus_mean_three"]
    for j, weighting in enumerate(("equal_chain", "event_weighted")):
        result[weighting] = {k: {"point": float(point[j, i]), "ci95": np.quantile(draws[:, j, i], [.025, .975]).tolist(),
                                  "bootstrap_sd": float(draws[:, j, i].std())} for i, k in enumerate(names)}
    return result


def self_test(legacy):
    legacy.self_test()
    a = [1, 0, 0, 0]; b = [2, 0, 0, 0]; n = [0, 0, 0, 0]
    special = np.array([n, a, b, n, a, [np.nan, 0, 0, 0], b, a,
                        [np.inf, 0, 0, 0], a, a, [-0., 0, 0, 0], n], np.float32)
    # Make two NaN payloads explicit. Bit grouping must retain their distinction.
    special = np.vstack([special, special[5], special[5]])
    special.view(np.uint32)[-1, 0] = 0x7fc00002
    x = np.tile(special, (40, 1))
    for raw in (special, x, np.array([n], np.float32), np.tile(np.array([a], np.float32), (200, 1)),
                np.array([n, a], np.float32), np.array([a, n, a], np.float32)):
        v, lengths = packetize(raw)
        for seed in SEEDS:
            y, details = permute_raw(raw, seed, 42)
            vy, ly = packetize(y)
            assert bit_multiset(v, lengths) == bit_multiset(vy, ly)
            np.testing.assert_array_equal(y.view(np.uint32), permute_raw(raw, seed, 42)[0].view(np.uint32))
            with np.errstate(invalid="ignore"):
                ids = legacy.detect(y)[0]
                np.testing.assert_array_equal(ids, literal_detect(y))
                for cut in range(len(y) + 1):
                    np.testing.assert_array_equal(legacy.detect(y[:cut])[0], ids[ids < cut])
            assert details["merged_packet_boundaries"] >= 0
    # Raw-distinct events can encode identically. Neutral bridges also allow repeats.
    for raw in (np.array([n, a, b], np.float32), np.array([n, a, n, a], np.float32)):
        ids = legacy.detect(raw)[0]
        np.testing.assert_array_equal(legacy.encode(raw[ids], np.zeros(4)), [1, 1])
    assert len(legacy.detect(np.array([a, n, a], np.float32))[0]) == 1
    assert len(legacy.detect(np.array([n, n], np.float32))[0]) == 0
    pi, R = legacy.priors([])
    np.testing.assert_allclose(pi, np.full(16, 1 / 16))
    np.testing.assert_allclose(R, np.full((16, 16), 1 / 16))
    # Changed denominators: aggregate each condition with its OWN number of events.
    fake = []
    for i, counts in enumerate(([2, 4, 3, 7], [9, 2, 11, 4], [0, 2, 3, 4])):
        fake.append({"chain": i, "half": i % 2, "class": 0, "conditions": {
            k: {"n": counts[j], "loss_sum_G_F_T": [0, (i + j + 1) * counts[j], 0]}
            for j, k in enumerate(CONDITIONS)}})
    bs = bootstrap(fake)
    assert bs["paired_complete_chains"] == 2 and bs["excluded"][0]["chain"] == 2
    assert bs["event_weighted"]["original"]["point"] == 20 / 11
    assert bs["event_weighted"][CONDITIONS[1]]["point"] == 14 / 6
    assert bootstrap([fake[-1]])["equal_chain"] is None
    return {"status": "PASS", "scope": "synthetic only; no source feature reads",
            "checks": ["raw-bit reconstruction/multisets/NaN payload/signed zero", "final partial and singleton blocks",
                       "literal/batch/prefix detector parity", "raw-distinct and neutral-bridge same-category events",
                       "no/one events and uniform empty-fit fallback", "legacy prequential parity/current-future independence",
                       "condition-specific denominators and paired-empty exclusions"]}


def source_identity(source):
    st = source.stat()
    h = hashlib.sha256()
    with source.open("rb") as f:
        f.seek(-8, 2); tail = f.read(8)
        assert tail[4:] == b"PAR1"
        length = struct.unpack("<I", tail[:4])[0]
        f.seek(-8 - length, 2)
        remaining = length + 8
        while remaining:
            data = f.read(min(2**20, remaining)); assert data
            h.update(data); remaining -= len(data)
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino, "device": st.st_dev,
            "footer_bytes": length + 8, "footer_sha256": h.hexdigest()}


def assert_old_stats(current, previous):
    assert current.keys() == previous.keys()
    for k in current:
        if previous[k] is None:
            assert current[k] is None
        else:
            np.testing.assert_allclose(current[k], previous[k], rtol=1e-12, atol=1e-12, err_msg=k)


def main(source_root, out, plan, legacy):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("detector-inputs.json", "detector-used-inputs.json", "detector-result.json"):
        assert not (out / name).exists(), name
    old_input = json.loads((HERE / plan["inputs"]["authoritative_selection"]).read_text())
    old_result = json.loads((HERE / plan["inputs"]["original_result"]).read_text())
    work_root = HERE.parents[2]
    files = []
    for old in old_input["files"]:
        if "/cache/" in old["path"]:
            path = source_root / "cache" / Path(old["path"]).name
        else:
            path = work_root / "research" / old["path"].split("/research/", 1)[1]
        assert sha(path) == old["sha256"], str(path)
        files.append({"path": str(path), "sha256": old["sha256"]})
    source = source_root / "datasets/train.parquet"
    identity = source_identity(source)
    assert all(identity[k] == old_input["source"][k] for k in identity)
    selected = old_input["selected_before_feature_reads"]
    assert len(selected) == len({s["chain"] for s in selected}) == 256
    cf = np.load(source_root / "cache/chain_folds.npz")
    cls = np.load(work_root / "research/artifacts/market-classes-v1.npz")["train"]
    forbidden = set(map(int, cf["chain"][cf["split"] == "valid"]))
    forbidden.update(map(int, cf["chain"][(cf["split"] == "train") & (cf["row_group"] == 528)]))
    prior = json.loads((work_root / "research/iterations/2026-10-01-deep/quant-inputs.json").read_text())
    forbidden.update(s["chain"] for s in prior["selected_before_feature_reads"])
    old_chains = {c["chain"]: c for c in old_result["chains"]}
    for s in selected:
        assert s["chain"] not in forbidden and s["row_group"] != 528
        at = (cf["split"] == "train") & (cf["row_group"] == s["row_group"])
        assert int(at.sum()) == 1 and int(cf["chain"][at][0]) == s["chain"]
        assert int(cls[s["row_group"]]) == s["class"]
        assert all(s[k] == old_chains[s["chain"]][k] for k in s)
    frozen_hashes = {"plan": sha(HERE / "detector-plan.json"), "script": sha(Path(__file__)),
                     "contract": sha(HERE / "contract.json"),
                     **{k: plan["inputs"][k] for k in plan["inputs"] if k.endswith("sha256")}}
    save(out / "detector-inputs.json", {"id": "DETECTOR02", "utc": datetime.now(timezone.utc).isoformat(),
         "selected_before_feature_reads": selected, "source": identity, "files": files, "hashes": frozen_hashes,
         "columns": plan["inputs"]["columns"], "targets_read": False})
    centers = {h: np.array(old_result["fitted"][str(h)]["centers"]) for h in (0, 1)}
    pf = pq.ParquetFile(source)
    records, raw_hashes = [], []
    for i, s in enumerate(selected):
        table = pf.read_row_group(s["row_group"], columns=plan["inputs"]["columns"], use_threads=False)
        x = np.column_stack([table.column(c).to_numpy() for c in plan["inputs"]["columns"]]).astype(np.float32)
        assert x.shape == (20000, 4)
        raw_hashes.append(dict(s, raw_four_float32_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        values, detector = detect_record(legacy, x)
        assert all(detector[k] == v for k, v in old_chains[s["chain"]]["detector"].items())
        vectors, lengths = packetize(x)
        original_packets = {"packets": len(vectors), "occupancy": occupancy(x),
             "block_raw_row_lengths_min_q25_median_q75_q95_max": quantiles([int(lengths[j:j+128].sum()) for j in range(0, len(lengths), 128)])}
        record = dict(s, values={"original": values}, detectors={"original": detector},
                      surrogate={}, original_packets=original_packets)
        for seed, condition in checked_zip(SEEDS, CONDITIONS[1:]):
            y, stats = permute_raw(x, seed, s["chain"])
            value, count = detect_record(legacy, y)
            record["values"][condition] = value
            record["detectors"][condition] = count
            record["surrogate"][condition] = stats
        records.append(record)
        if (i + 1) % 32 == 0:
            print(json.dumps({"stage": "read_and_redetect", "chains": i + 1, "elapsed_seconds": time.monotonic() - started}), flush=True)
    fitted = {}
    for condition in CONDITIONS:
        fitted[condition] = {}
        for half in (0, 1):
            marks = [legacy.encode(r["values"][condition], centers[half]) for r in records if r["half"] == half]
            pi, R = legacy.priors(marks)
            fitted[condition][half] = (pi, R)
            if condition == "original":
                np.testing.assert_allclose(pi, old_result["fitted"][str(half)]["pi"], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(R, old_result["fitted"][str(half)]["R"], rtol=1e-12, atol=1e-12)
    result = {"id": "DETECTOR02", "conditions": CONDITIONS, "chains": [], "fitted": {},
              "old_readonly_baseline": old_result["summary"]["original"], "targets_read": False, "WP_queries": 0,
              "interpretation": plan["question"], "checks": {"old_source_identity": True, "metadata": True,
                "whole_chain_exclusion": True, "raw_bit_preservation": True, "raw_prefix": True}}
    for condition in CONDITIONS:
        result["fitted"][condition] = {str(h): {"centers": centers[h].tolist(), "pi": fitted[condition][h][0].tolist(),
            "R": fitted[condition][h][1].tolist(), "fit_nonempty_chains": sum(bool(len(r["values"][condition])) for r in records if r["half"] == h)}
            for h in (0, 1)}
    for r in records:
        half = 1 - r["half"]
        entry = {k: v for k, v in r.items() if k != "values"}
        entry.update(fit_half=half, conditions={}, encoded_mark_sha256={})
        for condition in CONDITIONS:
            m = legacy.encode(r["values"][condition], centers[half])
            pi, R = fitted[condition][half]
            entry["conditions"][condition] = legacy.mark_stats(m, pi, R)
            entry["encoded_mark_sha256"][condition] = hashlib.sha256(m.tobytes()).hexdigest()
            if condition == "original":
                old = old_chains[r["chain"]]
                assert entry["encoded_mark_sha256"][condition] == old["encoded_mark_sha256"]
                assert_old_stats(entry["conditions"][condition], old["conditions"]["original"])
        result["chains"].append(entry)
    result["checks"]["original_encoded_marks_priors_and_per_chain_losses_reproduced"] = True
    result["summary"] = {k: legacy.summary(result["chains"], k) for k in CONDITIONS}
    np.testing.assert_allclose(result["summary"]["original"]["mean_per_chain_logloss_G_F_T"],
                               old_result["summary"]["original"]["mean_per_chain_logloss_G_F_T"], rtol=1e-12, atol=1e-12)
    result["by_half"] = {str(h): {k: legacy.summary([c for c in result["chains"] if c["half"] == h], k) for k in CONDITIONS} for h in (0, 1)}
    result["by_class"] = {str(cl): {k: legacy.summary([c for c in result["chains"] if c["class"] == cl], k) for k in CONDITIONS} for cl in range(4)}
    result["bootstrap"] = bootstrap(result["chains"])
    assert source_identity(source) == identity
    for f in files:
        assert sha(f["path"]) == f["sha256"]
    load_spec()
    assert sha(Path(__file__)) == frozen_hashes["script"]
    assert sha(HERE / "contract.json") == frozen_hashes["contract"]
    result["checks"]["source_and_inputs_unchanged_after"] = True
    result["status"] = "complete"
    result["elapsed_seconds"] = time.monotonic() - started
    save(out / "detector-used-inputs.json", {"raw_columns": plan["inputs"]["columns"], "chains": raw_hashes,
         "source_after": identity, "utc": datetime.now(timezone.utc).isoformat()})
    save(out / "detector-result.json", result)
    print(json.dumps({"id": result["id"], "elapsed_seconds": result["elapsed_seconds"],
                      "summary": result["summary"], "bootstrap": result["bootstrap"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/home/grinchenkoeg/WunderFunder"))
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    plan, legacy = load_spec()
    if args.self_test:
        print(json.dumps(self_test(legacy), indent=2))
    else:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("870-second internal budget")))
        signal.alarm(870)
        try:
            main(args.source_root.resolve(), args.out_dir.resolve(), plan, legacy)
        except BaseException as exc:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            save(args.out_dir / f"detector-failure-{stamp}.json", {
                "status": "failed", "exception": type(exc).__name__, "message": str(exc),
                "utc": stamp, "script_sha256": sha(Path(__file__)), "plan_sha256": PLAN_SHA,
                "interpretation": "Failure is not an empirical negative; no alternate recipe substituted."})
            raise
