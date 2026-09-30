"""Independent raw-packet and denominator tests; no empirical input/result is read."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def literal_packets(x):
    rows = [row.tobytes() for row in x]
    packets = []
    for row in rows:
        if packets and packets[-1][0] == row:
            packets[-1][1] += 1
        else:
            packets.append([row, 1])
    return packets


def literal_surrogate(x, seed, chain):
    packets = literal_packets(x)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"DETECTOR02:run128:{seed}:{chain}".encode()).digest()[:8], "little"))
    blocks = []
    for start in range(0, len(packets), 128):
        block = packets[start:start+128]
        order = rng.permutation(len(block))
        rows = [block[int(i)][0] for i in order for _ in range(block[int(i)][1])]
        expected = [p[0] for p in block for _ in range(p[1])]
        assert Counter(rows) == Counter(expected)
        blocks.extend(rows)
    return np.frombuffer(b"".join(blocks), dtype=np.float32).reshape(-1, 4).copy()


def literal_detector(x):
    out = []
    previous = None
    for i, row in enumerate(x):
        values = [float(v) for v in row]
        if previous is not None:
            good = all(math.isfinite(v) and abs(v) < 5.19 for v in previous+values)
            if good and any(a != b for a, b in zip(previous, values)) and max(values)-min(values) > 1e-6:
                out.append(i)
        previous = values
    return np.array(out, dtype=np.int64)


def main():
    start = time.monotonic()
    module = load("independent_review_detector", HERE/"detector_control.py")
    legacy = load("independent_review_old_detector", HERE.parent/"2026-10-01-eda/quant_mark_eda.py")
    atoms = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0],
                      [-0., 0, 0, 0], [np.nan, 0, 0, 0], [np.nan, 0, 0, 0],
                      [np.inf, 0, 0, 0]], dtype=np.float32)
    atoms.view(np.uint32)[4, 0] = 0x7fc00001
    atoms.view(np.uint32)[5, 0] = 0x7fc00002
    cases = [np.empty((0, 4), np.float32), np.repeat(atoms[1:2], 7, axis=0),
             atoms[[0, 1, 0, 1, 4, 2, 1]].copy(),
             np.concatenate([np.repeat(atoms[i % len(atoms):i % len(atoms)+1], 1+i % 5, axis=0) for i in range(259)]),
             atoms[np.tile([1, 2], 150)].copy()]
    observed_merger = False
    for raw in cases:
        v, n = module.packetize(raw)
        assert np.repeat(v, n, axis=0).tobytes() == raw.tobytes()
        for seed in module.SEEDS:
            actual, info = module.permute_raw(raw, seed, 1907)
            expected = literal_surrogate(raw, seed, 1907)
            assert actual.tobytes() == expected.tobytes()
            assert Counter(row.tobytes() for row in actual) == Counter(row.tobytes() for row in raw)
            assert info["merged_packet_boundaries"] == len(literal_packets(raw))-len(literal_packets(actual))
            observed_merger |= info["merged_packet_boundaries"] > 0
            expected_ids = literal_detector(actual)
            with np.errstate(invalid="ignore"):
                np.testing.assert_array_equal(legacy.detect(actual)[0], expected_ids)
                for cut in (0, 1, 2, 99, len(actual)//2, len(actual)):
                    cut = min(cut, len(actual))
                    np.testing.assert_array_equal(legacy.detect(actual[:cut])[0], expected_ids[expected_ids < cut])
    assert observed_merger
    # Hand-derived legal same-category emissions: distinct raw states and hidden neutral bridge.
    for raw, ids in ((atoms[[0, 1, 2]], [1, 2]), (atoms[[0, 1, 0, 1]], [1, 3]),
                     (atoms[[0, 1, 4, 1, 2]], [1, 4])):
        with np.errstate(invalid="ignore"):
            got = legacy.detect(raw)[0]
        np.testing.assert_array_equal(got, ids)
        np.testing.assert_array_equal(legacy.encode(raw[got], np.zeros(4)), np.ones(len(ids), int))
    chains = []
    for i in range(16):
        ns = [i+1, 3*i+2, 2*i+5, 33-i]
        if i == 15:
            ns[2] = 0
        gains = [.1+.01*i, .4-.003*i, -.1+.02*i, .23-.005*i]
        chains.append({"chain": i, "half": i//8, "class": (i % 8)//2,
            "conditions": {name: {"n": ns[j], "loss_sum_G_F_T": [0., (2+gains[j])*ns[j], 2*ns[j]]}
                           for j, name in enumerate(module.CONDITIONS)}})
    result = module.bootstrap(chains)
    assert result["paired_complete_chains"] == 15 and result["excluded"][0]["chain"] == 15
    macro, event = [], []
    for name in module.CONDITIONS:
        losses = [(c["conditions"][name]["loss_sum_G_F_T"][1]-c["conditions"][name]["loss_sum_G_F_T"][2], c["conditions"][name]["n"]) for c in chains[:15]]
        macro.append(sum(s/n for s, n in losses)/len(losses))
        event.append(sum(s for s, n in losses)/sum(n for s, n in losses))
    for weighting, expected in (("equal_chain", macro), ("event_weighted", event)):
        for name, point in zip(module.CONDITIONS, expected):
            np.testing.assert_allclose(result[weighting][name]["point"], point, atol=1e-14)
        np.testing.assert_allclose(result[weighting]["original_minus_mean_three"]["point"], expected[0]-sum(expected[1:])/3, atol=1e-14)
    empty = module.bootstrap([chains[-1]])
    assert empty["paired_complete_chains"] == 0 and empty["equal_chain"] is None
    receipt = {"status": "passed", "empirical_data_read": False, "tests": [
        "separately implemented literal byte-packet permutation agrees exactly for empty/singleton/partial/multiple blocks",
        "NaN payloads, signed zero, joint raw multiset and attached durations preserved",
        "allowed adjacent packet merges reflected in reconstructed maximal run count",
        "independent scalar detector and prefix parity, invalid bridge and legal repeated emitted categories",
        "separate condition-specific macro/event denominators and paired-complete exclusions"],
        "code_sha256": hashlib.sha256((HERE/"detector_control.py").read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds": time.monotonic()-start,
        "limits": "Synthetic correctness and source review; not empirical replication or physical simulator validation."}
    with (HERE/"independent-detector-check-result.json").open("x") as f:
        json.dump(receipt, f, indent=2); f.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
