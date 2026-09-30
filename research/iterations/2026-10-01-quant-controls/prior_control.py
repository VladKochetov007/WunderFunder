"""PRIOR02: one frozen matched-prior control; four train codes only, no targets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import subprocess
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-10-01-eda"
spec = importlib.util.spec_from_file_location("prior_qmark_eda", OLD / "quant_mark_eda.py")
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)
SEEDS = eda.SEEDS
CUTS = ("all", "first16", "after16")
ORIGINAL_MODELS = ("G", "F", "T_original", "T_independent", "static_R")
BLOCK_MODELS = ("F", "T_old_R", "T_matched_R")


def fingerprint(path):
    s = Path(path).stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino, "device": s.st_dev}


def footer_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        f.seek(-8, 2); tail = f.read(8)
        if tail[4:] != b"PAR1":
            raise ValueError("not a Parquet footer")
        length = struct.unpack("<I", tail[:4])[0]
        f.seek(-8-length, 2); remaining = length+8
        while remaining:
            data = f.read(min(2**20, remaining))
            if not data:
                raise ValueError("truncated footer")
            h.update(data); remaining -= len(data)
    return {"footer_bytes": length+8, "footer_sha256": h.hexdigest()}


def original_probabilities(m, pi, R):
    base = eda.probability_of_observed_mark(m, pi, R)
    independent = eda.probability_of_observed_mark(m, pi, np.tile(pi, (16, 1)))[:, 2]
    static = pi[m].copy()
    if len(m) > 1:
        static[1:] = R[m[:-1], m[1:]]
    return np.column_stack((base, independent, static))


def block_probabilities(m, pi, old_R, matched_R):
    old = eda.probability_of_observed_mark(m, pi, old_R)
    matched = eda.probability_of_observed_mark(m, pi, matched_R)
    return np.column_stack((old[:, 1:], matched[:, 2]))


def loss_record(probabilities):
    if not (np.isfinite(probabilities).all() and (probabilities > 0).all() and
            (probabilities <= 1+1e-12).all()):
        raise ValueError("invalid probability")
    loss = -np.log(probabilities)
    n = len(loss)
    return {"counts": [n, min(16, n), max(n-16, 0)],
            "loss_sums": np.stack((loss.sum(0), loss[:16].sum(0), loss[16:].sum(0))).tolist()}


def old_losses(record):
    return np.array([record[k] for k in ("loss_sum_G_F_T", "first16_loss_sum_G_F_T", "after16_loss_sum_G_F_T")])


def require_close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-12)


def contrasts(chains):
    original = np.array([c["conditions"]["original"]["loss_sums"] for c in chains])
    values = [original[:, :, 1]-original[:, :, 2], original[:, :, 3]-original[:, :, 2],
              original[:, :, 4]-original[:, :, 2]]
    names = ["original_F_minus_T_original", "original_T_independent_minus_T_original", "original_static_R_minus_T_original"]
    differences, gains = [], []
    for seed in SEEDS:
        block = np.array([c["conditions"][f"block128_{seed}"]["loss_sums"] for c in chains])
        differences.append(block[:, :, 1]-block[:, :, 2]); gains.append(block[:, :, 0]-block[:, :, 2])
        names.extend((f"block128_{seed}_old_minus_matched", f"block128_{seed}_F_minus_matched"))
        values.extend((differences[-1], gains[-1]))
    names.extend(("block128_mean_old_minus_matched", "block128_mean_F_minus_matched"))
    values.extend((np.mean(differences, axis=0), np.mean(gains, axis=0)))
    return names, np.stack(values, axis=-1)


def summarize(chains):
    names, values = contrasts(chains)  # chain, cut, contrast
    n = np.array([c["conditions"]["original"]["counts"] for c in chains])
    rng = np.random.default_rng(20261031)
    weights = np.zeros((1000, len(chains)), dtype=np.int64)
    strata = [np.array([i for i, c in enumerate(chains) if c["half"] == h and c["class"] == k])
              for h in (0, 1) for k in range(4)]
    if any(len(ix) != 32 for ix in strata):
        raise ValueError("fixed 8x32 stratum design mismatch")
    for b in range(len(weights)):
        ids = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])
        weights[b] = np.bincount(ids, minlength=len(chains))
    scopes = {"pooled": np.ones(len(chains), dtype=bool)}
    scopes.update({f"half_{h}": np.array([c["half"] == h for c in chains]) for h in (0, 1)})
    scopes.update({f"class_{k}": np.array([c["class"] == k for c in chains]) for k in range(4)})
    output = {}
    for scope, pick in scopes.items():
        report = {"chains_selected": int(pick.sum()), "slices": {}}
        for ci, cut in enumerate(CUTS):
            ix = np.flatnonzero(pick & (n[:, ci] > 0))
            if not len(ix):
                report["slices"][cut] = {"chains_with_events": 0}; continue
            count = n[ix, ci]
            w = weights[:, ix]
            if np.any(w.sum(1) == 0):
                raise ValueError("empty bootstrap scope; no silent omission")
            v = values[ix, ci]
            point_macro = (v/count[:, None]).mean(0)
            point_event = v.sum(0)/count.sum()
            boot_macro = w @ (v/count[:, None]) / w.sum(1)[:, None]
            boot_event = w @ v / (w @ count)[:, None]
            entry = {"chains_with_events": len(ix), "chains_without_events": int(pick.sum()-len(ix)),
                     "events": int(count.sum()), "event_count_min_median_max": [int(count.min()), float(np.median(count)), int(count.max())],
                     "model_losses": {}, "contrasts": {}}
            for condition in chains[0]["conditions"]:
                models = ORIGINAL_MODELS if condition == "original" else BLOCK_MODELS
                sums = np.array([chains[i]["conditions"][condition]["loss_sums"][ci] for i in ix])
                entry["model_losses"][condition] = {model: {"macro_chain": float((sums[:, j]/count).mean()),
                                                           "event_weighted": float(sums[:, j].sum()/count.sum())}
                                                         for j, model in enumerate(models)}
            for j, name in enumerate(names):
                entry["contrasts"][name] = {}
                for label, point, draws in (("macro_chain", point_macro, boot_macro), ("event_weighted", point_event, boot_event)):
                    entry["contrasts"][name][label] = {"point": float(point[j]),
                        "ci95": np.quantile(draws[:, j], [.025, .975]).tolist(), "bootstrap_sd": float(draws[:, j].std())}
            report["slices"][cut] = entry
        output[scope] = report
    return output


def self_test():
    eda.self_test()
    m = np.array([0, 0, 1, 15, 1, 1, 2, 0, 4, 3, 3, 3, 15, 7, 0, 0, 1, 1, 14])
    pi, R = eda.priors([m[:10], m[10:]])
    online = eda.online_probabilities(m, pi, R)
    ind = eda.online_probabilities(m, pi, np.tile(pi, (16, 1)))[:, 2]
    static = np.concatenate((pi[None, :], R[m[:-1]]), axis=0)
    full = np.concatenate((online, ind[:, None, :], static[:, None, :]), axis=1)
    np.testing.assert_allclose(full.sum(2), 1)
    assert (full > 0).all()
    require_close(full[np.arange(len(m)), :, m], original_probabilities(m, pi, R))
    other = m.copy(); other[5:] = 9
    other_static = np.concatenate((pi[None, :], R[other[:-1]]), axis=0)
    require_close(other_static[:6], static[:6])
    for prior in (R, np.tile(pi, (16, 1))):
        require_close(eda.online_probabilities(other, pi, prior)[:6], eda.online_probabilities(m, pi, prior)[:6])
    require_close(full[0], np.tile(pi, (5, 1)))
    for cut in (1, 16):
        require_close(original_probabilities(m[:cut], pi, R), original_probabilities(m, pi, R)[:cut])
    b = block_probabilities(m, pi, R, R)
    require_close(b[:, 1], b[:, 2])
    require_close(b[:, :2], online[np.arange(len(m)), :, m][:, 1:])
    sequences = [np.tile(m, 20), np.roll(np.tile(m, 18), 9)]
    original_pi, _ = eda.priors(sequences)
    for seed in SEEDS:
        shuffled = [eda.permute(x, "block128", seed, ch) for ch, x in enumerate(sequences)]
        shifted_pi, _ = eda.priors(shuffled)
        require_close(shifted_pi, original_pi)
        for a, z in zip(sequences, shuffled):
            require_close(loss_record(eda.probability_of_observed_mark(a, pi, R))["loss_sums"][0][1],
                          loss_record(eda.probability_of_observed_mark(z, pi, R))["loss_sums"][0][1])
    tiny = [np.array([0, 1, 0]), np.array([1, 1])]
    tiny_pi, tiny_R = eda.priors(tiny)
    expected_pi = (np.array([2/3, 4/3]+[0]*14)+1/16)/3
    table = np.zeros((16, 16)); table[0, 1] = .5; table[1, 0] = .5; table[1, 1] = 1
    require_close(tiny_pi, expected_pi)
    require_close(tiny_R, (table+expected_pi)/(table.sum(1)[:, None]+1))
    assert loss_record(np.empty((0, 5)))["counts"] == [0, 0, 0]
    print("PRIOR02 self_test passed: all controls online/prefix/reset, fitting permutation invariance, exact prior weighting, identical-prior parity")


def run(root, out):
    previous = root/"research/iterations/2026-10-01-eda"
    old_inputs_path = previous/"quant-mark-inputs.json"
    old_results_path = previous/"quant-mark-result.json"
    old_inputs, old_result = [json.loads(p.read_text()) for p in (old_inputs_path, old_results_path)]
    old_code_entry = [f for f in old_inputs["files"] if Path(f["path"]).name == "quant_mark_eda.py"]
    if len(old_code_entry) != 1 or eda.digest(OLD/"quant_mark_eda.py") != old_code_entry[0]["sha256"]:
        raise ValueError("previous evaluated code identity changed")
    selected = old_inputs["selected_before_feature_reads"]
    old_by_chain = {c["chain"]: c for c in old_result["chains"]}
    if len(selected) != 256 or len({c["chain"] for c in selected}) != 256 or set(old_by_chain) != {c["chain"] for c in selected}:
        raise ValueError("previous selected chain identity mismatch")
    for s in selected:
        c = old_by_chain[s["chain"]]
        if any(c[k] != s[k] for k in ("chain", "row_group", "class", "half")) or c["fit_half"] != 1-s["half"]:
            raise ValueError("previous half/class/row identity mismatch")
    source = root/"datasets/train.parquet"
    before = fingerprint(source); footer = footer_hash(source)
    if any(old_inputs["source"][k] != v for k, v in {**before, **footer}.items()):
        raise ValueError("original source identity changed")
    paths = [Path(__file__), HERE/"prior-plan.json", HERE/"contract.json", OLD/"quant_mark_eda.py",
             old_inputs_path, old_results_path, previous/"quant-mark-plan.json"]
    inventory = {"id": "PRIOR02", "selected": selected,
        "runtime_checkout_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "previous_recorded_remote_base_head": old_inputs["git_commit"],
        "provenance_note": "A checkout HEAD alone is not execution-code identity. Exact code and plan SHA256 identify both runs; old remote base HEAD did not contain SCP-staged QMARK code.",
        "files": [{"path": str(p.resolve()), "sha256": eda.digest(p)} for p in paths],
        "source": {"path": str(source.resolve()), **before, **footer, "whole_file_hashed": False}}
    eda.save_new(out/"prior-inputs.json", inventory)
    centers = {h: np.array(old_result["fitted"][str(h)]["centers"]) for h in (0, 1)}
    frozen = {h: (np.array(old_result["fitted"][str(h)]["pi"]), np.array(old_result["fitted"][str(h)]["R"])) for h in (0, 1)}
    pf = pq.ParquetFile(source); records = []
    columns = [f"i0_dv{j}" for j in range(4)]
    for s in selected:
        table = pf.read_row_group(s["row_group"], columns=columns, use_threads=False)
        raw = np.column_stack([table.column(c).to_numpy() for c in columns]).astype(np.float32)
        if raw.shape != (20000, 4):
            raise ValueError(f"raw shape {raw.shape}")
        ids, counts = eda.detect(raw); values = raw[ids[:4096]]
        old = old_by_chain[s["chain"]]
        marks = eda.encode(values, centers[1-s["half"]])
        if counts != old["detector"] or hashlib.sha256(marks.tobytes()).hexdigest() != old["encoded_mark_sha256"]:
            raise ValueError(f"detector/mark reproduction failed for chain {s['chain']}")
        np.testing.assert_array_equal(np.bincount(marks, minlength=16), old["conditions"]["original"]["counts"])
        records.append(dict(s, eval_marks=marks, fit_marks=eda.encode(values, centers[s["half"]])))
    matched = {}
    for h in (0, 1):
        fitting = [c for c in records if c["half"] == h]
        pi, R = frozen[h]
        check_pi, check_R = eda.priors([c["fit_marks"] for c in fitting])
        np.testing.assert_allclose(check_pi, pi, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(check_R, R, atol=1e-14, rtol=1e-12)
        for seed in SEEDS:
            permuted = [eda.permute(c["fit_marks"], "block128", seed, c["chain"]) for c in fitting]
            null_pi, null_R = eda.priors(permuted)
            np.testing.assert_allclose(null_pi, pi, atol=1e-14, rtol=1e-12)
            matched[h, seed] = null_R
    chains = []
    for c in records:
        h = 1-c["half"]; pi, R = frozen[h]; m = c["eval_marks"]
        p = original_probabilities(m, pi, R)
        entry = {k: c[k] for k in ("chain", "row_group", "class", "half")}
        entry.update(fit_half=h, encoded_mark_sha256=hashlib.sha256(m.tobytes()).hexdigest(),
                     conditions={"original": loss_record(p)})
        old = old_by_chain[c["chain"]]
        require_close(np.array(entry["conditions"]["original"]["loss_sums"])[:, :3], old_losses(old["conditions"]["original"]))
        for cut in (1, 16, 128):
            require_close(original_probabilities(m[:cut], pi, R), p[:cut])
        for seed in SEEDS:
            key = f"block128_{seed}"
            pm = eda.permute(m, "block128", seed, c["chain"])
            pp = block_probabilities(pm, pi, R, matched[h, seed])
            entry["conditions"][key] = loss_record(pp)
            require_close(np.array(entry["conditions"][key]["loss_sums"])[:, :2], old_losses(old["conditions"][key])[:, 1:])
            for cut in (1, 16, 128):
                require_close(block_probabilities(pm[:cut], pi, R, matched[h, seed]), pp[:cut])
        chains.append(entry)
    if fingerprint(source) != before:
        raise RuntimeError("source changed during read")
    summary = summarize(chains)
    return {"status": "complete", "id": "PRIOR02", "chains": chains,
        "condition_models": {"original": ORIGINAL_MODELS, "block128": BLOCK_MODELS},
        "slice_order": CUTS, "matched_R": {f"fit{h}_seed{seed}": matched[h, seed].tolist() for h in (0, 1) for seed in SEEDS},
        "summary": summary, "checks": {"source_identity": True, "mark_hashes_and_counts": True,
            "original_pi_R": True, "original_and_old_block_losses": True, "prefixes": True},
        "inventory_sha256": eda.digest(out/"prior-inputs.json"), "targets_read": False, "WP_queries": 0,
        "uncertainty": "Fixed nuisance fits and surrogates; paired whole-chain half/class bootstrap, no selection/fit/test-transfer correction"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE.parents[2])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    root = args.root.resolve(); out = root/"research/iterations/2026-10-01-quant-controls"
    for filename in ("prior-inputs.json", "prior-result.json"):
        if (out/filename).exists():
            raise FileExistsError(out/filename)
    started = time.monotonic()
    def alarm(signum, frame):
        raise TimeoutError(f"signal {signum}")
    signal.signal(signal.SIGTERM, alarm); signal.signal(signal.SIGALRM, alarm); signal.alarm(870)
    try:
        result = run(root, out)
    except Exception as exc:
        result = {"id": "PRIOR02", "status": "timeout" if isinstance(exc, TimeoutError) else "invalid_execution",
                  "error": f"{type(exc).__name__}: {exc}", "targets_read": False, "WP_queries": 0}
    finally:
        signal.alarm(0)
    result["elapsed_seconds"] = time.monotonic()-started
    eda.save_new(out/"prior-result.json", result)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "primary": result.get("summary", {}).get("pooled", {}).get("slices", {}).get("all", {}).get("contrasts", {}).get("block128_mean_old_minus_matched"),
                      "error": result.get("error")}, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
