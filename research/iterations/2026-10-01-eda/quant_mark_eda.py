"""QMARK-EDA-01: train-only causal code-mark statistics; no price outcomes or WP."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

KAPPA = 16.0
SEEDS = (20261011, 20261012, 20261013)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def detect(x):
    """Causal immediate-previous-row detector; invalid rows still advance raw state."""
    valid_row = np.isfinite(x).all(1) & (np.abs(x) < 5.19).all(1)
    valid = valid_row[1:] & valid_row[:-1]
    changed = np.any(x[1:] != x[:-1], axis=1)
    nonneutral = np.ptp(x[1:], axis=1) > 1e-6
    found = np.flatnonzero(valid & changed & nonneutral) + 1
    counts = {"rows": len(x), "valid_endpoint_pairs": int(valid.sum()),
              "invalid_extreme_pairs": int((~valid).sum()), "changed_pairs": int(changed.sum()),
              "valid_changed_neutral": int((valid & changed & ~nonneutral).sum()),
              "valid_same_code_holds": int((valid & ~changed).sum()),
              "detected_before_cap": len(found), "detected_after_cap": min(len(found), 4096),
              "detected_fraction": len(found) / max(len(x)-1, 1)}
    if len(found):
        counts.update(first_row=int(found[0]), last_row=int(found[-1]))
    else:
        counts.update(first_row=None, last_row=None)
    gap = np.diff(found)
    counts.update(gap1=int((gap == 1).sum()), gap2_8=int(((gap >= 2) & (gap <= 8)).sum()),
                  gap9plus=int((gap >= 9).sum()))
    if len(found) >= 3:
        counts["return_identical_after_one"] = int(np.all(x[found[2:]] == x[found[:-2]], axis=1).sum())
        counts["return_identical_denominator"] = len(found)-2
    else:
        counts["return_identical_after_one"] = counts["return_identical_denominator"] = 0
    return found, counts


def encode(values, centers):
    bits = values > centers
    return (bits * (2 ** np.arange(4))).sum(1).astype(np.int64)


def weighted_center(sequences):
    values, weights = [], []
    for x in sequences:
        if len(x):
            ids = np.linspace(0, len(x)-1, min(512, len(x))).astype(int)
            values.append(x[ids]); weights.append(np.full(len(ids), 1.0/len(ids)))
    if not values:
        raise ValueError("No fit-side detected marks")
    x, w = np.concatenate(values), np.concatenate(weights)
    center = []
    for j in range(4):
        order = np.argsort(x[:, j], kind="stable")
        at = min(np.searchsorted(np.cumsum(w[order]), .5*w.sum()), len(order)-1)
        center.append(float(x[order[at], j]))
    return np.array(center)


def priors(mark_sequences):
    total = np.zeros(16); transitions = np.zeros((16, 16)); nonempty = 0
    for m in mark_sequences:
        if len(m):
            total += np.bincount(m, minlength=16)/len(m); nonempty += 1
        if len(m) > 1:
            transitions += np.bincount(m[:-1]*16+m[1:], minlength=256).reshape(16, 16)/(len(m)-1)
    pi = (total+1/16)/(nonempty+1)
    R = (transitions+pi[None, :])/(transitions.sum(1)[:, None]+1)
    assert (pi > 0).all() and (R > 0).all()
    np.testing.assert_allclose(pi.sum(), 1)
    np.testing.assert_allclose(R.sum(1), 1)
    return pi, R


def ranks_before(labels):
    """Occurrence rank gives counts strictly before each current observation."""
    counts = np.zeros(len(labels), dtype=np.int64)
    for value in np.unique(labels):
        ids = np.flatnonzero(labels == value)
        counts[ids] = np.arange(len(ids))
    return counts


def probability_of_observed_mark(m, pi, R):
    if not len(m):
        return np.empty((0, 3))
    global_p = pi[m]
    frequency = (ranks_before(m)+KAPPA*pi[m])/(np.arange(len(m))+KAPPA)
    transition = frequency.copy()
    if len(m) > 1:
        previous, current = m[:-1], m[1:]
        transition[1:] = (ranks_before(previous*16+current)+KAPPA*R[previous, current])/(ranks_before(previous)+KAPPA)
    p = np.column_stack((global_p, frequency, transition))
    assert np.isfinite(p).all() and (p > 0).all() and (p <= 1+1e-12).all()
    return p


def online_probabilities(m, pi, R):
    counts = np.zeros(16); tr = np.zeros((16, 16)); context = np.zeros(16)
    predictions = []
    previous = None
    for mark in m:
        freq = (counts+KAPPA*pi)/(counts.sum()+KAPPA)
        trans = pi if previous is None else (tr[previous]+KAPPA*R[previous])/(context[previous]+KAPPA)
        predictions.append(np.stack((pi, freq, trans)))
        if previous is not None:
            tr[previous, mark] += 1; context[previous] += 1
        counts[mark] += 1; previous = mark
    return np.asarray(predictions).reshape(-1, 3, 16)


def mark_stats(m, pi, R):
    p = probability_of_observed_mark(m, pi, R)
    loss = -np.log(p)
    hist = np.bincount(m, minlength=16)
    freq = hist/max(len(m), 1)
    good = freq > 0
    entropy = float(-(freq[good]*np.log(freq[good])).sum())
    mi = same = None
    if len(m) > 1:
        table = np.bincount(m[:-1]*16+m[1:], minlength=256).reshape(16, 16)/(len(m)-1)
        independent = table.sum(1)[:, None]*table.sum(0)[None, :]
        used = table > 0
        mi = float((table[used]*np.log(table[used]/independent[used])).sum())
        same = float(np.mean(m[:-1] == m[1:]))
    return {"n": len(m), "n_first16": min(len(m), 16), "counts": hist.tolist(),
            "entropy": entropy, "lag1_mi": mi, "lag1_same": same,
            "loss_sum_G_F_T": loss.sum(0).tolist(),
            "first16_loss_sum_G_F_T": loss[:16].sum(0).tolist(),
            "after16_loss_sum_G_F_T": loss[16:].sum(0).tolist()}


def permute(m, kind, seed, chain):
    key = f"QMARK-EDA-01:{kind}:{seed}:{chain}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    out = m.copy()
    if kind == "full":
        out = rng.permutation(out)
    else:
        assert kind == "block128"
        for start in range(0, len(m), 128):
            out[start:start+128] = rng.permutation(out[start:start+128])
            np.testing.assert_array_equal(np.bincount(out[start:start+128], minlength=16),
                                          np.bincount(m[start:start+128], minlength=16))
    np.testing.assert_array_equal(np.bincount(out, minlength=16), np.bincount(m, minlength=16))
    return out


def summary(chains, condition):
    chosen = [c["conditions"][condition] for c in chains if c["conditions"][condition]["n"]]
    if not chosen:
        return {"chains": 0}
    n = np.array([c["n"] for c in chosen])
    total = np.array([c["loss_sum_G_F_T"] for c in chosen])
    per_chain = total/n[:, None]
    result = {"chains": len(chosen), "events": int(n.sum()),
              "events_min_median_max": [int(n.min()), float(np.median(n)), int(n.max())],
              "mean_per_chain_logloss_G_F_T": per_chain.mean(0).tolist(),
              "mean_per_chain_perplexity_from_mean_loss_G_F_T": np.exp(per_chain.mean(0)).tolist(),
              "event_weighted_logloss_G_F_T": (total.sum(0)/n.sum()).tolist(),
              "mean_chain_gain_F_minus_T": float((per_chain[:, 1]-per_chain[:, 2]).mean()),
              "event_weighted_gain_F_minus_T": float((total[:, 1]-total[:, 2]).sum()/n.sum()),
              "mean_empirical_entropy": float(np.mean([c["entropy"] for c in chosen])),
              "mean_lag1_MI": float(np.mean([c["lag1_mi"] for c in chosen if c["lag1_mi"] is not None])) if np.any(n > 1) else None,
              "mean_lag1_same": float(np.mean([c["lag1_same"] for c in chosen if c["lag1_same"] is not None])) if np.any(n > 1) else None}
    for label, count, field in (("first16", np.minimum(n, 16), "first16_loss_sum_G_F_T"),
                                ("after16", np.maximum(n-16, 0), "after16_loss_sum_G_F_T")):
        take = count > 0
        if take.any():
            sums = np.array([c[field] for c in chosen])[take]
            result[label] = {"chains": int(take.sum()), "events": int(count[take].sum()),
                "mean_chain_logloss_G_F_T": (sums/count[take, None]).mean(0).tolist(),
                "event_weighted_logloss_G_F_T": (sums.sum(0)/count[take].sum()).tolist()}
    return result


def bootstrap(chains, conditions):
    used = [c for c in chains if c["conditions"]["original"]["n"]]
    n = np.array([c["conditions"]["original"]["n"] for c in used])
    gains = np.array([[c["conditions"][key]["loss_sum_G_F_T"][1]-c["conditions"][key]["loss_sum_G_F_T"][2]
                       for key in conditions] for c in used])
    strata = []
    for half in (0, 1):
        for cls in range(4):
            ix = [i for i, c in enumerate(used) if c["half"] == half and c["class"] == cls]
            if ix:
                strata.append(np.array(ix))
    def statistics(ids):
        g = gains[ids]; events = n[ids]
        chain = (g/events[:, None]).mean(0)
        weighted = g.sum(0)/events.sum()
        return [chain[0], weighted[0], chain[0]-chain[1:4].mean(), chain[0]-chain[4:7].mean()]
    rng = np.random.default_rng(20261021)
    draws = np.array([statistics(np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])) for _ in range(1000)])
    point = statistics(np.arange(len(used)))
    names = ("chain_gain", "event_weighted_gain", "chain_original_minus_full_null", "chain_original_minus_block128_null")
    return {name: {"point": float(point[i]), "ci95": np.quantile(draws[:, i], [.025, .975]).tolist(),
                   "bootstrap_sd": float(draws[:, i].std())} for i, name in enumerate(names)}


def self_test():
    m = np.array([0, 0, 1, 15, 1, 1, 2, 0, 4, 3, 3, 3, 15, 7, 0, 0, 1, 1, 14])
    pi, R = priors([m[:10], m[10:]])
    all_p = online_probabilities(m, pi, R)
    np.testing.assert_allclose(all_p.sum(2), 1)
    assert (all_p > 0).all()
    selected = all_p[np.arange(len(m)), :, m]
    np.testing.assert_allclose(selected, probability_of_observed_mark(m, pi, R))
    other = m.copy(); other[5:] = 9
    np.testing.assert_allclose(online_probabilities(other, pi, R)[:6], all_p[:6])
    np.testing.assert_allclose(online_probabilities(m, pi, R)[0], np.stack([pi]*3))
    for cut in (1, 16):
        np.testing.assert_allclose(probability_of_observed_mark(m[:cut], pi, R), selected[:cut])
    for kind in ("full", "block128"):
        permute(np.tile(m, 20), kind, SEEDS[0], 7)
    raw = np.array([[0,0,0,0], [1,0,0,0], [1,0,0,0], [0,0,0,0], [1,0,0,0],
                    [np.nan,0,0,0], [2,0,0,0], [3,0,0,0]], dtype=np.float32)
    ids, _ = detect(raw)
    np.testing.assert_array_equal(ids, [1,4,7])
    for cut in range(1, len(raw)+1):
        np.testing.assert_array_equal(detect(raw[:cut])[0], ids[ids < cut])
    np.testing.assert_array_equal(encode(np.array([[0,0,0,0], [1,0,1,0]]), np.zeros(4)), [0,5])
    print("self_test passed: online/vectorized parity, current/future independence, reset, smoothing, prefixes, permutations, detector, ties")


def main(root):
    started = time.monotonic()
    out = root/"research/iterations/2026-10-01-eda"
    for name in ("quant-mark-inputs.json", "quant-mark-result.json"):
        if (out/name).exists():
            raise FileExistsError(out/name)
    cf_path = root/"cache/chain_folds.npz"
    class_path = root/"research/artifacts/market-classes-v1.npz"
    prior_path = root/"research/iterations/2026-10-01-deep/quant-inputs.json"
    cf = np.load(cf_path); classes = np.load(class_path)["train"]
    excluded = set(map(int, cf["chain"][cf["split"] == "valid"]))
    excluded.update(map(int, cf["chain"][(cf["split"] == "train") & (cf["row_group"] == 528)]))
    excluded.update(s["chain"] for s in json.loads(prior_path.read_text())["selected_before_feature_reads"])
    representatives = {}
    for g, ch in zip(cf["row_group"][cf["split"] == "train"], cf["chain"][cf["split"] == "train"], strict=True):
        if int(ch) not in excluded and int(g) != 528:
            representatives[int(ch)] = min(int(g), representatives.get(int(ch), int(g)))
    selected = []
    for cls in range(4):
        choices = [(hashlib.sha256(f"QMARK-EDA-01:{ch}".encode()).hexdigest(), ch, g)
                   for ch,g in representatives.items() if int(classes[g]) == cls]
        choices.sort(); assert len(choices) >= 64
        for i, (_, ch, g) in enumerate(choices[:64]):
            selected.append({"chain": ch, "row_group": g, "class": cls, "half": i % 2})
    assert len({s["chain"] for s in selected}) == 256
    source = root/"datasets/train.parquet"; stat = source.stat(); footer = hashlib.sha256()
    with source.open("rb") as f:
        f.seek(-8, 2); tail = f.read(8); assert tail[4:] == b"PAR1"
        length = struct.unpack("<I", tail[:4])[0]; f.seek(-8-length, 2); remaining = length+8
        while remaining:
            data = f.read(min(2**20, remaining)); assert data
            footer.update(data); remaining -= len(data)
    paths = [cf_path, class_path, prior_path, Path(__file__), out/"quant-mark-plan.json"]
    save_new(out/"quant-mark-inputs.json", {"selected_before_feature_reads": selected,
        "git_commit": subprocess.check_output(["git","rev-parse","HEAD"], cwd=root, text=True).strip(),
        "files": [{"path": str(p.resolve()), "sha256": digest(p)} for p in paths],
        "source": {"path": str(source.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                   "inode": stat.st_ino, "device": stat.st_dev,
                   "footer_bytes": length+8, "footer_sha256": footer.hexdigest(), "whole_file_hashed": False}})
    pf = pq.ParquetFile(source); records = []
    columns = [f"i0_dv{j}" for j in range(4)]
    for s in selected:
        table = pf.read_row_group(s["row_group"], columns=columns, use_threads=False)
        x = np.column_stack([table.column(c).to_numpy() for c in columns]).astype(np.float32)
        assert x.shape == (20000, 4), (s["row_group"], x.shape)
        ids, counts = detect(x)
        for cut in (99,5000,10000,19999):
            np.testing.assert_array_equal(detect(x[:cut])[0], ids[ids < cut])
        kept = ids[:4096]
        records.append(dict(s, values=x[kept].copy(), detector=counts))
    centers = {half: weighted_center([r["values"] for r in records if r["half"] == half]) for half in (0,1)}
    fitted = {}
    for half in (0,1):
        fitted[half] = priors([encode(r["values"], centers[half]) for r in records if r["half"] == half])
    result = {"id": "QMARK-EDA-01", "fitted": {str(h): {"centers": centers[h].tolist(), "pi": fitted[h][0].tolist(),
              "R": fitted[h][1].tolist()} for h in (0,1)}, "chains": [], "checks": {"raw_prefix": True,
              "mark_prefix": True, "permutations_preserve_counts": True}, "WP_queries": 0, "targets_read": False}
    conditions = ["original"]+[f"{kind}_{seed}" for kind in ("full","block128") for seed in SEEDS]
    for r in records:
        fit_half = 1-r["half"]; pi,R = fitted[fit_half]; center = centers[fit_half]
        m = encode(r["values"], center)
        entry = {k:r[k] for k in ("chain","row_group","class","half","detector")}
        entry.update(fit_half=fit_half, encoded_mark_sha256=hashlib.sha256(m.tobytes()).hexdigest(),
                     centers_disagree_marks=int((m != encode(r["values"], centers[r["half"]])).sum()),
                     bit_tie_counts=np.sum(r["values"] == center, axis=0).astype(int).tolist(), conditions={})
        probabilities = probability_of_observed_mark(m, pi, R)
        for cut in (1,16,128):
            cut = min(cut,len(m))
            np.testing.assert_allclose(probability_of_observed_mark(m[:cut], pi, R), probabilities[:cut])
        entry["conditions"]["original"] = mark_stats(m, pi, R)
        for kind in ("full","block128"):
            for seed in SEEDS:
                entry["conditions"][f"{kind}_{seed}"] = mark_stats(permute(m,kind,seed,r["chain"]), pi, R)
        result["chains"].append(entry)
    result["summary"] = {key: summary(result["chains"],key) for key in conditions}
    result["by_half"] = {str(h): {key: summary([c for c in result["chains"] if c["half"] == h],key)
                                   for key in conditions} for h in (0,1)}
    result["by_class"] = {str(c): {key: summary([r for r in result["chains"] if r["class"] == c],key)
                                    for key in conditions} for c in range(4)}
    result["bootstrap"] = bootstrap(result["chains"],conditions)
    original = result["summary"]["original"]
    gate = original["mean_chain_gain_F_minus_T"] >= .01*original["mean_per_chain_logloss_G_F_T"][1]
    gate &= all(result["by_half"][str(h)]["original"]["mean_chain_gain_F_minus_T"] > 0 for h in (0,1))
    gate &= result["bootstrap"]["chain_gain"]["ci95"][0] > 0
    gate &= result["bootstrap"]["chain_original_minus_block128_null"]["ci95"][0] > 0
    result["status"] = "nuisance_feature_study_nominated" if gate else "screen_failed"
    result["uncertainty"] = "Descriptive conditional chain sampling with fixed priors/centers/nulls; no nuisance-fit, selection or test-shift uncertainty"
    result["elapsed_seconds"] = time.monotonic()-started
    final_stat = source.stat()
    assert (stat.st_size,stat.st_mtime_ns,stat.st_ino,stat.st_dev) == (final_stat.st_size,final_stat.st_mtime_ns,final_stat.st_ino,final_stat.st_dev)
    save_new(out/"quant-mark-result.json",result)
    print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"],
                       "original":original,"bootstrap":result["bootstrap"]},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main(args.root.resolve())
