"""Q-DEEP-01: fixed train-feature geometry diagnostic; never reads labels/predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def exclusive(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def weighted_summary(error, weight):
    weight = weight / weight.sum()
    order = np.argsort(error)
    c = np.cumsum(weight[order])
    q = lambda p: float(error[order[min(np.searchsorted(c, p), len(order) - 1)]])
    return {"n": len(error), "coverage_01": float(weight[error <= .1].sum()),
            "median": q(.5), "p90": q(.9)}


def errors(unit, atoms):
    """Best one/two-atom projection; coefficients unconstrained, singular pairs skipped."""
    dot = unit @ atoms.T
    best = np.max(dot * dot, axis=1)
    one = np.sqrt(np.maximum(1 - np.minimum(best, 1), 0))
    for i in range(len(atoms)):
        for j in range(i):
            c = float(atoms[i] @ atoms[j])
            denominator = 1 - c * c
            if denominator <= 1e-10:
                continue  # a rank-one pair is already in the best-one result
            proj = (dot[:, i]**2 + dot[:, j]**2 - 2*c*dot[:, i]*dot[:, j])/denominator
            best = np.maximum(best, np.clip(proj, 0, 1))
    return one, np.sqrt(np.maximum(1 - np.minimum(best, 1), 0))


def fit_lines(delta, weights):
    scale = np.maximum(np.sqrt(np.average(delta * delta, axis=0, weights=weights)), 1e-6)
    z = delta / scale
    unit = z / np.linalg.norm(z, axis=1)[:, None]
    covariance = (z * weights[:, None]).T @ z
    _, vectors = np.linalg.eigh(covariance)
    pca = vectors[:, ::-1].T.copy()
    atoms = pca.copy()
    for _ in range(20):
        groups = np.argmax(np.abs(unit @ atoms.T), axis=1)
        for j in range(11):
            selected = groups == j
            if np.any(selected):
                u = unit[selected]
                gram = (u * weights[selected, None]).T @ u
                _, v = np.linalg.eigh(gram)
                atoms[j] = v[:, -1]
    return scale, atoms, pca


def self_test():
    rng = np.random.default_rng(72)
    atoms = np.eye(11)
    x = np.zeros((100, 11))
    x[:, 2] = rng.normal(size=100)
    x[:, 9] = rng.normal(size=100)
    x /= np.linalg.norm(x, axis=1)[:, None]
    one, two = errors(x, atoms)
    assert np.max(two) < 3e-8
    assert np.median(one) > .1
    repeated = np.array([atoms[2], atoms[2], atoms[9]])
    assert np.max(errors(x, repeated)[1]) < 3e-8
    y = rng.normal(size=(30, 11)); y /= np.linalg.norm(y, axis=1)[:, None]
    expected = np.sqrt(np.maximum(1-np.sort(y*y, axis=1)[:, -2:].sum(1), 0))
    np.testing.assert_allclose(errors(y, atoms)[1], expected, atol=3e-8)
    a = weighted_summary(np.array([0., .2, .3]), np.array([.5, .25, .25]))
    assert a["coverage_01"] == .5
    print("self_test passed: exact pairs, degenerate pairs, axis control, weighted coverage")


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("quant-inputs.json", "quant-result.json"):
        if (out/name).exists():
            raise FileExistsError(out/name)
    cf_path = root/"cache/chain_folds.npz"
    cls_path = root/"research/artifacts/market-classes-v1.npz"
    cf = np.load(cf_path)
    classes = np.load(cls_path)["train"]
    representatives = {}
    for g, ch in zip(cf["row_group"][cf["split"] == "train"],
                     cf["chain"][cf["split"] == "train"], strict=True):
        if int(g) != 528:
            representatives[int(ch)] = min(int(g), representatives.get(int(ch), int(g)))
    selected = []
    for c in range(4):
        groups = [(hashlib.sha256(f"Q-DEEP-01:{ch}".encode()).hexdigest(), ch, g)
                  for ch, g in representatives.items() if int(classes[g]) == c]
        groups.sort()
        assert len(groups) >= 16
        for i, (_, ch, g) in enumerate(groups[:16]):
            selected.append({"chain": ch, "row_group": g, "class": c, "half": i % 2})
    assert len({s["chain"] for s in selected}) == 64
    source = root/"datasets/train.parquet"
    stat = source.stat()
    footer = hashlib.sha256()
    with source.open("rb") as f:
        f.seek(-8, 2); tail = f.read(8)
        assert tail[4:] == b"PAR1"
        length = struct.unpack("<I", tail[:4])[0]
        f.seek(-8-length, 2)
        remaining = length + 8
        while remaining:
            chunk = f.read(min(2**20, remaining)); assert chunk
            footer.update(chunk); remaining -= len(chunk)
    paths = [cf_path, cls_path, Path(__file__), out/"quant-plan.json"]
    exclusive(out/"quant-inputs.json", {"selected_before_feature_reads": selected,
        "files": [{"path": str(p.resolve()), "sha256": sha(p)} for p in paths],
        "source": {"path": str(source.resolve()), "bytes": stat.st_size,
                   "mtime_ns": stat.st_mtime_ns, "footer_bytes": length+8,
                   "footer_sha256": footer.hexdigest(), "whole_file_hashed": False},
        "data_access": "Only train i0_p, i0_v, i0_dv columns; no outcomes or predictions"})
    columns = [f"i0_p{j}" for j in range(22)] + [f"i0_v{j}" for j in range(22)] + [f"i0_dv{j}" for j in range(4)]
    pf = pq.ParquetFile(source)
    records = {side: {half: [] for half in (0, 1)} for side in (0, 1)}
    counts = []
    for s in selected:
        table = pf.read_row_group(s["row_group"], columns=columns, use_threads=False)
        x = np.column_stack([table.column(c).to_numpy() for c in columns]).astype(np.float64)
        quiet = np.all(x[1:, :22] == x[:-1, :22], axis=1)
        dv = x[:, 44:48]
        trade = np.any(dv[1:] != dv[:-1], axis=1) & (np.ptp(dv[1:], axis=1) > 1e-6)
        both = (np.linalg.norm(np.diff(x[:, 22:33], axis=0), axis=1) > 1e-7) & (np.linalg.norm(np.diff(x[:, 33:44], axis=0), axis=1) > 1e-7)
        for side in (0, 1):
            v = x[:, 22+side*11:33+side*11]
            delta = np.diff(v, axis=0)
            finite = np.isfinite(v).all(1)
            bounded = (np.abs(v) < 5.19).all(1)
            valid = finite[1:] & finite[:-1] & bounded[1:] & bounded[:-1]
            nonzero = np.linalg.norm(delta, axis=1) > 1e-7
            item = dict(s, side=side, total_rows=len(delta), quiet_rows=int(quiet.sum()),
                        zero_or_tiny=int((~nonzero).sum()), extreme_or_nonfinite=int((~valid).sum()),
                        quiet_eligible=int((quiet & valid & nonzero).sum()), both_sides_changed=int((quiet & valid & nonzero & both).sum()))
            for isquiet, cap in ((True, 256), (False, 64)):
                ids = np.flatnonzero(valid & nonzero & (quiet == isquiet))
                if len(ids):
                    ids = ids[np.linspace(0, len(ids)-1, min(cap, len(ids))).astype(int)]
                    records[side][s["half"]].append({"d": delta[ids], "prev": np.linalg.norm(v[:-1][ids], axis=1),
                        "trade": trade[ids], "class": np.full(len(ids), s["class"]),
                        "chain": np.full(len(ids), s["chain"]), "quiet": np.full(len(ids), isquiet),
                        "weight": np.full(len(ids), 1./len(ids))})
            counts.append(item)
    packed = {side: {half: {k: np.concatenate([r[k] for r in records[side][half]])
               for k in ("d", "prev", "trade", "class", "chain", "quiet", "weight")}
               for half in (0, 1)} for side in (0, 1)}
    result = {"counts": counts, "fits": [], "stability": [], "WP_queries": 0, "labels_read": False}
    fits = {}
    rng = np.random.default_rng(20261001)
    random_basis = np.linalg.qr(rng.normal(size=(11, 11)))[0].T
    gates = []
    for side in (0, 1):
        for half in (0, 1):
            a = packed[side][half]; keep = a["quiet"]
            if np.unique(a["chain"][keep]).size < 24:
                raise RuntimeError("insufficient diagnostic chains with quiet increments")
            scale, atoms, pca = fit_lines(a["d"][keep], a["weight"][keep])
            fits[side, half] = (scale, atoms)
            state_median = float(np.median(a["prev"][keep]))
            b = packed[side][1-half]
            z = b["d"]/scale; unit = z / np.linalg.norm(z, axis=1)[:, None]
            entry = {"side": side, "fit_half": half, "eval_half": 1-half, "scale": scale.tolist(),
                     "atoms": atoms.tolist(), "condition": float(np.linalg.cond(atoms)),
                     "rank": int(np.linalg.matrix_rank(atoms)), "controls": {}}
            for name, basis in (("learned", atoms), ("raw", np.eye(11)), ("pca", pca), ("random", random_basis)):
                one, two = errors(unit, basis)
                metrics = {}
                slices = {"quiet": b["quiet"], "price_code_change": ~b["quiet"],
                          "trade": b["quiet"] & b["trade"], "nontrade": b["quiet"] & ~b["trade"],
                          "state_low": b["quiet"] & (b["prev"] <= state_median),
                          "state_high": b["quiet"] & (b["prev"] > state_median)}
                slices.update({f"class_{c}": b["quiet"] & (b["class"] == c) for c in range(4)})
                for label, mask in slices.items():
                    if mask.any():
                        # Weights make each sequence equal for full quiet sample; sliced weights retain its share.
                        metrics[label] = {"one": weighted_summary(one[mask], b["weight"][mask]),
                                          "two": weighted_summary(two[mask], b["weight"][mask]),
                                          "sampled_event_two": weighted_summary(two[mask], np.ones(mask.sum()))}
                entry["controls"][name] = metrics
            cov = {name: entry["controls"][name]["quiet"]["two"]["coverage_01"] for name in ("learned", "raw", "pca")}
            entry["representation_geometry_gate"] = bool(cov["learned"] >= .8 and
                cov["learned"] - max(cov["raw"], cov["pca"]) >= .1 and entry["rank"] == 11 and entry["condition"] < 100)
            gates.append(entry["representation_geometry_gate"])
            result["fits"].append(entry)
        original = []
        for half in (0, 1):
            scale, atoms = fits[side, half]
            mapped = atoms * scale
            original.append(mapped / np.linalg.norm(mapped, axis=1)[:, None])
        similarity = np.abs(original[0] @ original[1].T)
        i, j = linear_sum_assignment(-similarity)
        matched = similarity[i, j]
        stable = int((matched >= .95).sum()) >= 9
        gates.append(stable)
        result["stability"].append({"side": side, "matched_abs_cosines": matched.tolist(),
                                    "at_least_095": int((matched >= .95).sum()), "gate": stable})
    final_stat = source.stat()
    assert (stat.st_size, stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns)
    result["status"] = "geometry_gate_passed" if all(gates) else "screen_failed"
    result["physical_queue_identification"] = "unresolved; not established by sparse reconstruction"
    exclusive(out/"quant-result.json", result)
    print(json.dumps({"status": result["status"], "fits": [{"side": e["side"], "fit_half": e["fit_half"],
         "condition": e["condition"], "coverage": {n:e["controls"][n]["quiet"]["two"]["coverage_01"] for n in e["controls"]}}
         for e in result["fits"]], "stability": result["stability"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main(args.root.resolve(), args.root.resolve()/"research/iterations/2026-10-01-deep")
